// No-device regression. Timings start at executor publication and end at FakeDevice calls.
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>
#include "control/gimbal_command_executor.hpp"

namespace {
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
constexpr int center = 320;
constexpr int right = 512;
constexpr int left = 128;
constexpr int speed_cap = 10;
constexpr auto response_budget = 150ms;

void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

struct Command {
    int x, y, speed;
    bool stop;
    Clock::time_point at;
};

class FakeDevice final : public IGimbalDevice {
    std::mutex mutex_;
    std::condition_variable changed_;
    std::vector<Command> commands_;

    void record(int x, int y, int speed, bool stop)
    {
        const auto at = Clock::now();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            commands_.push_back({x, y, speed, stop, at});
        }
        changed_.notify_all();
    }

public:
    void move(int x, int y) override { record(x, y, 0, false); }
    void move_with_speed(int x, int y, int speed) override { record(x, y, speed, false); }
    void stop() override { record(0, 0, 0, true); }
    void home() override {}

    std::size_t mark()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return commands_.size();
    }

    // Direction 0 means STOP.
    Command wait(std::size_t first, Clock::time_point deadline, int direction,
                 const char* failure)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        std::optional<Command> found;
        changed_.wait_until(lock, deadline, [&] {
            for (; first < commands_.size(); ++first) {
                const auto& command = commands_[first];
                if (command.at > deadline) return false;
                const bool match = direction == 0 ? command.stop :
                    !command.stop && command.x == direction && command.y == 0 && command.speed >= 1;
                if (match) { found = command; return true; }
            }
            return false;
        });
        require(found.has_value(), failure);
        return *found;
    }

    std::vector<Command> since(std::size_t first)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return {commands_.begin() + first, commands_.end()};
    }
};

ControlConfig config()
{
    ControlConfig config;
    config.enabled = true;
    auto& a = config.automatic;
    a.controller = AutoControllerType::PD;
    a.pid_x.kp = a.pid_y.kp = 30;
    a.pid_x.ki = a.pid_y.ki = 0;
    a.pid_x.kd = a.pid_y.kd = 6;
    a.deadband_x = a.deadband_y = .06;
    a.speed_max = speed_cap;
    return config;
}

struct Rig {
    RuntimeControl runtime;
    FakeDevice* device = nullptr;
    GimbalCommandExecutor executor;
    std::uint64_t frame_id = 0;

    Rig() : executor(config(), make_device(), runtime) { executor.connected(); }

    std::unique_ptr<FakeDevice> make_device()
    {
        auto fake = std::make_unique<FakeDevice>();
        device = fake.get();
        return fake;
    }

    Clock::time_point publish(std::optional<int> x, TrackLead lead = {})
    {
        auto frame = std::make_shared<FrameContext>();
        frame->metadata.frame_id = ++frame_id;
        frame->image = cv::Mat(480, 640, CV_16UC1);
        auto result = std::make_shared<TrackingResult>();
        result->frame = frame;
        result->aim_y = .08;
        if (x) {
            Track track;
            track.track_id = 7;
            track.x = float(*x - 20);
            track.y = 240 - .08f * 80;
            track.width = 40;
            track.height = 80;
            track.confidence = .9f;
            result->tracks.push_back(track);
        }
        result->lead = lead;
        const auto published = Clock::now();
        frame->metadata.captured_at = published;
        executor.publish(result);
        return published;
    }

    std::uint64_t select()
    {
        publish(center);
        require(executor.select(7) == ControlResult::Ok, "select observed ID 7");
        return executor.snapshot().command_generation;
    }

    void hold(int x, TrackLead lead = {})
    {
        for (int frame = 0; frame < 5; ++frame) {
            publish(x, lead);
            std::this_thread::sleep_for(40ms);
        }
    }
};

double elapsed(Clock::time_point start, const Command& command)
{
    return std::chrono::duration<double, std::milli>(command.at - start).count();
}

enum class LeadMode { off, opposing, stale };

void fast_motion(LeadMode mode, const char* name)
{
    Rig rig;
    const auto first_epoch = rig.select();
    const auto current_epoch = mode == LeadMode::stale ? rig.select() : first_epoch;
    require(mode != LeadMode::stale || current_epoch != first_epoch, "stale epoch fixture");
    const auto lead = [&](double dx) {
        return mode == LeadMode::off ? TrackLead{} : TrackLead{7, dx, 0, first_epoch};
    };
    const auto before = rig.device->mark();
    rig.publish(center, lead(60));
    std::this_thread::sleep_for(50ms);
    require(rig.device->since(before).empty(), "centered target must stay still");

    const auto right_at = rig.publish(right, lead(-60));
    const auto right_command = rig.device->wait(before, right_at + response_budget, 1, "right jump not followed");
    rig.hold(right, lead(-60));
    const auto reverse = rig.device->mark();
    const auto left_at = rig.publish(left, lead(60));
    const auto reversal_stop = rig.device->wait(reverse, left_at + response_budget, 0, "reversal STOP missing");
    const auto left_command = rig.device->wait(reverse, left_at + response_budget, -1, "left jump not followed");
    require(reversal_stop.at <= left_command.at, "reversal must STOP before the first left command");
    for (const auto& command : rig.device->since(reverse))
        require(command.stop || (command.x == -1 && command.y == 0), "old direction emitted after the left jump");

    // At the large step both forecasts can hit the same cap. Use a 64 px offset
    // as a second witness that a current opposing lead acts and a stale one does not.
    const auto before_center = rig.device->mark();
    const auto center_at = rig.publish(center);
    rig.device->wait(before_center, center_at + response_budget, 0, "centering STOP missing");
    const auto witness = rig.device->mark();
    rig.hold(center + 64, lead(-60));
    int settled_speed = 0;
    for (const auto& command : rig.device->since(witness))
        if (!command.stop) settled_speed = command.speed;
    require(settled_speed == (mode == LeadMode::opposing ? 1 : 6), "opposing/stale lead speed witness");
    for (const auto& command : rig.device->since(before))
        require(command.stop || (command.speed >= 1 && command.speed <= speed_cap), "automatic speed cap");
    std::printf("PASS %-13s right=%6.2f ms reversal_STOP=%6.2f ms left=%6.2f ms witness_SPD=%d\n",
                name, elapsed(right_at, right_command), elapsed(left_at, reversal_stop),
                elapsed(left_at, left_command), settled_speed);
}

void one_frame_gap()
{
    constexpr auto gap_frame_interval = 60ms;
    Rig rig;
    const auto epoch = rig.select();
    auto mark = rig.device->mark();
    auto at = rig.publish(right);
    rig.device->wait(mark, at + response_budget, 1, "gap fixture did not start moving");
    mark = rig.device->mark();
    const auto gap = rig.publish(std::nullopt, {7, 60, 0, epoch});
    const auto stop = rig.device->wait(mark, gap + gap_frame_interval, 0, "one missing frame did not STOP");
    std::this_thread::sleep_until(gap + gap_frame_interval);
    const auto state = rig.executor.snapshot();
    require(state.lost && state.selected_id == 7 && state.command_generation != epoch,
            "gap must preserve the selected ID and invalidate the old forecast");
    mark = rig.device->mark();
    const auto returned = rig.publish(right, {7, -60, 0, epoch});
    const auto resumed = rig.device->wait(mark, returned + response_budget, 1, "same ID did not resume");
    std::printf("PASS one-frame gap STOP=%6.2f ms resume_after_return=%6.2f ms\n",
                elapsed(gap, stop), elapsed(returned, resumed));
}

void stop_authority(bool disconnect)
{
    Rig rig;
    const auto epoch = rig.select();
    auto mark = rig.device->mark();
    auto at = rig.publish(right, {7, 60, 0, epoch});
    rig.device->wait(mark, at + response_budget, 1, "STOP fixture did not start moving");
    mark = rig.device->mark();
    at = Clock::now();
    if (disconnect) rig.executor.disconnected();
    else rig.executor.halt();
    const auto stop = rig.device->wait(mark, at + response_budget, 0, "halt/disconnect did not STOP");
    mark = rig.device->mark();
    rig.executor.ping();
    rig.hold(left, {7, -60, 0, epoch});
    const auto state = rig.executor.snapshot();
    require(rig.device->since(mark).empty() && !state.automatic && !state.moving && state.selected_id == 0,
            "fresh same-ID observations/old LEAD restarted after halt/disconnect");
    std::printf("PASS %-13s STOP=%6.2f ms no restart\n", disconnect ? "disconnect" : "halt", elapsed(at, stop));
}
} // namespace

int main()
{
    try {
        fast_motion(LeadMode::off, "lead off");
        fast_motion(LeadMode::opposing, "opposing lead");
        fast_motion(LeadMode::stale, "stale lead");
        one_frame_gap();
        stop_authority(false);
        stop_authority(true);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
    std::puts("Times are executor-to-FakeDevice command times. Physical response was not measured.");
}
