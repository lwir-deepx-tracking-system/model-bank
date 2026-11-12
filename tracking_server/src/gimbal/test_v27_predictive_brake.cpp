// Camera control regression using only an in-memory device.
#include <algorithm>
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

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

struct Command {
    int x, y, speed;
};

class FakeDevice final : public IGimbalDevice {
    std::mutex mutex_;
    std::condition_variable signal_;
    std::vector<Command> commands_;
    bool pause_next_ = false, paused_ = false, released_ = false;
public:
    void move(int x, int y) override { move_with_speed(x, y, 0); }
    void move_with_speed(int x, int y, int speed) override {
        std::unique_lock<std::mutex> lock(mutex_);
        commands_.push_back({x, y, speed});
        if ((x || y) && pause_next_) {
            pause_next_ = false;
            paused_ = true;
            signal_.notify_all();
            signal_.wait_for(lock, 500ms, [this] { return released_; });
            paused_ = false;
        }
    }
    void pause_next_motion() {
        std::lock_guard<std::mutex> lock(mutex_);
        pause_next_ = true;
        released_ = false;
    }
    bool wait_paused() {
        std::unique_lock<std::mutex> lock(mutex_);
        return signal_.wait_for(lock, 100ms, [this] { return paused_; });
    }
    void release_motion() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        signal_.notify_all();
    }
    void stop() override { move_with_speed(0, 0, 0); }
    void home() override {}
    std::vector<Command> commands() {
        std::lock_guard<std::mutex> lock(mutex_);
        return commands_;
    }
};

ControlConfig options(int speed_min)
{
    ControlConfig config;
    config.enabled = true;
    auto& a = config.automatic;
    a.controller = AutoControllerType::PD;
    a.pid_x.kp = a.pid_y.kp = 30;
    a.pid_x.ki = a.pid_y.ki = 0;
    a.pid_x.kd = a.pid_y.kd = 6;
    a.deadband_x = a.deadband_y = 0.06;
    a.speed_min = speed_min;
    a.speed_max = 10;
    a.period_ms = 5;
    return config;
}

SmoothingConfig smoothing(bool enabled)
{
    SmoothingConfig config;
    config.predictive_brake = enabled;
    return config;
}

struct Rig {
    RuntimeControl runtime;
    FakeDevice* recorder = nullptr;
    GimbalCommandExecutor executor;
    std::uint64_t frame_id = 0;
    Clock::time_point publication = Clock::now();

    Rig(bool enabled, int speed_min = 1)
        : executor(options(speed_min), device(), runtime, smoothing(enabled))
    {
        executor.connected();
        frame(center);
        require(executor.select(7) == ControlResult::Ok, "selection failed");
    }

    std::unique_ptr<FakeDevice> device() {
        auto device = std::make_unique<FakeDevice>();
        recorder = device.get();
        return device;
    }

    void frame(std::optional<double> x, double y = 240, bool healthy = true, int step = 1,
               bool forecast = false, std::chrono::milliseconds interval = 45ms) {
        // Capture time has a fixed cadence. Allow the asynchronous worker to
        // observe each publication before replacing its latest-result slot.
        publication += interval;
        std::this_thread::sleep_until(publication);
        auto context = std::make_shared<FrameContext>();
        frame_id += step;
        context->metadata.frame_id = frame_id;
        context->metadata.captured_at = publication - 20ms;
        context->image = cv::Mat(480, 640, CV_16UC1);
        auto result = std::make_shared<TrackingResult>();
        result->frame = context;
        if (x) {
            Track track;
            track.track_id = 7;
            track.x = static_cast<float>(*x - 20);
            track.y = static_cast<float>(y - 40);
            track.width = 40; track.height = 80; track.confidence = 0.9f;
            result->tracks.push_back(track);
        }
        if (forecast) result->lead = {7, 1, 0, executor.snapshot().command_generation};
        executor.publish(result, healthy);
        std::this_thread::sleep_for(25ms);
    }

    Command last() const { return recorder->commands().back(); }

    std::size_t stops() const {
        const auto commands = recorder->commands();
        return static_cast<std::size_t>(std::count_if(commands.begin(), commands.end(),
            [](const auto& command) { return command.x == 0 && command.y == 0; }));
    }
};

void approaches_stop_early()
{
    for (const int side : {-1, 1}) {
        for (const bool enabled : {false, true}) {
            Rig rig(enabled);
            for (const double error : {27.0, 27.0, 23.0}) rig.frame(center + side * error);
            require(rig.last().x == side, "fixture failed to establish horizontal movement");
            const auto before = rig.stops();
            rig.frame(center + side * 21.0);
            require(rig.stops() == before + (enabled ? 1 : 0), "near-center STOP timing mismatch");
            require(rig.last().x == (enabled ? 0 : side), "disabled must follow outside measured band");
            rig.frame(center + side * 18.0);
            require(rig.stops() == before + 1 && rig.last().x == 0,
                    "measured deadband must stop either condition without repeating STOP");
        }
    }
}

void escape_and_bound()
{
    for (const bool outward : {false, true}) {
        Rig rig(true);
        for (const double error : {27.0, 27.0, 23.0, 21.0}) rig.frame(center + error);
        require(rig.last().x == 0, "early STOP required for latch test");
        const auto count = rig.stops();
        if (outward) {
            rig.frame(center + 22.0);
            require(rig.last().x == 1 && rig.stops() == count,
                    "first outward observation must release even inside shell");
            rig.frame(center + 65.0);
            require(rig.last().x == 1, "rapid outward observation must follow");
        } else {
            for (const double error : {20.7, 20.5}) {
                rig.frame(center + error);
                require(rig.last().x == 0 && rig.stops() == count, "hold resubmitted STOP or released early");
            }
            rig.frame(center + 20.3);
            require(rig.last().x == 1 && rig.stops() == count,
                    "same-side hold must release after 100ms of advancing captures");
            for (const double error : {20.1, 19.9}) rig.frame(center + error);
            require(rig.last().x == 1 && rig.stops() == count, "one shell visit must not brake twice");
        }
    }
}

void unsupported_approaches()
{
    for (const int condition : {0, 1, 2, 3, 4, 5}) {
        Rig rig(true, condition == 2 ? 8 : 1);
        const double y = condition == 1 ? 270 : 240;
        for (const double error : {27.0, 27.0, 23.0}) rig.frame(center + error, y);
        const auto before = rig.stops();
        if (condition == 5) rig.executor.publish_unsafe_observation();
        rig.frame(center + 21.0, y, condition != 3, condition == 0 ? 2 : 1, condition == 4,
                  condition == 0 ? 110ms : 45ms);
        require(rig.last().x == 1 && rig.stops() == before,
                "stale interval, vertical demand, high SPD, invalid observation, or LEAD applied predictive STOP");
        if (condition == 1) require(rig.last().y != 0, "two-axis fixture has no Y");
        if (condition == 2) require(rig.last().speed >= 8, "high SPD fixture is below range");
    }
}

void bounded_gaps_and_mailbox_loss()
{
    {
        Rig rig(true);
        for (const double error : {27.0, 27.0, 23.0}) rig.frame(center + error);
        rig.frame(center + 21.0, 240, true, 2);
        require(rig.last().x == 0, "increasing IDs with a bounded capture interval must qualify");
    }

    for (const bool missing : {false, true}) {
        Rig rig(true);
        rig.frame(center + 38.0);
        rig.frame(center + 38.0);
        const auto before = rig.stops();
        rig.recorder->pause_next_motion();
        rig.frame(center + 27.0);
        const bool paused = rig.recorder->wait_paused();
        if (paused) {
            rig.frame(missing ? std::nullopt : std::optional<double>(center + 24.0));
            rig.frame(center + 21.0);
        }
        rig.recorder->release_motion();
        require(paused, "fixture did not suspend the executor for mailbox replacement");
        std::this_thread::sleep_for(30ms);
        require(rig.stops() == before + (missing ? 0 : 1),
                "overwritten valid or missing observation did not preserve the right braking evidence");
        require(rig.last().x == (missing ? 1 : 0), "explicit loss was hidden by mailbox replacement");
    }
}

void derivative_reversal()
{
    for (const bool enabled : {false, true}) {
        Rig rig(enabled);
        for (const double error : {60.0, 60.0, 50.0, 40.0}) rig.frame(center + error);
        require(rig.last().x == 1, "derivative fixture reversed before reaching the shell");
        const auto before = rig.stops();
        rig.frame(center + 21.0);
        require(rig.stops() == before + 1, "both paths must STOP at the derivative-driven reversal");
        require(rig.last().x == (enabled ? 0 : -1),
                "predictive braking must hold instead of applying a premature reverse move");
        if (enabled) {
            rig.frame(center + 20.5);
            require(rig.last().x == 0 && rig.stops() == before + 1,
                    "a closing frame must preserve the derivative-reversal hold");
            rig.frame(center + 22.0);
            require(rig.last().x != 0 && rig.stops() == before + 1,
                    "the first outward sample must release the derivative-reversal hold");
        }
    }
}

void authority_and_loss()
{
    for (const bool reset : {false, true}) {
        Rig rig(true);
        for (const double error : {27.0, 27.0, 23.0, 21.0}) rig.frame(center + error);
        require(rig.last().x == 0, "early STOP required for authority test");
        if (reset) rig.executor.halt();
        else rig.frame(std::nullopt);
        require(rig.last().x == 0, "missing/invalidated input must remain stopped");
        require(reset ? !rig.executor.snapshot().automatic : rig.executor.snapshot().lost,
                "loss/authority state did not follow baseline");
        if (reset) require(rig.executor.select(7) == ControlResult::Ok, "reselection failed");
        rig.frame(center + 20.5);
        require(rig.last().x == 1, "old latch survived loss or reselection");
    }
}
}

int main()
{
    try {
        approaches_stop_early();
        escape_and_bound();
        unsupported_approaches();
        bounded_gaps_and_mailbox_loss();
        derivative_reversal();
        authority_and_loss();
        std::puts("PASS predictive braking, baseline parity, release, and safety gates (FakeDevice)");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
