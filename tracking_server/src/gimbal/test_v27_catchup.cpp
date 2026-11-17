// Camera control regression: executor timings are measured with a FakeDevice.
#include <algorithm>
#include <chrono>
#include <cmath>
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

void require(bool pass, const char* message)
{
    if (!pass) throw std::runtime_error(message);
}

struct Command {
    int x, y, speed;
    Clock::time_point time;
};

class FakeDevice final : public IGimbalDevice {
    std::mutex mutex_;
    std::vector<Command> commands_;
public:
    void move(int x, int y) override { move_with_speed(x, y, 0); }
    void move_with_speed(int x, int y, int speed) override {
        std::lock_guard<std::mutex> lock(mutex_);
        commands_.push_back({x, y, speed, Clock::now()});
    }
    void stop() override { move_with_speed(0, 0, 0); }
    void home() override {}
    std::vector<Command> commands() {
        std::lock_guard<std::mutex> lock(mutex_);
        return commands_;
    }
};

ControlConfig options(int cap)
{
    ControlConfig config;
    config.enabled = true;
    auto& a = config.automatic;
    a.controller = AutoControllerType::PD;
    a.pid_x.kp = a.pid_y.kp = 30;
    a.pid_x.ki = a.pid_y.ki = 0;
    a.pid_x.kd = a.pid_y.kd = 6;
    a.deadband_x = a.deadband_y = 0.06;
    a.speed_max = cap;
    return config;
}

SmoothingConfig smoothing(int speed, int grace = 100)
{
    SmoothingConfig config;
    config.catchup_speed = speed;
    config.observation_grace_ms = grace;
    return config;
}

Track person(double x, double y = 240)
{
    Track t;
    t.track_id = 7;
    t.x = static_cast<float>(x - 20);
    t.y = static_cast<float>(y - 40);
    t.width = 40; t.height = 80; t.confidence = 0.9f;
    return t;
}

struct Rig {
    RuntimeControl runtime;
    FakeDevice* recorder = nullptr;
    GimbalCommandExecutor executor;
    std::uint64_t frame_id = 0;

    Rig(int speed, int grace = 100, int cap = 100)
        : executor(options(cap), device(), runtime, smoothing(speed, grace))
    {
        executor.connected();
        frame(center);
        require(executor.select(7) == ControlResult::Ok, "selection failed");
    }
    std::unique_ptr<FakeDevice> device() {
        auto result = std::make_unique<FakeDevice>();
        recorder = result.get();
        return result;
    }
    void frame(std::optional<double> x, double y = 240, bool healthy = true) {
        auto context = std::make_shared<FrameContext>();
        context->metadata.frame_id = ++frame_id;
        context->metadata.captured_at = Clock::now();
        context->image = cv::Mat(480, 640, CV_16UC1);
        auto result = std::make_shared<TrackingResult>();
        result->frame = context;
        if (x) result->tracks.push_back(person(*x, y));
        executor.publish(result, healthy);
    }
};

double elapsed(Clock::time_point first, Clock::time_point last)
{
    return std::chrono::duration<double, std::milli>(last - first).count();
}

struct MotionResult { double start_ms, reach_knee_ms; };

MotionResult outward_motion(int speed, double px_per_second, Clock::duration cadence,
                            Clock::duration phase)
{
    Rig rig(speed);
    const auto start = Clock::now() + cadence + phase;
    const double dx = px_per_second * std::chrono::duration<double>(cadence).count();
    int moving_frame = 0;
    for (int i = 0; i <= 10; ++i) {
        std::this_thread::sleep_until(start + i * cadence);
        if (!moving_frame && i * dx > 19.2) moving_frame = i;
        rig.frame(center + std::min(i * dx, 126.0));
    }
    std::this_thread::sleep_for(cadence);
    auto commands = rig.recorder->commands();
    const auto first = std::find_if(commands.begin(), commands.end(), [](auto c) { return c.x == 1; });
    const auto fast = std::find_if(first, commands.end(), [](auto c) { return c.x == 1 && c.speed >= 7; });
    require(first != commands.end() && first->speed == 1, "first outward command must be SPD 1");
    require(fast != commands.end(), "sustained motion failed to reach SPD 7");
    require(std::all_of(first, commands.end(), [](auto c) { return c.x == 1 && c.y == 0; }),
            "outward fixture reversed, stopped, or moved Y");
    const MotionResult result{elapsed(start + moving_frame * cadence, first->time),
                              elapsed(first->time, fast->time)};
    rig.frame(190);
    std::this_thread::sleep_for(140ms);
    commands = rig.recorder->commands();
    const auto reverse = std::find_if(commands.begin() + 1, commands.end(), [](auto c) { return c.x == -1; });
    require(reverse != commands.end() && reverse->speed == 1 && (reverse - 1)->x == 0,
            "reversal must STOP then restart at SPD 1");
    rig.executor.halt();
    require(rig.recorder->commands().back().speed == 0, "halt must send STOP");
    return result;
}

void spike_return()
{
    for (bool sustained : {false, true}) {
        Rig rig(7);
        for (int x : {338, 375}) {
            std::this_thread::sleep_for(45ms);
            rig.frame(x);
        }
        std::this_thread::sleep_for(45ms);
        rig.frame(sustained ? 397 : center);
        std::this_thread::sleep_for(35ms);
        const auto commands = rig.recorder->commands();
        require(commands.size() >= 3 && commands[1].speed == 1,
                "identical first displacement must be followed from SPD 1");
        const bool reached = std::any_of(commands.begin(), commands.end(), [](auto c) { return c.speed >= 7; });
        require(reached == sustained, "only the continuing displacement should trigger high output");
        require(sustained || commands.back().speed == 0, "one-frame displacement must stop on return");
    }
}

struct Samples {
    AutoTrackingController controller{options(100).automatic, smoothing(7)};
    FrameMetadata meta;

    void push(double x, double y = 240, bool healthy = true, int step = 1,
              TrackLead lead = {}, std::chrono::milliseconds dt = 45ms) {
        if (meta.frame_id == 0) meta.captured_at = Clock::now();
        meta.frame_id += step;
        meta.captured_at += dt;
        controller.calculate(person(x, y), 640, 480, meta, 1, lead, 0.5, healthy);
    }
    bool take_x() { return controller.take_catchup(1, 0); }
};

void eligibility()
{
    Samples sustained;
    sustained.push(338); sustained.push(356); sustained.push(374);
    require(sustained.take_x(), "sustained valid outward motion must qualify");
    sustained.push(374, 240, true, 0, {}, 0ms);
    require(!sustained.take_x(), "reused frame renewed eligibility");
    sustained.push(392, 250);
    require(!sustained.controller.take_catchup(1, -1), "shared-speed boost cannot start an unqualified Y axis");

    Samples slow;
    slow.push(420); slow.push(425); slow.push(430); slow.push(430);
    require(!slow.take_x(), "slow off-center motion must not qualify despite large P demand");
    Samples forecast;
    forecast.push(320); forecast.push(332); forecast.push(344, 240, true, 1, {7, 200, 0, 1});
    require(!forecast.take_x(), "forecast must not make measured demand qualify");
    Samples broken;
    broken.push(338); broken.push(356); broken.push(374, 240, true, 2);
    require(!broken.take_x(), "nonadjacent observation renewed eligibility");
    broken.push(392, 240, false); broken.push(410); broken.push(428);
    require(!broken.take_x(), "unhealthy frame contributed to motion confirmation");
    broken.push(446, 240, true, 1, {}, 300ms);
    require(!broken.take_x(), "stale history renewed eligibility");
}

void horizontal_only()
{
    for (bool diagonal : {false, true}) {
        const auto x = [diagonal](double dx) { return center + (diagonal ? dx : 0); };
        const auto y = [](double dx) { return 240 + dx * 0.75; }; // Same normalized axis error.
        Samples samples;
        for (double dx : {0.0, 18.0, 55.0, 77.0}) samples.push(x(dx), y(dx));
        require(samples.controller.take_catchup(diagonal ? 1 : 0, -1),
                "fixture must meet the observed motion and PD conditions");
        for (int speed : {0, 7}) {
            Rig rig(speed);
            for (double dx : {18.0, 55.0, 77.0}) {
                std::this_thread::sleep_for(45ms);
                rig.frame(x(dx), y(dx));
            }
            std::this_thread::sleep_for(35ms);
            const auto commands = rig.recorder->commands();
            require(commands.size() >= 3 && commands[1].speed == 1,
                    "vertical or diagonal motion must start and ramp normally");
            require(std::all_of(commands.begin() + 1, commands.end(), [diagonal](const auto& c) {
                return c.x == int(diagonal) && c.y == -1 && c.speed > 0 && c.speed < 7;
            }), "vertical or diagonal motion skipped the normal low-speed ramp");
        }
    }
}

void invalid_grace_and_cap()
{
    Rig grace(7);
    for (int x : {338, 356}) {
        std::this_thread::sleep_for(45ms);
        grace.frame(x);
    }
    std::this_thread::sleep_for(45ms);
    grace.frame(std::nullopt);
    std::this_thread::sleep_for(40ms);
    grace.frame(404);
    std::this_thread::sleep_for(32ms);
    grace.executor.publish_unsafe_observation();
    std::this_thread::sleep_for(35ms);
    const auto commands = grace.recorder->commands();
    require(commands.size() >= 3 && commands.back().speed == 0, "unsafe input did not STOP");
    require(std::none_of(commands.begin(), commands.end(), [](auto c) { return c.speed >= 7; }),
            "missing observation triggered a boost on return");

    Rig low_cap(7, 0, 5);
    for (int x : {338, 356, 374, 392, 410, 428, 446}) {
        std::this_thread::sleep_for(45ms);
        low_cap.frame(x);
    }
    std::this_thread::sleep_for(35ms);
    const auto limited = low_cap.recorder->commands();
    require(std::any_of(limited.begin(), limited.end(), [](auto c) { return c.speed == 5; }),
            "cap fixture failed to reach SPD 5");
    require(std::none_of(limited.begin(), limited.end(), [](auto c) { return c.speed > 5; }),
            "catch-up exceeded configured cap");
}
}

int main()
{
    try {
        eligibility();
        spike_return();
        horizontal_only();
        invalid_grace_and_cap();
        for (double rate : {240.0, 400.0}) {
            for (auto period : {38ms, 45ms}) {
                for (auto phase : {0ms, 12ms}) {
                    const auto off = outward_motion(0, rate, period, phase);
                    const auto on = outward_motion(7, rate, period, phase);
                    std::printf("%.0f px/s, %.0f ms frame, %.0f ms phase: first SPD 1 %.1f/%.1f ms, "
                                "SPD 1 to 7 %.1f/%.1f ms (off/on)\n", rate,
                                elapsed({}, Clock::time_point(period)), elapsed({}, Clock::time_point(phase)),
                                off.start_ms, on.start_ms, off.reach_knee_ms, on.reach_knee_ms);
                }
            }
        }
        std::puts("PASS catch-up conditions; times are FakeDevice command timings");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
