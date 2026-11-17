// No hardware: exercise the actual executor against a recording IGimbalDevice.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>
#include "control/gimbal_command_executor.hpp"

namespace {
using Clock = std::chrono::steady_clock;
struct Command { int x, y, speed; };
struct Recorder {
    std::mutex mutex;
    std::vector<Command> commands;
    void add(int x, int y, int speed) {
        std::lock_guard<std::mutex> lock(mutex);
        commands.push_back({x, y, speed});
    }
    std::vector<Command> take() {
        std::lock_guard<std::mutex> lock(mutex);
        auto copy = commands; commands.clear(); return copy;
    }
};
struct FakeDevice : IGimbalDevice {
    Recorder& recorder;
    explicit FakeDevice(Recorder& r) : recorder(r) {}
    void move(int x, int y) override { recorder.add(x, y, 0); }
    void move_with_speed(int x, int y, int speed) override { recorder.add(x, y, speed); }
    void stop() override { recorder.add(0, 0, 0); }
    void home() override {}
};
int failures = 0;
void check(bool ok, const char* name) {
    std::printf("%s %s\n", ok ? "ok" : "FAIL", name);
    failures += !ok;
}
struct Rig {
    Recorder recorder;
    RuntimeControl runtime;
    std::unique_ptr<GimbalCommandExecutor> executor;
    std::uint64_t frame_id = 0;
    Rig() {
        ControlConfig config; config.enabled = true;
        config.automatic.controller = AutoControllerType::PD;
        config.automatic.deadband_x = config.automatic.deadband_y = 0.06;
        config.automatic.pid_x.kp = config.automatic.pid_y.kp = 30;
        config.automatic.pid_x.kd = config.automatic.pid_y.kd = 0;
        config.automatic.speed_rate = 1000;
        config.automatic.period_ms = config.poll_interval_ms = 10;
        executor = std::make_unique<GimbalCommandExecutor>(config,
            std::make_unique<FakeDevice>(recorder), runtime);
        executor->connected();
    }
    void frame(std::uint64_t epoch, double ex = .2, double ey = 0, double dx = 60, double dy = 0) {
        auto context = std::make_shared<FrameContext>();
        context->image = cv::Mat(480, 640, CV_16UC1);
        context->metadata.frame_id = ++frame_id;
        context->metadata.captured_at = Clock::now();
        auto result = std::make_shared<TrackingResult>(); result->frame = context;
        Track person; person.track_id = 7; person.confidence = .9f;
        person.width = 40; person.height = 80;
        person.x = float(320 + ex * 320 - 20); person.y = float(240 + ey * 240 - 40);
        result->tracks.push_back(person); result->lead = {7, dx, dy, epoch};
        executor->publish(result);
        executor->ping(); // serialized after ingestion
    }
    std::vector<Command> feed(std::uint64_t epoch, double ex = .2, double ey = 0, double dx = 60, double dy = 0) {
        recorder.take();
        for (int i = 0; i < 6; ++i) {
            frame(epoch, ex, ey, dx, dy);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        return recorder.take();
    }
};
int max_speed(const std::vector<Command>& commands) {
    int speed = 0;
    for (const auto& c : commands) speed = std::max(speed, c.speed);
    return speed;
}
void selection_epoch() {
    Rig r;
    r.frame(0);
    check(r.executor->select(7) == ControlResult::Ok, "select observed ID");
    const auto epoch = r.executor->snapshot().command_generation;
    check(max_speed(r.feed(epoch)) >= 11, "current selection's forecast reaches the executor");
    check(r.executor->select(7) == ControlResult::Ok, "reselect the same observed ID");
    const auto next = r.executor->snapshot().command_generation;
    check(next != epoch && max_speed(r.feed(epoch)) <= 6,
          "in-flight forecast from the previous same-ID selection is ignored");
    check(max_speed(r.feed(next)) >= 11, "fresh selection epoch accepts a new forecast");
    r.executor->disconnected(); r.recorder.take();
    r.feed(next);
    check(!r.executor->snapshot().automatic && !r.executor->snapshot().moving,
          "forecast packets cannot revive a disconnected control session");
}
void shared_speed() {
    for (bool swap : {false, true}) {
        Rig r;
        const double ex = swap ? 10.0 / 320 : .5;
        const double ey = swap ? .5 : 10.0 / 240;
        r.frame(0, ex, ey, 0, 0);
        check(r.executor->select(7) == ControlResult::Ok, "select shared-speed test ID");
        const auto epoch = r.executor->snapshot().command_generation;
        bool ok = true; int seen = 0;
        for (double lead : {20., -60., 60., 0.}) {
            for (const auto& c : r.feed(epoch, ex, ey, swap ? lead : 0, swap ? 0 : lead)) {
                if (c.speed) { ++seen; ok &= swap ? c.x == 0 && c.y == -1 : c.y == 0 && c.x == 1; }
            }
        }
        check(ok && seen, "shared SPD never activates the centred axis when its forecast toggles");
    }
}
}
int main() {
    selection_epoch(); shared_speed();
    return failures ? 1 : 0;
}
