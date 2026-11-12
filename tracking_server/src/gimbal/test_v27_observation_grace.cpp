// No device: run the actual v27 executor with a recording IGimbalDevice.
// From this folder, use the CORE sources in V27_CHANGES.md:
// c++ -std=c++17 -O2 -pthread -Iv27/include -Iteam/include -Isim/stub test_v27_observation_grace.cpp $CORE -o test_v27_observation_grace
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <limits>
#include <mutex>
#include <vector>
#include "control/gimbal_command_executor.hpp"

namespace {
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

struct Command { int x, y, speed; Clock::time_point when; };
struct Recorder : IGimbalDevice {
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<Command> commands;
    bool block_move = false, blocked = false;

    void add(int x, int y, int speed) {
        std::unique_lock<std::mutex> lock(mutex);
        commands.push_back({x, y, speed, Clock::now()});
        if ((x || y) && block_move) {
            blocked = true;
            changed.notify_all();
            changed.wait(lock, [this] { return !block_move; });
        }
        changed.notify_all();
    }
    void move(int x, int y) override { add(x, y, 0); }
    void move_with_speed(int x, int y, int speed) override { add(x, y, speed); }
    void stop() override { add(0, 0, 0); }
    void home() override {}
    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        commands.clear();
    }
    std::vector<Command> read() {
        std::lock_guard<std::mutex> lock(mutex);
        return commands;
    }
    bool wait_command(bool moving, Clock::duration timeout) {
        std::unique_lock<std::mutex> lock(mutex);
        return changed.wait_for(lock, timeout, [&] {
            for (const auto& c : commands) if (moving == bool(c.x || c.y)) return true;
            return false;
        });
    }
    bool await_block() {
        std::unique_lock<std::mutex> lock(mutex);
        return changed.wait_for(lock, 200ms, [&] { return blocked; });
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        block_move = false;
        changed.notify_all();
    }
};

Track person(int id = 7, float center = 384) {
    Track t;
    t.track_id = id; t.confidence = .9f;
    t.width = 40; t.height = 80; t.x = center - 20; t.y = 200;
    return t;
}
struct Rig {
    RuntimeControl runtime;
    Recorder* recorder = nullptr;
    std::unique_ptr<GimbalCommandExecutor> executor;
    std::uint64_t frame_id = 0;
    Rig(int grace = 100, int timeout = 250) {
        ControlConfig config; config.enabled = true;
        config.automatic.deadband_x = config.automatic.deadband_y = .06;
        config.automatic.pid_x.kp = config.automatic.pid_y.kp = 30;
        config.automatic.speed_rate = 1000;
        config.automatic.speed_min = 6;
        config.automatic.period_ms = config.poll_interval_ms = 5;
        config.automatic.input_timeout_ms = timeout;
        SmoothingConfig smooth; smooth.observation_grace_ms = grace;
        auto device = std::make_unique<Recorder>();
        recorder = device.get();
        executor = std::make_unique<GimbalCommandExecutor>(config, std::move(device), runtime, smooth);
        executor->connected();
        recorder->clear();
    }
    ~Rig() { recorder->release(); }

    void frame(std::vector<Track> tracks, bool healthy = true, Clock::time_point capture = Clock::now(),
               std::uint64_t id = 0, TrackLead lead = {}) {
        auto context = std::make_shared<FrameContext>();
        context->image = cv::Mat(480, 640, CV_16UC1);
        context->metadata.frame_id = id ? id : ++frame_id;
        context->metadata.captured_at = capture;
        auto result = std::make_shared<TrackingResult>();
        result->frame = context;
        result->model_generation = runtime.generation.load();
        result->tracker_epoch = runtime.tracker_epoch.load();
        result->tracks = std::move(tracks);
        result->lead = lead;
        executor->publish(result, healthy);
    }
    bool start(Clock::duration cadence = 60ms, Clock::duration age = 0ms) {
        const auto last_capture = Clock::now() - age;
        frame({person()}, true, last_capture - cadence);
        const auto selected = executor->select(7);
        frame({person()}, true, last_capture);
        executor->snapshot();
        const bool moving = recorder->wait_command(true, 100ms);
        recorder->clear();
        return selected == ControlResult::Ok && moving;
    }
    bool gap(std::vector<Track> tracks = {}) {
        frame(std::move(tracks));
        const auto s = executor->snapshot();
        return s.automatic && s.lost && s.moving && s.selected_id == 7;
    }
};

int failures = 0;
void check(bool ok, const char* name) {
    std::printf("%s %s\n", ok ? "ok" : "FAIL", name);
    failures += !ok;
}

void recovery_and_other_id() {
    Rig r;
    const bool ready = r.start();
    const auto epoch = r.executor->snapshot().command_generation;
    const bool held = r.gap({person(8, 100)});
    const auto lost = r.executor->snapshot();
    const auto during = r.recorder->read();
    r.frame({person(7, 384)}, true, Clock::now(), 0, {7, 60, 0, epoch});
    const auto recovered = r.executor->snapshot();
    check(ready && held && during.empty() && lost.command_generation != epoch &&
          recovered.automatic && !recovered.lost && recovered.moving && recovered.selected_id == 7,
          "one gap holds the exact sent output, invalidates old LEAD and accepts the same ID immediately");
    std::this_thread::sleep_for(35ms);
    r.frame({person(7, 384)}, true, Clock::now(), 0, {7, 60, 0, epoch});
    r.executor->snapshot();
    const auto after = r.recorder->read();
    bool no_old_forecast = true;
    for (const auto& c : after) no_old_forecast &= c.x == 1 && c.y == 0 && c.speed <= 6;
    check(no_old_forecast, "old forecast cannot change the returned observation's command");
    check(r.gap({person(8)}) && !r.recorder->wait_command(false, 0ms),
          "two healthy same-ID observations can establish a new one-frame hold");
    r.frame({person(8)});
    check(r.recorder->wait_command(false, 45ms) && r.executor->snapshot().selected_id == 7,
          "a second distinct missing frame stops without adopting the other ID");
}
void deadlines() {
    {
        Rig r;
        const bool started = r.start(40ms);
        const auto before = Clock::now();
        const auto stamp = Clock::now();
        r.frame({}, true, stamp);
        const auto lost = r.executor->snapshot();
        r.frame({}, true, stamp, r.frame_id);
        r.frame({}, true, stamp, r.frame_id);
        const bool stopped = r.recorder->wait_command(false, 130ms);
        const auto commands = r.recorder->read();
        check(started && lost.lost && lost.moving && stopped && commands.size() == 1 &&
              commands[0].when - before >= 50ms && commands[0].when - before <= 110ms,
              "capture cadence limits hold and exact retransmission does not renew it");
    }
    {
        Rig r(45);
        const bool started = r.start(90ms);
        const auto before = Clock::now();
        const bool held = r.gap();
        const bool stopped = r.recorder->wait_command(false, 120ms);
        const auto commands = r.recorder->read();
        check(started && held && stopped && commands.size() == 1 &&
              commands[0].when - before >= 35ms && commands[0].when - before <= 90ms,
              "configured limit caps a longer capture cadence");
    }
    {
        Rig r;
        const bool started = r.start(60ms, 175ms);
        const auto before = Clock::now();
        const bool held = r.gap();
        const bool stopped = r.recorder->wait_command(false, 110ms);
        const auto commands = r.recorder->read();
        check(started && held && stopped && commands.size() == 1 &&
              commands[0].when - before >= 35ms && commands[0].when - before <= 85ms,
              "last measured capture timeout caps the hold");
    }
}
void unsafe_observations() {
    const auto run = [](const char* name, auto loss) {
        Rig r;
        const bool started = r.start();
        loss(r);
        check(started && r.recorder->wait_command(false, 45ms) &&
              r.executor->snapshot().lost && !r.executor->snapshot().moving, name);
    };
    run("OBS0 cannot hold or move even if a row is present", [](Rig& r) { r.frame({person()}, false); });
    run("stale capture cannot qualify", [](Rig& r) { r.frame({}, true, Clock::now() - 500ms); });
    run("future capture cannot qualify", [](Rig& r) { r.frame({person()}, true, Clock::now() + 500ms); });
    run("a repeated capture with a new frame ID stops", [](Rig& r) {
        const auto stamp = Clock::now();
        r.frame({person()}, true, stamp);
        r.executor->snapshot();
        r.frame({}, true, stamp);
    });
    run("a skipped frame number cannot establish a single-frame miss", [](Rig& r) {
        r.frame({}, true, Clock::now(), r.frame_id + 2);
    });
    run("an invalid selected row is not a missing observation", [](Rig& r) {
        auto invalid = person();
        invalid.x = std::numeric_limits<float>::quiet_NaN();
        r.frame({invalid});
    });
    run("a tracker-predicted selected row cannot hold movement", [](Rig& r) {
        auto predicted = person();
        predicted.predicted = true;
        r.frame({predicted});
    });
    run("duplicate selected rows cannot hold movement", [](Rig& r) { r.frame({person(), person()}); });

    Rig disabled(0);
    check(disabled.start() && !disabled.gap() && disabled.recorder->wait_command(false, 45ms),
          "default option immediately stops on missing detection");
}

void authorities() {
    const auto run = [](const char* name, auto revoke) {
        Rig r;
        const bool held = r.start() && r.gap();
        revoke(r);
        check(held && r.recorder->wait_command(false, 45ms) &&
              !r.executor->snapshot().moving && !r.executor->snapshot().automatic, name);
    };
    run("manual mode stops a held command", [](Rig& r) { r.executor->manual(); });
    run("halt stops a held command", [](Rig& r) { r.executor->halt(); });
    run("disconnect stops a held command", [](Rig& r) { r.executor->disconnected(); });
    run("model generation change stops a held command", [](Rig& r) {
        r.runtime.generation.fetch_add(1);
        r.frame({person()});
    });
    run("tracker generation change stops a held command", [](Rig& r) {
        r.runtime.tracker_epoch.fetch_add(1);
        r.frame({person()});
    });
    Rig r;
    const bool held = r.start() && r.gap({person(8)});
    const auto invalid = r.executor->select(9);
    const auto selected = r.executor->select(8);
    check(held && invalid == ControlResult::InvalidTarget && selected == ControlResult::Ok &&
          r.recorder->wait_command(false, 45ms) && r.executor->snapshot().selected_id == 8,
          "only explicit selection of an observed different ID can replace the old ID");
}
void overwritten_barriers() {
    const auto run = [](const char* name, auto invalid) {
        Rig r;
        {
            std::lock_guard<std::mutex> lock(r.recorder->mutex);
            r.recorder->block_move = true;
        }
        r.frame({person()});
        const auto selected = r.executor->select(7);
        const bool blocked = r.recorder->await_block();
        invalid(r);
        r.frame({person()});
        r.recorder->release();
        check(selected == ControlResult::Ok && blocked && r.recorder->wait_command(false, 90ms), name);
    };
    run("OBS0 survives replacement in the latest-only mailbox", [](Rig& r) { r.frame({}, false); });
    run("future timestamp survives replacement in the latest-only mailbox",
        [](Rig& r) { r.frame({}, true, Clock::now() + 500ms); });
    run("missing frames overwritten by a same-ID observation still force STOP", [](Rig& r) {
        r.frame({});
        r.frame({});
    });
    Rig r;
    {
        std::lock_guard<std::mutex> lock(r.recorder->mutex);
        r.recorder->block_move = true;
    }
    r.frame({person()});
    const auto selected = r.executor->select(7);
    const bool blocked = r.recorder->await_block();
    r.recorder->clear();
    r.frame({person()});
    r.executor->publish_unsafe_observation();
    r.recorder->release();
    const bool stopped = r.recorder->wait_command(false, 90ms);
    check(selected == ControlResult::Ok && blocked && stopped &&
          !r.recorder->wait_command(true, 25ms),
          "malformed input drops any queued pre-barrier observation");
}
}

int main() {
    recovery_and_other_id();
    deadlines();
    unsafe_observations();
    authorities();
    overwritten_barriers();
    return failures ? 1 : 0;
}
