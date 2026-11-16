// v27 search when lost (--search 1): run the actual executor with a recording IGimbalDevice, frames fed in real time.
// From this folder, use the CORE sources in V27_CHANGES.md:
// c++ -std=c++17 -O2 -pthread -Iv27/include -Iteam/include -Isim/stub test_v27_search.cpp $CORE -o test_v27_search && ./test_v27_search
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>
#include "control/gimbal_command_executor.hpp"

namespace {
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

struct Command { int x, y, speed; Clock::time_point when; };
struct Recorder : IGimbalDevice {
    std::mutex mutex;
    std::vector<Command> commands;
    void add(int x, int y, int speed) { std::lock_guard<std::mutex> lock(mutex); commands.push_back({x, y, speed, Clock::now()}); }
    void move(int x, int y) override { add(x, y, 0); }
    void move_with_speed(int x, int y, int speed) override { add(x, y, speed); }
    void stop() override { add(0, 0, 0); }
    void home() override {}
    void clear() { std::lock_guard<std::mutex> lock(mutex); commands.clear(); }
    std::vector<Command> read() { std::lock_guard<std::mutex> lock(mutex); return commands; }
};

Track person(int id, float center_x, float center_y = 240)
{
    Track t;
    t.track_id = id; t.confidence = .9f;
    t.width = 40; t.height = 80; t.x = center_x - 20; t.y = center_y - 40;
    return t;
}

// Search settings scaled down so a whole pattern takes well under a second (SPD 3 = 160 px/s in the bookkeeping).
SmoothingConfig fast_search()
{
    SmoothingConfig s;
    s.search = true;
    s.search_grace_ms = 60;
    s.search_follow_ms = 150;
    s.search_follow_px = 1000;
    s.search_follow_speed = 10;
    s.search_speed = 3;
    s.search_sweep_px = 24;
    s.search_widen = 1.5;
    s.search_max_px = 40;
    s.search_timeout_ms = 1500;
    return s;
}

struct Rig {
    RuntimeControl runtime;
    Recorder* recorder = nullptr;
    std::unique_ptr<GimbalCommandExecutor> executor;
    std::atomic<std::uint64_t> frame_id{0};
    std::mutex scene_mutex;
    std::vector<Track> scene;
    TrackVelocity velocity;   // VEL of the frames (predictive-control velocity)
    std::atomic<bool> feeding{true}, closing{false}, pinging{true}, healthy{true};
    std::thread feeder;
    explicit Rig(SmoothingConfig smooth, int reacquire_ms = 3000, int heartbeat_ms = 1500)
    {
        ControlConfig config; config.enabled = true;
        config.heartbeat_timeout_ms = heartbeat_ms;
        config.automatic.deadband_x = config.automatic.deadband_y = .06;
        config.automatic.pid_x.kp = config.automatic.pid_y.kp = 30;
        config.automatic.speed_rate = 1000;
        config.automatic.speed_min = 1;
        config.automatic.speed_max = 30;
        config.automatic.period_ms = config.poll_interval_ms = 5;
        config.automatic.input_timeout_ms = 250;
        config.automatic.reacquire_timeout_ms = reacquire_ms;
        auto device = std::make_unique<Recorder>();
        recorder = device.get();
        executor = std::make_unique<GimbalCommandExecutor>(config, std::move(device), runtime, smooth);
        executor->connected();
        feeder = std::thread([this] {
            auto next_ping = Clock::now();
            while (!closing) {
                if (feeding) publish();
                if (pinging && Clock::now() >= next_ping) { executor->ping(); next_ping = Clock::now() + 100ms; }
                std::this_thread::sleep_for(20ms);
            }
        });
    }
    ~Rig() { closing = true; feeder.join(); }
    void set(std::vector<Track> tracks) { std::lock_guard<std::mutex> lock(scene_mutex); scene = std::move(tracks); }
    void publish()
    {
        auto context = std::make_shared<FrameContext>();
        context->image = cv::Mat(480, 640, CV_16UC1);
        context->metadata.frame_id = ++frame_id;
        context->metadata.captured_at = Clock::now();
        auto result = std::make_shared<TrackingResult>();
        result->frame = context;
        result->model_generation = runtime.generation.load();
        result->tracker_epoch = runtime.tracker_epoch.load();
        { std::lock_guard<std::mutex> lock(scene_mutex); result->tracks = scene; result->velocity = velocity; }
        executor->publish(result, healthy);
    }
    // Follow id 7 standing right of centre until the gimbal moves right, then the person vanishes.
    bool follow_then_lose(float x = 520)
    {
        set({person(7, x)});
        std::this_thread::sleep_for(60ms);
        if (executor->select(7) != ControlResult::Ok) return false;
        std::this_thread::sleep_for(150ms);
        const auto s = executor->snapshot();
        if (!s.automatic || !s.moving) return false;
        recorder->clear();
        set({});
        return true;
    }
};

int failures = 0;
void check(bool ok, const char* name) { std::printf("%s %s\n", ok ? "ok" : "FAIL", name); failures += !ok; }

std::vector<Command> moves(const std::vector<Command>& c)
{
    std::vector<Command> m;
    for (const auto& x : c) if (x.x || x.y) m.push_back(x);
    return m;
}

// Open-loop pan offset (picture px) of a command list, the same bookkeeping as the executor.
double max_offset(const std::vector<Command>& c, Clock::time_point end)
{
    double pan = 0, worst = 0;
    for (std::size_t i = 0; i < c.size(); ++i) {
        const auto until = i + 1 < c.size() ? c[i + 1].when : end;
        pan += c[i].x * search_picture_px_per_s(c[i].speed) * std::chrono::duration<double>(until - c[i].when).count();
        worst = std::max(worst, std::abs(pan));
    }
    return worst;
}

void off_by_default()
{
    Rig r{SmoothingConfig{}};
    const bool ready = r.follow_then_lose();
    const auto lost_at = Clock::now();
    std::this_thread::sleep_for(500ms);
    const auto c = r.recorder->read();
    const auto s = r.executor->snapshot();
    check(ready && !c.empty() && !c.front().x && !c.front().y && c.front().when - lost_at < 60ms && moves(c).empty() &&
          s.automatic && s.lost && !s.moving && s.search == 0,
          "default (no --search): loss STOPs at once and never moves again (team rule)");
}

void pattern()
{
    const auto cfg = fast_search();
    Rig r{cfg};
    const bool ready = r.follow_then_lose();
    const auto lost_at = Clock::now();
    std::this_thread::sleep_for(30ms);
    const auto during_grace = r.executor->snapshot();
    std::this_thread::sleep_for(1700ms);
    const auto c = r.recorder->read();
    const auto m = moves(c);
    const auto s = r.executor->snapshot();
    bool grace_ok = !c.empty() && !c.front().x && c.front().when - lost_at < 60ms && !m.empty() &&
                    m.front().when - lost_at >= std::chrono::milliseconds(cfg.search_grace_ms - 25);
    // Direction sequence of the moves: follow-through right, then legs left, right, left ... ending stopped.
    std::vector<int> dirs;
    for (const auto& x : m) if (dirs.empty() || dirs.back() != x.x) dirs.push_back(x.x);
    bool alternating = dirs.size() >= 3 && dirs[0] == 1 && dirs[1] == -1 && dirs[2] == 1;
    for (std::size_t i = 1; i < dirs.size(); ++i) alternating = alternating && dirs[i] == -dirs[i - 1];
    bool through_stop = true;   // every direction change goes through STOP
    for (std::size_t i = 1; i < c.size(); ++i)
        if (c[i].x && c[i - 1].x && c[i].x != c[i - 1].x) through_stop = false;
    bool speeds = true;
    for (const auto& x : m) speeds = speeds && x.speed >= 1 && x.speed <= cfg.search_follow_speed && !x.y;
    const double follow_px = search_picture_px_per_s(10) * cfg.search_follow_ms / 1000.0;
    const double worst = max_offset(c, Clock::now());
    check(ready && during_grace.search == 1 && during_grace.lost && !during_grace.moving && grace_ok,
          "search: STOP at the loss, no motion during the grace");
    check(alternating && through_stop && speeds,
          "search: follow-through in the last direction, then alternating sweep legs, each turn through STOP");
    check(worst <= follow_px + cfg.search_max_px + 30,
          "search: the open-loop excursion stays within follow-through + widest leg");
    check(!c.empty() && !c.back().x && !c.back().y && s.automatic && s.lost && !s.moving && s.search == 4 &&
          s.selected_id == 7, "search: the pattern ends stopped and holds, the selection is kept for the reacquire window");
}

void no_follow_through_when_centred()
{
    auto cfg = fast_search();
    Rig r{cfg};
    r.set({person(7, 330)});   // inside the deadband: the gimbal stands still
    std::this_thread::sleep_for(60ms);
    const bool selected = r.executor->select(7) == ControlResult::Ok;
    std::this_thread::sleep_for(100ms);
    r.recorder->clear();
    r.set({});
    std::this_thread::sleep_for(250ms);
    const auto m = moves(r.recorder->read());
    check(selected && !m.empty() && m.front().x == 1 && m.front().speed <= cfg.search_speed,
          "search: a person lost at the centre starts the sweep at sweep speed on the side last seen");
}

// The follow-through goes where the predictive-control velocity says, at the SPD that covers it.
void follow_through_from_velocity()
{
    const auto run = [](float x, double vx) {
        Rig r{fast_search()};
        { std::lock_guard<std::mutex> lock(r.scene_mutex); r.velocity = TrackVelocity{7, vx, 0}; }
        r.set({person(7, x)});
        std::this_thread::sleep_for(60ms);
        const bool selected = r.executor->select(7) == ControlResult::Ok;
        std::this_thread::sleep_for(150ms);
        r.recorder->clear();
        r.set({});
        std::this_thread::sleep_for(250ms);
        const auto m = moves(r.recorder->read());
        int first = m.empty() ? 0 : m.front().x, top = 0;
        for (const auto& c : m) { if (c.x != first) break; top = std::max(top, c.speed); }
        return std::make_tuple(selected, first, top);
    };
    const auto [a_ok, a_dir, a_top] = run(330, 250.0);   // centred (gimbal still), person walking right at 250 px/s
    const auto [b_ok, b_dir, b_top] = run(520, -200.0);  // gimbal turning right, but the person is estimated walking left
    const auto [c_ok, c_dir, c_top] = run(520, 5.0);     // slower than 15 px/s: the last output decides (right)
    check(a_ok && a_dir == 1 && a_top == 8, "search: follow-through from the predictive velocity (+250 px/s -> right, SPD 8 = 277 px/s)");
    check(b_ok && b_dir == -1 && b_top == 6, "search: the predictive velocity wins over the last output (-200 px/s -> left, SPD 6)");
    check(c_ok && c_dir == 1 && c_top <= 10, "search: a velocity below 15 px/s falls back to the last following output");
}

void blind_stops()
{
    auto cfg = fast_search();
    cfg.search_max_px = 400; cfg.search_sweep_px = 400; cfg.search_timeout_ms = 2500;   // a long first leg
    Rig r{cfg};
    const bool ready = r.follow_then_lose();
    std::this_thread::sleep_for(350ms);
    const bool moving = r.executor->snapshot().moving;
    r.feeding = false;   // no frames: the camera froze
    std::this_thread::sleep_for(320ms);
    const auto blind = r.executor->snapshot();
    r.recorder->clear();
    std::this_thread::sleep_for(150ms);
    const bool still = moves(r.recorder->read()).empty();
    r.feeding = true;
    std::this_thread::sleep_for(150ms);
    const auto again = r.executor->snapshot();
    r.healthy = false;   // frames go on, but the server marks them OBS 0 (frozen camera, replay, camera reset)
    std::this_thread::sleep_for(60ms);
    const auto unhealthy = r.executor->snapshot();
    r.recorder->clear();
    std::this_thread::sleep_for(150ms);
    const bool still2 = moves(r.recorder->read()).empty();
    r.healthy = true;
    std::this_thread::sleep_for(150ms);
    const auto again2 = r.executor->snapshot();
    check(ready && moving && !blind.moving && still && again.moving && again.search >= 2,
          "search: stops within the input timeout without fresh frames, continues when frames return");
    check(!unhealthy.moving && still2 && again2.moving && again2.search >= 2,
          "search: stops at once on frames marked unhealthy (OBS 0), continues on healthy ones");
}

void same_id_back()
{
    auto cfg = fast_search();
    cfg.search_max_px = 400; cfg.search_sweep_px = 400; cfg.search_timeout_ms = 2500;
    Rig r{cfg};
    const bool ready = r.follow_then_lose();
    std::this_thread::sleep_for(300ms);
    r.set({person(8, 100)});   // another person: never followed by the bridge
    std::this_thread::sleep_for(200ms);
    const auto other = r.executor->snapshot();
    r.set({person(7, 330), person(8, 100)});
    std::this_thread::sleep_for(150ms);
    const auto back = r.executor->snapshot();
    check(ready && other.automatic && other.lost && other.selected_id == 7 && other.search >= 2 &&
          back.automatic && !back.lost && back.selected_id == 7 && back.search == 0,
          "search: another ID never ends it; the same ID ends it and following resumes");
}

void authority()
{
    {
        Rig r{fast_search(), 3000, 300};
        const bool ready = r.follow_then_lose();
        std::this_thread::sleep_for(250ms);
        r.pinging = false;
        std::this_thread::sleep_for(450ms);
        r.recorder->clear();
        std::this_thread::sleep_for(100ms);
        const auto s = r.executor->snapshot();
        check(ready && !s.automatic && !s.moving && s.search == 0 && moves(r.recorder->read()).empty(),
              "search: a lost heartbeat stops it and drops the selection");
    }
    {
        Rig r{fast_search()};
        const bool ready = r.follow_then_lose();
        std::this_thread::sleep_for(250ms);
        r.executor->halt();
        r.recorder->clear();
        std::this_thread::sleep_for(100ms);
        const auto s = r.executor->snapshot();
        check(ready && !s.automatic && !s.moving && s.search == 0 && moves(r.recorder->read()).empty(),
              "search: STOP/HOME (H) ends it at once");
    }
    {
        auto cfg = fast_search();
        cfg.search_timeout_ms = 5000; cfg.search_max_px = 4000; cfg.search_sweep_px = 4000;
        Rig r{cfg, 700};
        const bool ready = r.follow_then_lose();
        std::this_thread::sleep_for(800ms);
        const auto s = r.executor->snapshot();
        check(ready && !s.automatic && !s.moving && s.search == 0, "search: the reacquire timeout still drops the selection");
    }
}

void settings()
{
    const auto rejects = [](auto change) {
        auto s = fast_search(); change(s);
        try { Rig r{s}; } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    check(rejects([](SmoothingConfig& s) { s.search_speed = 0; }) &&
          rejects([](SmoothingConfig& s) { s.search_speed = 31; }) &&
          rejects([](SmoothingConfig& s) { s.search_timeout_ms = 50; }) &&
          rejects([](SmoothingConfig& s) { s.search_widen = 0.5; }) &&
          rejects([](SmoothingConfig& s) { s.search_max_px = 10; }) &&
          !rejects([](SmoothingConfig&) {}),
          "search: out-of-range settings are refused");
}
}

int main()
{
    off_by_default();
    pattern();
    no_follow_through_when_centred();
    follow_through_from_velocity();
    blind_stops();
    same_id_back();
    authority();
    settings();
    return failures ? 1 : 0;
}
