// v27 coast when lost (--coast 1): run the actual executor with a recording IGimbalDevice, frames fed in real time.
// From this folder, use the CORE sources in V27_CHANGES.md:
// c++ -std=c++17 -O2 -pthread -Iv27/include -Iteam/include -Isim/stub test_v27_coast.cpp $CORE -o test_v27_coast && ./test_v27_coast
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <optional>
#include <thread>
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

TrackPrediction predicted(int id, double center_x, double center_y = 240)
{
    TrackPrediction p;
    p.track_id = id; p.width = 40; p.height = 80; p.x = center_x - 20; p.y = center_y - 40;
    return p;
}

SmoothingConfig coast_on(int ms = 300)
{
    SmoothingConfig s;
    s.coast = true;
    s.coast_ms = ms;
    return s;
}

struct Rig {
    RuntimeControl runtime;
    Recorder* recorder = nullptr;
    std::unique_ptr<GimbalCommandExecutor> executor;
    std::atomic<std::uint64_t> frame_id{0};
    std::mutex scene_mutex;
    std::vector<Track> scene;
    TrackPrediction prediction;   // PRED of the frames
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
    void set(std::vector<Track> tracks, std::optional<TrackPrediction> p = std::nullopt)
    {
        std::lock_guard<std::mutex> lock(scene_mutex);
        scene = std::move(tracks);
        prediction = p ? *p : TrackPrediction{};
    }
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
        { std::lock_guard<std::mutex> lock(scene_mutex); result->tracks = scene; result->prediction = prediction; }
        executor->publish(result, healthy);
    }
    // Follow id 7 standing right of centre until the gimbal moves right, then the person vanishes; the frames then carry
    // the prediction p (none = no PRED).
    bool follow_then_lose(std::optional<TrackPrediction> p, float x = 520)
    {
        set({person(7, x)});
        std::this_thread::sleep_for(60ms);
        if (executor->select(7) != ControlResult::Ok) return false;
        std::this_thread::sleep_for(150ms);
        const auto s = executor->snapshot();
        if (!s.automatic || !s.moving) return false;
        recorder->clear();
        set({}, p);
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

bool has_stop(const std::vector<Command>& c)
{
    for (const auto& x : c) if (!x.x && !x.y) return true;
    return false;
}

void off_by_default()
{
    Rig r{SmoothingConfig{}};
    const bool ready = r.follow_then_lose(predicted(7, 560));
    const auto lost_at = Clock::now();
    std::this_thread::sleep_for(300ms);
    const auto c = r.recorder->read();
    const auto s = r.executor->snapshot();
    check(ready && !c.empty() && !c.front().x && !c.front().y && c.front().when - lost_at < 60ms && moves(c).empty() &&
          s.automatic && s.lost && !s.moving,
          "default (no --coast): a PRED suffix is ignored, the loss STOPs at once (team rule)");
}

void coasts_then_stops()
{
    Rig r{coast_on(300)};
    const bool ready = r.follow_then_lose(predicted(7, 560));
    const auto lost_at = Clock::now();
    std::this_thread::sleep_for(150ms);
    const auto during = r.executor->snapshot();
    const auto early = r.recorder->read();
    std::this_thread::sleep_for(300ms);
    const auto c = r.recorder->read();
    const auto s = r.executor->snapshot();
    bool right = !moves(early).empty();
    for (const auto& x : moves(early)) right = right && x.x == 1;
    Clock::time_point first_stop{};
    for (const auto& x : c) if (!x.x && !x.y) { first_stop = x.when; break; }
    check(ready && right && !has_stop(early) && during.moving && during.lost && during.automatic && during.selected_id == 7,
          "coast: no STOP at the loss; it keeps turning toward the predicted place (right), reported lost");
    check(first_stop != Clock::time_point{} && first_stop - lost_at >= 280ms && first_stop - lost_at < 400ms &&
          !s.moving && s.lost && s.automatic && s.selected_id == 7,
          "coast: STOP after --coast-ms, the selection is kept for the reacquire window");
}

void follows_the_prediction()
{
    Rig r{coast_on(600)};
    const bool ready = r.follow_then_lose(predicted(7, 100));   // the person is predicted on the left: reverse through STOP
    std::this_thread::sleep_for(250ms);
    const auto m = moves(r.recorder->read());
    const auto s = r.executor->snapshot();
    check(ready && !m.empty() && m.back().x == -1 && s.moving,
          "coast: the predicted place decides the direction (prediction left of the centre -> turn left)");
}

void no_prediction_no_coast()
{
    for (int which = 0; which < 3; ++which) {
        Rig r{coast_on(500)};
        std::optional<TrackPrediction> p;
        if (which == 1) p = predicted(8, 560);   // another track's prediction
        if (which == 2) { p = predicted(7, 560); p->width = 0; }   // malformed box
        const bool ready = r.follow_then_lose(p);
        const auto lost_at = Clock::now();
        std::this_thread::sleep_for(200ms);
        const auto c = r.recorder->read();
        check(ready && !c.empty() && !c.front().x && c.front().when - lost_at < 60ms && moves(c).empty(),
              which == 0 ? "coast: no PRED -> STOP at once" : which == 1 ? "coast: PRED of another id -> STOP at once"
                                                                          : "coast: an invalid PRED box -> STOP at once");
    }
}

void safety()
{
    {
        Rig r{coast_on(800)};
        const bool ready = r.follow_then_lose(predicted(7, 560));
        std::this_thread::sleep_for(100ms);
        const bool moving = r.executor->snapshot().moving;
        r.healthy = false;   // frames marked OBS 0
        std::this_thread::sleep_for(60ms);
        const auto s = r.executor->snapshot();
        r.recorder->clear();
        std::this_thread::sleep_for(100ms);
        check(ready && moving && !s.moving && moves(r.recorder->read()).empty(), "coast: an unhealthy frame (OBS 0) stops it at once");
    }
    {
        Rig r{coast_on(800)};
        const bool ready = r.follow_then_lose(predicted(7, 560));
        std::this_thread::sleep_for(100ms);
        r.feeding = false;   // camera froze
        std::this_thread::sleep_for(300ms);
        const auto s = r.executor->snapshot();
        r.recorder->clear();
        std::this_thread::sleep_for(100ms);
        check(ready && !s.moving && moves(r.recorder->read()).empty(), "coast: no fresh frame -> STOP within the input timeout");
    }
    {
        Rig r{coast_on(2000), 3000, 300};
        const bool ready = r.follow_then_lose(predicted(7, 560));
        std::this_thread::sleep_for(100ms);
        r.pinging = false;
        std::this_thread::sleep_for(450ms);
        r.recorder->clear();
        std::this_thread::sleep_for(100ms);
        const auto s = r.executor->snapshot();
        check(ready && !s.automatic && !s.moving && moves(r.recorder->read()).empty(), "coast: a lost heartbeat stops it and drops the selection");
    }
    {
        Rig r{coast_on(2000)};
        const bool ready = r.follow_then_lose(predicted(7, 560));
        std::this_thread::sleep_for(100ms);
        r.executor->halt();
        r.recorder->clear();
        std::this_thread::sleep_for(100ms);
        const auto s = r.executor->snapshot();
        check(ready && !s.automatic && !s.moving && moves(r.recorder->read()).empty(), "coast: STOP/HOME (H) ends it at once");
    }
    {
        Rig r{coast_on(2000)};
        const bool ready = r.follow_then_lose(predicted(7, 560));
        std::this_thread::sleep_for(100ms);
        r.executor->manual();
        r.recorder->clear();
        std::this_thread::sleep_for(100ms);
        const auto s = r.executor->snapshot();
        check(ready && !s.automatic && !s.moving && moves(r.recorder->read()).empty(), "coast: manual mode ends it at once");
    }
    {
        Rig r{coast_on(2000), 400};
        const bool ready = r.follow_then_lose(predicted(7, 560));
        std::this_thread::sleep_for(550ms);
        const auto s = r.executor->snapshot();
        check(ready && !s.automatic && !s.moving, "coast: the reacquire timeout still drops the selection");
    }
}

void same_id_back()
{
    Rig r{coast_on(800)};
    const bool ready = r.follow_then_lose(predicted(7, 560));
    std::this_thread::sleep_for(100ms);
    r.set({person(8, 560)}, predicted(7, 560));   // another person: never followed
    std::this_thread::sleep_for(100ms);
    const auto other = r.executor->snapshot();
    r.set({person(7, 330), person(8, 560)});
    std::this_thread::sleep_for(100ms);
    const auto back = r.executor->snapshot();
    check(ready && other.lost && other.selected_id == 7 && back.automatic && !back.lost && back.selected_id == 7,
          "coast: another ID never ends it; the same ID ends it and following resumes");
}

void then_search()
{
    SmoothingConfig s = coast_on(200);
    s.search = true;
    s.search_grace_ms = 60; s.search_follow_ms = 150; s.search_follow_px = 1000; s.search_follow_speed = 10;
    s.search_speed = 3; s.search_sweep_px = 24; s.search_widen = 1.5; s.search_max_px = 40; s.search_timeout_ms = 1500;
    Rig r{s};
    const bool ready = r.follow_then_lose(predicted(7, 560));
    std::this_thread::sleep_for(120ms);
    const auto during = r.executor->snapshot();
    int seen_follow = 0, seen_sweep = 0;
    for (int i = 0; i < 40; ++i) {
        const auto x = r.executor->snapshot();
        seen_follow += x.search == 2; seen_sweep += x.search == 3;
        std::this_thread::sleep_for(10ms);
    }
    check(ready && during.moving && during.search == 0 && seen_sweep > 0 && seen_follow == 0,
          "coast + search: the coast first, then the sweep (its follow-through is skipped)");
}

void settings()
{
    const auto rejects = [](int ms) {
        try { Rig r{coast_on(ms)}; } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    check(rejects(0) && rejects(5000) && !rejects(800), "coast: out-of-range --coast-ms is refused");
}
}

int main()
{
    off_by_default();
    coasts_then_stops();
    follows_the_prediction();
    no_prediction_no_coast();
    safety();
    same_id_back();
    then_search();
    settings();
    return failures ? 1 : 0;
}
