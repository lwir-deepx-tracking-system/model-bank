// v27 seek when lost (--seek 1): run the actual executor with a recording IGimbalDevice, frames fed in real time.
// From this folder, use the CORE sources in V27_CHANGES.md:
// c++ -std=c++17 -O2 -pthread -Iv27/include -Iteam/include -Isim/stub test_v27_seek.cpp $CORE -o test_v27_seek && ./test_v27_seek
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

TrackPrediction box(int id, double center_x, double center_y = 240)
{
    TrackPrediction p;
    p.track_id = id; p.width = 40; p.height = 80; p.x = center_x - 20; p.y = center_y - 40;
    return p;
}

SmoothingConfig seek_on(int ms = 600, int speed = 8)
{
    SmoothingConfig s;
    s.seek = true;
    s.seek_ms = ms;
    s.seek_speed = speed;
    return s;
}

struct Rig {
    RuntimeControl runtime;
    Recorder* recorder = nullptr;
    std::unique_ptr<GimbalCommandExecutor> executor;
    std::atomic<std::uint64_t> frame_id{0};
    std::mutex scene_mutex;
    std::vector<Track> scene;
    TrackPrediction prediction, seek;   // PRED / SEEK of the frames
    std::atomic<bool> feeding{true}, closing{false}, pinging{true}, healthy{true};
    std::thread feeder;
    explicit Rig(SmoothingConfig smooth, int reacquire_ms = 3000, int heartbeat_ms = 1500, double speed_rate = 1000)
    {
        ControlConfig config; config.enabled = true;
        config.heartbeat_timeout_ms = heartbeat_ms;
        config.automatic.deadband_x = config.automatic.deadband_y = .06;
        config.automatic.pid_x.kp = config.automatic.pid_y.kp = 30;
        config.automatic.speed_rate = speed_rate;
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
    void set(std::vector<Track> tracks, std::optional<TrackPrediction> s = std::nullopt, std::optional<TrackPrediction> p = std::nullopt)
    {
        std::lock_guard<std::mutex> lock(scene_mutex);
        scene = std::move(tracks);
        seek = s ? *s : TrackPrediction{};
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
        { std::lock_guard<std::mutex> lock(scene_mutex); result->tracks = scene; result->prediction = prediction; result->seek = seek; }
        executor->publish(result, healthy);
    }
    // Follow id 7 standing right of centre until the gimbal moves right, then the person vanishes; the frames then carry
    // the SEEK box s (none = no SEEK) and the PRED box p.
    bool follow_then_lose(std::optional<TrackPrediction> s, std::optional<TrackPrediction> p = std::nullopt, float x = 520)
    {
        set({person(7, x)});
        std::this_thread::sleep_for(60ms);
        if (executor->select(7) != ControlResult::Ok) return false;
        std::this_thread::sleep_for(150ms);
        const auto st = executor->snapshot();
        if (!st.automatic || !st.moving) return false;
        recorder->clear();
        set({}, s, p);
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

int max_speed(const std::vector<Command>& c)
{
    int m = 0;
    for (const auto& x : c) if (x.x || x.y) m = std::max(m, x.speed);
    return m;
}

void off_by_default()
{
    Rig r{SmoothingConfig{}};
    const bool ready = r.follow_then_lose(box(7, 600));
    const auto lost_at = Clock::now();
    std::this_thread::sleep_for(300ms);
    const auto c = r.recorder->read();
    const auto s = r.executor->snapshot();
    check(ready && !c.empty() && !c.front().x && !c.front().y && c.front().when - lost_at < 60ms && moves(c).empty() &&
          s.automatic && s.lost && !s.moving && s.search == 0,
          "default (no --seek): a SEEK suffix is ignored, the loss STOPs at once (team rule)");
}

void follows_capped_then_stops()
{
    Rig r{seek_on(400, 8)};
    const bool ready = r.follow_then_lose(box(7, 620));   // 300 px right of the centre: the PD demand is 28
    const auto lost_at = Clock::now();
    std::this_thread::sleep_for(200ms);
    const auto during = r.executor->snapshot();
    const auto early = r.recorder->read();
    std::this_thread::sleep_for(350ms);
    const auto c = r.recorder->read();
    const auto s = r.executor->snapshot();
    bool right = !moves(early).empty();
    for (const auto& x : moves(early)) right = right && x.x == 1 && x.y == 0;
    Clock::time_point last_stop{};
    for (const auto& x : c) if (!x.x && !x.y) last_stop = x.when;
    int seek_max = moves(c).empty() ? 0 : moves(c).back().speed;   // once the first frame without the person had time to arrive (one feeder period)
    for (const auto& x : c) if ((x.x || x.y) && x.when - lost_at > 25ms) seek_max = std::max(seek_max, x.speed);
    std::printf("      seek at a 300 px error (PD demand 28): highest SPD sent while seeking %d, last SPD %d\n", seek_max, moves(c).empty() ? -1 : moves(c).back().speed);
    check(ready && right && during.moving && during.lost && during.automatic && during.selected_id == 7 && during.search == 5,
          "seek: it turns toward the SEEK box (right), reported lost and SEARCH 5");
    check(!moves(c).empty() && moves(c).back().speed == 8 && seek_max == 8,
          "seek: the speed is cut to --seek-speed at once (a faster following move was under way) and stays there");
    check(last_stop != Clock::time_point{} && last_stop - lost_at >= 380ms && last_stop - lost_at < 500ms &&
          !s.moving && s.lost && s.automatic && s.selected_id == 7 && s.search == 0,
          "seek: STOP after --seek-ms, the selection is kept for the reacquire window, SEARCH 0");
}

void never_above_the_cap_from_rest()
{
    Rig r{seek_on(2000, 5), 3000, 1500, 150};
    r.set({person(7, 330)});   // inside the deadband: the gimbal stands still while following
    std::this_thread::sleep_for(60ms);
    const bool ready = r.executor->select(7) == ControlResult::Ok;
    std::this_thread::sleep_for(100ms);
    const bool still = !r.executor->snapshot().moving;
    r.recorder->clear();
    r.set({}, box(7, 620));
    std::this_thread::sleep_for(500ms);
    const auto m = moves(r.recorder->read());
    bool ramp = !m.empty() && m.front().speed == 1;
    for (std::size_t i = 1; i < m.size(); ++i) ramp = ramp && m[i].speed >= m[i - 1].speed && m[i].speed - m[i - 1].speed <= 2;
    check(ready && still && ramp && max_speed(m) == 5,
          "seek from rest: starts at the minimum speed, ramps with --speed-rate, never above --seek-speed");
}

void turns_through_stop()
{
    Rig r{seek_on(2000, 8)};
    const bool ready = r.follow_then_lose(box(7, 620));
    std::this_thread::sleep_for(150ms);
    const bool right = r.executor->snapshot().moving;
    r.recorder->clear();
    r.set({}, box(7, 20));   // the server's next leg: the other side
    std::this_thread::sleep_for(200ms);
    const auto c = r.recorder->read();
    const auto m = moves(c);
    const bool stop_first = !c.empty() && !c.front().x && !c.front().y;
    bool left = !m.empty() && m.front().speed == 1;
    for (const auto& x : m) left = left && x.x == -1;
    check(ready && right && stop_first && left && max_speed(m) <= 8,
          "seek: a change of direction is a STOP first, then the new direction from the minimum speed");
}

void inside_the_deadband_it_stops()
{
    Rig r{seek_on(2000, 8)};
    const bool ready = r.follow_then_lose(box(7, 620));
    std::this_thread::sleep_for(150ms);
    r.recorder->clear();
    r.set({}, box(7, 325));   // the leg's end is reached: the SEEK point is at the centre
    std::this_thread::sleep_for(150ms);
    const auto c = r.recorder->read();
    const auto s = r.executor->snapshot();
    check(ready && has_stop(c) && moves(c).empty() && !s.moving && s.lost && s.search == 5,
          "seek: with the SEEK point at the centre the gimbal stops and waits there (still SEARCH 5)");
}

void coast_goes_first()
{
    SmoothingConfig sm = seek_on(700, 8);
    sm.coast = true; sm.coast_ms = 250;
    Rig r{sm};
    const bool ready = r.follow_then_lose(box(7, 20), box(7, 620));   // PRED right, SEEK left
    std::this_thread::sleep_for(150ms);
    const auto during_coast = r.executor->snapshot();
    const auto early = moves(r.recorder->read());
    bool right = !early.empty();
    for (const auto& x : early) right = right && x.x == 1;
    std::this_thread::sleep_for(300ms);
    const auto during_seek = r.executor->snapshot();
    const auto m = moves(r.recorder->read());
    check(ready && right && during_coast.search == 0 && during_seek.search == 5 && !m.empty() && m.back().x == -1 && m.back().speed <= 8,
          "coast + seek: the PRED box is followed for --coast-ms (uncapped), then the SEEK box (capped, through STOP)");
}

void no_seek_box_no_motion()
{
    for (int which = 0; which < 3; ++which) {
        Rig r{seek_on(800)};
        std::optional<TrackPrediction> p;
        if (which == 1) p = box(8, 600);   // another track's box
        if (which == 2) { p = box(7, 600); p->width = 0; }   // malformed box
        const bool ready = r.follow_then_lose(p);
        const auto lost_at = Clock::now();
        std::this_thread::sleep_for(200ms);
        const auto c = r.recorder->read();
        check(ready && !c.empty() && !c.front().x && c.front().when - lost_at < 60ms && moves(c).empty(),
              which == 0 ? "seek: no SEEK -> STOP at once" : which == 1 ? "seek: SEEK of another id -> STOP at once"
                                                                         : "seek: an invalid SEEK box -> STOP at once");
    }
    {
        Rig r{seek_on(2000)};
        const bool ready = r.follow_then_lose(box(7, 600));
        std::this_thread::sleep_for(120ms);
        const bool moving = r.executor->snapshot().moving;
        r.set({});   // the server ended the search
        std::this_thread::sleep_for(80ms);
        const auto s = r.executor->snapshot();
        r.recorder->clear();
        std::this_thread::sleep_for(100ms);
        check(ready && moving && !s.moving && s.search == 0 && moves(r.recorder->read()).empty(),
              "seek: when the frames stop carrying SEEK the gimbal stops at once");
    }
}

void safety()
{
    {
        Rig r{seek_on(2000)};
        const bool ready = r.follow_then_lose(box(7, 600));
        std::this_thread::sleep_for(100ms);
        const bool moving = r.executor->snapshot().moving;
        r.healthy = false;   // frames marked OBS 0
        std::this_thread::sleep_for(60ms);
        const auto s = r.executor->snapshot();
        r.recorder->clear();
        std::this_thread::sleep_for(100ms);
        check(ready && moving && !s.moving && moves(r.recorder->read()).empty(), "seek: an unhealthy frame (OBS 0) stops it at once");
    }
    {
        Rig r{seek_on(2000)};
        const bool ready = r.follow_then_lose(box(7, 600));
        std::this_thread::sleep_for(100ms);
        r.feeding = false;   // camera froze
        std::this_thread::sleep_for(300ms);
        const auto s = r.executor->snapshot();
        r.recorder->clear();
        std::this_thread::sleep_for(100ms);
        check(ready && !s.moving && moves(r.recorder->read()).empty(), "seek: no fresh frame -> STOP within the input timeout");
    }
    {
        Rig r{seek_on(2000), 3000, 300};
        const bool ready = r.follow_then_lose(box(7, 600));
        std::this_thread::sleep_for(100ms);
        r.pinging = false;
        std::this_thread::sleep_for(450ms);
        r.recorder->clear();
        std::this_thread::sleep_for(100ms);
        const auto s = r.executor->snapshot();
        check(ready && !s.automatic && !s.moving && s.search == 0 && moves(r.recorder->read()).empty(),
              "seek: a lost heartbeat stops it and drops the selection");
    }
    {
        Rig r{seek_on(2000)};
        const bool ready = r.follow_then_lose(box(7, 600));
        std::this_thread::sleep_for(100ms);
        r.executor->halt();
        r.recorder->clear();
        std::this_thread::sleep_for(100ms);
        const auto s = r.executor->snapshot();
        check(ready && !s.automatic && !s.moving && s.search == 0 && moves(r.recorder->read()).empty(), "seek: STOP/HOME (H) ends it at once");
    }
    {
        Rig r{seek_on(2000)};
        const bool ready = r.follow_then_lose(box(7, 600));
        std::this_thread::sleep_for(100ms);
        r.executor->manual();
        r.recorder->clear();
        std::this_thread::sleep_for(100ms);
        const auto s = r.executor->snapshot();
        check(ready && !s.automatic && !s.moving && moves(r.recorder->read()).empty(), "seek: manual mode ends it at once");
    }
    {
        Rig r{seek_on(2000), 400};
        const bool ready = r.follow_then_lose(box(7, 600));
        std::this_thread::sleep_for(550ms);
        const auto s = r.executor->snapshot();
        check(ready && !s.automatic && !s.moving, "seek: the reacquire timeout still drops the selection");
    }
}

void same_id_back()
{
    Rig r{seek_on(2000)};
    const bool ready = r.follow_then_lose(box(7, 600));
    std::this_thread::sleep_for(100ms);
    r.set({person(8, 560)}, box(7, 600));   // another person: never followed
    std::this_thread::sleep_for(100ms);
    const auto other = r.executor->snapshot();
    r.set({person(7, 330), person(8, 560)}, box(7, 600));   // the id is back (a late SEEK on the same frame is ignored)
    std::this_thread::sleep_for(100ms);
    const auto back = r.executor->snapshot();
    check(ready && other.lost && other.selected_id == 7 && other.search == 5 &&
          back.automatic && !back.lost && back.selected_id == 7 && back.search == 0,
          "seek: another ID never ends it; the same ID ends it and following resumes (SEARCH 0)");
}

void settings()
{
    const auto rejects = [](SmoothingConfig s) {
        try { Rig r{s}; } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    SmoothingConfig both = seek_on();
    both.search = true;
    check(rejects(seek_on(0)) && rejects(seek_on(20000)) && rejects(seek_on(600, 0)) && rejects(seek_on(600, 31)) &&
          rejects(both) && !rejects(seek_on(4500, 8)),
          "seek: out-of-range --seek-ms / --seek-speed and --seek with --search are refused");
}
}

int main()
{
    off_by_default();
    follows_capped_then_stops();
    never_above_the_cap_from_rest();
    turns_through_stop();
    inside_the_deadband_it_stops();
    coast_goes_first();
    no_seek_box_no_motion();
    safety();
    same_id_back();
    settings();
    return failures ? 1 : 0;
}
