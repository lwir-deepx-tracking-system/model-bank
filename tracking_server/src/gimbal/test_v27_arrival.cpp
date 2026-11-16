// v27 arrival control (SmoothingConfig::arrival_control, gimbal_bridge_v27 --arrival-control): the control step runs when a
// tracking result arrives instead of at the worker's next 25 ms period. The real executor with a recording IGimbalDevice.
//  1. on the period   results that arrive exactly when the period is due give the SAME command list (time, X, Y, SPD) with
//                     the option off and on, over a long scenario (ramps, deadband, reversals, both axes, LEAD, coast, losses)
//  2. between periods a result is acted on at its own instant (off: 0..25 ms later); the SPD ramp, the soft deadband stop, the
//                     reversal hold, the coast time, the input timeout, the reacquire timeout and the heartbeat take the same
//                     real time with the option off and on
//  3. real clock      time from publish() to the device command, option off and on (printed; bounds are loose)
// Parts 1 and 2 run on a clock this file controls: std::chrono::steady_clock::now() is replaced (g++ / libstdc++), time moves
// only when the test says so, and after every move the test waits until the worker has done everything due at that instant.
// Part 1 needs "the result is taken in first, then the period is found due" at one and the same instant. For that the
// executor's worker thread has its own view of the time, and at such an instant that view moves on at the moment the worker
// takes the result in (it then releases the previous result; the hook is the deleter of the result, see Rig::publish).
// From this folder, with the CORE sources of V27_CHANGES.md:
// c++ -std=c++17 -O2 -pthread -Iv27/include -Iteam/include -Isim/stub test_v27_arrival.cpp $CORE -o test_v27_arrival && ./test_v27_arrival
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>
#include "control/gimbal_command_executor.hpp"

#if !defined(__GLIBCXX__)
#error "test_v27_arrival.cpp replaces std::chrono::steady_clock::now() and is written for g++ / libstdc++"
#endif

namespace {
std::atomic<bool> g_fake{false};
std::atomic<long long> g_now_ns{0};      // the controlled time, as this (the test's) thread sees it
std::atomic<long long> g_worker_ns{0};   // ... as every other thread (the executor's worker) sees it
std::thread::id g_test_thread;
long long real_ns()
{
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
}

namespace std { namespace chrono { inline namespace _V2 {
steady_clock::time_point steady_clock::now() noexcept
{
    if (!g_fake.load()) return time_point(nanoseconds(real_ns()));
    return time_point(nanoseconds(std::this_thread::get_id() == g_test_thread ? g_now_ns.load() : g_worker_ns.load()));
}
}}}

namespace {
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

int failures = 0;
void check(bool ok, const char* name) { std::printf("%s %s\n", ok ? "ok" : "FAIL", name); failures += !ok; }

struct Command {
    int x, y, speed;
    double ms;
    bool stop() const { return !x && !y; }
    bool operator==(const Command& o) const { return x == o.x && y == o.y && speed == o.speed && ms == o.ms; }
};

struct Recorder : IGimbalDevice {
    std::mutex mutex;
    std::vector<Command> commands;
    long long origin_ns = 0;
    void add(int x, int y, int speed)
    {
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
        std::lock_guard<std::mutex> lock(mutex);
        commands.push_back({x, y, speed, double(ns - origin_ns) / 1e6});
    }
    void move(int x, int y) override { add(x, y, 0); }
    void move_with_speed(int x, int y, int speed) override { add(x, y, speed); }
    void stop() override { add(0, 0, 0); }
    void home() override {}
    std::vector<Command> read() { std::lock_guard<std::mutex> lock(mutex); return commands; }
};

struct Options {
    bool arrival = false;
    bool p_only = false;          // P controller (default: the live PD 30 / 6)
    double speed_rate = 150;      // the live --speed-rate
    double stop_rate = 0;
    int reacquire_ms = 10000;
    int heartbeat_ms = 100000000;
};

// px from the picture centre of a 640x480 frame; the box is 40x80 and aimed at its centre
struct Scene {
    bool person = true;
    double x = 0, y = 0;
    bool pred = false;       // PRED suffix for the followed id (coast)
    double pred_x = 0;
    double lead = 0;         // LEAD dx in px (0 = none)
};

struct Rig {
    RuntimeControl runtime;
    Recorder* recorder = nullptr;
    std::unique_ptr<GimbalCommandExecutor> executor;
    long long origin_ns = 0;
    std::uint64_t frame_id = 0, epoch = 0;
    ControlSnapshot state;
    Rig(const Options& o, bool fake)
    {
        g_fake = fake;
        // Far in the real future: a libstdc++ that waits on the absolute steady time then never times out by itself.
        origin_ns = fake ? real_ns() + 100000LL * 1000000000LL : real_ns();
        g_test_thread = std::this_thread::get_id();
        g_now_ns = origin_ns; g_worker_ns = origin_ns;
        ControlConfig config; config.enabled = true;
        config.heartbeat_timeout_ms = o.heartbeat_ms;
        auto& a = config.automatic;
        a.controller = o.p_only ? AutoControllerType::P : AutoControllerType::PID;   // the live options: --controller pid --ki 0 --kd 6
        a.pid_x.kp = a.pid_y.kp = 30; a.pid_x.ki = a.pid_y.ki = 0; a.pid_x.kd = a.pid_y.kd = 6;
        a.deadband_x = a.deadband_y = .06;
        a.speed_min = 1; a.speed_max = 100; a.speed_rate = o.speed_rate;
        a.period_ms = config.poll_interval_ms = 25;
        a.input_timeout_ms = 250; a.reacquire_timeout_ms = o.reacquire_ms;
        SmoothingConfig s;   // the live options: --deadband-exit 0.10 --axis-reversal 1 --coast 1 --coast-ms 800 --far-gain 1.6 --far-start-px 240 --far-full-px 320
        s.deadband_exit_x = s.deadband_exit_y = .10; s.axis_reversal = true;
        s.coast = true; s.coast_ms = 800;
        s.far_gain = 1.6; s.far_start_px = 240; s.far_full_px = 320;
        s.stop_rate = o.stop_rate;
        s.arrival_control = o.arrival;
        auto device = std::make_unique<Recorder>();
        recorder = device.get();
        recorder->origin_ns = origin_ns;
        executor = std::make_unique<GimbalCommandExecutor>(config, std::move(device), runtime, s);
        executor->connected();
        barrier();
    }
    void at(double ms) { g_now_ns.store(origin_ns + std::llround(ms * 1e6)); g_worker_ns.store(g_now_ns.load()); }
    // Part 1: the instant begins for this thread only; the worker's view follows when it takes in the result published next.
    void at_for_result(double ms) { g_now_ns.store(origin_ns + std::llround(ms * 1e6)); }
    void worker_catches_up() { g_worker_ns.store(g_now_ns.load()); }
    // Two synchronous round trips through the worker: everything due at this instant has been done when it returns.
    void barrier() { executor->snapshot(); state = executor->snapshot(); epoch = state.command_generation; }
    void publish(const Scene& s)
    {
        auto context = std::make_shared<FrameContext>();
        context->image = cv::Mat(480, 640, CV_16UC1);
        context->metadata.frame_id = ++frame_id;
        context->metadata.captured_at = Clock::now() - 30ms;
        // The worker releases the previous result when it takes in a new one (ingest); from then on it sees the new instant.
        std::shared_ptr<TrackingResult> result(new TrackingResult, [](TrackingResult* p) { delete p; g_worker_ns.store(g_now_ns.load()); });
        result->frame = context;
        result->model_generation = runtime.generation.load();
        result->tracker_epoch = runtime.tracker_epoch.load();
        if (s.person) {
            Track t;
            t.track_id = 7; t.confidence = .9f; t.width = 40; t.height = 80;
            t.x = float(320 + s.x - 20); t.y = float(240 + s.y - 40);
            result->tracks.push_back(t);
        }
        if (s.pred) { result->prediction.track_id = 7; result->prediction.x = 320 + s.pred_x - 20; result->prediction.y = 200;
                      result->prediction.width = 40; result->prediction.height = 80; }
        if (s.lead != 0) result->lead = {7, s.lead, 0, epoch};
        executor->publish(result, true);
    }
};

// ---------- 1. results that arrive exactly when the period is due
struct Random {
    unsigned long long s;
    explicit Random(unsigned seed) : s(seed * 2654435761ULL + 88172645463325252ULL) {}
    unsigned next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return unsigned(s >> 16); }
    int below(int n) { return int(next() % unsigned(n)); }
    double uniform(double a, double b) { return a + (b - a) * (next() % 100000) / 100000.0; }
};

std::vector<Command> on_period_run(bool arrival, unsigned seed)
{
    Options o; o.arrival = arrival; o.reacquire_ms = 1500;
    Rig r(o, true);            // the worker's period starts at 0 ms: due at 25, 50, 75, ...
    Random rng(seed);
    Scene s; s.x = 150;
    int next_frame = 1, frames = 0, left = 0, mode = 0;
    double vx = 0;
    for (int k = 1; k <= 2400; ++k) {   // 60 s
        if (k != next_frame) { r.at(25.0 * k); r.barrier(); continue; }
        r.at_for_result(25.0 * k);
        const int gap = 1 + (rng.below(5) + 3) / 3;   // 2 or 3 periods between results, sometimes 1... (1 + {1,1,1,2,2})
        const double dt = gap * .025;
        if (left <= 0) {
            mode = rng.below(9); left = 4 + rng.below(12);
            const int side = s.x >= 0 ? 1 : -1;
            s.person = true; s.pred = false; s.lead = 0; vx = 0;
            if (mode == 0) vx = -side * rng.uniform(100, 450);                 // approach the centre
            else if (mode == 1) vx = side * rng.uniform(200, 900);             // outward
            else if (mode == 2) s.x = -s.x + rng.uniform(-20, 20);             // other side: reversal
            else if (mode == 3) s.y = (rng.below(2) ? 1 : -1) * rng.uniform(40, 170);   // vertical demand too
            else if (mode == 4) s.y = 0;
            else if (mode == 5) { s.person = false; s.pred = true; s.pred_x = s.x; left = 8 + rng.below(20); }   // coast
            else if (mode == 6) { s.person = false; left = 2 + rng.below(6); }   // lost, no prediction
            else if (mode == 7) s.x = (rng.below(2) ? 1 : -1) * rng.uniform(200, 310);   // far (far gain)
            // mode 8: hold
        }
        if (mode == 5) s.pred_x = std::clamp(s.pred_x + rng.uniform(-200, 200) * dt, -300.0, 300.0);
        else if (s.person) {
            s.x = std::clamp(s.x + vx * dt + rng.uniform(-2, 2), -310.0, 310.0);
            if (mode == 0 && std::abs(s.x) < 15) vx = 0;
            s.lead = rng.below(4) == 0 ? std::clamp(vx * .2, -55.0, 55.0) : 0;
        }
        --left;
        r.publish(s);          // the worker takes it in, then finds the period due at the same instant
        r.barrier();
        r.worker_catches_up(); // (the very first result has no previous one to release)
        if (!r.state.automatic && s.person && frames >= 2 && r.executor->select(7) == ControlResult::Ok) r.barrier();
        ++frames;
        next_frame = k + gap;
    }
    return r.recorder->read();
}

void on_period_same_commands()
{
    const auto off = on_period_run(false, 20261115), on = on_period_run(true, 20261115);
    const bool same = off == on && off == on_period_run(false, 20261115);   // and the same again: the run is deterministic
    int stops = 0, vertical = 0, top = 0;
    for (const auto& c : off) { stops += c.stop(); vertical += c.y != 0; top = std::max(top, c.speed); }
    std::printf("   on the period: %zu commands off, %zu on; %d STOP, %d with a vertical move, top SPD %d\n",
                off.size(), on.size(), stops, vertical, top);
    if (!same)
        for (std::size_t i = 0; i < std::min(off.size(), on.size()); ++i)
            if (!(off[i] == on[i])) {
                std::printf("   first difference at %zu: off %.0f ms %d %d %d, on %.0f ms %d %d %d\n", i,
                            off[i].ms, off[i].x, off[i].y, off[i].speed, on[i].ms, on[i].x, on[i].y, on[i].speed);
                break;
            }
    check(off.size() > 400 && stops >= 20 && vertical >= 20 && top >= 30, "on the period: the scenario exercises ramps, STOPs, both axes and the far gain");
    check(same, "on the period: results arriving exactly when the period is due give the identical command list, option off and on");
}

// ---------- 2. results between the periods (every 33 ms from 7 ms on), time in 1 ms steps
const int FIRST_MS = 7, FRAME_MS = 33;
bool frame_due(int t) { return t >= FIRST_MS && (t - FIRST_MS) % FRAME_MS == 0; }

// scene(t) for a due frame; every(t) after each ms. Selects id 7 after the third frame.
void run(Rig& r, int until_ms, const std::function<std::optional<Scene>(int)>& scene, const std::function<void(int)>& every = {})
{
    int frames = 0;
    bool selected = false;
    for (int t = 1; t <= until_ms; ++t) {
        r.at(t);
        r.barrier();                       // the period timer and every time limit due at this instant
        if (frame_due(t)) {
            if (const auto s = scene(t)) { r.publish(*s); r.barrier(); ++frames; }
            if (!selected && frames == 3) { selected = r.executor->select(7) == ControlResult::Ok; r.barrier(); }
        }
        if (every) every(t);
    }
}

double first_stop_after(const std::vector<Command>& c, double ms)
{
    for (const auto& x : c) if (x.stop() && x.ms >= ms) return x.ms;
    return -1;
}

void between_periods_latency()
{
    double mean[2] = {0, 0}, worst[2] = {0, 0};
    bool every_frame[2] = {true, true};
    for (const bool arrival : {false, true}) {
        Options o; o.arrival = arrival; o.p_only = true; o.speed_rate = 1e6;   // SPD = the demand of the newest frame at once
        Rig r(o, true);
        run(r, 4000, [](int t) { Scene s; s.x = ((t - FIRST_MS) / FRAME_MS) % 2 ? 200 : 100; return std::optional<Scene>(s); });
        const auto c = r.recorder->read();
        int n = 0;
        for (int t = FIRST_MS + 8 * FRAME_MS; t + FRAME_MS <= 4000; t += FRAME_MS) {
            const auto it = std::find_if(c.begin(), c.end(), [t](const Command& x) { return x.ms >= t; });
            if (it == c.end() || it->ms >= t + FRAME_MS || it->stop()) { every_frame[arrival] = false; continue; }
            mean[arrival] += it->ms - t; worst[arrival] = std::max(worst[arrival], it->ms - t); ++n;
        }
        mean[arrival] /= std::max(1, n);
    }
    std::printf("   result -> command: off mean %.1f ms, worst %.0f ms; on mean %.1f ms, worst %.0f ms (controlled clock)\n",
                mean[0], worst[0], mean[1], worst[1]);
    check(every_frame[0] && every_frame[1], "between periods: every result changes the command, option off and on");
    check(mean[0] > 9 && mean[0] < 16 && worst[0] <= 25, "between periods, off: a result waits for the period (0..25 ms, about 12.5 on average)");
    check(mean[1] == 0 && worst[1] == 0, "between periods, on: the command leaves at the instant the result arrives");
}

void between_periods_ramp()
{
    double reach[2] = {-1, -1};
    bool law[2] = {true, true};
    int steps[2] = {0, 0};
    for (const bool arrival : {false, true}) {
        Options o; o.arrival = arrival;
        Rig r(o, true);
        run(r, 1500, [](int) { Scene s; s.x = 300; return std::optional<Scene>(s); });   // demand 28.1 x far gain 1.45 = 40.8
        const auto c = r.recorder->read();
        const auto first = std::find_if(c.begin(), c.end(), [](const Command& x) { return x.x == 1; });
        if (first == c.end() || first->speed != 1) { law[arrival] = false; continue; }
        for (auto it = first; it != c.end(); ++it) {
            const double expected = std::min(40.78125, 1 + .150 * (it->ms - first->ms));   // --speed-rate 150 SPD/s on the clock
            if (it->x != 1 || it->y != 0 || std::abs(it->speed - expected) > .5 + 1e-6) law[arrival] = false;
            if (it->speed == 41 && reach[arrival] < 0) reach[arrival] = it->ms - first->ms;
            ++steps[arrival];
        }
    }
    std::printf("   ramp SPD 1 -> 41 at 150 SPD/s: off %.0f ms in %d commands, on %.0f ms in %d commands (263.3 ms + rounding to a step)\n",
                reach[0], steps[0], reach[1], steps[1]);
    check(law[0] && law[1], "ramp: every command equals 1 + 150 SPD/s x the time since the first move, option off and on");
    check(reach[0] >= 263 && reach[0] <= 289 && reach[1] >= 263 && reach[1] <= 289, "ramp: SPD 41 is reached 263..289 ms after the first move, option off and on");
}

void between_periods_soft_stop()
{
    const int centre_at = FIRST_MS + 45 * FRAME_MS;   // the first frame inside the deadband
    double took[2] = {-1, -1};
    bool law[2] = {true, true};
    for (const bool arrival : {false, true}) {
        Options o; o.arrival = arrival; o.stop_rate = 40;
        Rig r(o, true);
        run(r, centre_at + 800, [centre_at](int t) { Scene s; s.x = t >= centre_at ? 5 : 200; return std::optional<Scene>(s); });
        const auto c = r.recorder->read();
        const double stop = first_stop_after(c, centre_at);
        took[arrival] = stop - centre_at;
        int n = 0;
        for (const auto& x : c) {
            if (x.ms < centre_at || x.ms >= stop) continue;
            // 18.75 SPD falling at --stop-rate 40 SPD/s; the previous output was at most one period before or after the frame
            if (x.x != 1 || std::abs(x.speed - (18.75 - .040 * (x.ms - centre_at))) > 1.5 + 1e-6) law[arrival] = false;
            ++n;
        }
        if (n < 10) law[arrival] = false;
    }
    std::printf("   soft deadband stop from SPD 18.75 at 40 SPD/s: STOP after %.0f ms off, %.0f ms on (431 ms +- one period)\n", took[0], took[1]);
    check(law[0] && law[1], "soft stop: SPD falls by stop-rate x real time, option off and on");
    check(took[0] >= 400 && took[0] <= 460 && took[1] >= 400 && took[1] <= 460, "soft stop: STOP 400..460 ms after the frame in the deadband, option off and on");
}

void between_periods_time_limits()
{
    const int change = FIRST_MS + 30 * FRAME_MS;   // 997 ms: following at full demand by then
    // (a) the person is gone, the frames go on: STOP, then the selection is dropped after --reacquire-ms
    double stop_a[2], drop_a[2];
    // (b) no frames at all: STOP when the last frame is 250 ms old
    double stop_b[2];
    // (c) the person is gone but the frames carry PRED: coast for --coast-ms 800, then STOP
    double stop_c[2]; bool coasted[2];
    // (d) the heartbeat stops: STOP and selection dropped after the heartbeat time
    double stop_d[2];
    // (e) one axis reverses while the other moves: the reversed axis waits --reversal-hold-ms 500
    double hold_e[2];
    for (const bool arrival : {false, true}) {
        {
            Options o; o.arrival = arrival; o.reacquire_ms = 1000;
            Rig r(o, true);
            double dropped = -1;
            run(r, change + 1300, [change](int t) { Scene s; s.x = 250; s.person = t < change; return std::optional<Scene>(s); },
                [&](int t) { if (t > change && dropped < 0 && !r.state.automatic) dropped = t; });
            stop_a[arrival] = first_stop_after(r.recorder->read(), change) - change;
            drop_a[arrival] = dropped - change;
        }
        {
            Options o; o.arrival = arrival;
            Rig r(o, true);
            run(r, change + 500, [change](int t) { Scene s; s.x = 250; return t <= change ? std::optional<Scene>(s) : std::nullopt; });
            stop_b[arrival] = first_stop_after(r.recorder->read(), change) - change;
        }
        {
            Options o; o.arrival = arrival;
            Rig r(o, true);
            bool moving = true;
            run(r, change + 1100, [change](int t) { Scene s; s.x = 250; s.person = t < change; s.pred = !s.person; s.pred_x = 200; return std::optional<Scene>(s); },
                [&](int t) { if (t > change + 30 && t < change + 790 && !r.state.moving) moving = false; });
            stop_c[arrival] = first_stop_after(r.recorder->read(), change) - change;
            coasted[arrival] = moving;
        }
        {
            Options o; o.arrival = arrival; o.heartbeat_ms = 500;
            Rig r(o, true);
            run(r, change + 800, [](int) { Scene s; s.x = 250; return std::optional<Scene>(s); },
                [&](int t) { if (t % 100 == 0 && t <= 1000) { r.executor->ping(); r.barrier(); } });
            stop_d[arrival] = first_stop_after(r.recorder->read(), 1000) - 1000;
            if (r.state.automatic) stop_d[arrival] = -2;
        }
        {
            Options o; o.arrival = arrival;
            Rig r(o, true);
            run(r, change + 900, [change](int t) { Scene s; s.x = t < change ? 250 : -100; s.y = 150; return std::optional<Scene>(s); });
            double halted = -1, rejoined = -1;
            for (const auto& x : r.recorder->read()) {
                if (x.ms < change) continue;
                if (halted < 0 && x.x == 0 && x.y != 0) halted = x.ms;
                if (halted >= 0 && rejoined < 0 && x.x == -1) rejoined = x.ms;
            }
            hold_e[arrival] = halted >= 0 && rejoined >= 0 ? rejoined - halted : -1;
        }
    }
    std::printf("   person gone: STOP after %.0f ms off, %.0f ms on; selection dropped %.0f / %.0f ms after the frame (reacquire 1000)\n",
                stop_a[0], stop_a[1], drop_a[0], drop_a[1]);
    std::printf("   frames stop: STOP %.0f ms off, %.0f ms on after the last frame arrived (30 ms old; input timeout 250)\n", stop_b[0], stop_b[1]);
    std::printf("   coast: STOP %.0f ms off, %.0f ms on after the frame without the person (coast 800)\n", stop_c[0], stop_c[1]);
    std::printf("   heartbeat: STOP %.0f ms off, %.0f ms on after the last heartbeat (500)\n", stop_d[0], stop_d[1]);
    std::printf("   reversal hold: the reversed axis rejoins %.0f ms off, %.0f ms on after it halted (500)\n", hold_e[0], hold_e[1]);
    check(stop_a[0] > 0 && stop_a[0] <= 25 && stop_a[1] == 0, "person gone: STOP within one period (off) / at the frame (on)");
    check(drop_a[0] >= 1000 && drop_a[0] <= 1051 && drop_a[1] >= 1000 && drop_a[1] <= 1026, "reacquire timeout: 1000 ms after the loss was seen, within one period, option off and on");
    check(stop_b[0] >= 220 && stop_b[0] <= 245 && stop_b[1] >= 220 && stop_b[1] <= 245, "input timeout: STOP when the last frame is 250 ms old, within one period, option off and on (the timer still runs)");
    check(coasted[0] && coasted[1] && stop_c[0] >= 800 && stop_c[0] <= 851 && stop_c[1] >= 800 && stop_c[1] <= 826, "coast: moves on for 800 ms after the loss was seen, then STOP, option off and on");
    check(stop_d[0] >= 500 && stop_d[0] <= 526 && stop_d[1] >= 500 && stop_d[1] <= 526, "heartbeat: STOP and selection dropped 500 ms after the last heartbeat, within one period, option off and on");
    check(hold_e[0] >= 500 && hold_e[0] <= 526 && hold_e[1] >= 500 && hold_e[1] <= 526, "reversal hold: 500 ms on the clock, within one period, option off and on");
}

// ---------- 3. the real clock: publish() -> device command
void real_clock_latency()
{
    double median[2] = {0, 0}, p90[2] = {0, 0}, mean[2] = {0, 0}, worst[2] = {0, 0};
    for (const bool arrival : {false, true}) {
        Options o; o.arrival = arrival; o.p_only = true; o.speed_rate = 1e6;
        Rig r(o, false);
        std::vector<double> sent;
        const auto start = Clock::now();
        for (int i = 0; i < 130; ++i) {
            std::this_thread::sleep_until(start + i * 33400us);
            Scene s; s.x = i % 2 ? 200 : 100;
            sent.push_back(double(real_ns() - r.origin_ns) / 1e6);
            r.publish(s);
            if (i == 3) r.executor->select(7);
            if (i % 3 == 0) r.executor->ping();
        }
        std::this_thread::sleep_for(60ms);
        const auto c = r.recorder->read();
        std::vector<double> delay;
        for (std::size_t i = 10; i + 1 < sent.size(); ++i) {
            const auto it = std::find_if(c.begin(), c.end(), [&](const Command& x) { return x.ms >= sent[i]; });
            if (it != c.end() && it->ms < sent[i + 1]) delay.push_back(it->ms - sent[i]);
        }
        if (delay.size() < 100) { check(false, "real clock: a command for every result"); return; }
        std::sort(delay.begin(), delay.end());
        median[arrival] = delay[delay.size() / 2]; p90[arrival] = delay[delay.size() * 9 / 10]; worst[arrival] = delay.back();
        for (const double d : delay) mean[arrival] += d / delay.size();
    }
    std::printf("   real clock, result -> device command: off mean %.2f median %.2f p90 %.2f worst %.2f ms; on mean %.3f median %.3f p90 %.3f worst %.3f ms\n",
                mean[0], median[0], p90[0], worst[0], mean[1], median[1], p90[1], worst[1]);
    check(mean[0] > 6 && mean[0] < 19, "real clock, off: about half a period of waiting (6..19 ms mean)");
    check(median[1] < 2 && p90[1] < 3, "real clock, on: the command leaves within 2 ms of the result (median), 3 ms (90 %)");
}
}

int main()
{
    check(!SmoothingConfig{}.arrival_control, "default SmoothingConfig: arrival control off (the team rule)");
    on_period_same_commands();
    between_periods_latency();
    between_periods_ramp();
    between_periods_soft_stop();
    between_periods_time_limits();
    real_clock_latency();
    std::printf(failures ? "FAILED %d\n" : "ALL OK\n", failures);
    return failures ? 1 : 0;
}
