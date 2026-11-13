// v27: unit tests of the motion-smoothing changes in v27/ (patched copies of the team's AutoTrackingController and
// GimbalCommandExecutor), without the bridge, the GUI code or a gimbal. Exit code 0 only when every check passes.
//  1. hysteresis    AutoTrackingController alone, deterministic (no threads, no clock)
//  2. defaults      SmoothingConfig{} keeps the team rules (re-move above the deadband, STOP of both axes on reversal, immediate deadband STOP)
//  3. reversal      GimbalCommandExecutor with a recording fake device: --axis-reversal halts only the reversing axis
//  4. soft stop     GimbalCommandExecutor: --stop-rate steps the speed down before the deadband STOP
// Build: see V27_CHANGES.md (needs OpenCV core headers, or sim/stub on a machine without OpenCV).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "control/gimbal_command_executor.hpp"

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

AutoControlConfig control_config()
{
    AutoControlConfig a;
    a.controller = AutoControllerType::PD;
    a.pid_x.kp = a.pid_y.kp = 30; a.pid_x.kd = a.pid_y.kd = 6; a.pid_x.ki = a.pid_y.ki = 0;
    a.deadband_x = a.deadband_y = 0.06;
    a.min_confidence = 0.1;
    a.speed_max = 30;
    a.reacquire_timeout_ms = 10000;
    return a;
}

// one 40x80 person whose centre is at normalized error (ex, ey) of a 640x480 image (+ey = below the centre)
Track person(double ex, double ey)
{
    Track t; t.track_id = 7; t.width = 40; t.height = 80; t.confidence = 0.9f;
    t.x = float(320 + ex * 320 - 20); t.y = float(240 + ey * 240 - 40);
    return t;
}

// ---------- 1 + 2a: hysteresis on the controller alone
bool moves(AutoTrackingController& c, double ex, std::uint64_t& id, std::chrono::steady_clock::time_point& t)
{
    FrameMetadata m; m.frame_id = ++id; t += std::chrono::milliseconds(33); m.captured_at = t;
    return !c.calculate(person(ex, 0.0), 640, 480, m).stop();
}

void test_hysteresis()
{
    for (const bool with_exit : {false, true}) {
        SmoothingConfig s; if (with_exit) s.deadband_exit_x = s.deadband_exit_y = 0.10;
        AutoTrackingController c(control_config(), s);
        std::uint64_t id = 0; auto t = std::chrono::steady_clock::now() - std::chrono::seconds(10);
        const bool far = moves(c, 0.50, id, t);      // outside everything
        const bool centre = moves(c, 0.01, id, t);   // inside the deadband
        const bool mid = moves(c, 0.08, id, t);      // above deadband 0.06, below exit 0.10
        const bool out = moves(c, 0.12, id, t);      // above exit 0.10
        if (with_exit) {
            check(far && !centre && !mid && out, "hysteresis: held at 0.08 after centring, moves again at 0.12");
        } else {
            check(far && !centre && mid && out, "defaults: no hysteresis, moves again at 0.08 (team rule)");
        }
    }
}

// ---------- 5: --aim-screen-y: the aim point is held at that height of the picture instead of the centre
void test_aim_screen_y()
{
    const auto demand = [](double screen_y, double head_y_px) {
        SmoothingConfig s; s.aim_screen_y = screen_y;
        AutoTrackingController c(control_config(), s);
        Track t; t.track_id = 7; t.width = 40; t.height = 80; t.confidence = 0.9f; t.x = 300; t.y = float(head_y_px - 0.5 * 80);   // aim_y 0.5: the aim point is the box centre
        FrameMetadata m; m.frame_id = 1; m.captured_at = std::chrono::steady_clock::now() - std::chrono::seconds(1);
        return c.calculate(t, 640, 480, m);
    };
    check(demand(0.5, 240).stop(), "aim-screen-y default 0.5: a box centred on the picture centre is at rest (team rule)");
    const auto low = demand(0.5, 160);
    check(!low.stop() && std::fabs(low.error_y - (160 - 240) / 240.0) < 1e-6, "aim-screen-y 0.5: aim point at 160 px is 80 px above the target");
    check(demand(1.0 / 3.0, 160).stop(), "aim-screen-y 1/3: a box whose aim point is at 160 px (one third) is at rest");
    const auto mid = demand(1.0 / 3.0, 240);
    check(!mid.stop() && std::fabs(mid.error_y - (240 - 160) / 240.0) < 1e-6, "aim-screen-y 1/3: aim point at the centre is 80 px below the target");
}

// ---------- 6: --far-gain / --brake-px-per-spd shape the speed demand; defaults leave it unchanged
void test_far_gain_and_brake()
{
    const auto speed = [](double far_gain, double brake, double ex_px) {
        SmoothingConfig s; s.far_gain = far_gain; s.brake_px_per_spd = brake;
        AutoControlConfig a = control_config(); a.speed_max = 100;
        AutoTrackingController c(a, s);
        FrameMetadata m; m.frame_id = 1; m.captured_at = std::chrono::steady_clock::now() - std::chrono::seconds(1);
        return c.calculate(person(ex_px / 320.0, 0.0), 640, 480, m).speed;
    };
    const double p150 = 30.0 * 150.0 / 320.0;   // the PD demand of a first frame (no D yet) at 150 px
    check(std::fabs(speed(1.0, 0.0, 150) - p150) < 1e-9, "far gain / brake off: speed = PD demand (team rule)");
    check(std::fabs(speed(2.0, 0.0, 50) - 30.0 * 50 / 320.0) < 1e-9, "far gain 2: no change below far_start (64 px)");
    check(std::fabs(speed(2.0, 0.0, 200) - 2.0 * 30.0 * 200 / 320.0) < 1e-9, "far gain 2: full gain beyond far_full (160 px)");
    check(std::fabs(speed(2.0, 0.0, 112) - 1.5 * 30.0 * 112 / 320.0) < 1e-9, "far gain 2: half way at 112 px");
    check(std::fabs(speed(1.0, 4.0, 150) - std::min(p150, (150 - 19.2) / 4.0)) < 1e-9, "brake 4: no cap where the cap is above the demand");
    check(std::fabs(speed(4.0, 4.0, 150) - (150 - 19.2) / 4.0) < 1e-9, "far gain 4 + brake 4: capped at (|e| - deadband) / 4");
    check(speed(4.0, 40.0, 40) >= 1.0, "brake never asks less than the minimum speed while moving");
}

// ---------- executor with a recording fake device
struct Command { int x, y, speed; bool stop; std::chrono::steady_clock::time_point at{}; };
struct Recorder {
    std::mutex m; std::vector<Command> log;
    void add(Command c) { c.at = std::chrono::steady_clock::now(); std::lock_guard<std::mutex> l(m); log.push_back(c); }
    std::vector<Command> since(std::size_t n) { std::lock_guard<std::mutex> l(m); return {log.begin() + std::min(n, log.size()), log.end()}; }
    std::size_t size() { std::lock_guard<std::mutex> l(m); return log.size(); }
};
struct FakeDevice : IGimbalDevice {
    Recorder& r;
    explicit FakeDevice(Recorder& rec) : r(rec) {}
    void move(int x, int y) override { r.add({x, y, 0, false}); }
    void move_with_speed(int x, int y, int s) override { r.add({x, y, s, false}); }
    void stop() override { r.add({0, 0, 0, true}); }
    void home() override {}
};

class Rig {
public:
    Recorder rec;
    RuntimeControl runtime;
    std::unique_ptr<GimbalCommandExecutor> ex;
    std::uint64_t id = 0;
    explicit Rig(SmoothingConfig s)
    {
        ControlConfig c; c.enabled = true; c.gimbal.speed = 20; c.automatic = control_config();
        ex = std::make_unique<GimbalCommandExecutor>(c, std::make_unique<FakeDevice>(rec), runtime, s);
        ex->connected();
    }
    ~Rig() { ex->shutdown(); }
    // frames at ~30 fps with the person at (ex, ey) for this long
    void feed(double ex_, double ey_, double seconds)
    {
        const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
        while (std::chrono::steady_clock::now() < end) {
            auto f = std::make_shared<FrameContext>();
            f->metadata.frame_id = ++id; f->metadata.captured_at = std::chrono::steady_clock::now();
            f->image = cv::Mat(480, 640, CV_16UC1);
            auto r = std::make_shared<TrackingResult>(); r->frame = f; r->tracks.push_back(person(ex_, ey_));
            ex->publish(r); ex->ping();
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
    }
    bool follow(double ex_, double ey_, double seconds)
    {
        feed(ex_, ey_, 0.1);
        if (ex->select(7) != ControlResult::Ok) return false;
        feed(ex_, ey_, seconds);
        return true;
    }
};

// ---------- 2b + 3: direction reversal of the Y axis while X still has a large error
// e = size of the error on both axes (0.6 -> shared SPD 18, 0.1 -> SPD 3)
std::vector<Command> after_reversal(bool axis_reversal, double e = 0.6, int hold_ms = 500, double seconds = 0.4)
{
    SmoothingConfig s; s.axis_reversal = axis_reversal; s.reversal_hold_ms = hold_ms;
    Rig r(s);
    if (!r.follow(e, -e, 0.6)) return {};              // right + up: X=+1, Y=+1
    const auto mark = r.rec.size();
    r.feed(e, e, seconds);                             // jumps below the centre: Y reverses, X keeps
    auto got = r.rec.since(mark);
    while (!got.empty() && !got.front().stop && got.front().x == 1 && got.front().y == 1) got.erase(got.begin());
    return got;
}

void test_reversal()
{
    const auto team = after_reversal(false);
    check(!team.empty() && team.front().stop, "defaults: a reversal is an immediate STOP of both axes (team rule)");
    const auto v27 = after_reversal(true);
    check(!v27.empty() && !v27.front().stop && v27.front().x == 1 && v27.front().y == 0,
          "axis_reversal: Y halts alone, X keeps moving (no STOP)");
    bool waits = true;
    for (const auto& c : v27) if (!c.stop && c.y == -1) waits = false;
    check(waits, "axis_reversal: while X runs fast (SPD 18 > join 5) Y stays halted for the first 0.4 s (hold 500 ms)");

    // bounded wait: with hold 300 ms the halted axis rejoins after about 0.3 s even though X is still fast
    const auto bounded = after_reversal(true, 0.6, 300, 0.9);
    double waited = -1;
    if (!bounded.empty() && !bounded.front().stop)
        for (const auto& c : bounded) if (!c.stop && c.y == -1) { waited = std::chrono::duration<double>(c.at - bounded.front().at).count(); break; }
    check(waited >= 0.25 && waited <= 0.55, "axis_reversal: Y rejoins after reversal_hold_ms (300 ms) at the latest");
    std::printf("      waited %.3f s\n", waited);

    // slow motion (shared SPD 3 <= join 5): Y halts for one period, then rejoins downward at once, without a STOP
    const auto slow = after_reversal(true, 0.1);
    bool ok = !slow.empty() && !slow.front().stop && slow.front().y == 0;
    int rejoin_speed = -1;
    for (const auto& c : slow) { if (c.stop) { ok = false; break; } if (c.y == -1) { rejoin_speed = c.speed; break; } }
    check(ok && rejoin_speed >= 1 && rejoin_speed <= 5, "axis_reversal: at SPD <= join the axis rejoins in the new direction without a STOP");
}

// ---------- 2c + 4: deadband entry after the speed has ramped up
std::vector<int> speeds_into_stop(double stop_rate)
{
    SmoothingConfig s; s.stop_rate = stop_rate;
    Rig r(s);
    if (!r.follow(0.97, 0.0, 1.0)) return {-1};
    const auto mark = r.rec.size();
    r.feed(0.0, 0.0, 0.6);
    std::vector<int> spd;
    for (const auto& c : r.rec.since(mark)) { if (c.stop) { spd.push_back(0); break; } spd.push_back(c.speed); }
    return spd;
}

void test_soft_stop()
{
    const auto team = speeds_into_stop(0);
    check(!team.empty() && team.back() == 0 && team.size() <= 2, "defaults: deadband entry is an immediate STOP (team rule)");
    const auto soft = speeds_into_stop(100);
    bool down = soft.size() >= 4 && soft.back() == 0;
    for (std::size_t i = 1; down && i + 1 < soft.size(); ++i) down = soft[i] <= soft[i - 1];
    check(down, "stop_rate 100: speed steps down before the STOP");
    std::printf("      speeds into STOP: team");
    for (int v : team) std::printf(" %d", v);
    std::printf(" | stop_rate 100");
    for (int v : soft) std::printf(" %d", v);
    std::printf("\n");
}
}

int main()
{
    test_hysteresis();
    test_reversal();
    test_soft_stop();
    test_aim_screen_y();
    test_far_gain_and_brake();
    std::printf(failures ? "FAILED %d\n" : "ALL OK\n", failures);
    return failures ? 1 : 0;
}
