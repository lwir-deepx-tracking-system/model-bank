// Deterministic v27 controller tests. Build with v27/src/control/auto_tracking_controller.cpp
// and team/src/control/pid_controller.cpp; use sim/stub when OpenCV is unavailable.
#include <cmath>
#include <cstdio>
#include <limits>

#include "control/auto_tracking_controller.hpp"

namespace {
int failures = 0;

void check(bool pass, const char* name)
{
    std::printf("%s  %s\n", pass ? "ok" : "FAIL", name);
    if (!pass) ++failures;
}

bool near(double a, double b)
{
    return std::abs(a - b) < 1e-9;
}

AutoControlConfig config()
{
    AutoControlConfig c;
    c.controller = AutoControllerType::PID;
    c.pid_x.ki = c.pid_y.ki = 2;
    c.pid_x.kd = c.pid_y.kd = 2;
    c.pid_x.derivative_filter_ms = c.pid_y.derivative_filter_ms = 0;
    return c;
}

Track person()
{
    Track t;
    t.track_id = 7;
    t.x = 380;
    t.y = 250;
    t.width = 40;
    t.height = 80;
    t.confidence = 0.9f;
    return t;
}

FrameMetadata frame(std::uint64_t id)
{
    FrameMetadata metadata;
    metadata.frame_id = id;
    metadata.captured_at = std::chrono::steady_clock::time_point(std::chrono::milliseconds(100 * id));
    return metadata;
}

void test_observed_derivative()
{
    AutoTrackingController c(config());
    auto t = person();
    c.calculate(t, 640, 480, frame(1));
    const auto enabled = c.calculate(t, 640, 480, frame(2), 0, {7, 32, 24});
    const auto changed = c.calculate(t, 640, 480, frame(3), 0, {7, -32, -12});
    const auto disabled = c.calculate(t, 640, 480, frame(4));
    check(near(enabled.pid_x.p, 10.5) && near(enabled.pid_x.i, 0.07) &&
          near(enabled.error_y, 74.0 / 240.0), "P and I use the forecast in image pixels");
    check(near(changed.pid_x.p, 4.5) && near(disabled.pid_x.p, 7.5) &&
          near(enabled.pid_x.d, 0) && near(changed.pid_x.d, 0) && near(disabled.pid_x.d, 0) &&
          near(enabled.pid_y.d, 0) && near(changed.pid_y.d, 0) && near(disabled.pid_y.d, 0),
          "enabling, changing and removing lead leave the stationary derivative at zero");

    t.x += 32;
    t.y += 24;
    const auto moving = c.calculate(t, 640, 480, frame(5), 0, {7, -40, -40});
    check(near(moving.pid_x.d, 2) && near(moving.pid_y.d, 2),
          "observed motion still contributes to the derivative while lead changes");
    const auto duplicate = c.calculate(t, 640, 480, frame(5), 0, {7, 60, 60});
    check(near(duplicate.pid_x.value, moving.pid_x.value) && near(duplicate.pid_y.value, moving.pid_y.value),
          "a duplicate frame does not recalculate the forecast or derivative");
    t.track_id = 8;
    const auto new_id = c.calculate(t, 640, 480, frame(6), 0, {8, 40, 40});
    check(near(new_id.pid_x.d, 0) && near(new_id.pid_y.d, 0),
          "new track ID retains the derivative reset");
}

void test_lead_validation()
{
    const auto t = person();
    AutoTrackingController baseline(config());
    const auto measured = baseline.calculate(t, 640, 480, frame(1));
    const TrackLead invalid[] = {
        {8, 40, 40}, {7, 60.01, 0}, {7, 0, -61},
        {7, std::numeric_limits<double>::infinity(), 0},
        {7, 0, std::numeric_limits<double>::quiet_NaN()},
    };
    for (const auto& lead : invalid) {
        AutoTrackingController c(config());
        const auto d = c.calculate(t, 640, 480, frame(1), 0, lead);
        check(d.x == measured.x && d.y == measured.y &&
              near(d.error_x, measured.error_x) && near(d.error_y, measured.error_y),
              "wrong ID, non-finite or excessive lead leaves the observation usable");
    }
    AutoTrackingController c(config());
    auto near_edge = t;
    near_edge.x = 600;
    const auto beyond_frame = c.calculate(near_edge, 640, 480, frame(1), 0, {7, 60, -60});
    check(beyond_frame.x == 1 && near(beyond_frame.error_x, 360.0 / 320.0),
          "valid observation stays valid when bounded lead moves its forecast beyond the edge");
    near_edge.x = 650;
    check(c.calculate(near_edge, 640, 480, frame(2), 0, {7, -60, 0}).stop(),
          "a forecast cannot make an out-of-frame observed box valid");
}

void test_existing_reference_without_lead()
{
    for (const float top : {-30.f, -5.f, 250.f, 455.f, 475.f}) {
        auto observed = person();
        observed.y = top;
        auto old_bridge_box = observed;
        constexpr double aim_y = 0.08;
        old_bridge_box.y += static_cast<float>((aim_y - 0.5) * old_bridge_box.height);
        AutoTrackingController old_path(config()), new_path(config());
        const auto expected = old_path.calculate(old_bridge_box, 640, 480, frame(1));
        const auto actual = new_path.calculate(observed, 640, 480, frame(1), 0, {}, aim_y);
        check(old_path.valid(old_bridge_box, 640, 480) == new_path.valid(observed, 640, 480, aim_y) &&
              actual.x == expected.x && actual.y == expected.y && near(actual.speed, expected.speed) &&
              near(actual.error_y, expected.error_y) && observed.y == top,
              "unshifted row preserves old no-lead validity and commands at vertical edges");
    }
    const auto c = config();
    PidController legacy(c.pid_x, c.controller, c.speed_max);
    legacy.calculate(0.2, 0);
    const auto out = legacy.calculate(0.3, 0.1);
    check(near(out.p, 9) && near(out.i, 0.06) && near(out.d, 2),
          "two-argument PID still derives all three terms from the same error");
}

void test_observation_gates()
{
    auto cfg = config();
    cfg.deadband_x = cfg.deadband_y = 0.06;
    cfg.pid_x.ki = cfg.pid_y.ki = cfg.pid_x.kd = cfg.pid_y.kd = 0;
    SmoothingConfig smoothing;
    smoothing.deadband_exit_x = smoothing.deadband_exit_y = 0.10;
    for (bool swap : {false, true}) {
        AutoTrackingController c(cfg, smoothing);
        auto t = person();
        // Strong axis has normalized error .5; weak axis is only 10 pixels off centre.
        t.x = swap ? 310 : 460;
        t.y = swap ? 320 : 210;
        c.calculate(t, 640, 480, frame(1));
        bool held = true;
        std::uint64_t id = 1;
        for (double lead : {60., -60., 20., 0.}) {
            const auto d = c.calculate(t, 640, 480, frame(++id), 0,
                {7, swap ? lead : 0., swap ? 0. : lead});
            held = held && (swap ? d.x == 0 && d.y == -1 : d.y == 0 && d.x == 1);
        }
        check(held, "lead cannot activate the centred axis at the strong axis's shared speed");
    }
    AutoTrackingController c(cfg);
    auto t = person(); t.x = 340; // measured error +40px, lead -60px crosses centre.
    const auto d = c.calculate(t, 640, 480, frame(1), 0, {7, -60, 0});
    check(d.x == 1 && near(d.pid_x.value, 3.75),
          "lead alone cannot reverse the measured controller's direction");
    cfg.pid_x.kd = 2;
    AutoTrackingController damping(cfg);
    t.x = 364; damping.calculate(t, 640, 480, frame(1));
    t.x = 344; // D=-1.25, measured P=4.125; predicted P can shrink enough to reverse.
    const auto b = damping.calculate(t, 640, 480, frame(2), 0, {7, -40, 0});
    check(b.x == 1 && near(b.pid_x.value, 2.875),
          "direction guard checks P plus D, even when forecast stays on the same side");
}
}

int main()
{
    test_observed_derivative();
    test_lead_validation();
    test_existing_reference_without_lead();
    test_observation_gates();
    return failures != 0;
}
