// v27: closed-loop real-time simulation of the GimbalCommandExecutor (v27/ copy) with a virtual gimbal, to compare the
// team rules (all options 0) with the v27 motion smoothing. No gimbal, no camera, no bridge. Build: see V27_CHANGES.md.
// Plant: angular rate = K * SPD (deg/s) per commanded axis direction; camera 30 fps, fixed latency, box noise.
// usage: sim <stop_rate> <deadband_exit> <seconds> [axis_reversal 0|1]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <random>
#include <thread>
#include <vector>
#include "control/gimbal_command_executor.hpp"
#include "common/logger.hpp"

static std::mutex m;
static int cx = 0, cy = 0, cspd = 0;           // last command seen by the virtual gimbal
static int n_cmd = 0, n_stop = 0;
struct Dev : IGimbalDevice {
    void move(int x, int y) override { std::lock_guard<std::mutex> l(m); cx = x; cy = y; cspd = 20; ++n_cmd; }
    void move_with_speed(int x, int y, int s) override { std::lock_guard<std::mutex> l(m); cx = x; cy = y; cspd = s; ++n_cmd; }
    void stop() override { std::lock_guard<std::mutex> l(m); cx = cy = 0; cspd = 0; ++n_cmd; ++n_stop; }
    void home() override {}
};

int main(int argc, char** argv)
{
    if (argc < 4) { std::fprintf(stderr, "usage: sim <stop_rate> <deadband_exit> <seconds> [axis_reversal 0|1]\n"); return 2; }
    const double stop_rate = std::atof(argv[1]), exit_db = std::atof(argv[2]), secs = std::atof(argv[3]);
    const double K = 0.5;          // deg/s per SPD unit (assumption; vendor unit unknown)
    const double FOV = 50.0;       // horizontal field of view, deg
    const double LAT = 0.12;       // capture -> executor latency, s
    using C = std::chrono::steady_clock;

    ControlConfig c; c.enabled = true; c.gimbal.speed = 20; c.gimbal.speed_limit = 100;
    auto& a = c.automatic;
    a.controller = AutoControllerType::PD; a.pid_x.kp = a.pid_y.kp = 30; a.pid_x.ki = a.pid_y.ki = 0; a.pid_x.kd = a.pid_y.kd = 6;
    a.pid_x.integral_limit = a.pid_y.integral_limit = 3; a.deadband_x = a.deadband_y = 0.06; a.speed_max = 100; a.min_confidence = 0.1;
    a.reacquire_timeout_ms = 10000;
    SmoothingConfig sm; sm.stop_rate = stop_rate; sm.deadband_exit_x = sm.deadband_exit_y = exit_db;
    sm.axis_reversal = argc > 4 && std::atoi(argv[4]) != 0;

    if (std::getenv("SIMDBG")) Logger::set_level("debug");
    RuntimeControl rt;
    GimbalCommandExecutor ex(c, std::make_unique<Dev>(), rt, sm);
    ex.connected();

    std::mt19937 rng(7); std::normal_distribution<double> noise(0, 0.006);   // box-centre jitter (normalized)
    struct Cap { C::time_point t; double ex, ey; };
    std::vector<Cap> pending;
    double gx = 0, gy = 0, prev_rate = 0;   // gimbal angles (deg)
    double jerk = 0, max_dv = 0, ex_abs = 0; int samples = 0, cycles = 0; bool was_moving = false;
    const auto t0 = C::now(); auto last = t0, next_frame = t0; std::uint64_t fid = 0; bool selected = false;
    std::printf("t,target_x,gimbal_x,err_x,spd\n");
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        const auto now = C::now(); const double t = std::chrono::duration<double>(now - t0).count();
        if (t > secs) break;
        const double dt = std::chrono::duration<double>(now - last).count(); last = now;
        // target: 15 deg right, 5 deg up at start; from 3 s to 6 s it walks right at 4 deg/s, then stands
        const double tx = 15 + (t > 3 ? 4 * (std::min(t, 6.0) - 3) : 0), ty = 5;
        int x, y, s; { std::lock_guard<std::mutex> l(m); x = cx; y = cy; s = cspd; }
        const double rate = x * K * s;
        gx += rate * dt; gy += y * K * s * dt;
        const double dv = std::abs(rate - prev_rate); jerk += dv; max_dv = std::max(max_dv, dv); prev_rate = rate;
        const bool moving = x || y; if (moving && !was_moving) ++cycles; was_moving = moving;
        if (now >= next_frame) {   // capture
            next_frame += std::chrono::microseconds(33333);
            const double nx = (tx - gx) / (FOV / 2) + noise(rng), ny = -(ty - gy) / (FOV * 0.75 / 2) + noise(rng); // image y down
            pending.push_back({now, nx, ny});
        }
        while (!pending.empty() && std::chrono::duration<double>(now - pending.front().t).count() >= LAT) {   // delivery
            const Cap cap = pending.front(); pending.erase(pending.begin());
            auto f = std::make_shared<FrameContext>(); f->metadata.frame_id = ++fid; f->metadata.captured_at = cap.t;
            f->image = cv::Mat(480, 640, CV_16UC1);
            auto r = std::make_shared<TrackingResult>(); r->frame = f;
            Track tr; tr.track_id = 1; tr.width = 40; tr.height = 80; tr.confidence = 0.9f;
            tr.x = float(320 + cap.ex * 320 - 20); tr.y = float(240 + cap.ey * 240 - 40);
            r->tracks.push_back(tr);
            ex.publish(r);
            if (!selected) { ex.ping(); std::this_thread::sleep_for(std::chrono::milliseconds(5)); selected = ex.select(1) == ControlResult::Ok; }
        }
        static double last_ping = 0; if (t - last_ping > 0.5) { ex.ping(); last_ping = t; }
        static double last_log = 0; if (t - last_log >= 0.05) { last_log = t; std::printf("%.2f,%.2f,%.2f,%.3f,%d\n", t, tx, gx, (tx - gx) / (FOV / 2), x * s); }
        if (t > 1.5) { ex_abs += std::abs(tx - gx); ++samples; }
    }
    std::fprintf(stderr, "stop_rate=%g exit=%g axis=%d | commands=%d STOPs=%d start-stop cycles=%d | max |dRate|=%.1f deg/s per step | sum|dRate|=%.0f | mean |err| after 1.5 s=%.2f deg | final err x=%.2f y=%.2f deg\n",
                 stop_rate, exit_db, int(sm.axis_reversal), n_cmd, n_stop, cycles, max_dv, jerk, ex_abs / std::max(1, samples), (15 + 4 * (std::min(secs, 6.0) - 3) - gx), 5 - gy);
    ex.shutdown();
}
