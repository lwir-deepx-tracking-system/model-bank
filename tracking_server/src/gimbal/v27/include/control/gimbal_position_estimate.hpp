#pragma once

#include <algorithm>
#include <chrono>

struct GimbalPositionEstimate {
    using Clock = std::chrono::steady_clock;

    void command(Clock::time_point now, int x, int y, int speed)
    {
        advance(now);
        x_ = x;
        y_ = y;
        speed_ = speed;
    }

    void advance(Clock::time_point now)
    {
        if (last_ != Clock::time_point{} && speed_ > 0) {
            // Display scale, estimated from elapsed submitted commands. Never used for control.
            const double elapsed = std::chrono::duration<double>(now - last_).count();
            const double units = elapsed * (std::max(5, speed_) / 5.0);
            pan = std::clamp(pan + x_ * units * (100.0 / 40), 0.0, 100.0);
            tilt = std::clamp(tilt + y_ * units * (100.0 / 14), 0.0, 100.0);
        }
        last_ = now;
    }

    double pan = 50.0;
    double tilt = 50.0;

private:
    Clock::time_point last_{};
    int x_ = 0, y_ = 0, speed_ = 0;
};
