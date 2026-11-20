#pragma once

#include <memory>
#include <string>
#include <stdexcept>
#include <chrono>
#include <optional>
#include "config/app_config.hpp"

// Device strategy: application commands never contain vendor JSON or serial FDs.
struct GimbalPose {
    double pan = 0, tilt = 0;
    std::chrono::steady_clock::time_point received_at{};
};

class IGimbalDevice {
public:
    virtual ~IGimbalDevice() = default;
    virtual void move(int x_direction, int y_direction) = 0;
    // Implementations without variable speed must reject AUTO, never silently ignore SPD.
    virtual void move_with_speed(int, int, int) { throw std::logic_error("Variable gimbal speed unsupported"); }
    virtual void stop() = 0;
    virtual void home() = 0; // Request only; does not establish physical completion.
    virtual void enable_pose_feedback() {} // optional, once during bridge startup
    virtual std::optional<GimbalPose> read_pose() { return std::nullopt; } // nonblocking
};

// Shared encoder also permits testing vendor commands without a physical device.
class WaveshareCommands {
public:
    static std::string move(const GimbalConfig&, int x_direction, int y_direction);
    static std::string move(const GimbalConfig&, int x_direction, int y_direction, int speed);
    static std::string stop();
    static std::string home(const GimbalConfig&);
};

class GimbalDeviceFactory {
public:
    static std::unique_ptr<IGimbalDevice> create(const GimbalConfig&);
};
