// v27 copy of team/include/control/auto_tracking_controller.hpp: changed only where marked "v27" (motion smoothing, gimbal_bridge_v27). See ../../../V27_CHANGES.md.
#pragma once

#include "common/track.hpp"
#include "control/pid_controller.hpp"
#include "control/smoothing_config.hpp"   // v27

struct GimbalDemand {
    int x = 0, y = 0; // command coordinates: +X right, +Y up
    double speed = 0;
    double error_x = 0, error_y = 0; // image coordinates: +Y down
    PidOutput pid_x, pid_y; // signed outputs in image coordinates
    bool stop() const { return x == 0 && y == 0; }
};

// Per-target controller state, independent of TCP, Serial and wall-clock time.
// Converts image coordinates (+Y down) to command coordinates (+Y up).
class AutoTrackingController {
    AutoControlConfig config_;
    SmoothingConfig smoothing_;              // v27
    PidController x_, y_;
    bool initialized_ = false;
    bool hold_x_ = false, hold_y_ = false;   // v27: axis inside the deadband (hysteresis state)
    int track_id_ = -1, width_ = 0, height_ = 0;
    std::uint64_t generation_ = 0;
    FrameMetadata previous_{};
    GimbalDemand cached_{};
public:
    explicit AutoTrackingController(AutoControlConfig config, SmoothingConfig smoothing = {})
        : config_(config), smoothing_(smoothing), x_(config.pid_x, config.controller, config.speed_max),
          y_(config.pid_y, config.controller, config.speed_max) {}
    bool valid(const Track&, int width, int height) const;
    GimbalDemand calculate(const Track&, int width, int height,
                          const FrameMetadata&, std::uint64_t model_generation = 0);
    void set_applied_output(double image_x, double image_y);
    void reset();
};
