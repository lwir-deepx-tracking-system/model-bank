#pragma once

#include "config/app_config.hpp"

struct PidOutput {
    double p = 0, i = 0, d = 0, value = 0;
};

// One axis, no clock or I/O dependencies. Call once for each new measurement.
class PidController {
    PidAxisConfig config_;
    AutoControllerType type_;
    double max_output_;
    bool initialized_ = false, feedback_valid_ = false;
    double previous_error_ = 0, integral_ = 0, derivative_ = 0;
    double requested_ = 0, applied_ = 0;
public:
    PidController(PidAxisConfig config, AutoControllerType type, double max_output)
        : config_(config), type_(type), max_output_(max_output) {}
    PidOutput calculate(double error, double dt_seconds);
    // Command actually submitted after common SPD conversion and rate limiting.
    // This is command-space feedback, never motor/encoder feedback.
    void set_applied_output(double value);
    void reset();
};
