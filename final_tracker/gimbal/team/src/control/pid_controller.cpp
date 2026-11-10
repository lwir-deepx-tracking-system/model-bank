#include "control/pid_controller.hpp"

#include <algorithm>
#include <cmath>

void PidController::reset()
{
    initialized_ = feedback_valid_ = false;
    previous_error_ = integral_ = derivative_ = requested_ = applied_ = 0;
}
void PidController::set_applied_output(double value)
{
    if (!std::isfinite(value)) return;
    applied_ = value;
    feedback_valid_ = true;
}
PidOutput PidController::calculate(double error, double dt_seconds)
{
    if (!std::isfinite(error) || !std::isfinite(dt_seconds) || dt_seconds < 0) {
        reset(); return {};
    }
    const bool use_i = type_ == AutoControllerType::PI || type_ == AutoControllerType::PID;
    const bool use_d = type_ == AutoControllerType::PD || type_ == AutoControllerType::PID;
    PidOutput output;
    output.p = config_.kp * error;
    if (initialized_ && dt_seconds > 0) {
        if (use_d && config_.kd > 0) {
            const double raw = (error - previous_error_) / dt_seconds;
            const double alpha = dt_seconds / (config_.derivative_filter_ms / 1000.0 + dt_seconds);
            derivative_ += alpha * (raw - derivative_);
        }
        output.d = use_d ? config_.kd * derivative_ : 0;
        if (use_i && config_.ki > 0) {
            const double candidate = std::clamp(integral_ + config_.ki * error * dt_seconds,
                                               -config_.integral_limit, config_.integral_limit);
            const double increment = candidate - integral_;
            const double raw = output.p + candidate + output.d;
            const double saturation_error = raw - std::clamp(raw, -max_output_, max_output_);
            // Freeze integration that pushes further into output saturation or
            // a still-limited actuator command; allow integration that unwinds it.
            const bool saturated = increment * saturation_error > 0;
            const double unmet = requested_ - applied_;
            // SPD is rounded to an integer. A sub-half-unit rounding residual
            // must not permanently freeze I before it can reach the next unit.
            const bool rate_limited = feedback_valid_ && std::abs(unmet) > 0.500001 && increment * unmet > 0;
            if (!saturated && !rate_limited) integral_ = candidate;
        }
    }
    if (!use_i) integral_ = 0;
    if (!use_d) derivative_ = 0;
    output.i = integral_;
    output.value = std::clamp(output.p + output.i + output.d, -max_output_, max_output_);
    if (!std::isfinite(output.p) || !std::isfinite(output.i) ||
        !std::isfinite(output.d) || !std::isfinite(output.value)) {
        reset(); return {};
    }
    previous_error_ = error;
    requested_ = output.value;
    initialized_ = true;
    return output;
}
