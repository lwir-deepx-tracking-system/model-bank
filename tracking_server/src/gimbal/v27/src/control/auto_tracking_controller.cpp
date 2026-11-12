// v27 copy of team/src/control/auto_tracking_controller.cpp: changed only where marked "v27" (motion smoothing, gimbal_bridge_v27). See ../../../V27_CHANGES.md.
#include "control/auto_tracking_controller.hpp"

#include <algorithm>
#include <cmath>

namespace {
double observed_y(const Track& t, double aim_y)
{
    // Use the bridge's existing box reference without changing the observation.
    const float y = t.y + static_cast<float>((aim_y - 0.5) * t.height);
    return double(y) + double(t.height) / 2;
}
}

bool AutoTrackingController::valid(const Track& t, int width, int height, double aim_y) const
{
    if (width <= 0 || height <= 0 || t.track_id <= 0 || t.predicted ||
        !std::isfinite(t.x) || !std::isfinite(t.y) ||
        !std::isfinite(t.width) || !std::isfinite(t.height) ||
        !std::isfinite(t.confidence) || t.width <= 0 || t.height <= 0 ||
        t.confidence < config_.min_confidence || t.confidence > 1)
        return false;
    const double cx = double(t.x) + double(t.width) / 2;
    const double cy = observed_y(t, aim_y);
    return cx >= 0 && cx < width && cy >= 0 && cy < height;
}

void AutoTrackingController::reset()
{
    x_.reset(); y_.reset();
    initialized_ = false;
    hold_x_ = hold_y_ = false;   // v27
    track_id_ = -1; width_ = height_ = 0;
    generation_ = 0; previous_ = {}; cached_ = {};
}

void AutoTrackingController::set_applied_output(double image_x, double image_y)
{
    x_.set_applied_output(image_x); y_.set_applied_output(image_y);
}

GimbalDemand AutoTrackingController::calculate(const Track& t, int width, int height,
    const FrameMetadata& metadata, std::uint64_t model_generation, const TrackLead& lead,
    double aim_y)
{
    if (!valid(t, width, height, aim_y) || metadata.frame_id == 0 ||
        metadata.captured_at == std::chrono::steady_clock::time_point{}) {
        reset(); return {};
    }
    if (initialized_ && (t.track_id != track_id_ || model_generation != generation_ ||
        width != width_ || height != height_)) reset();
    double dt = 0;
    if (initialized_) {
        if (metadata.frame_id == previous_.frame_id) return cached_;
        if (metadata.frame_id < previous_.frame_id || metadata.captured_at <= previous_.captured_at) {
            // Retain the last accepted frame watermark: repeated ticks of the
            // rejected frame must not become a new first measurement after reset.
            x_.reset(); y_.reset(); cached_ = {};
            return cached_;
        }
        dt = std::chrono::duration<double>(metadata.captured_at - previous_.captured_at).count();
        if (dt >= config_.input_timeout_ms / 1000.0) { reset(); dt = 0; }
    }
    const bool angular = config_.fov_x_deg > 0 && config_.fov_y_deg > 0;
    const auto error = [angular](double from_center, double half_extent, double fov) {
        const double normalized = from_center / half_extent;
        if (!angular) return normalized;
        constexpr double radians = 3.14159265358979323846 / 180;
        return std::atan(normalized * std::tan(fov * radians / 2));
    };
    const double x = double(t.x) + double(t.width) / 2 - width / 2.0;
    const double y = observed_y(t, aim_y) - height / 2.0;
    const double measured_x = error(x, width / 2.0, config_.fov_x_deg);
    const double measured_y = error(y, height / 2.0, config_.fov_y_deg);
    GimbalDemand d;
    d.error_x = measured_x;
    d.error_y = measured_y;
    if (lead.matches(t.track_id)) {
        d.error_x = error(x + lead.dx_px, width / 2.0, config_.fov_x_deg);
        d.error_y = error(y + lead.dy_px, height / 2.0, config_.fov_y_deg);
    }
    // v27: only observed error may release an axis from the deadband. Otherwise a
    // small forecast can wake a centred axis at the other axis's large shared SPD.
    // Enter at |e| <= deadband; leave above max(deadband, exit).
    const auto held = [](bool& hold, double error, double enter, double exit) {
        const double leave = std::max(enter, exit);
        hold = hold ? std::abs(error) <= leave : std::abs(error) <= enter;
        return hold;
    };
    const auto predicted_pid = [dt](PidController& pid, double predicted, double observed) {
        auto measured_only = pid;
        const auto measured_output = measured_only.calculate(observed, dt, observed);
        const auto forecast_output = pid.calculate(predicted, dt, observed);
        // A forecast may brake or accelerate an allowed direction, but must not
        // create a start/reversal that the measured PID would not request. Compare
        // the whole PID (including D), not just the sign of the positional error.
        if ((forecast_output.value > 0 && measured_output.value <= 0) ||
            (forecast_output.value < 0 && measured_output.value >= 0)) {
            pid = measured_only;
            return measured_output;
        }
        return forecast_output;
    };
    if (held(hold_x_, measured_x, config_.deadband_x, smoothing_.deadband_exit_x)) x_.reset();
    else d.pid_x = predicted_pid(x_, d.error_x, measured_x);
    if (held(hold_y_, measured_y, config_.deadband_y, smoothing_.deadband_exit_y)) y_.reset();
    else d.pid_y = predicted_pid(y_, d.error_y, measured_y);
    const auto direction = [](double value) { return value > 0 ? 1 : value < 0 ? -1 : 0; };
    d.x = direction(d.pid_x.value);
    // Image Y grows downward; the shared GUI/device command contract is +Y up.
    d.y = -direction(d.pid_y.value);
    if (!d.stop())
        d.speed = std::clamp(std::max(std::abs(d.pid_x.value), std::abs(d.pid_y.value)),
                             double(config_.speed_min), double(config_.speed_max));
    initialized_ = true; track_id_ = t.track_id;
    width_ = width; height_ = height; generation_ = model_generation;
    previous_ = metadata; cached_ = d;
    return d;
}
