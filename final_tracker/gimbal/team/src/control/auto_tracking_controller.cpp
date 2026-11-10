#include "control/auto_tracking_controller.hpp"

#include <algorithm>
#include <cmath>

bool AutoTrackingController::valid(const Track& t, int width, int height) const
{
    if (width <= 0 || height <= 0 || t.track_id <= 0 || t.predicted ||
        !std::isfinite(t.x) || !std::isfinite(t.y) ||
        !std::isfinite(t.width) || !std::isfinite(t.height) ||
        !std::isfinite(t.confidence) || t.width <= 0 || t.height <= 0 ||
        t.confidence < config_.min_confidence || t.confidence > 1)
        return false;
    const double cx = double(t.x) + double(t.width) / 2;
    const double cy = double(t.y) + double(t.height) / 2;
    return cx >= 0 && cx < width && cy >= 0 && cy < height;
}

void AutoTrackingController::reset()
{
    x_.reset(); y_.reset();
    initialized_ = false;
    track_id_ = -1; width_ = height_ = 0;
    generation_ = 0; previous_ = {}; cached_ = {};
}

void AutoTrackingController::set_applied_output(double image_x, double image_y)
{
    x_.set_applied_output(image_x); y_.set_applied_output(image_y);
}

GimbalDemand AutoTrackingController::calculate(const Track& t, int width, int height,
    const FrameMetadata& metadata, std::uint64_t model_generation)
{
    if (!valid(t, width, height) || metadata.frame_id == 0 ||
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
    GimbalDemand d;
    d.error_x = (double(t.x) + double(t.width) / 2 - width / 2.0) / (width / 2.0);
    d.error_y = (double(t.y) + double(t.height) / 2 - height / 2.0) / (height / 2.0);
    if (config_.fov_x_deg > 0 && config_.fov_y_deg > 0) {
        constexpr double radians = 3.14159265358979323846 / 180;
        d.error_x = std::atan(d.error_x * std::tan(config_.fov_x_deg * radians / 2));
        d.error_y = std::atan(d.error_y * std::tan(config_.fov_y_deg * radians / 2));
    }
    if (std::abs(d.error_x) <= config_.deadband_x) x_.reset();
    else d.pid_x = x_.calculate(d.error_x, dt);
    if (std::abs(d.error_y) <= config_.deadband_y) y_.reset();
    else d.pid_y = y_.calculate(d.error_y, dt);
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
