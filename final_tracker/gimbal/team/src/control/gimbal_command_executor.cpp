#include "control/gimbal_command_executor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "common/logger.hpp"

GimbalCommandExecutor::GimbalCommandExecutor(ControlConfig config,
    std::unique_ptr<IGimbalDevice> device, RuntimeControl& runtime)
    : config_(config), runtime_(runtime), device_(std::move(device)),
      controller_(config.automatic), model_generation_(runtime.generation.load()),
      tracker_epoch_(runtime.tracker_epoch.load())
{
    if (!config.enabled || !device_) throw std::invalid_argument("Tracking control needs an enabled device");
    worker_ = std::thread([this] { run(); });
    const bool started = invoke([this] { stop("startup"); return !state_.error; });
    if (!started) {
        { std::lock_guard<std::mutex> lock(mutex_); closing_ = true; }
        wake_.notify_one(); worker_.join();
        throw std::runtime_error("Gimbal startup STOP failed");
    }
}
GimbalCommandExecutor::~GimbalCommandExecutor()
{
    shutdown();
    { std::lock_guard<std::mutex> lock(mutex_); closing_ = true; }
    wake_.notify_one();
    worker_.join();
}
void GimbalCommandExecutor::run()
{
    auto next = Clock::now();
    for (;;) {
        std::packaged_task<void()> request;
        TrackingResultPtr input;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait_until(lock, next, [this] { return closing_ || !requests_.empty() || pending_; });
            if (closing_) break;
            input = std::move(pending_);
            if (!requests_.empty()) { request = std::move(requests_.front()); requests_.pop_front(); }
        }
        if (input) ingest(std::move(input));
        if (request.valid()) request(); // mode/STOP requests precede any AUTO output
        if (Clock::now() >= next) {
            tick();
            next = Clock::now() + std::chrono::milliseconds(
                std::min(config_.automatic.period_ms, config_.poll_interval_ms));
        }
    }
}
void GimbalCommandExecutor::publish(TrackingResultPtr input)
{
    if (!input || !input->frame) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pending_ || input->tracker_epoch > pending_->tracker_epoch ||
            (input->tracker_epoch == pending_->tracker_epoch &&
             (input->model_generation > pending_->model_generation ||
              (input->model_generation == pending_->model_generation &&
               input->frame->metadata.frame_id > pending_->frame->metadata.frame_id))))
            pending_ = std::move(input);
    }
    wake_.notify_one();
}
void GimbalCommandExecutor::ingest(TrackingResultPtr input)
{
    if (input->model_generation != runtime_.generation.load() ||
        input->tracker_epoch != runtime_.tracker_epoch.load()) return;
    if (input->tracker_epoch != tracker_epoch_) {
        invalidate("tracker generation"); latest_.reset(); last_frame_id_ = 0;
        tracker_epoch_ = input->tracker_epoch;
    }
    if (input->model_generation != model_generation_) {
        invalidate("model generation"); latest_.reset(); last_frame_id_ = 0;
        model_generation_ = input->model_generation;
    }
    if (input->frame->metadata.frame_id <= last_frame_id_) return;
    last_frame_id_ = input->frame->metadata.frame_id;
    latest_ = std::move(input);
}
bool GimbalCommandExecutor::alive() const
{
    return connected_ && Clock::now() - ping_ < std::chrono::milliseconds(config_.heartbeat_timeout_ms);
}
bool GimbalCommandExecutor::ready() const
{
    return !stopped_ && !state_.error && alive() && !runtime_.stop_requested.load() &&
        runtime_.state.load() == RuntimeState::Running &&
        runtime_.generation.load() == model_generation_ &&
        runtime_.tracker_epoch.load() == tracker_epoch_;
}
bool GimbalCommandExecutor::fresh() const
{
    if (!latest_ || !latest_->frame || latest_->bypass ||
        latest_->model_generation != runtime_.generation.load() ||
        latest_->tracker_epoch != runtime_.tracker_epoch.load() ||
        latest_->frame->image.type() != CV_16UC1 || latest_->frame->image.empty()) return false;
    const auto captured = latest_->frame->metadata.captured_at;
    const auto now = Clock::now();
    return captured != Clock::time_point{} && captured <= now &&
        now - captured < std::chrono::milliseconds(config_.automatic.input_timeout_ms);
}
const Track* GimbalCommandExecutor::target(std::uint32_t id) const
{
    if (!fresh() || id == 0 || id > std::uint32_t(std::numeric_limits<int>::max())) return nullptr;
    for (const auto& t : latest_->tracks)
        if (t.track_id == static_cast<int>(id) &&
            controller_.valid(t, latest_->frame->image.cols, latest_->frame->image.rows)) return &t;
    return nullptr;
}
void GimbalCommandExecutor::failure(const std::exception& e)
{
    Logger::error(std::string("[Control] device error: ") + e.what());
    controller_.reset();
    state_.error = true;
    state_.automatic = false; state_.selected_id = 0; state_.lost = false;
    ++state_.command_generation;
    try { device_->stop(); } catch (const std::exception& stop_error) {
        Logger::error(std::string("[Control] STOP failed: ") + stop_error.what());
    }
    state_.moving = false; x_ = y_ = sent_speed_ = 0; limited_speed_ = 0;
}
void GimbalCommandExecutor::stop(const char* reason)
{
    // No ramp: safety/Deadband STOP is immediate command submission.
    try {
        device_->stop();
        controller_.set_applied_output(0, 0);
        Logger::debug(std::string("[Control] STOP reason=") + reason);
        state_.moving = false; x_ = y_ = sent_speed_ = 0; limited_speed_ = 0;
        last_output_ = Clock::now();
    } catch (const std::exception& e) { failure(e); }
}
void GimbalCommandExecutor::invalidate(const char* reason)
{
    controller_.reset();
    ++state_.command_generation;
    state_.automatic = false; state_.selected_id = 0; state_.lost = false;
    lost_since_ = {}; next_auto_ = {};
    if (state_.moving) stop(reason);
}
void GimbalCommandExecutor::send(int x, int y, int speed, bool automatic)
{
    if (x == x_ && y == y_ && speed == sent_speed_) return;
    try {
        if (automatic) device_->move_with_speed(x, y, speed);
        else device_->move(x, y);
        x_ = x; y_ = y; sent_speed_ = speed; state_.moving = x || y;
    } catch (const std::exception& e) { failure(e); }
}
void GimbalCommandExecutor::tick()
{
    if (runtime_.generation.load() != model_generation_ ||
        runtime_.tracker_epoch.load() != tracker_epoch_ ||
        runtime_.state.load() != RuntimeState::Running || runtime_.stop_requested.load()) {
        invalidate("runtime/model/tracker transition"); latest_.reset(); last_frame_id_ = 0;
        model_generation_ = runtime_.generation.load();
        tracker_epoch_ = runtime_.tracker_epoch.load();
        return;
    }
    if (!alive()) { if (state_.automatic || state_.moving) invalidate("heartbeat/disconnect"); return; }
    if (!state_.automatic || state_.error) return;
    const auto now = Clock::now();
    const Track* t = target(state_.selected_id);
    // Enforce the deadline before accepting a late reappearance.
    if (state_.lost && now - lost_since_ >= std::chrono::milliseconds(config_.automatic.reacquire_timeout_ms)) {
        invalidate("reacquire timeout"); return;
    }
    if (!t) {
        if (!state_.lost) { controller_.reset(); state_.lost = true; lost_since_ = now; ++state_.command_generation; }
        if (state_.moving) stop("target lost/stale");
        return;
    }
    state_.lost = false; lost_since_ = {};
    const auto demand = controller_.calculate(*t, latest_->frame->image.cols, latest_->frame->image.rows,
        latest_->frame->metadata, latest_->model_generation);
    if (demand.stop()) { if (state_.moving) stop("deadband"); return; }
    if (now < next_auto_) return;
    next_auto_ = now + std::chrono::milliseconds(config_.automatic.period_ms);
    // Reverse via STOP and restart from the minimum speed on the next period.
    if ((x_ && demand.x && x_ != demand.x) || (y_ && demand.y && y_ != demand.y)) {
        stop("direction reversal"); return;
    }
    const double dt = last_output_ == Clock::time_point{} ? config_.automatic.period_ms / 1000.0 :
        std::min(std::chrono::duration<double>(now - last_output_).count(), config_.automatic.period_ms / 1000.0);
    const double step = config_.automatic.speed_rate * dt;
    // The minimum is the first usable device command; subsequent changes ramp.
    const double baseline = state_.moving ? limited_speed_ : config_.automatic.speed_min;
    limited_speed_ = state_.moving ?
        std::clamp(demand.speed, std::max(double(config_.automatic.speed_min), baseline - step), baseline + step) :
        double(config_.automatic.speed_min);
    last_output_ = now;
    if (!ready()) { invalidate("control authority"); return; }
    send(demand.x, demand.y, std::clamp(int(std::lround(limited_speed_)),
         config_.automatic.speed_min, config_.automatic.speed_max), true);
    controller_.set_applied_output(x_ * sent_speed_, -y_ * sent_speed_);
    Logger::debug("[Control] frame=" + std::to_string(latest_->frame->metadata.frame_id) +
        " target=" + std::to_string(state_.selected_id) + " ex=" + std::to_string(demand.error_x) +
        " ey=" + std::to_string(demand.error_y) + " SPD=" + std::to_string(sent_speed_) +
        " ux=" + std::to_string(demand.pid_x.value) + " uy=" + std::to_string(demand.pid_y.value) +
        " ix=" + std::to_string(demand.pid_x.i) + " iy=" + std::to_string(demand.pid_y.i) +
        " dx=" + std::to_string(demand.pid_x.d) + " dy=" + std::to_string(demand.pid_y.d) +
        " x=" + std::to_string(x_) + " y=" + std::to_string(y_) +
        " model_generation=" + std::to_string(model_generation_) + " generation=" + std::to_string(state_.command_generation));
}
void GimbalCommandExecutor::connected() { invoke([this] { invalidate("connect"); connected_ = !stopped_; ping_ = Clock::now(); }); }
void GimbalCommandExecutor::disconnected() { invoke([this] { connected_ = false; invalidate("disconnect"); }, true); }
void GimbalCommandExecutor::ping() { invoke([this] { ping_ = Clock::now(); }); }
ControlResult GimbalCommandExecutor::select(std::uint32_t id)
{
    return invoke([this, id] {
        if (!ready()) return ControlResult::NotReady;
        if (!target(id)) return ControlResult::InvalidTarget;
        invalidate("AUTO selection");
        if (state_.error) return ControlResult::NotReady;
        state_.selected_id = id; state_.automatic = true;
        return ControlResult::Ok;
    });
}
ControlResult GimbalCommandExecutor::manual()
{
    return invoke([this] {
        if (state_.automatic) invalidate("MANUAL"); // preserve already-manual motion
        return state_.error ? ControlResult::NotReady : ControlResult::Ok;
    });
}
void GimbalCommandExecutor::move(int x, int y)
{
    invoke([this, x, y] {
        if (x < -1 || x > 1 || y < -1 || y > 1 || state_.error) return;
        if (state_.automatic) invalidate("keyboard override");
        if (state_.error || ((x || y) && !ready())) return;
        if (!x && !y) { if (state_.moving) stop("key release"); return; }
        send(x, y, config_.gimbal.speed, false);
    });
}
void GimbalCommandExecutor::halt() { invoke([this] { invalidate("halt/HOME"); }, true); }
void GimbalCommandExecutor::suspend() { invoke([this] { invalidate("model switch"); latest_.reset(); last_frame_id_ = 0; }, true); }
void GimbalCommandExecutor::shutdown() { invoke([this] { stopped_ = true; connected_ = false; invalidate("shutdown"); }, true); }
ControlSnapshot GimbalCommandExecutor::snapshot() { return invoke([this] { return state_; }); }
