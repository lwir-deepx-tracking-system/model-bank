// v27 copy of team/src/control/gimbal_command_executor.cpp: changed only where marked "v27" (motion smoothing, gimbal_bridge_v27). See ../../../V27_CHANGES.md.
#include "control/gimbal_command_executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include "common/logger.hpp"

namespace {
void trace_output(const FrameMetadata& frame, int track_id, bool catchup_eligible,
                  bool catchup_applied, double previous_limited_speed, double slew_limited_speed,
                  double wanted_speed, int final_speed, int x, int y, bool sent) noexcept try
{
    static const bool enabled = [] {
        const char* value = std::getenv("LWIR_CONTROL_TRACE");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    if (!enabled) return;
    const auto captured_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        frame.captured_at.time_since_epoch()).count();
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::boolalpha << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "{\"event\":\"control_output\",\"frame_id\":" << frame.frame_id
        << ",\"captured_ns\":" << captured_ns << ",\"track_id\":" << track_id
        << ",\"catchup_eligible\":" << catchup_eligible << ",\"catchup_applied\":" << catchup_applied
        << ",\"previous_limited_speed\":" << previous_limited_speed
        << ",\"slew_limited_speed\":" << slew_limited_speed << ",\"wanted_speed\":" << wanted_speed
        << ",\"final_speed\":" << final_speed << ",\"x\":" << x << ",\"y\":" << y
        << ",\"sent\":" << sent << "}\n";
    std::fputs(out.str().c_str(), stderr);
} catch (...) {
    // A diagnostics formatting failure must not interrupt control.
}
}

GimbalCommandExecutor::GimbalCommandExecutor(ControlConfig config,
    std::unique_ptr<IGimbalDevice> device, RuntimeControl& runtime, SmoothingConfig smoothing)
    : config_(config), smoothing_(smoothing), runtime_(runtime), device_(std::move(device)),
      controller_(config.automatic, smoothing), model_generation_(runtime.generation.load()),
      tracker_epoch_(runtime.tracker_epoch.load())
{
    if (!config.enabled || !device_) throw std::invalid_argument("Tracking control needs an enabled device");
    if (smoothing_.catchup_speed < 0 || smoothing_.catchup_speed > SmoothingConfig::max_catchup_speed)
        throw std::invalid_argument("--catchup-speed must be within 0..7");
    if (smoothing_.observation_grace_ms < 0 || smoothing_.observation_grace_ms > 100 ||
        (smoothing_.observation_grace_ms > 0 &&
         smoothing_.observation_grace_ms >= config_.automatic.input_timeout_ms))
        throw std::invalid_argument("--observation-grace-ms must be within 0..100 and below --input-timeout-ms when enabled");
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
        Observation input;
        bool unsafe = false;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            const auto wake_at = grace_deadline_ == Clock::time_point{} ? next : std::min(next, grace_deadline_);
            wake_.wait_until(lock, wake_at, [this] {
                return closing_ || !requests_.empty() || pending_.result || pending_unsafe_;
            });
            if (closing_) break;
            input = std::move(pending_);
            unsafe = pending_unsafe_;
            pending_unsafe_ = false;
            if (!requests_.empty()) { request = std::move(requests_.front()); requests_.pop_front(); }
        }
        const auto now = Clock::now();
        if (unsafe) {
            latest_healthy_ = false;
            controller_.reset_catchup();
            if (smoothing_.observation_grace_ms > 0 && state_.automatic) {
                mark_lost(now);
                if (state_.moving) stop("unsafe observation");
            }
        }
        if (grace_deadline_ != Clock::time_point{}) {
            if (!ready()) invalidate("control authority");
            else if (now >= grace_deadline_) stop("observation grace deadline");
        }
        const bool resumed = input.result && ingest(std::move(input));
        if (request.valid()) request(); // mode/STOP requests precede any AUTO output
        if (resumed || Clock::now() >= next) {
            tick();
            next = Clock::now() + std::chrono::milliseconds(
                std::min(config_.automatic.period_ms, config_.poll_interval_ms));
        }
    }
}
void GimbalCommandExecutor::publish(TrackingResultPtr input, bool healthy_live)
{
    if (!input || !input->frame) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto received = Clock::now();
        if ((smoothing_.observation_grace_ms > 0 || smoothing_.catchup_speed > 0) &&
            input->model_generation == runtime_.generation.load() &&
            input->tracker_epoch == runtime_.tracker_epoch.load()) {
            if (published_model_ != input->model_generation || published_tracker_ != input->tracker_epoch) {
                published_model_ = input->model_generation;
                published_tracker_ = input->tracker_epoch;
                published_frame_ = 0;
                published_capture_ = {};
            }
            const auto id = input->frame->metadata.frame_id;
            const auto captured = input->frame->metadata.captured_at;
            const bool timely = captured > Clock::time_point{} && captured <= received &&
                received - captured < std::chrono::milliseconds(config_.automatic.input_timeout_ms);
            const bool advances = id > published_frame_ && captured > published_capture_;
            const bool repeated = id == published_frame_ && captured == published_capture_;
            healthy_live = healthy_live && timely && (advances || repeated) && !input->bypass &&
                input->frame->image.type() == CV_16UC1 && !input->frame->image.empty();
            // An unsafe sample has to be visible even if a newer frame replaces it.
            if (!healthy_live) {
                pending_unsafe_ = true;
                if (smoothing_.observation_grace_ms > 0)
                    pending_ = {}; // A pre-barrier queued frame cannot restart AUTO.
            }
            if (id > published_frame_) published_frame_ = id;
            if (timely && advances) published_capture_ = captured;
        }
        const auto& queued = pending_.result;
        if (!queued || input->tracker_epoch > queued->tracker_epoch ||
            (input->tracker_epoch == queued->tracker_epoch &&
             (input->model_generation > queued->model_generation ||
              (input->model_generation == queued->model_generation &&
               input->frame->metadata.frame_id > queued->frame->metadata.frame_id))))
            pending_ = {std::move(input), received, healthy_live};
    }
    wake_.notify_one();
}
void GimbalCommandExecutor::publish_unsafe_observation()
{
    if (smoothing_.observation_grace_ms == 0 && smoothing_.catchup_speed == 0) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (smoothing_.observation_grace_ms > 0) pending_ = {};
        pending_unsafe_ = true;
    }
    wake_.notify_one();
}
bool GimbalCommandExecutor::ingest(Observation observation)
{
    auto input = std::move(observation.result);
    if (input->model_generation != runtime_.generation.load() ||
        input->tracker_epoch != runtime_.tracker_epoch.load()) return false;
    if (input->tracker_epoch != tracker_epoch_) {
        invalidate("tracker generation"); latest_.reset(); last_frame_id_ = 0;
        tracker_epoch_ = input->tracker_epoch;
    }
    if (input->model_generation != model_generation_) {
        invalidate("model generation"); latest_.reset(); last_frame_id_ = 0;
        model_generation_ = input->model_generation;
    }
    const auto id = input->frame->metadata.frame_id;
    if (id <= last_frame_id_) return false; // Duplicate frames never renew grace.
    const bool adjacent = last_frame_id_ != 0 && id - last_frame_id_ == 1;
    last_frame_id_ = id;
    latest_ = std::move(input);
    latest_healthy_ = observation.healthy_live;
    if (smoothing_.catchup_speed > 0 && (!latest_healthy_ || !adjacent)) controller_.reset_catchup();
    if (smoothing_.observation_grace_ms == 0) return false;

    if (!state_.automatic || !ready()) return false;
    const auto now = Clock::now();
    const auto captured = latest_->frame->metadata.captured_at;
    const auto prior = last_observed_capture_;
    const auto cadence = observation_cadence_;
    bool present = false;
    const bool observed = target(state_.selected_id, &present) != nullptr;
    const bool holding = grace_deadline_ != Clock::time_point{};
    if (observed) {
        // A latest-slot or wire gap may have hidden multiple missing results.
        if (!adjacent && state_.moving) {
            mark_lost(now);
            stop("observation sequence gap");
        }
        last_observed_capture_ = captured;
        observation_cadence_ = adjacent && prior != Clock::time_point{} &&
            now - prior < std::chrono::milliseconds(config_.automatic.input_timeout_ms) ?
            captured - prior : Clock::duration{};
        if (!holding || !adjacent) return false;
        grace_deadline_ = {};
        state_.lost = false; lost_since_ = {};
        lead_after_frame_ = id;
        next_auto_ = {};
        return true; // Use this measured same-ID result without waiting for the next control period.
    }

    if (!present && !state_.lost && state_.moving && latest_healthy_ && fresh() && adjacent &&
        cadence > Clock::duration::zero()) {
        const auto budget = std::min<Clock::duration>(
            std::chrono::milliseconds(smoothing_.observation_grace_ms), cadence + cadence / 2);
        const auto input_deadline = prior + std::chrono::milliseconds(config_.automatic.input_timeout_ms);
        const auto reacquire_deadline = now + std::chrono::milliseconds(config_.automatic.reacquire_timeout_ms);
        const auto deadline = std::min({observation.received + budget, input_deadline, reacquire_deadline});
        if (deadline > now) grace_deadline_ = deadline;
    }
    mark_lost(now);
    if (state_.moving && (holding || grace_deadline_ == Clock::time_point{}))
        stop("target lost/stale");
    return false;
}
void GimbalCommandExecutor::mark_lost(Clock::time_point now)
{
    last_observed_capture_ = {};
    observation_cadence_ = {};
    if (state_.lost) return;
    controller_.reset();
    state_.lost = true;
    lost_since_ = now;
    ++state_.command_generation;
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
    if ((smoothing_.observation_grace_ms > 0 && !latest_healthy_) ||
        !latest_ || !latest_->frame || latest_->bypass ||
        latest_->model_generation != runtime_.generation.load() ||
        latest_->tracker_epoch != runtime_.tracker_epoch.load() ||
        latest_->frame->image.type() != CV_16UC1 || latest_->frame->image.empty()) return false;
    const auto captured = latest_->frame->metadata.captured_at;
    const auto now = Clock::now();
    return captured != Clock::time_point{} && captured <= now &&
        now - captured < std::chrono::milliseconds(config_.automatic.input_timeout_ms);
}
const Track* GimbalCommandExecutor::target(std::uint32_t id, bool* present) const
{
    if (!fresh() || id == 0 || id > std::uint32_t(std::numeric_limits<int>::max())) return nullptr;
    const Track* found = nullptr;
    for (const auto& t : latest_->tracks) {
        if (t.track_id != static_cast<int>(id)) continue;
        if (present) *present = true;
        const bool valid = controller_.valid(t, latest_->frame->image.cols,
            latest_->frame->image.rows, latest_->aim_y);
        if (smoothing_.observation_grace_ms == 0) {
            if (valid) return &t;
        } else {
            if (!valid || found) return nullptr;
            found = &t;
        }
    }
    return found;
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
    position_estimate_.command(Clock::now(), 0, 0, 0);
    state_.moving = false; x_ = y_ = sent_speed_ = 0; limited_speed_ = 0;
    hold_x_ = hold_y_ = false;   // v27
}
void GimbalCommandExecutor::stop(const char* reason)
{
    grace_deadline_ = {};
    controller_.reset_catchup();
    // No ramp: safety/Deadband STOP is immediate command submission.
    try {
        device_->stop();
        position_estimate_.command(Clock::now(), 0, 0, 0);
        controller_.set_applied_output(0, 0);
        Logger::debug(std::string("[Control] STOP reason=") + reason);
        state_.moving = false; x_ = y_ = sent_speed_ = 0; limited_speed_ = 0;
        hold_x_ = hold_y_ = false;   // v27
        last_output_ = Clock::now();
    } catch (const std::exception& e) { failure(e); }
}
void GimbalCommandExecutor::invalidate(const char* reason)
{
    controller_.reset();
    ++state_.command_generation;
    state_.automatic = false; state_.selected_id = 0; state_.lost = false;
    lost_since_ = {}; next_auto_ = {};
    grace_deadline_ = {}; last_observed_capture_ = {}; observation_cadence_ = {};
    if (state_.moving) stop(reason);
}
void GimbalCommandExecutor::send(int x, int y, int speed, bool automatic)
{
    if (x == x_ && y == y_ && speed == sent_speed_) return;
    try {
        if (automatic) device_->move_with_speed(x, y, speed);
        else device_->move(x, y);
        position_estimate_.command(Clock::now(), x, y, speed);
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
        mark_lost(now);
        if (state_.moving && grace_deadline_ == Clock::time_point{}) stop("target lost/stale");
        return;
    }
    if (state_.lost && smoothing_.observation_grace_ms > 0)
        lead_after_frame_ = latest_->frame->metadata.frame_id;
    state_.lost = false; lost_since_ = {};
    const auto lead = latest_->frame->metadata.frame_id > lead_after_frame_ &&
        latest_->lead.control_epoch == state_.command_generation ? latest_->lead : TrackLead{};
    const bool catchup_observed = smoothing_.catchup_speed > 0 && latest_healthy_ &&
        (smoothing_.observation_grace_ms > 0 ||
         std::count_if(latest_->tracks.begin(), latest_->tracks.end(), [t](const Track& row) {
             return row.track_id == t->track_id;
         }) == 1);
    const auto demand = controller_.calculate(*t, latest_->frame->image.cols, latest_->frame->image.rows,
        latest_->frame->metadata, latest_->model_generation, lead, latest_->aim_y, catchup_observed);
    if (demand.stop()) {
        if (!state_.moving) return;
        controller_.reset_catchup();
        if (smoothing_.stop_rate <= 0) { stop("deadband"); return; }   // team original: immediate STOP
    }
    if (now < next_auto_) return;
    next_auto_ = now + std::chrono::milliseconds(config_.automatic.period_ms);
    const double dt = last_output_ == Clock::time_point{} ? config_.automatic.period_ms / 1000.0 :
        std::min(std::chrono::duration<double>(now - last_output_).count(), config_.automatic.period_ms / 1000.0);
    if (demand.stop()) {
        // v27 soft deadband stop: keep the last direction, lower SPD by stop_rate per second,
        // STOP once SPD would fall below speed_min. Every other stop path stays immediate.
        limited_speed_ -= smoothing_.stop_rate * dt;
        last_output_ = now;
        if (limited_speed_ < config_.automatic.speed_min + 0.5) { stop("deadband"); return; }
        if (!ready()) { invalidate("control authority"); return; }
        send(x_, y_, std::clamp(int(std::lround(limited_speed_)),
             config_.automatic.speed_min, config_.automatic.speed_max), true);
        controller_.set_applied_output(x_ * sent_speed_, -y_ * sent_speed_);
        return;
    }
    const bool reverse_x = x_ && demand.x && x_ != demand.x;
    const bool reverse_y = y_ && demand.y && y_ != demand.y;
    int want_x = demand.x, want_y = demand.y;
    double want_speed = demand.speed;
    if (!smoothing_.axis_reversal) {
        // Team original: reverse via STOP of both axes and restart from the minimum speed on the next period.
        if (reverse_x || reverse_y) { stop("direction reversal"); return; }
    } else {
        // v27 per-axis reversal: only the reversing axis halts; the other axis keeps moving.
        // The halted axis rejoins (either direction) once the shared SPD is down to
        // reversal_join_speed, so it normally never restarts at a high speed; after
        // reversal_hold_ms it rejoins anyway, so it cannot be kept waiting without bound.
        if (reverse_x) { hold_x_ = true; hold_x_since_ = now; }
        if (reverse_y) { hold_y_ = true; hold_y_since_ = now; }
        const double join = std::max(double(config_.automatic.speed_min), smoothing_.reversal_join_speed);
        const auto limit = std::chrono::milliseconds(smoothing_.reversal_hold_ms);
        if (hold_x_ && !reverse_x && (!demand.x || limited_speed_ <= join || now - hold_x_since_ >= limit)) hold_x_ = false;
        if (hold_y_ && !reverse_y && (!demand.y || limited_speed_ <= join || now - hold_y_since_ >= limit)) hold_y_ = false;
        if (hold_x_) want_x = 0;
        if (hold_y_) want_y = 0;
        if (!want_x && !want_y) { stop("direction reversal"); return; }
        // Shared SPD follows only the axes that actually move.
        want_speed = std::clamp(std::max(want_x ? std::abs(demand.pid_x.value) : 0.0,
                                         want_y ? std::abs(demand.pid_y.value) : 0.0),
                                double(config_.automatic.speed_min), double(config_.automatic.speed_max));
    }
    const double step = config_.automatic.speed_rate * dt;
    // The minimum is the first usable device command; subsequent changes ramp.
    const double baseline = state_.moving ? limited_speed_ : config_.automatic.speed_min;
    limited_speed_ = state_.moving ?
        std::clamp(want_speed, std::max(double(config_.automatic.speed_min), baseline - step), baseline + step) :
        double(config_.automatic.speed_min);
    const double slew_limited_speed = limited_speed_;
    bool catchup_eligible = false, catchup_applied = false;
    if (smoothing_.catchup_speed > 0) {
        const int knee = std::min(smoothing_.catchup_speed, config_.automatic.speed_max);
        const bool observed = controller_.take_catchup(want_x, want_y);
        // Only horizontal catch-up was measured. The shared speed stays on the normal ramp
        // during vertical motion, axis changes, braking, and reversal.
        catchup_eligible = observed && state_.moving && want_x != 0 && want_y == 0 &&
            want_x == x_ && want_y == y_ &&
            !hold_x_ && !hold_y_ && want_speed >= knee && want_speed > baseline;
        catchup_applied = catchup_eligible && std::lround(limited_speed_) < knee;
        if (catchup_applied) limited_speed_ = knee;
    }
    last_output_ = now;
    if (!ready()) { invalidate("control authority"); return; }
    const int speed = std::clamp(int(std::lround(limited_speed_)),
         config_.automatic.speed_min, config_.automatic.speed_max);
    const bool changed = want_x != x_ || want_y != y_ || speed != sent_speed_;
    send(want_x, want_y, speed, true);
    const bool sent = changed && !state_.error;
    trace_output(latest_->frame->metadata, t->track_id, catchup_eligible, catchup_applied && sent,
                 baseline, slew_limited_speed, want_speed, sent_speed_, x_, y_, sent);
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
        lead_after_frame_ = last_frame_id_;
        state_.selected_id = id; state_.automatic = true;
        if (smoothing_.observation_grace_ms > 0)
            last_observed_capture_ = latest_->frame->metadata.captured_at;
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
ControlSnapshot GimbalCommandExecutor::snapshot()
{
    return invoke([this] {
        position_estimate_.advance(Clock::now());
        state_.pan_estimate = position_estimate_.pan;
        state_.tilt_estimate = position_estimate_.tilt;
        return state_;
    });
}
