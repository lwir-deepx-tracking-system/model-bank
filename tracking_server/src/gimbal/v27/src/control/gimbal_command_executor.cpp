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
bool control_output_trace_enabled()
{
    static const bool enabled = [] {
        const char* value = std::getenv("LWIR_CONTROL_TRACE");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

void trace_predictive_brake(const FrameMetadata& frame, int track_id, int direction, int speed,
                            double error, double closing, double reserve, bool sent) noexcept try
{
    if (!control_output_trace_enabled()) return;
    const auto captured_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        frame.captured_at.time_since_epoch()).count();
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::boolalpha << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "{\"event\":\"predictive_brake\",\"frame_id\":" << frame.frame_id
        << ",\"captured_ns\":" << captured_ns << ",\"track_id\":" << track_id
        << ",\"prior_x\":" << direction << ",\"prior_speed\":" << speed
        << ",\"error_px\":" << error << ",\"closing_px_per_s\":" << closing
        << ",\"reserve_px\":" << reserve << ",\"sent\":" << sent << "}\n";
    std::fputs(out.str().c_str(), stderr);
} catch (...) {
    // A diagnostics formatting failure must not interrupt control.
}

void trace_output(const FrameMetadata& frame, int track_id, bool catchup_eligible,
                  bool catchup_applied, double previous_limited_speed, double slew_limited_speed,
                  double wanted_speed, int final_speed, int x, int y, bool sent) noexcept try
{
    if (!control_output_trace_enabled()) return;
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
    if (smoothing_.search) {   // v27 search when lost
        const auto& s = smoothing_;
        if (s.search_grace_ms < 0 || s.search_follow_ms < 0 || s.search_timeout_ms <= s.search_grace_ms ||
            !(s.search_follow_px >= 0) || !(s.search_sweep_px > 0) || !(s.search_widen >= 1.0 && s.search_widen <= 4.0) ||
            !(s.search_max_px >= s.search_sweep_px) || !(s.search_tilt_px >= 0) ||
            s.search_speed < config_.automatic.speed_min || s.search_speed > config_.automatic.speed_max ||
            s.search_follow_speed < config_.automatic.speed_min || s.search_follow_speed > config_.automatic.speed_max)
            throw std::invalid_argument("search settings out of range");
    }
    if (smoothing_.coast && (smoothing_.coast_ms <= 0 || smoothing_.coast_ms > 3000))   // v27 coast when lost
        throw std::invalid_argument("--coast-ms must be within 1..3000");
    if (smoothing_.seek && (smoothing_.search || smoothing_.seek_ms <= 0 || smoothing_.seek_ms > 10000 ||   // v27 seek when lost
        smoothing_.seek_speed < config_.automatic.speed_min || smoothing_.seek_speed > config_.automatic.speed_max))
        throw std::invalid_argument("--seek: not with --search 1, --seek-ms within 1..10000, --seek-speed within --speed-min..--speed-max");
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
        bool unsafe = false, reset_brake = false;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            const auto wake_at = grace_deadline_ == Clock::time_point{} ? next : std::min(next, grace_deadline_);
            wake_.wait_until(lock, wake_at, [this] {
                return closing_ || !requests_.empty() || pending_.result || pending_unsafe_;
            });
            if (closing_) break;
            input = std::move(pending_);
            unsafe = pending_unsafe_;
            reset_brake = pending_brake_reset_;
            pending_unsafe_ = pending_brake_reset_ = false;
            if (!requests_.empty()) { request = std::move(requests_.front()); requests_.pop_front(); }
        }
        const auto now = Clock::now();
        if (reset_brake) brake_ = {};
        if (unsafe) {
            latest_healthy_ = false;
            controller_.reset_catchup();
            brake_ = {};
            if (smoothing_.observation_grace_ms > 0 && state_.automatic) {
                mark_lost(now);
                if (state_.moving) stop("unsafe observation");
            }
        }
        if (grace_deadline_ != Clock::time_point{}) {
            if (!ready()) invalidate("control authority");
            else if (now >= grace_deadline_) stop("observation grace deadline");
        }
        arrived_ = false;
        const bool resumed = input.result && ingest(std::move(input));
        if (request.valid()) request(); // mode/STOP requests precede any AUTO output
        // v27 arrival control: a newly taken-in frame is acted on now instead of at the next period (0..period_ms later).
        // The period timer restarts here, so time-outs, ramps, coast and search keep running every period_ms.
        const bool arrival = smoothing_.arrival_control && arrived_;
        if (resumed || arrival || Clock::now() >= next) {
            tick(arrival);
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
        if ((smoothing_.observation_grace_ms > 0 || smoothing_.catchup_speed > 0 ||
             smoothing_.predictive_brake || smoothing_.search || smoothing_.coast || smoothing_.seek) &&   // v27 search / coast / seek move only on validated live frames
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
            if (smoothing_.predictive_brake && healthy_live) {
                const auto selected = static_cast<int>(published_brake_target_.load());
                if (selected > 0) {
                    const auto matches = [selected](const Track& row) { return row.track_id == selected; };
                    const auto first = std::find_if(input->tracks.begin(), input->tracks.end(), matches);
                    const auto& image = input->frame->image;
                    if (first == input->tracks.end() ||
                        !controller_.valid(*first, image.cols, image.rows, input->aim_y) ||
                        std::any_of(first + 1, input->tracks.end(), matches))
                        pending_brake_reset_ = true;
                }
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
    if (smoothing_.observation_grace_ms == 0 && smoothing_.catchup_speed == 0 &&
        !smoothing_.predictive_brake && !smoothing_.search && !smoothing_.coast && !smoothing_.seek) return;
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
    arrived_ = true;   // v27 arrival control
    if (smoothing_.catchup_speed > 0 && (!latest_healthy_ || !adjacent)) controller_.reset_catchup();
    if (smoothing_.predictive_brake && (!latest_healthy_ ||
        (state_.automatic && !target(state_.selected_id)))) brake_ = {};
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
    if (state_.moving && !searching() && !coasting_ && (holding || grace_deadline_ == Clock::time_point{}))
        stop("target lost/stale");
    return false;
}
void GimbalCommandExecutor::mark_lost(Clock::time_point now)
{
    last_observed_capture_ = {};
    observation_cadence_ = {};
    brake_ = {};
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
    brake_ = {};
    state_.error = true;
    published_brake_target_.store(0);
    state_.automatic = false; state_.selected_id = 0; state_.lost = false;
    search_ = {}; state_.search = 0;   // v27
    coasting_ = coasted_ = seeking_ = false;      // v27 coast / seek
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
    brake_ = {};
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
    published_brake_target_.store(0);
    state_.automatic = false; state_.selected_id = 0; state_.lost = false;
    lost_since_ = {}; next_auto_ = {};
    grace_deadline_ = {}; last_observed_capture_ = {}; observation_cadence_ = {};
    brake_ = {};
    search_ = {}; state_.search = 0; follow_x_ = follow_speed_ = 0; follow_error_x_ = follow_vx_ = 0; follow_v_at_ = {};   // v27
    coasting_ = coasted_ = seeking_ = false;   // v27 coast / seek
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
bool GimbalCommandExecutor::apply_predictive_brake(const Track& track, const GimbalDemand& demand,
                                                    const TrackLead& lead)
{
    const bool forecast = lead.matches(track.track_id) && (lead.dx_px != 0 || lead.dy_px != 0);
    const auto duplicate_id = std::count_if(latest_->tracks.begin(), latest_->tracks.end(),
        [&track](const Track& row) { return row.track_id == track.track_id; }) != 1;
    if (!latest_healthy_ || forecast || duplicate_id || y_ || demand.y) {
        brake_ = {};
        return false;
    }

    using namespace std::chrono_literals;
    constexpr int max_brake_speed = 7;
    constexpr double preview_seconds = 0.050;
    constexpr double max_reserve_at_640 = 4.0;
    const auto& frame = latest_->frame->metadata;
    const int width = latest_->frame->image.cols;
    if (brake_.width != width) brake_ = {};
    if (frame.frame_id == brake_.previous.frame_id)
        return brake_.hold_since != Clock::time_point{};

    const double half_width = width / 2.0;
    const double position = double(track.x) + double(track.width) / 2 - half_width;
    const int side = position > 0 ? 1 : position < 0 ? -1 : 0;
    double band = config_.automatic.deadband_x;
    if (config_.automatic.fov_x_deg > 0 && config_.automatic.fov_y_deg > 0) {
        constexpr double radians = 3.14159265358979323846 / 180.0;
        band = std::tan(band) / std::tan(config_.automatic.fov_x_deg * radians / 2);
    }
    const double outside = std::abs(position) - band * half_width;
    const double max_reserve = max_reserve_at_640 * width / 640.0;
    const auto dt = frame.captured_at - brake_.previous.captured_at;
    // The latest-only executor can skip a valid publication. Its observed timing
    // remains usable, but the publisher independently records explicit target loss.
    const bool consecutive = brake_.previous.frame_id && frame.frame_id > brake_.previous.frame_id &&
        dt >= 10ms && dt <= 100ms;
    const double closing = consecutive && side * brake_.position > 0 ?
        side * (brake_.position - position) / std::chrono::duration<double>(dt).count() : 0.0;
    const double reserve = std::min(max_reserve, preview_seconds * std::min(closing, brake_.closing_speed));
    const bool in_shell = std::isfinite(band) && band >= 0 && outside > 0 && outside <= max_reserve;
    // A derivative-driven reversal near center may be a correction for travel
    // already underway. Stop first, then let new observations decide direction.
    const bool derivative_reversal = demand.x == -side &&
        side * (demand.pid_x.p + demand.pid_x.i) > 0 && side * demand.pid_x.d < 0;
    const bool approach_demand = demand.x == side || derivative_reversal;
    brake_.previous = frame;
    brake_.width = width;
    brake_.position = position;
    brake_.closing_speed = closing;

    if (!in_shell || brake_.used_direction != side) {
        brake_.used_direction = 0;
        brake_.hold_since = {};
    } else if (brake_.hold_since != Clock::time_point{}) {
        // Re-evaluate a hold only with new observations. Any outward motion releases
        // immediately; duplicates cannot prolong it past the existing freshness check.
        if (consecutive && closing >= 0 && approach_demand && frame.captured_at - brake_.hold_since < 100ms)
            return true;
        brake_.hold_since = {};
    }
    if (brake_.used_direction || !in_shell || outside > reserve ||
        !state_.moving || x_ != side || !approach_demand ||
        sent_speed_ < 1 || sent_speed_ > max_brake_speed)
        return false;

    if (!ready()) { invalidate("control authority"); return true; }
    const int prior_speed = sent_speed_;
    stop("predictive brake");
    const bool sent = !state_.error;
    if (sent) brake_ = {frame, width, position, closing, side, frame.captured_at};
    trace_predictive_brake(frame, track.track_id, side, prior_speed, position, closing, reserve, sent);
    return true;
}

// v27 search when lost (SmoothingConfig::search). Called by tick() on every period while the followed ID is missing.
// Open loop: the pose offset from the loss is estimated from the commanded SPD (search_picture_px_per_s); it only bounds
// and centres the pattern. Every safety condition of following still applies (tick() checked heartbeat and mode; this
// checks frame freshness/health and control authority before each output).
void GimbalCommandExecutor::search_step(Clock::time_point now)
{
    using std::chrono::milliseconds;
    auto& s = search_;
    const auto& c = smoothing_;
    if (s.phase == Search::None) {
        s = {};
        s.phase = Search::Grace;
        s.lost_at = lost_since_ == Clock::time_point{} ? now : lost_since_;
        s.last_step = now;
        // Follow-through: where the person should be. First choice = the server's predictive-control velocity of the
        // person against the background (VEL, predictive_lead.py), at the SPD whose measured picture speed covers it;
        // without one (lead data missing/old, slower than 15 px/s) = the last following output.
        const bool velocity = follow_v_at_ != Clock::time_point{} && s.lost_at - follow_v_at_ <= milliseconds(300) &&
            std::abs(follow_vx_) >= SmoothingConfig::search_min_velocity;
        if (velocity) {
            s.dir_x = follow_vx_ > 0 ? 1 : -1;
            int spd = config_.automatic.speed_min;
            while (spd < c.search_follow_speed && search_picture_px_per_s(spd) < std::abs(follow_vx_)) ++spd;
            s.follow_speed = std::clamp(spd, config_.automatic.speed_min, c.search_follow_speed);
        } else {
            s.dir_x = follow_x_;
            s.follow_speed = std::clamp(follow_speed_, config_.automatic.speed_min, c.search_follow_speed);
        }
        s.side = s.dir_x ? s.dir_x : follow_error_x_ < 0 ? -1 : 1;
        if (coasted_) {   // v27 coast: the coast already went where the person should be; sweep around the place it ended
            s.dir_x = 0;
            s.side = x_ ? x_ : follow_error_x_ < 0 ? -1 : 1;
        }
        Logger::debug("[Control] search: target lost, follow-through direction " + std::to_string(s.dir_x) +
                      (velocity ? " from the predictive velocity " + std::to_string(follow_vx_) + " px/s" : " from the last output"));
    }
    const double dt = std::max(0.0, std::chrono::duration<double>(now - s.last_step).count());
    s.last_step = now;
    if (state_.moving) {   // the command applied since the previous step
        const double px = search_picture_px_per_s(sent_speed_) * dt;
        s.pan += x_ * px;
        s.tilt += y_ * px;
    }
    const auto halt = [this](const char* why) { if (state_.moving) stop(why); };
    const auto report = [this, &s] { state_.search = int(s.phase); };
    if (grace_deadline_ != Clock::time_point{}) { report(); return; }   // observation grace owns the output until its deadline
    const auto elapsed = now - s.lost_at;
    if (s.phase == Search::Done) { halt("search done"); report(); return; }
    if (elapsed >= milliseconds(c.search_timeout_ms)) {
        s.phase = Search::Done; halt("search timeout"); report(); return;
    }
    // Blind (no fresh healthy frame): never move. The pattern continues when frames are good again.
    if (!fresh() || !latest_healthy_) { s.blind = true; halt("search: no fresh healthy frame"); report(); return; }
    s.blind = false;
    if (elapsed < milliseconds(c.search_grace_ms)) { halt("target lost/stale"); report(); return; }
    if (!ready()) { invalidate("control authority"); return; }
    const double step = config_.automatic.speed_rate * std::min(dt, config_.automatic.period_ms / 1000.0);
    const auto drive = [&](int x, int y, int speed) {
        const double want = std::clamp(speed, config_.automatic.speed_min, config_.automatic.speed_max);
        const bool turn = state_.moving && ((x_ && x && x_ != x) || (y_ && y && y_ != y));
        if (turn) { stop("search turn"); return; }   // as the team rule: a reversal goes through STOP
        limited_speed_ = state_.moving ?
            std::clamp(want, std::max(double(config_.automatic.speed_min), limited_speed_ - step), limited_speed_ + step) :
            double(config_.automatic.speed_min);
        last_output_ = now;
        send(x, y, std::clamp(int(std::lround(limited_speed_)), config_.automatic.speed_min, config_.automatic.speed_max), true);
        controller_.set_applied_output(x_ * sent_speed_, -y_ * sent_speed_);
    };
    if (s.phase == Search::Grace) {
        s.phase = s.dir_x ? Search::Follow : Search::Sweep;
        s.follow_since = now;
        s.leg = 0;
    }
    if (s.phase == Search::Follow) {
        if (now - s.follow_since < milliseconds(c.search_follow_ms) && std::abs(s.pan) < c.search_follow_px &&
            std::abs(s.pan) < c.search_max_px) {
            drive(s.dir_x, 0, s.follow_speed);
            report();
            return;
        }
        s.phase = Search::Sweep;
        s.leg = 0;
    }
    // Sweep: leg k goes to start * (-1)^k * A_k, A_k = min(max, sweep * widen^k). After a follow-through the first leg
    // turns back across the place of the loss; without one it starts on the side where the person was last seen.
    // The widest leg is swept to both sides, then the pattern ends.
    const int start = s.dir_x ? -s.side : s.side;
    const auto amplitude_of = [&c](int leg) {
        return std::min(c.search_max_px, c.search_sweep_px * std::pow(c.search_widen, leg));
    };
    int last_leg = 0;
    while (last_leg < 62 && amplitude_of(last_leg) < c.search_max_px) ++last_leg;
    ++last_leg;
    for (;;) {
        if (s.leg > last_leg) { s.phase = Search::Done; halt("search pattern complete"); report(); return; }
        const double amplitude = amplitude_of(s.leg);
        const int dir = (s.leg % 2 == 0) ? start : -start;
        const double target = dir * amplitude;
        if (dir * (target - s.pan) <= 0) { ++s.leg; continue; }   // this leg's end is reached
        int y = 0;
        if (c.search_tilt_px > 0) {   // small tilt: up on even legs, down on odd legs
            const double tilt_target = (s.leg % 2 == 0 ? 1 : -1) * c.search_tilt_px;
            if (std::abs(tilt_target - s.tilt) > 4.0) y = tilt_target > s.tilt ? 1 : -1;
        }
        drive(dir, y, c.search_speed);
        report();
        return;
    }
}

// v27 coast when lost (SmoothingConfig::coast): the server's predicted box of the selected ID as a Track for the normal
// controller, or nullptr. Only on a fresh, healthy frame that carries PRED for the selected ID, under control authority, and
// for at most coast_ms after the loss (lost_since_). The caller has called mark_lost().
const Track* GimbalCommandExecutor::coast_target(Clock::time_point now)
{
    if (!latest_ || !fresh() || !latest_healthy_ || !ready() || state_.selected_id == 0 ||
        now - lost_since_ >= std::chrono::milliseconds(smoothing_.coast_ms)) return nullptr;
    const auto& p = latest_->prediction;
    if (!p.matches(static_cast<int>(state_.selected_id))) return nullptr;
    Track t;
    t.track_id = p.track_id; t.class_id = 0; t.confidence = 1.0f;
    t.x = float(p.x); t.y = float(p.y); t.width = float(p.width); t.height = float(p.height);
    if (!controller_.valid(t, latest_->frame->image.cols, latest_->frame->image.rows, latest_->aim_y)) return nullptr;
    if (!coasting_) Logger::debug("[Control] coast: following the predicted place of " + std::to_string(state_.selected_id));
    coast_track_ = t;
    coasting_ = coasted_ = true;
    if (seeking_) { seeking_ = false; state_.search = 0; }
    return &coast_track_;
}

// v27 seek when lost (SmoothingConfig::seek): the server's SEEK box of the selected ID as a Track for the normal controller,
// or nullptr. The same conditions as the coast (fresh healthy frame that names the selected ID, control authority), for at
// most seek_ms after the loss. tick() caps the speed and turns through STOP while this box is followed.
const Track* GimbalCommandExecutor::seek_target(Clock::time_point now)
{
    if (!latest_ || !fresh() || !latest_healthy_ || !ready() || state_.selected_id == 0 ||
        now - lost_since_ >= std::chrono::milliseconds(smoothing_.seek_ms)) return nullptr;
    const auto& p = latest_->seek;
    if (!p.matches(static_cast<int>(state_.selected_id))) return nullptr;
    Track t;
    t.track_id = p.track_id; t.class_id = 0; t.confidence = 1.0f;
    t.x = float(p.x); t.y = float(p.y); t.width = float(p.width); t.height = float(p.height);
    if (!controller_.valid(t, latest_->frame->image.cols, latest_->frame->image.rows, latest_->aim_y)) return nullptr;
    if (!seeking_) Logger::debug("[Control] seek: looking where the server points for " + std::to_string(state_.selected_id));
    // The server moves the SEEK point in steps (the next turn of its sweep). A step is a new target, not motion of a person:
    // the controller starts again from the new point, so its D term cannot answer the step with a kick the wrong way.
    if (coasted_ && (std::abs(t.x - coast_track_.x) > 32.0f || std::abs(t.y - coast_track_.y) > 32.0f)) controller_.reset();
    coast_track_ = t;
    coasting_ = coasted_ = seeking_ = true;
    state_.search = 5;
    return &coast_track_;
}

void GimbalCommandExecutor::tick(bool arrival)
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
        if (smoothing_.coast && search_.phase == Search::None) t = coast_target(now);   // v27 coast
        if (!t && smoothing_.seek) t = seek_target(now);   // v27 seek: after (or without) the coast
        if (!t) {
            coasting_ = false;
            if (seeking_) { seeking_ = false; state_.search = 0; }
            if (smoothing_.search) { search_step(now); return; }   // v27
            if (state_.moving && grace_deadline_ == Clock::time_point{}) stop("target lost/stale");
            return;
        }
    }
    const bool coast = t == &coast_track_;   // v27 coast / seek: still lost (the reacquire timeout runs), no lead/brake/catch-up
    const bool seek = coast && seeking_;     // v27 seek: SPD capped, a turn goes through STOP
    if (!coast) {
        coasting_ = coasted_ = false;
        if (seeking_) { seeking_ = false; state_.search = 0; }
        if (state_.lost && smoothing_.observation_grace_ms > 0)
            lead_after_frame_ = latest_->frame->metadata.frame_id;
        if (search_.phase != Search::None) {   // v27: the same ID is back; the search output is the starting point of following
            Logger::debug(std::string("[Control] search ended: target back"));
            search_ = {}; state_.search = 0;
        }
        state_.lost = false; lost_since_ = {};
        follow_x_ = x_; follow_speed_ = sent_speed_;   // v27: the output while the target is seen
        if (latest_->velocity.matches(t->track_id)) { follow_vx_ = latest_->velocity.vx; follow_v_at_ = now; }
    }
    const auto lead = !coast && latest_->frame->metadata.frame_id > lead_after_frame_ &&
        latest_->lead.control_epoch == state_.command_generation ? latest_->lead : TrackLead{};
    const bool catchup_observed = !coast && smoothing_.catchup_speed > 0 && latest_healthy_ &&
        (smoothing_.observation_grace_ms > 0 ||
         std::count_if(latest_->tracks.begin(), latest_->tracks.end(), [t](const Track& row) {
             return row.track_id == t->track_id;
         }) == 1);
    const auto demand = controller_.calculate(*t, latest_->frame->image.cols, latest_->frame->image.rows,
        latest_->frame->metadata, latest_->model_generation, lead, latest_->aim_y, catchup_observed);
    if (!coast) follow_error_x_ = demand.error_x;   // v27
    if (smoothing_.predictive_brake && !coast && apply_predictive_brake(*t, demand, lead)) return;
    if (demand.stop()) {
        if (!state_.moving) return;
        controller_.reset_catchup();
        if (smoothing_.stop_rate <= 0) { stop("deadband"); return; }   // team original: immediate STOP
    }
    // v27 arrival control: the step for a new frame is not held back by the output period. The ramp below uses the real
    // time since the previous output (dt), so SPD per second is the same however the steps are spaced.
    if (!arrival && now < next_auto_) return;
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
    if (seek && (reverse_x || reverse_y)) { stop("seek turn"); return; }   // v27 seek: both axes halt, the new direction starts from rest
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
        // Shared SPD follows only the axes that actually move. speed_x/speed_y carry --far-gain and
        // --brake-px-per-spd (they equal |pid value| when both are off), so those options work here too.
        want_speed = std::clamp(std::max(want_x ? demand.speed_x : 0.0,
                                         want_y ? demand.speed_y : 0.0),
                                double(config_.automatic.speed_min), double(config_.automatic.speed_max));
    }
    if (seek) want_speed = std::min(want_speed, double(smoothing_.seek_speed));   // v27 seek: a speed at which the detector still sees people
    const double step = config_.automatic.speed_rate * dt;
    // The minimum is the first usable device command; subsequent changes ramp.
    const double baseline = state_.moving ? limited_speed_ : config_.automatic.speed_min;
    limited_speed_ = state_.moving ?
        std::clamp(want_speed, std::max(double(config_.automatic.speed_min), baseline - step), baseline + step) :
        double(config_.automatic.speed_min);
    if (seek) limited_speed_ = std::min(limited_speed_, double(smoothing_.seek_speed));   // v27 seek: the cap holds at once, also when a faster move was under way
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
        if (smoothing_.predictive_brake) published_brake_target_.store(id);
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
