// v27 copy of team/include/control/gimbal_command_executor.hpp: changed only where marked "v27" (motion smoothing, gimbal_bridge_v27). See ../../../V27_CHANGES.md.
#pragma once

#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include "control/auto_tracking_controller.hpp"
#include "control/gimbal_device.hpp"
#include "control/gimbal_position_estimate.hpp"
#include "pipeline/runtime_control.hpp"

enum class ControlResult { Ok, InvalidTarget, NotReady };
struct ControlSnapshot {
    bool automatic = false;
    bool lost = false;
    bool moving = false;
    bool error = false;
    std::uint32_t selected_id = 0;
    std::uint64_t command_generation = 0;
    double pan_estimate = 50.0;
    double tilt_estimate = 50.0;
};

// One worker owns device I/O and every mutable control decision. GUI requests
// complete only after execution; computed AUTO moves are never queued for replay.
class GimbalCommandExecutor {
    using Clock = std::chrono::steady_clock;
    ControlConfig config_;
    SmoothingConfig smoothing_;   // v27
    RuntimeControl& runtime_;
    std::unique_ptr<IGimbalDevice> device_;
    AutoTrackingController controller_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::packaged_task<void()>> requests_;
    struct Observation {
        TrackingResultPtr result;
        Clock::time_point received{};
        bool healthy_live = false;
    };
    Observation pending_;
    bool pending_unsafe_ = false;
    // Publication order is checked before the latest-only slot can discard a sample.
    std::uint64_t published_model_ = 0, published_tracker_ = 0, published_frame_ = 0;
    Clock::time_point published_capture_{};
    bool closing_ = false;
    std::thread worker_;

    // Worker-owned state only.
    ControlSnapshot state_;
    GimbalPositionEstimate position_estimate_;
    TrackingResultPtr latest_;
    bool latest_healthy_ = false;
    Clock::time_point last_observed_capture_{}, grace_deadline_{};
    Clock::duration observation_cadence_{};
    std::uint64_t model_generation_ = 0, last_frame_id_ = 0;
    std::uint64_t lead_after_frame_ = 0; // A fresh selection cannot reuse the previous selection's forecast.
    std::uint64_t tracker_epoch_ = 0;
    bool connected_ = false;
    bool stopped_ = false;
    Clock::time_point ping_{}, lost_since_{}, last_output_{}, next_auto_{};
    int x_ = 0, y_ = 0, sent_speed_ = 0;
    double limited_speed_ = 0;
    bool hold_x_ = false, hold_y_ = false;   // v27: axis halted by a per-axis reversal, waiting to rejoin
    Clock::time_point hold_x_since_{}, hold_y_since_{};

    void run();
    bool ingest(Observation);
    void mark_lost(Clock::time_point now);
    void tick();
    bool alive() const;
    bool ready() const;
    bool fresh() const;
    const Track* target(std::uint32_t, bool* present = nullptr) const;
    void invalidate(const char* reason);
    void stop(const char* reason);
    void failure(const std::exception&);
    void send(int x, int y, int speed, bool automatic);
    template<class F> auto invoke(F fn, bool priority = false) -> decltype(fn()) {
        using R = decltype(fn());
        auto task = std::make_shared<std::packaged_task<R()>>(std::move(fn));
        auto result = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (priority) requests_.emplace_front([task] { (*task)(); });
            else requests_.emplace_back([task] { (*task)(); });
        }
        wake_.notify_one();
        return result.get();
    }
public:
    // v27: optional smoothing settings; the default reproduces the team executor exactly.
    GimbalCommandExecutor(ControlConfig, std::unique_ptr<IGimbalDevice>, RuntimeControl&, SmoothingConfig = {});
    ~GimbalCommandExecutor();
    GimbalCommandExecutor(const GimbalCommandExecutor&) = delete;
    GimbalCommandExecutor& operator=(const GimbalCommandExecutor&) = delete;
    void publish(TrackingResultPtr, bool healthy_live = false); // bounded latest-result mailbox
    void publish_unsafe_observation(); // malformed input without a usable frame
    void connected();
    void disconnected();
    void ping();
    ControlResult select(std::uint32_t id);
    ControlResult manual();
    void halt(); // cancel AUTO and stop manual motion (unsupported HOME, etc.)
    void move(int x, int y);
    void suspend(); // model switch: cancel authority but keep the TCP session alive
    void shutdown();
    ControlSnapshot snapshot();
};
