#pragma once

#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include "control/auto_tracking_controller.hpp"
#include "control/gimbal_device.hpp"
#include "pipeline/runtime_control.hpp"

enum class ControlResult { Ok, InvalidTarget, NotReady };
struct ControlSnapshot {
    bool automatic = false;
    bool lost = false;
    bool moving = false;
    bool error = false;
    std::uint32_t selected_id = 0;
    std::uint64_t command_generation = 0;
};

// One worker owns device I/O and every mutable control decision. GUI requests
// complete only after execution; computed AUTO moves are never queued for replay.
class GimbalCommandExecutor {
    using Clock = std::chrono::steady_clock;
    ControlConfig config_;
    RuntimeControl& runtime_;
    std::unique_ptr<IGimbalDevice> device_;
    AutoTrackingController controller_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::packaged_task<void()>> requests_;
    TrackingResultPtr pending_;
    bool closing_ = false;
    std::thread worker_;

    // Worker-owned state only.
    ControlSnapshot state_;
    TrackingResultPtr latest_;
    std::uint64_t model_generation_ = 0, last_frame_id_ = 0;
    std::uint64_t tracker_epoch_ = 0;
    bool connected_ = false;
    bool stopped_ = false;
    Clock::time_point ping_{}, lost_since_{}, last_output_{}, next_auto_{};
    int x_ = 0, y_ = 0, sent_speed_ = 0;
    double limited_speed_ = 0;

    void run();
    void ingest(TrackingResultPtr);
    void tick();
    bool alive() const;
    bool ready() const;
    bool fresh() const;
    const Track* target(std::uint32_t) const;
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
    GimbalCommandExecutor(ControlConfig, std::unique_ptr<IGimbalDevice>, RuntimeControl&);
    ~GimbalCommandExecutor();
    GimbalCommandExecutor(const GimbalCommandExecutor&) = delete;
    GimbalCommandExecutor& operator=(const GimbalCommandExecutor&) = delete;
    void publish(TrackingResultPtr); // bounded latest-result mailbox
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
