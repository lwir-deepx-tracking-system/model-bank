#include "pipeline/runtime_control.hpp"

void RuntimeControl::set_detection_quiesced(bool value)
{
    {   // mutex lock_guard용 스코프
        std::lock_guard<std::mutex> lock(mutex_);
        detection_quiesced_ = value;
    }
    condition_.notify_all();
}

bool RuntimeControl::wait_detection_quiesced()
{
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return detection_quiesced_ || stop_requested.load(); });
    return detection_quiesced_ && !stop_requested.load();
}

void RuntimeControl::request_stop()
{
    stop_requested.store(true);
    state.store(RuntimeState::Stopping);
    condition_.notify_all();
}
