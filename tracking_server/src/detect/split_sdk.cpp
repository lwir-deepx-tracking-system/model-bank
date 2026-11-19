// ARM diagnostic backend for same-frame dual DX-RT submission and collection.
// The caller owns both input and output buffers through the return of run().
#include <dxrt/dxrt_c_api.h>
#include "split_sdk.hpp"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <cstring>
#include <vector>
#include <time.h>

namespace {
using Engines = std::array<dxrt_engine_t, 2>;
using Sizes = std::array<std::uint64_t, 2>;

thread_local std::string error_text;

std::uint64_t now_ns(clockid_t clock) {
    timespec stamp{};
    return clock_gettime(clock, &stamp) == 0
        ? std::uint64_t(stamp.tv_sec) * 1000000000ULL + stamp.tv_nsec : 0;
}

void note_error(const char* step, int core) {
    error_text = std::string(step) + " core " + std::to_string(core) +
        ": " + dxrt_last_error_message();
}

struct Pair {
    Engines engine{};
    Sizes sizes{};
    std::mutex lock;
    bool poisoned = false;
    bool pending = false;
    LwirSplitTiming timing{};
    std::array<std::vector<dxrt_tensor_info_t>, 2> expected, received;
    std::mutex worker_lock;
    std::condition_variable worker_ready, worker_done;
    std::thread worker;
    void* right_destination = nullptr;
    const char* right_error = nullptr;
    bool work_ready = false;
    bool work_done = false;
    bool stop_worker = false;
    bool parallel_wait = false;

    const char* collect(int core, void* destination);
    void wait_worker();

    ~Pair() {
        if (worker.joinable()) {
            {
                std::lock_guard guard(worker_lock);
                stop_worker = true;
            }
            worker_ready.notify_one();
            worker.join();
        }
        for (int core = 0; core < 2; ++core) {
            if (!engine[core]) continue;
            if (pending) dxrt_engine_wait(engine[core], timing.job_id[core], nullptr);
            dxrt_engine_destroy(engine[core]);
        }
    }
};

bool same_tensor(const dxrt_tensor_info_t& a, const dxrt_tensor_info_t& b) {
    if (a.type != b.type || a.ndim != b.ndim || a.elem_size != b.elem_size ||
        a.size_in_bytes != b.size_in_bytes || std::strncmp(a.name, b.name, sizeof(a.name)) != 0)
        return false;
    if (a.ndim < 0 || a.ndim > DXRT_MAX_TENSOR_DIMS) return false;
    for (int i = 0; i < a.ndim; ++i)
        if (a.shape[i] != b.shape[i]) return false;
    return true;
}

const char* Pair::collect(int core, void* destination) {
    timing.wait_start[core] = now_ns(CLOCK_MONOTONIC);
    const auto cpu = now_ns(CLOCK_THREAD_CPUTIME_ID);
    auto& metadata = received[core];
    const auto& layout = expected[core];
    int count = static_cast<int>(metadata.size());
    const auto code = destination
        ? dxrt_engine_wait_with_tensor_info(engine[core], timing.job_id[core], destination, metadata.data(), &count)
        : dxrt_engine_wait(engine[core], timing.job_id[core], nullptr);
    timing.wait_done[core] = now_ns(CLOCK_MONOTONIC);
    timing.wait_cpu[core] = now_ns(CLOCK_THREAD_CPUTIME_ID) - cpu;
    if (code != DXRT_OK) return "DX-RT wait failed";
    if (!destination) return nullptr;
    bool same = count == static_cast<int>(layout.size());
    for (int i = 0; same && i < count; ++i)
        same = same_tensor(metadata[i], layout[i]);
    dxrt_engine_get_npu_inference_time(engine[core], &timing.npu_us[core]);
    dxrt_engine_get_latency(engine[core], &timing.latency_us[core]);
    return same ? nullptr : "Runtime output layout changed";
}

void Pair::wait_worker() {
    std::unique_lock guard(worker_lock);
    for (;;) {
        worker_ready.wait(guard, [this] { return stop_worker || work_ready; });
        if (stop_worker) return;
        void* output = right_destination;
        work_ready = false;
        guard.unlock();
        const char* error = collect(1, output);
        guard.lock();
        right_error = error;
        work_done = true;
        worker_done.notify_one();
    }
}
}

extern "C" {

// Separate engine-local job IDs are intentionally paired with their core indices.
void* lwir_split_open(const char* model, std::uint64_t* input_bytes, std::uint64_t* output_bytes) {
    if (!model || !input_bytes || !output_bytes) {
        error_text = "Missing split engine arguments";
        return nullptr;
    }
    try {
        auto pair = std::make_unique<Pair>();
        for (int core = 0; core < 2; ++core) {
            dxrt_options_t option;
            dxrt_options_init(&option);
            option.device_id = 0;
            option.bound_option = static_cast<std::uint32_t>(core + 1);
            option.buffer_count = 1;
            option.use_ort = 0;
            auto& engine = pair->engine[core];
            if (dxrt_engine_create(model, &option, &engine) != DXRT_OK) {
                note_error("create", core);
                return nullptr;
            }
            std::uint64_t input = 0, output = 0;
            if (dxrt_engine_get_input_size(engine, &input) != DXRT_OK ||
                dxrt_engine_get_output_size(engine, &output) != DXRT_OK) {
                note_error("get buffer size", core);
                return nullptr;
            }
            if (input == 0 || output == 0 ||
                (core == 1 && (input != pair->sizes[0] || output != pair->sizes[1]))) {
                error_text = "Unexpected or different tile buffer sizes";
                return nullptr;
            }
            pair->sizes = {input, output};
            int count = 0;
            if (dxrt_engine_get_output_tensor_info(engine, nullptr, &count) != DXRT_OK || count <= 0 || count > 64) {
                note_error("get output count", core);
                return nullptr;
            }
            pair->expected[core].resize(count);
            pair->received[core].resize(count);
            if (dxrt_engine_get_output_tensor_info(engine, pair->expected[core].data(), &count) != DXRT_OK ||
                count != static_cast<int>(pair->expected[core].size())) {
                note_error("get output layout", core);
                return nullptr;
            }
        }
        *input_bytes = pair->sizes[0];
        *output_bytes = pair->sizes[1];
        return pair.release();
    } catch (const std::exception& error) {
        error_text = error.what();
        return nullptr;
    }
}

int lwir_split_metadata(void* handle, int core, int output, void* infos, int* count) {
    if (!handle || core < 0 || core > 1 || !count) {
        error_text = "Invalid metadata arguments";
        return -1;
    }
    auto engine = static_cast<Pair*>(handle)->engine[core];
    auto* metadata = static_cast<dxrt_tensor_info_t*>(infos);
    auto result = output ? dxrt_engine_get_output_tensor_info(engine, metadata, count)
                         : dxrt_engine_get_input_tensor_info(engine, metadata, count);
    if (result == DXRT_OK) return 0;
    note_error("get tensor metadata", core);
    return -1;
}

int lwir_split_set_parallel_wait(void* handle, int enabled) {
    if (!handle) return -1;
    try {
        auto& pair = *static_cast<Pair*>(handle);
        std::lock_guard guard(pair.lock);
        if (pair.pending || pair.poisoned) {
            error_text = "Cannot change wait mode with pending work or a failed engine";
            return -1;
        }
        if (enabled && !pair.worker.joinable())
            pair.worker = std::thread(&Pair::wait_worker, &pair);
        pair.parallel_wait = enabled != 0;
        return 0;
    } catch (const std::exception& error) {
        error_text = error.what();
        return -1;
    }
}

int lwir_split_submit(void* handle, const void* left, const void* right, std::size_t input_bytes,
                      LwirSplitTiming* timing) {
    if (!handle || !left || !right || !timing) {
        error_text = "Missing split input buffers";
        return -1;
    }
    try {
        auto& pair = *static_cast<Pair*>(handle);
        std::lock_guard guard(pair.lock);
        if (pair.poisoned || pair.pending || input_bytes != pair.sizes[0]) {
            error_text = "Wrong split input size, pending work, or a failed engine";
            return -1;
        }
        auto& timed = pair.timing;
        timed = {};
        timed.job_id[0] = timed.job_id[1] = -1;
        const std::array<const void*, 2> inputs{left, right};
        for (int core = 0; core < 2; ++core) {
            timed.submit_start[core] = now_ns(CLOCK_MONOTONIC);
            const auto code = dxrt_engine_run_async(
                pair.engine[core], inputs[core], nullptr, nullptr, &timed.job_id[core]);
            timed.submit_done[core] = now_ns(CLOCK_MONOTONIC);
            if (code != DXRT_OK) {
                try {
                    note_error("submit", core);
                } catch (...) {
                    if (core == 1) dxrt_engine_wait(pair.engine[0], timed.job_id[0], nullptr);
                    pair.poisoned = true;
                    throw;
                }
                if (core == 1) dxrt_engine_wait(pair.engine[0], timed.job_id[0], nullptr);
                pair.poisoned = true;
                return -1;
            }
        }
        pair.pending = true;
        *timing = timed;
        return 0;
    } catch (const std::exception& error) {
        error_text = error.what();
        return -1;
    }
}

int lwir_split_wait(void* handle, void* left_output, void* right_output, std::size_t output_bytes,
                    LwirSplitTiming* timing) {
    if (!handle) {
        error_text = "No split engine";
        return -1;
    }
    try {
        auto& pair = *static_cast<Pair*>(handle);
        std::lock_guard guard(pair.lock);
        if (!pair.pending) {
            error_text = "No pending split work";
            return -1;
        }
        const bool valid = left_output && right_output && left_output != right_output &&
                           timing && output_bytes == pair.sizes[1];
        const char* left_error;
        const char* right_error;
        if (valid && pair.parallel_wait) {
            {
                std::lock_guard guard(pair.worker_lock);
                pair.right_destination = right_output;
                pair.work_done = false;
                pair.work_ready = true;
            }
            pair.worker_ready.notify_one();
            left_error = pair.collect(0, left_output);
            std::unique_lock guard(pair.worker_lock);
            pair.worker_done.wait(guard, [&pair] { return pair.work_done; });
            right_error = pair.right_error;
        } else {
            left_error = pair.collect(0, valid ? left_output : nullptr);
            right_error = pair.collect(1, valid ? right_output : nullptr);
        }
        const char* first_error = !valid ? "Wrong or aliased split output buffers"
                                        : (left_error ? left_error : right_error);
        pair.pending = false;
        pair.poisoned = first_error != nullptr;
        if (timing) *timing = pair.timing;
        if (first_error) {
            error_text = first_error;
            if (valid) error_text += left_error ? " on core 0" : " on core 1";
        }
        return first_error ? -1 : 0;
    } catch (const std::exception& error) {
        error_text = error.what();
        return -1;
    }
}

int lwir_split_run(void* handle, const void* left, const void* right, std::size_t input_bytes,
                   void* left_output, void* right_output, std::size_t output_bytes,
                   LwirSplitTiming* timing) {
    if (lwir_split_submit(handle, left, right, input_bytes, timing) != 0) return -1;
    return lwir_split_wait(handle, left_output, right_output, output_bytes, timing);
}

const char* lwir_split_error() { return error_text.c_str(); }

void lwir_split_close(void* handle) { delete static_cast<Pair*>(handle); }

}
