#include <dxrt/dxrt_c_api.h>
#include "split_sdk.hpp"

#include <array>
#include <cassert>
#include <cstring>
#include <string>
#include <vector>

namespace {
struct FakeEngine {
    int core;
    bool busy = false;
    std::uint8_t first_input = 0;
};
std::vector<int> events;
int submit_failure = -1, wait_failure = -1, destroyed = 0;

FakeEngine& get(dxrt_engine_t handle) { return *reinterpret_cast<FakeEngine*>(handle); }
}

extern "C" {
const char* dxrt_last_error_message() { return "injected"; }

dxrt_status_t dxrt_engine_create(const char*, const dxrt_options_t* opt, dxrt_engine_t* out) {
    assert(opt->device_id == 0 && opt->buffer_count == 1 && opt->use_ort == 0);
    assert(1 <= opt->bound_option && opt->bound_option <= 2);
    *out = reinterpret_cast<dxrt_engine_t>(new FakeEngine{int(opt->bound_option) - 1});
    return DXRT_OK;
}

void dxrt_engine_destroy(dxrt_engine_t engine) {
    assert(!get(engine).busy);
    ++destroyed;
    delete &get(engine);
}

dxrt_status_t dxrt_engine_get_input_size(dxrt_engine_t, std::uint64_t* size) {
    *size = 4;
    return DXRT_OK;
}

dxrt_status_t dxrt_engine_get_output_size(dxrt_engine_t, std::uint64_t* size) {
    *size = 8;
    return DXRT_OK;
}

dxrt_status_t dxrt_engine_get_input_tensor_info(dxrt_engine_t, dxrt_tensor_info_t* data, int* count) {
    if (data) {
        assert(*count >= 1);
        *data = {};
        std::strcpy(data->name, "input");
        data->ndim = 1; data->shape[0] = 4; data->type = 2;
        data->size_in_bytes = 4; data->elem_size = 1;
    }
    *count = 1;
    return DXRT_OK;
}

dxrt_status_t dxrt_engine_get_output_tensor_info(dxrt_engine_t, dxrt_tensor_info_t* data, int* count) {
    if (data) {
        assert(*count >= 1);
        *data = {};
        std::strcpy(data->name, "output");
        data->ndim = 1; data->shape[0] = 2; data->type = 1;
        data->size_in_bytes = 8; data->elem_size = 4;
    }
    *count = 1;
    return DXRT_OK;
}

dxrt_status_t dxrt_engine_run_async(dxrt_engine_t handle, const void* input, void*, void*, int* job) {
    auto& engine = get(handle);
    assert(!engine.busy);
    events.push_back(engine.core);
    if (engine.core == submit_failure) return DXRT_ERR_INVALID_ARG;
    engine.busy = true;
    engine.first_input = *static_cast<const std::uint8_t*>(input);
    *job = 7;  // Engine-local jobs can have the same ID.
    return DXRT_OK;
}

dxrt_status_t dxrt_engine_wait(dxrt_engine_t handle, int job, void* output) {
    auto& engine = get(handle);
    assert(job == 7 && engine.busy);
    events.push_back(engine.core + 2);
    engine.busy = false;
    if (output) std::memset(output, engine.first_input + engine.core, 8);
    return engine.core == wait_failure ? DXRT_ERR_INVALID_ARG : DXRT_OK;
}

dxrt_status_t dxrt_engine_get_npu_inference_time(dxrt_engine_t handle, std::uint32_t* latency) {
    *latency = 12000 + get(handle).core;
    return DXRT_OK;
}

dxrt_status_t dxrt_engine_get_latency(dxrt_engine_t handle, int* latency) {
    *latency = 15000 + get(handle).core;
    return DXRT_OK;
}

dxrt_status_t dxrt_engine_wait_with_tensor_info(dxrt_engine_t engine, int job, void* output,
                                                dxrt_tensor_info_t* data, int* count) {
    dxrt_engine_get_output_tensor_info(engine, data, count);
    return dxrt_engine_wait(engine, job, output);
}
}

int main() {
    std::uint64_t input_bytes = 0, output_bytes = 0;
    const std::array<std::uint8_t, 4> left{10}, right{20};
    std::array<std::uint8_t, 8> a{}, b{};
    LwirSplitTiming timing{};

    auto opened = [&] { return lwir_split_open("fake-model", &input_bytes, &output_bytes); };
    auto run = [&](void* pair) {
        return lwir_split_run(pair, left.data(), right.data(), input_bytes,
                              a.data(), b.data(), output_bytes, &timing);
    };

    auto* pair = opened();
    assert(pair && input_bytes == 4 && output_bytes == 8);
    int total = 0;
    assert(lwir_split_metadata(pair, 1, 0, nullptr, &total) == 0 && total == 1);
    dxrt_tensor_info_t desc{};
    assert(lwir_split_metadata(pair, 1, 1, &desc, &total) == 0 && desc.size_in_bytes == 8);
    assert(run(pair) == 0);
    assert((events == std::vector<int>{0, 1, 2, 3}));
    assert(a.front() == 10 && a.back() == 10 && b.front() == 21 && b.back() == 21);
    assert(timing.job_id[0] == 7 && timing.job_id[1] == 7);
    assert(timing.npu_us[0] == 12000 && timing.npu_us[1] == 12001);
    assert(timing.latency_us[0] == 15000 && timing.latency_us[1] == 15001);
    assert(lwir_split_submit(pair, left.data(), right.data(), input_bytes, &timing) == 0);
    assert(lwir_split_submit(pair, left.data(), right.data(), input_bytes, &timing) == -1);
    lwir_split_close(pair);

    events.clear();
    submit_failure = 1;
    pair = opened();
    assert(run(pair) == -1 && std::string(lwir_split_error()) == "submit core 1: injected");
    assert((events == std::vector<int>{0, 1, 2}));
    assert(run(pair) == -1 && events.size() == 3);
    lwir_split_close(pair);

    events.clear();
    submit_failure = -1;
    wait_failure = 0;
    pair = opened();
    assert(run(pair) == -1 && std::string(lwir_split_error()) == "DX-RT wait failed on core 0");
    assert((events == std::vector<int>{0, 1, 2, 3}));
    lwir_split_close(pair);
    assert(destroyed == 6);
}
