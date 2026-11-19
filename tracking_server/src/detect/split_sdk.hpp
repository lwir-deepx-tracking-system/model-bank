#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

struct LwirSplitTiming {
    std::uint64_t submit_start[2], submit_done[2];
    std::uint64_t wait_start[2], wait_done[2], wait_cpu[2];
    std::uint32_t npu_us[2];
    int latency_us[2];
    int job_id[2];
};

void* lwir_split_open(const char* model, std::uint64_t* input_bytes, std::uint64_t* output_bytes);
int lwir_split_metadata(void* handle, int core, int output, void* infos, int* count);
int lwir_split_set_parallel_wait(void* handle, int enabled);
int lwir_split_submit(void* handle, const void* left, const void* right, std::size_t input_bytes,
                      LwirSplitTiming* timing);
int lwir_split_wait(void* handle, void* left_output, void* right_output, std::size_t output_bytes,
                    LwirSplitTiming* timing);
int lwir_split_run(void* handle, const void* left, const void* right, std::size_t input_bytes,
                   void* left_output, void* right_output, std::size_t output_bytes,
                   LwirSplitTiming* timing);
const char* lwir_split_error();
void lwir_split_close(void* handle);

}
