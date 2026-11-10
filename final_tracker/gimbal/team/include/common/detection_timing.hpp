#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

// 비동기 요청 하나에 귀속되는 Detection 내부 단계별 측정 시각.
// steady_clock을 사용해 시스템 시각 변경의 영향을 받지 않는다.
struct DetectionTiming
{
    struct NpuDeviceTiming
    {
        std::string task_name;
        int device_id = 0;
        double input_format_us = 0.0;
        double h2d_us = 0.0;
        double inference_core_all_us = 0.0;
        double inference_core_0_us = 0.0;
        double inference_core_1_us = 0.0;
        double inference_core_2_us = 0.0;
        double d2h_us = 0.0;
        double output_format_us = 0.0;
        double total_us = 0.0;
    };

    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    std::uint64_t frame_id = 0;
    TimePoint preprocess_started_at{};
    TimePoint preprocess_finished_at{};
    TimePoint inference_started_at{};
    TimePoint inference_finished_at{};
    TimePoint postprocess_started_at{};
    TimePoint postprocess_finished_at{};

    // DXRT job profiler가 제공하는 장치 내부 구간별 시간이다.
    // 여러 NPU task/device가 있으면 한 job 안의 유효한 값을 합산한다.
    double npu_input_format_duration_us = 0.0;
    double h2d_duration_us = 0.0;
    double npu_processing_duration_ms = 0.0;
    double d2h_duration_us = 0.0;
    double npu_output_format_duration_us = 0.0;
    double npu_task_total_duration_us = 0.0;
    bool npu_profiler_valid = false;
    std::vector<NpuDeviceTiming> npu_devices;
    double cpu_task_us = 0.0;
    bool cpu_task_valid = false;

    double preprocess_ms() const
    {
        return elapsed_ms(preprocess_started_at, preprocess_finished_at);
    }

    // dxrun Latency의 비교 대상으로 RunAsync 직전부터 callback 진입까지 측정한다.
    double latency_ms() const
    {
        return elapsed_ms(inference_started_at, inference_finished_at);
    }

    // DXRT profiler가 측정한 순수 NPU core 실행 시간이다.
    double npu_processing_ms() const
    {
        return npu_processing_duration_ms;
    }

    double postprocess_ms() const
    {
        return elapsed_ms(postprocess_started_at, postprocess_finished_at);
    }

private:
    static double elapsed_ms(TimePoint started_at, TimePoint finished_at)
    {
        return std::chrono::duration<double, std::milli>(
            finished_at - started_at).count();
    }
};
