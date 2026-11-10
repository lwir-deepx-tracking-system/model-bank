#pragma once

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <optional>
#include <vector>

#include "common/frame_metadata.hpp"

// 프레임별 큐 대기, 처리, 필요한 경우 전체 지연을 기록한다.
struct StageMetric
{
    std::uint64_t frame_id = 0;
    std::optional<double> queue_wait_ms;
    double processing_ms = 0.0;
    std::optional<double> e2e_ms;
};

// 장시간 실행 시 측정 기록의 메모리 사용량을 제한한다.
constexpr std::size_t kMaxStageMetrics = 10000;

// 시각 정보를 저장 컨테이너와 독립적인 한 프레임 측정값으로 변환한다.
inline StageMetric make_stage_metric(
    const FrameMetadata& metadata,
    std::chrono::steady_clock::time_point enqueued_at,
    std::chrono::steady_clock::time_point queue_started_at,
    std::chrono::steady_clock::time_point processing_started_at,
    std::chrono::steady_clock::time_point finished_at,
    bool include_e2e = false)
{
    StageMetric metric;
    metric.frame_id = metadata.frame_id;
    metric.processing_ms =
        std::chrono::duration<double, std::milli>(
            finished_at - processing_started_at).count();
    if (enqueued_at != std::chrono::steady_clock::time_point{})
        metric.queue_wait_ms =
            std::chrono::duration<double, std::milli>(
                queue_started_at - enqueued_at).count();
    if (include_e2e && metadata.captured_at != std::chrono::steady_clock::time_point{})
        metric.e2e_ms =
            std::chrono::duration<double, std::milli>(finished_at - metadata.captured_at).count();
    return metric;
}

// 저장 sink가 없는 독립 worker 테스트에서 기존 메모리 기록 방식을 지원한다.
inline void record_stage_metric(
    std::vector<StageMetric>& metrics, const FrameMetadata& metadata,
    std::chrono::steady_clock::time_point enqueued_at,
    std::chrono::steady_clock::time_point queue_started_at,
    std::chrono::steady_clock::time_point processing_started_at,
    std::chrono::steady_clock::time_point finished_at, bool include_e2e = false)
{
    if (metrics.size() < kMaxStageMetrics)
        metrics.push_back(make_stage_metric(metadata, enqueued_at, queue_started_at,
            processing_started_at, finished_at, include_e2e));
}
