#pragma once

#include "common/stage_metric.hpp"
#include "common/detection_timing.hpp"

enum class MetricStage { Camera, Detection, Tracking, Control };

// 처리 스레드는 측정값만 전달하고 파일·디렉터리 관리는 저장 구현에 맡긴다.
class IMeasurementSink {
public:
    virtual ~IMeasurementSink() = default;
    virtual void activate(std::uint64_t generation, int model_id) noexcept = 0;
    virtual void record(std::uint64_t generation, MetricStage stage, const StageMetric& metric) noexcept = 0;
    virtual void record_detection(std::uint64_t generation, const DetectionTiming& timing) noexcept = 0;
};
