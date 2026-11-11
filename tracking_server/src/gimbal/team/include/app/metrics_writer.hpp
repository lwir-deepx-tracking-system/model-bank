#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include "common/measurement_sink.hpp"
#include <map>
#include <vector>

// YAML 해석 없이 저장 경로와 설정 사본의 원본·상대 목적지만 전달한다.
struct MetricsWriterOptions
{
    std::filesystem::path                                                output_root;
    std::map<int, std::filesystem::path>                                 model_configs;
    std::vector<std::pair<std::filesystem::path, std::filesystem::path>> snapshots;
};

// 실행 디렉터리·설정 사본·모델별 CSV와 비동기 저장 스레드를 소유한다.
class MetricsWriter final : public IMeasurementSink
{
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    explicit MetricsWriter(const MetricsWriterOptions &options);
    ~MetricsWriter();
    // 새 모델 세대를 등록하고 이전 모델의 파일 버퍼를 닫도록 요청한다.
    void activate(std::uint64_t generation, int model_id) noexcept override;
    // 처리 스레드에서 단계별 측정값을 작은 큐 이벤트로 전달한다.
    void record(std::uint64_t generation, MetricStage stage, const StageMetric &metric) noexcept override;
    // 큰 profiler 구조체 대신 CSV에 필요한 숫자만 추려 전달한다.
    void record_detection(std::uint64_t generation, const DetectionTiming &timing) noexcept override;
    // 전환 완료를 보고하기 전에 큐 앞부분의 기록을 파일 버퍼에서 내보낸다.
    void flush();
    // 모든 생산자가 종료한 뒤 남은 큐를 기록하고 파일을 닫는다.
    void close() noexcept;
    // 큐를 폐기하고 이번 실행 결과만 삭제한다. worker의 늦은 제출은 무시한다.
    void                         discard();
    bool                         failed() const noexcept;
    std::string                  error_message() const;
    const std::filesystem::path &directory() const;
};
