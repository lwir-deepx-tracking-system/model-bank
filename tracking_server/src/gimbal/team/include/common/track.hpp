#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "common/frame.hpp"


// Tracker가 생성한 객체의 Tracking 결과를 저장한다.
struct Track
{
    // Tracker가 객체에 부여한 고유 ID
    int track_id = -1;

    // Tracking된 객체의 Bounding Box
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;

    // 객체 Class
    int class_id = -1;

    // Tracking 결과의 신뢰도
    float confidence = 0.0f;

    // Lost-track extrapolations may be displayed but must not drive AUTO.
    bool predicted = false;
};

// Optional control input for the observed track in this frame, in image pixels.
struct TrackLead
{
    int track_id = 0;
    double dx_px = 0;
    double dy_px = 0;
    std::uint64_t control_epoch = 0; // The executor selection/session that produced this forecast.

    bool matches(int id) const
    {
        return id > 0 && track_id == id && std::isfinite(dx_px) && std::isfinite(dy_px) &&
            std::abs(dx_px) <= 60.0 && std::abs(dy_px) <= 60.0;
    }
};

// 한 프레임의 추적 목록과 GUI 표시에 필요한 원본 프레임을 함께 전달한다.
struct TrackingResult
{
    FrameContextPtr frame;
    std::vector<Track> tracks;
    std::chrono::steady_clock::time_point enqueued_at{};
    std::uint64_t model_generation = 0;
    std::uint64_t tracker_epoch = 0; // 로컬 추적 세대; PC 통신에는 포함하지 않는다.
    bool bypass = false;
    TrackLead lead;
    double aim_y = 0.5; // Control reference, measured from the top of the observed box.
};

// GUI 경로와 제어 경로가 같은 결과를 복사 없이 공유한다.
using TrackingResultPtr = std::shared_ptr<const TrackingResult>;
