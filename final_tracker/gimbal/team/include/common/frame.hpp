#pragma once

#include <chrono>
#include <memory>

#include <opencv2/core.hpp>

#include "common/frame_metadata.hpp"

// Camera가 만든 원본 영상과 식별 정보.
// Detection 이후 GUI 송신까지 같은 객체를 공유한다.
struct FrameContext
{
    FrameMetadata metadata;
    cv::Mat image;   // 640x480, 16-bit single channel raw
};

// 파이프라인은 원본 프레임을 수정하지 않고 shared_ptr로 수명을 공유한다.
using FrameContextPtr = std::shared_ptr<const FrameContext>;

// Camera -> Detection queue에서 사용하는 메시지.
// enqueued_at은 프레임 자체가 아닌 각 queue 메시지에 속한다.
struct FrameMessage
{
    FrameContextPtr frame;
    std::chrono::steady_clock::time_point enqueued_at{};
};
