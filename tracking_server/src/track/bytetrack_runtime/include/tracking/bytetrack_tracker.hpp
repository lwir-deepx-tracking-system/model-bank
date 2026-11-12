#pragma once

#include <array>
#include <deque>
#include <memory>
#include <vector>

#include "config/app_config.hpp"
#include "tracking/camera_motion.hpp"
#include "tracking/kalman_filter.hpp"
#include "tracking/tracker.hpp"

namespace bytetrack
{
enum class TrackState
{
    New,
    Tracked,
    Lost,
    Removed
};

// Tracker 내부에서 관리하는 Track 하나의 상태
struct STrack
{
    int track_id = 0;
    TrackState state = TrackState::New;
    bool is_activated = false;

    KalmanState kf{};
    std::array<float, 4> det_tlwh{};  // 검출 bbox (x, y, width, height)

    float score = 0.0f;
    int class_id = -1;
    int source_index = -1; // LWIR I/O: detection row stored on the track

    int frame_id = 0;       // 마지막으로 검출과 매칭된 프레임
    int start_frame = 0;
    int tracklet_len = 0;

    // ---- OC-SORT용 (oru / ocm_weight / ocr 중 하나라도 켜면 기록) ----
    // 매칭된 실제 검출 박스 기록. 카메라 움직임(CMC)을 적용해 항상 현재 화면 좌표로 유지한다.
    // (det_tlwh는 화면 밖 판정용으로 검출 당시 좌표 그대로 둔다)
    struct Observation
    {
        int frame_id = 0;
        std::array<float, 4> tlwh{};  // x, y, width, height
    };
    std::deque<Observation> observations;  // 마지막 관측과 그 이전 ocm_delta_t 프레임 안의 관측만 남긴다

    // ORU: Tracked → Lost로 바뀐 순간의 Kalman 상태 (놓친 첫 프레임의 예측·CMC까지 적용, update 전)
    KalmanState lost_kf{};
    bool has_lost_kf = false;
};

using STrackPtr = std::shared_ptr<STrack>;
}  // namespace bytetrack


// ByteTrack 알고리즘을 이용해 Detection을 Track으로 연결한다.
// Detection의 (x, y)는 bbox 좌상단, (width, height)는 크기로 가정한다.
class ByteTrackTracker : public Tracker
{
private:
    TrackingConfig config_;
    float new_track_threshold_ = 0.6F;
    KalmanFilter kalman_;

    int frame_id_ = 0;
    int next_track_id_ = 0;
    int max_time_lost_ = 30;

    CameraMotion camera_motion_;   // 다음 track() 한 번에만 적용할 카메라 움직임
    float image_width_ = 0.0f;     // 영상 크기. 0이면 모름 (화면 밖 판정을 하지 않음)
    float image_height_ = 0.0f;

    std::vector<bytetrack::STrackPtr> tracked_stracks_;
    std::vector<bytetrack::STrackPtr> lost_stracks_;

    void activate(bytetrack::STrack& track);
    void apply_detection(bytetrack::STrack& track, const bytetrack::STrack& det);
    // OC-SORT 기능 중 하나라도 켜져 있는지 (관측 기록 여부)
    bool observation_centric() const { return config_.oru || config_.ocm_weight > 0.0F || config_.ocr; }
    // 관측 기록에 박스를 추가하고 오래된 기록을 정리한다.
    void record_observation(bytetrack::STrack& track, const std::array<float, 4>& tlwh) const;
    // ORU: 저장한 Lost 직전 상태에서 가상 관측으로 놓친 구간을 다시 계산한다 (새 검출 update 직전 상태까지).
    void re_update(bytetrack::STrack& track, const bytetrack::STrack& det) const;
    // Kalman update의 측정 노이즈 배율 (nsa_kalman이 꺼져 있으면 1)
    double measurement_noise_scale(float score) const;

public:
    // AppConfig가 검증한 runtime tracking 설정을 적용한다.
    explicit ByteTrackTracker(const TrackingConfig& config);

    ~ByteTrackTracker() override = default;

    // Detection 목록을 이용해 ByteTrack Tracking을 수행한다.
    std::vector<Track> track(
        const std::vector<Detection>& detections
    ) override;

    // ByteTrack 내부 상태를 초기화한다.
    void reset() override;

    // 다음 track() 호출에 적용할 카메라 움직임(직전 프레임 → 현재 프레임)을 지정한다.
    // track()이 끝나면 지워지므로 프레임마다 다시 지정한다. 호출하지 않으면 보정하지 않는다.
    void set_camera_motion(const CameraMotion& motion) { camera_motion_ = motion; }

    // 영상 크기를 알려 준다. 지정하면 놓친 Track의 예측 박스가 화면 밖으로 나갔을 때
    // (보이는 넓이가 절반 미만) lost_output_frames 출력에서 제외한다. 지정하지 않으면 제외하지 않는다.
    void set_image_size(float width, float height)
    {
        image_width_ = width;
        image_height_ = height;
    }

    const TrackingConfig& config() const { return config_; }
};
