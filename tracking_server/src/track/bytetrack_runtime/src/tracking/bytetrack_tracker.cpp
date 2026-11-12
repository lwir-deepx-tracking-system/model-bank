#include "tracking/bytetrack_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_set>
#include <utility>

#include "common/logger.hpp"

using bytetrack::STrack;
using bytetrack::STrackPtr;
using bytetrack::TrackState;

namespace
{

using Box = std::array<float, 4>;  // x, y, width, height

// Kalman 상태에서 현재 bbox를 계산한다.
Box track_box(const STrack& t)
{
    const auto& m = t.kf.mean;
    const float h = static_cast<float>(m[3]);
    const float w = static_cast<float>(t.kf.xywh ? m[2] : m[2] * m[3]);
    return {static_cast<float>(m[0]) - w / 2.0f,
            static_cast<float>(m[1]) - h / 2.0f, w, h};
}

// 검출 박스(tlwh) → Kalman 측정값. XYAH는 [cx, cy, w/h, h], XYWH는 [cx, cy, w, h]
KalmanMeasurement to_measurement(const Box& b, KalmanFormat format)
{
    return {b[0] + b[2] / 2.0, b[1] + b[3] / 2.0,
            format == KalmanFormat::XYWH ? static_cast<double>(b[2]) : static_cast<double>(b[2]) / b[3], b[3]};
}

float iou(const Box& a, const Box& b)
{
    const float x1 = std::max(a[0], b[0]);
    const float y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[0] + a[2], b[0] + b[2]);
    const float y2 = std::min(a[1] + a[3], b[1] + b[3]);
    const float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float uni = a[2] * a[3] + b[2] * b[3] - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

// cost[i][j] = 1 - IoU(tracks[i]의 예측 bbox, dets[j]의 검출 bbox)
std::vector<std::vector<float>> iou_distance(
    const std::vector<STrackPtr>& tracks,
    const std::vector<STrackPtr>& dets)
{
    std::vector<std::vector<float>> cost(
        tracks.size(), std::vector<float>(dets.size(), 1.0f));
    for (size_t i = 0; i < tracks.size(); ++i)
    {
        const Box tb = track_box(*tracks[i]);
        for (size_t j = 0; j < dets.size(); ++j)
            cost[i][j] = 1.0f - iou(tb, dets[j]->det_tlwh);
    }
    return cost;
}

// 박스의 각 변을 (폭, 높이) × scale만큼 넓힌다. max_px > 0이면 한 변의 확장량을 그 값으로 제한한다.
Box buffered(const Box& b, float scale, float max_px)
{
    float ex = b[2] * scale, ey = b[3] * scale;
    if (max_px > 0.0f)
    {
        ex = std::min(ex, max_px);
        ey = std::min(ey, max_px);
    }
    return {b[0] - ex, b[1] - ey, b[2] + 2.0f * ex, b[3] + 2.0f * ey};
}

// Buffered IoU distance: cost[i][j] = 1 - IoU(넓힌 Track 예측 박스, 넓힌 검출 박스).
// small_size > 0이면 Track 박스의 짧은 변이 그보다 큰 Track은 대상에서 빼고 cost 1(매칭 불가)로 둔다.
std::vector<std::vector<float>> buffered_iou_distance(
    const std::vector<STrackPtr>& tracks,
    const std::vector<STrackPtr>& dets,
    const TrackingConfig& c)
{
    std::vector<std::vector<float>> cost(
        tracks.size(), std::vector<float>(dets.size(), 1.0f));
    for (size_t i = 0; i < tracks.size(); ++i)
    {
        const Box tb = track_box(*tracks[i]);
        if (c.buffer_small_size > 0.0f && std::min(tb[2], tb[3]) > c.buffer_small_size)
            continue;
        const Box tbb = buffered(tb, c.buffer_scale, c.buffer_max_px);
        for (size_t j = 0; j < dets.size(); ++j)
            cost[i][j] = 1.0f - iou(tbb, buffered(dets[j]->det_tlwh, c.buffer_scale, c.buffer_max_px));
    }
    return cost;
}

void fuse_score(std::vector<std::vector<float>>& cost, const std::vector<STrackPtr>& dets)
{
    for (auto& row : cost)
        for (size_t j = 0; j < dets.size(); ++j)
            row[j] = 1.0f - (1.0f - row[j]) * dets[j]->score;
}

// Hungarian 알고리즘 (n <= m). 반환: 행 i에 배정된 열 번호
std::vector<int> hungarian(const std::vector<std::vector<double>>& a)
{
    const int n = static_cast<int>(a.size());
    const int m = static_cast<int>(a[0].size());
    const double inf = std::numeric_limits<double>::infinity();

    std::vector<double> u(n + 1, 0.0), v(m + 1, 0.0);
    std::vector<int> p(m + 1, 0), way(m + 1, 0);

    for (int i = 1; i <= n; ++i)
    {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(m + 1, inf);
        std::vector<char> used(m + 1, 0);
        do
        {
            used[j0] = 1;
            const int i0 = p[j0];
            double delta = inf;
            int j1 = 0;
            for (int j = 1; j <= m; ++j)
            {
                if (used[j])
                    continue;
                const double cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j])
                {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta)
                {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= m; ++j)
            {
                if (used[j])
                {
                    u[p[j]] += delta;
                    v[j] -= delta;
                }
                else
                {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do
        {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int> result(n, -1);
    for (int j = 1; j <= m; ++j)
        if (p[j] != 0)
            result[p[j] - 1] = j - 1;
    return result;
}

struct MatchResult
{
    std::vector<std::pair<size_t, size_t>> matches;
    std::vector<size_t> unmatched_rows;
    std::vector<size_t> unmatched_cols;
};

// 헝가리안은 assign_cost로 짝을 정하고, gate_cost가 thresh 이하인 쌍만 매칭으로 인정한다.
// (OCM처럼 순위용 비용과 허용 판단용 비용이 다를 때 쓴다. 둘이 같으면 linear_assignment와 같다)
MatchResult linear_assignment_gated(
    const std::vector<std::vector<float>>& assign_cost,
    const std::vector<std::vector<float>>& cost,
    size_t rows, size_t cols, float thresh)
{
    MatchResult result;
    std::vector<char> row_used(rows, 0), col_used(cols, 0);

    if (rows > 0 && cols > 0)
    {
        const bool transpose = rows > cols;
        const size_t n = transpose ? cols : rows;
        const size_t m = transpose ? rows : cols;
        constexpr double kInvalid = 1e5;

        std::vector<std::vector<double>> a(n, std::vector<double>(m));
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < m; ++j)
            {
                const float g = transpose ? cost[j][i] : cost[i][j];
                const float c = transpose ? assign_cost[j][i] : assign_cost[i][j];
                a[i][j] = g > thresh ? kInvalid : c;
            }

        const auto assign = hungarian(a);
        for (size_t i = 0; i < n; ++i)
        {
            if (assign[i] < 0)
                continue;
            const size_t j = static_cast<size_t>(assign[i]);
            const size_t r = transpose ? j : i;
            const size_t c = transpose ? i : j;
            if (cost[r][c] > thresh)
                continue;
            result.matches.emplace_back(r, c);
            row_used[r] = 1;
            col_used[c] = 1;
        }
    }

    for (size_t r = 0; r < rows; ++r)
        if (!row_used[r])
            result.unmatched_rows.push_back(r);
    for (size_t c = 0; c < cols; ++c)
        if (!col_used[c])
            result.unmatched_cols.push_back(c);
    return result;
}

// cost가 thresh 이하인 쌍만 매칭으로 인정한다.
MatchResult linear_assignment(
    const std::vector<std::vector<float>>& cost,
    size_t rows, size_t cols, float thresh)
{
    return linear_assignment_gated(cost, cost, rows, cols, thresh);
}

// ---- OC-SORT 보조 함수 ----

using Observation = STrack::Observation;

// 박스에 카메라 움직임을 적용한다: 중심은 회전·확대 후 평행이동, 크기는 확대 비율만큼.
Box warp_box(const Box& b, const CameraMotion& m)
{
    const double cx = b[0] + b[2] / 2.0, cy = b[1] + b[3] / 2.0;
    const double k = m.scale();
    const double nx = m.a * cx + m.b * cy + m.tx, ny = m.c * cx + m.d * cy + m.ty;
    const float w = static_cast<float>(b[2] * k), h = static_cast<float>(b[3] * k);
    return {static_cast<float>(nx) - w / 2.0f, static_cast<float>(ny) - h / 2.0f, w, h};
}

// 단위 방향 벡터 (from 중심 → to 중심). 거의 움직이지 않았으면 false
bool direction(const Box& from, const Box& to, double& ux, double& uy)
{
    const double dx = (to[0] + to[2] / 2.0) - (from[0] + from[2] / 2.0);
    const double dy = (to[1] + to[3] / 2.0) - (from[1] + from[3] / 2.0);
    const double n = std::sqrt(dx * dx + dy * dy);
    if (n < 1e-6)
        return false;
    ux = dx / n;
    uy = dy / n;
    return true;
}

// frame 이상인 관측 중 가장 이른 것 (before 미만으로 한정). 없으면 nullptr
const Observation* earliest_since(const std::deque<Observation>& obs, int frame, int before)
{
    for (const auto& o : obs)
        if (o.frame_id >= frame && o.frame_id < before)
            return &o;
    return nullptr;
}

// Track의 이동 방향: 마지막 관측 기준 delta_t 프레임 전 관측(없으면 그 사이 가장 이른 관측,
// 그것도 없으면 바로 앞 관측) → 마지막 관측. 관측이 하나뿐이면 false
bool track_direction(const STrack& t, int delta_t, double& ux, double& uy)
{
    const auto& obs = t.observations;
    if (obs.size() < 2)
        return false;
    const Observation& last = obs.back();
    const Observation* prev = earliest_since(obs, last.frame_id - delta_t, last.frame_id);
    if (!prev)
        prev = &obs[obs.size() - 2];
    return direction(prev->tlwh, last.tlwh, ux, uy);
}

// OCM 방향 일치도: (π/2 - 두 방향 사이 각) / π  → 같은 방향 +0.5, 수직 0, 반대 -0.5
double direction_consistency(double ux, double uy, double vx, double vy)
{
    constexpr double kPi = 3.14159265358979323846;
    const double cos_angle = std::clamp(ux * vx + uy * vy, -1.0, 1.0);
    return (kPi / 2.0 - std::acos(cos_angle)) / kPi;
}

// 두 목록을 track_id 기준으로 합친다.
std::vector<STrackPtr> joint(
    const std::vector<STrackPtr>& a, const std::vector<STrackPtr>& b)
{
    std::vector<STrackPtr> out = a;
    std::unordered_set<int> ids;
    for (const auto& t : a)
        ids.insert(t->track_id);
    for (const auto& t : b)
        if (ids.insert(t->track_id).second)
            out.push_back(t);
    return out;
}

// a에서 b에 있는 track_id를 뺀다.
std::vector<STrackPtr> sub(
    const std::vector<STrackPtr>& a, const std::vector<STrackPtr>& b)
{
    std::unordered_set<int> ids;
    for (const auto& t : b)
        ids.insert(t->track_id);
    std::vector<STrackPtr> out;
    for (const auto& t : a)
        if (!ids.count(t->track_id))
            out.push_back(t);
    return out;
}

// tracked와 lost에 거의 같은 위치의 Track이 있으면 짧게 유지된 쪽을 지운다.
void remove_duplicates(
    std::vector<STrackPtr>& tracked, std::vector<STrackPtr>& lost)
{
    std::vector<char> drop_a(tracked.size(), 0), drop_b(lost.size(), 0);
    for (size_t i = 0; i < tracked.size(); ++i)
        for (size_t j = 0; j < lost.size(); ++j)
        {
            if (1.0f - iou(track_box(*tracked[i]), track_box(*lost[j])) >= 0.15f)
                continue;
            const int age_a = tracked[i]->frame_id - tracked[i]->start_frame;
            const int age_b = lost[j]->frame_id - lost[j]->start_frame;
            if (age_a > age_b)
                drop_b[j] = 1;
            else
                drop_a[i] = 1;
        }

    std::vector<STrackPtr> keep_a, keep_b;
    for (size_t i = 0; i < tracked.size(); ++i)
        if (!drop_a[i])
            keep_a.push_back(tracked[i]);
    for (size_t j = 0; j < lost.size(); ++j)
        if (!drop_b[j])
            keep_b.push_back(lost[j]);
    tracked.swap(keep_a);
    lost.swap(keep_b);
}

}  // namespace


// NSA Kalman (StrongSORT): 측정 노이즈 공분산 R을 (1 - 검출 점수)배 한다.
// 점수가 높을수록 검출을 더 믿고, 낮은(가려진) 검출은 Track 상태를 덜 흔든다.
// update()는 표준편차에 곱하므로 sqrt를 넘긴다. 점수 1에서 R이 0이 되지 않게 (1 - 점수)를 kNsaMinRatio 이상으로 둔다.
double ByteTrackTracker::measurement_noise_scale(float score) const
{
    if (!config_.nsa_kalman)
        return 1.0;
    constexpr double kNsaMinRatio = 0.05;
    const double ratio = std::max(1.0 - static_cast<double>(score), kNsaMinRatio);
    return std::sqrt(std::min(ratio, 1.0));
}


ByteTrackTracker::ByteTrackTracker(const TrackingConfig& config)
    : config_(config),
      new_track_threshold_(config.new_track_threshold > 0.0F ?
                               config.new_track_threshold :
                               std::min(config.track_threshold + 0.1F, 1.0F)),
      kalman_(config.kalman_format == "xywh" ? KalmanFormat::XYWH : KalmanFormat::XYAH)
{
    max_time_lost_ = config_.track_buffer;

    Logger::info(
        "[Tracking] ByteTrackTracker 생성 완료"
    );

    std::ostringstream values;
    values << "track_threshold=" << config_.track_threshold
           << ", low_threshold=" << config_.low_threshold
           << ", new_track_threshold=" << new_track_threshold_
           << ", match_threshold=" << config_.match_threshold
           << ", low_match_threshold=" << config_.low_match_threshold
           << ", unconfirmed_match_threshold=" << config_.unconfirmed_match_threshold
           << ", track_buffer=" << config_.track_buffer
           << " (Lost 유지 " << max_time_lost_ << "프레임)";
    if (config_.fuse_score)
        values << ", fuse_score=true";
    if (config_.nsa_kalman)
        values << ", nsa_kalman=true";
    if (config_.kalman_format != "xyah")
        values << ", kalman_format=" << config_.kalman_format;
    if (config_.lost_output_frames > 0)
        values << ", lost_output_frames=" << config_.lost_output_frames;
    if (config_.buffer_scale > 0.0F)
        values << ", buffer_scale=" << config_.buffer_scale
               << ", buffer_max_px=" << config_.buffer_max_px
               << ", buffer_small_size=" << config_.buffer_small_size
               << ", buffer_snap=" << (config_.buffer_snap ? "true" : "false");
    if (config_.oru)
        values << ", oru=true";
    if (config_.ocm_weight > 0.0F)
        values << ", ocm_weight=" << config_.ocm_weight << ", ocm_delta_t=" << config_.ocm_delta_t;
    if (config_.ocr)
        values << ", ocr=true";
    Logger::debug(
        "[Tracking] ByteTrack 설정 값: " + values.str()
    );
}


// 새 Track에 ID를 부여하고 Kalman 상태를 만든다.
void ByteTrackTracker::activate(STrack& t)
{
    t.track_id = ++next_track_id_;
    t.kf = kalman_.initiate(to_measurement(t.det_tlwh, kalman_.format()));
    t.state = TrackState::Tracked;
    t.is_activated = (frame_id_ == 1);  // 첫 프레임이 아니면 다음 프레임에 확정
    t.tracklet_len = 0;
    t.frame_id = frame_id_;
    t.start_frame = frame_id_;
    if (observation_centric())
        record_observation(t, t.det_tlwh);
}


void ByteTrackTracker::record_observation(STrack& t, const std::array<float, 4>& tlwh) const
{
    t.observations.push_back({frame_id_, tlwh});
    // OCM 방향 계산에 필요한 범위(마지막 관측 기준 ocm_delta_t 프레임)만 남긴다. 바로 앞 관측 하나는 항상 둔다.
    const int oldest = frame_id_ - config_.ocm_delta_t;
    while (t.observations.size() > 2 && t.observations.front().frame_id < oldest)
        t.observations.pop_front();
}


// ORU (Observation-centric Re-Update)
// 저장한 상태 = 마지막 관측 프레임(t.frame_id)의 다음 프레임까지 예측한 상태(update 전)다.
// 놓친 프레임마다 마지막 관측 → 새 검출을 직선 보간한 가상 관측으로 update하고 한 프레임씩 예측해,
// 현재 프레임의 예측 상태까지 다시 만든다. 이어서 apply_detection이 실제 검출로 update한다.
// 관측과 저장 상태는 매 프레임 CMC를 적용해 두었으므로 모두 현재 화면 좌표에서 계산한다.
void ByteTrackTracker::re_update(STrack& t, const STrack& det) const
{
    const int gap = frame_id_ - t.frame_id;  // Lost였으므로 2 이상
    if (gap < 2 || t.observations.empty())
        return;
    const Box from = t.observations.back().tlwh;
    const Box& to = det.det_tlwh;

    KalmanState s = t.lost_kf;
    for (int k = 1; k < gap; ++k)
    {
        if (k > 1)
            kalman_.predict(s);
        const float r = static_cast<float>(k) / static_cast<float>(gap);
        const Box v = {from[0] + (to[0] - from[0]) * r, from[1] + (to[1] - from[1]) * r,
                       from[2] + (to[2] - from[2]) * r, from[3] + (to[3] - from[3]) * r};
        kalman_.update(s, to_measurement(v, kalman_.format()));
    }
    kalman_.predict(s);
    t.kf = s;
}


// 매칭된 검출로 Track을 갱신한다. Lost였던 Track도 같은 ID로 되살아난다.
void ByteTrackTracker::apply_detection(STrack& t, const STrack& det)
{
    t.tracklet_len = (t.state == TrackState::Tracked) ? t.tracklet_len + 1 : 0;
    if (config_.oru && t.state == TrackState::Lost && t.has_lost_kf)
        re_update(t, det);
    t.has_lost_kf = false;
    kalman_.update(t.kf, to_measurement(det.det_tlwh, kalman_.format()), measurement_noise_scale(det.score));
    t.state = TrackState::Tracked;
    t.is_activated = true;
    t.score = det.score;
    t.class_id = det.class_id;
    t.source_index = det.source_index; // LWIR I/O: store the matched detection row
    t.frame_id = frame_id_;
    t.det_tlwh = det.det_tlwh;  // 마지막으로 매칭된 검출 박스 (화면 밖 판정에 사용)
    if (observation_centric())
        record_observation(t, det.det_tlwh);
}


// Detection 목록을 이용해 ByteTrack Tracking을 수행한다.
std::vector<Track> ByteTrackTracker::track(
    const std::vector<Detection>& detections)
{
    ++frame_id_;

    // 1. Detection Confidence 기준 분류
    std::vector<STrackPtr> high_dets, low_dets;
    for (const auto& d : detections)
    {
        if (d.width <= 0.0f || d.height <= 0.0f)
            continue;

        auto det = std::make_shared<STrack>();
        det->det_tlwh = {d.x, d.y, d.width, d.height};
        det->score = d.confidence;
        det->class_id = d.class_id;
        det->source_index = d.source_index; // LWIR I/O: copy the input detection row

        if (d.confidence >= config_.track_threshold)
            high_dets.push_back(std::move(det));
        else if (d.confidence > config_.low_threshold)
            low_dets.push_back(std::move(det));
    }

    std::vector<STrackPtr> unconfirmed, confirmed;
    for (const auto& t : tracked_stracks_)
        (t->is_activated ? confirmed : unconfirmed).push_back(t);

    // 2. 기존 Track Prediction (Lost Track 포함)
    std::vector<STrackPtr> pool = joint(confirmed, lost_stracks_);
    for (auto& t : pool)
    {
        if (t->state != TrackState::Tracked)
        {
            t->kf.mean[7] = 0.0;  // 놓친 Track은 높이 변화 속도를 멈춘다
            if (t->kf.xywh)
                t->kf.mean[6] = 0.0;  // XYWH는 폭 변화 속도도 멈춘다 (BoT-SORT와 같음)
        }
        kalman_.predict(t->kf);
    }

    // 2-1. 카메라 움직임 보정 (CMC): 예측 위치를 카메라가 움직인 만큼 옮긴다.
    //      미확정 Track은 예측하지 않지만 위치는 같이 옮겨야 다음 검출과 겹친다.
    if (!camera_motion_.is_identity())
    {
        for (auto& t : pool)
            apply_camera_motion(t->kf, camera_motion_);
        for (auto& t : unconfirmed)
            apply_camera_motion(t->kf, camera_motion_);
        // OC-SORT: 과거 관측과 ORU 저장 상태도 현재 화면 좌표로 옮긴다
        if (observation_centric())
            for (const auto* list : {&pool, &unconfirmed})
                for (auto& t : *list)
                {
                    for (auto& o : t->observations)
                        o.tlwh = warp_box(o.tlwh, camera_motion_);
                    if (t->has_lost_kf)
                        apply_camera_motion(t->lost_kf, camera_motion_);
                }
    }
    camera_motion_ = CameraMotion{};

    std::vector<STrackPtr> activated, refind, lost_new, removed;
    auto on_match = [&](const STrackPtr& t, const STrackPtr& det)
    {
        const bool was_tracked = (t->state == TrackState::Tracked);
        apply_detection(*t, *det);
        (was_tracked ? activated : refind).push_back(t);
    };

    // 3-1. 1차 Association: 모든 Track ↔ high detection
    auto cost1 = iou_distance(pool, high_dets);
    if (config_.fuse_score)
        fuse_score(cost1, high_dets);
    // OCM: 순위용 비용에서 방향 일치도를 뺀다. 허용 판단은 cost1(IoU 기반)로만 한다.
    //   Track 방향  : delta_t 프레임 전 관측 → 마지막 관측
    //   후보 방향   : 지금 기준 delta_t 프레임 안의 가장 이른 관측(없으면 마지막 관측) → 검출
    std::vector<std::vector<float>> assign1 = cost1;
    if (config_.ocm_weight > 0.0f)
        for (size_t i = 0; i < pool.size(); ++i)
        {
            const STrack& t = *pool[i];
            double ux = 0.0, uy = 0.0;
            if (!track_direction(t, config_.ocm_delta_t, ux, uy))
                continue;
            const Observation* origin = earliest_since(t.observations, frame_id_ - config_.ocm_delta_t, frame_id_);
            const Box from = origin ? origin->tlwh : t.observations.back().tlwh;
            for (size_t j = 0; j < high_dets.size(); ++j)
            {
                if (cost1[i][j] > config_.match_threshold)
                    continue;  // IoU로 이미 탈락한 쌍은 순위를 계산할 필요가 없다
                double vx = 0.0, vy = 0.0;
                if (!direction(from, high_dets[j]->det_tlwh, vx, vy))
                    continue;
                assign1[i][j] -= static_cast<float>(config_.ocm_weight * high_dets[j]->score *
                                                    direction_consistency(ux, uy, vx, vy));
            }
        }
    const auto m1 = linear_assignment_gated(
        assign1, cost1, pool.size(), high_dets.size(), config_.match_threshold);
    for (const auto& [r, c] : m1.matches)
        on_match(pool[r], high_dets[c]);

    std::vector<size_t> left_pool = m1.unmatched_rows;
    std::vector<size_t> left_high = m1.unmatched_cols;

    // 3-1b. Buffered IoU (켰을 때만): 1차에서 남은 Track ↔ 남은 high detection을 넓힌 박스로 다시 매칭.
    //       원래 IoU로 맞는 쌍은 이미 1차에서 정해졌으므로, 빠르거나 작아서 겹치지 않던 쌍만 구제한다.
    if (config_.buffer_scale > 0.0f && !left_pool.empty() && !left_high.empty())
    {
        std::vector<STrackPtr> bt, bd;
        for (size_t r : left_pool)
            bt.push_back(pool[r]);
        for (size_t c : left_high)
            bd.push_back(high_dets[c]);
        auto costb = buffered_iou_distance(bt, bd, config_);
        if (config_.fuse_score)
            fuse_score(costb, bd);
        const auto mb = linear_assignment(costb, bt.size(), bd.size(), config_.match_threshold);
        for (const auto& [r, c] : mb.matches)
        {
            const bool re_updated = config_.oru && bt[r]->state == TrackState::Lost && bt[r]->has_lost_kf;
            on_match(bt[r], bd[c]);
            if (re_updated)
                continue;  // ORU가 놓친 구간을 다시 계산했으므로 snap하지 않는다
            // 넓힌 박스로만 맞은 쌍은 예측이 크게 빗나간 경우다. Kalman 업데이트는 틀린 예측과 검출의
            // 중간 위치를 내므로, buffer_snap이면 위치(cx, cy, a, h)를 검출 박스로 맞춘다.
            // (프레임 누락에는 유리, 검출이 흔들리는 작은 표적에는 평활이 사라져 불리)
            if (config_.buffer_snap)
            {
                const KalmanMeasurement z = to_measurement(bd[c]->det_tlwh, kalman_.format());
                for (int k = 0; k < 4; ++k)
                    bt[r]->kf.mean[k] = z[k];
            }
        }

        std::vector<size_t> next_pool, next_high;
        for (size_t r : mb.unmatched_rows)
            next_pool.push_back(left_pool[r]);
        for (size_t c : mb.unmatched_cols)
            next_high.push_back(left_high[c]);
        left_pool = std::move(next_pool);
        left_high = std::move(next_high);
    }

    // 3-2. 2차 Association: 남은 Tracked Track ↔ low detection (ByteTrack 핵심)
    std::vector<STrackPtr> remain_tracked;
    for (size_t r : left_pool)
        if (pool[r]->state == TrackState::Tracked)
            remain_tracked.push_back(pool[r]);

    const auto m2 = linear_assignment(
        iou_distance(remain_tracked, low_dets), remain_tracked.size(),
        low_dets.size(), config_.low_match_threshold);
    for (const auto& [r, c] : m2.matches)
        on_match(remain_tracked[r], low_dets[c]);

    // 3-2b. OCR (켰을 때만): 아직 짝이 없는 Track(이번에 놓칠 Track + 이미 Lost인 Track)을
    //       Kalman 예측 대신 마지막 관측 박스로 남은 high 검출과 다시 매칭한다.
    std::unordered_set<const STrack*> recovered;
    if (config_.ocr && !left_high.empty())
    {
        std::vector<STrackPtr> ot, od;
        for (size_t r : m2.unmatched_rows)
            ot.push_back(remain_tracked[r]);
        for (size_t r : left_pool)
            if (pool[r]->state == TrackState::Lost)
                ot.push_back(pool[r]);
        for (size_t c : left_high)
            od.push_back(high_dets[c]);

        std::vector<std::vector<float>> costr(ot.size(), std::vector<float>(od.size(), 1.0f));
        for (size_t i = 0; i < ot.size(); ++i)
        {
            if (ot[i]->observations.empty())
                continue;
            const Box last = ot[i]->observations.back().tlwh;
            for (size_t j = 0; j < od.size(); ++j)
                costr[i][j] = 1.0f - iou(last, od[j]->det_tlwh);
        }
        const auto mr = linear_assignment(costr, ot.size(), od.size(), config_.match_threshold);
        std::unordered_set<const STrack*> used_dets;
        for (const auto& [r, c] : mr.matches)
        {
            on_match(ot[r], od[c]);
            recovered.insert(ot[r].get());
            used_dets.insert(od[c].get());
        }
        std::vector<size_t> next_high;
        for (size_t c : left_high)
            if (!used_dets.count(high_dets[c].get()))
                next_high.push_back(c);
        left_high = std::move(next_high);
    }

    for (size_t r : m2.unmatched_rows)
    {
        auto& t = remain_tracked[r];
        if (recovered.count(t.get()))
            continue;
        if (t->state != TrackState::Lost)
        {
            t->state = TrackState::Lost;
            lost_new.push_back(t);
            if (config_.oru)
            {
                t->lost_kf = t->kf;  // 놓친 첫 프레임의 예측 상태 (ORU에서 여기부터 다시 계산)
                t->has_lost_kf = true;
            }
        }
    }

    // 3-3. 미확정 Track(직전 프레임에 생긴 Track) ↔ 남은 high detection
    std::vector<STrackPtr> remain_high;
    for (size_t c : left_high)
        remain_high.push_back(high_dets[c]);

    auto cost3 = iou_distance(unconfirmed, remain_high);
    if (config_.fuse_score)
        fuse_score(cost3, remain_high);
    const auto m3 = linear_assignment(
        cost3, unconfirmed.size(), remain_high.size(), config_.unconfirmed_match_threshold);
    for (const auto& [r, c] : m3.matches)
    {
        apply_detection(*unconfirmed[r], *remain_high[c]);
        activated.push_back(unconfirmed[r]);
    }
    for (size_t r : m3.unmatched_rows)
    {
        unconfirmed[r]->state = TrackState::Removed;
        removed.push_back(unconfirmed[r]);
    }

    // 4. Track 상태 및 ID 갱신
    // 4-1. 매칭되지 않은 high detection으로 새 Track 생성
    for (size_t c : m3.unmatched_cols)
    {
        auto& det = remain_high[c];
        if (det->score < new_track_threshold_)
            continue;
        activate(*det);
        activated.push_back(det);
    }

    // 4-2. 오래 못 찾은 Lost Track 삭제
    for (auto& t : lost_stracks_)
    {
        if (frame_id_ - t->frame_id > max_time_lost_)
        {
            t->state = TrackState::Removed;
            removed.push_back(t);
        }
    }

    // 4-3. 목록 정리
    std::vector<STrackPtr> new_tracked;
    for (const auto& t : tracked_stracks_)
        if (t->state == TrackState::Tracked)
            new_tracked.push_back(t);
    new_tracked = joint(new_tracked, activated);
    new_tracked = joint(new_tracked, refind);

    lost_stracks_ = sub(lost_stracks_, new_tracked);
    lost_stracks_.insert(lost_stracks_.end(), lost_new.begin(), lost_new.end());
    lost_stracks_ = sub(lost_stracks_, removed);

    remove_duplicates(new_tracked, lost_stracks_);
    tracked_stracks_ = std::move(new_tracked);

    // 5. Track 목록 생성 (확정된 Track만 출력)
    std::vector<Track> tracks;
    for (const auto& t : tracked_stracks_)
    {
        if (!t->is_activated)
            continue;
        const Box b = track_box(*t);
        Track out;
        out.track_id = t->track_id;
        out.x = b[0];
        out.y = b[1];
        out.width = b[2];
        out.height = b[3];
        out.class_id = t->class_id;
        out.confidence = t->score;
        out.source_index = t->frame_id == frame_id_ ? t->source_index : -1; // LWIR I/O: return the row only for a current-frame match
        tracks.push_back(out);
    }
    for (const auto& t : lost_stracks_)
    {
        if (!t->is_activated || frame_id_ - t->frame_id > config_.lost_output_frames)
            continue;
        const Box b = track_box(*t);
        // 화면 밖으로 나가는 중인 Track은 예측 박스를 내지 않는다 (사라진 물체 자리에 남는 헛출력 방지).
        // 기억(Lost 유지)은 그대로 하므로 다시 들어오면 같은 ID로 이어진다.
        //   1) 예측 박스가 화면 안에 절반도 안 보이면 제외
        //   2) 마지막 검출 박스가 화면 경계에 닿아 있고, 예측 위치가 그 경계 바깥쪽으로 움직였으면 제외
        //      (검출 박스는 경계에서 잘려 작아지므로 예측 박스가 화면 안에 남아 있어도 실제로는 나가는 중)
        if (image_width_ > 0.0f && image_height_ > 0.0f)
        {
            const float vis_w = std::min(b[0] + b[2], image_width_) - std::max(b[0], 0.0f);
            const float vis_h = std::min(b[1] + b[3], image_height_) - std::max(b[1], 0.0f);
            const float area = b[2] * b[3];
            if (vis_w <= 0.0f || vis_h <= 0.0f || area <= 0.0f || vis_w * vis_h < 0.5f * area)
                continue;

            constexpr float kEdge = 2.0f;  // 경계에 닿았다고 보는 거리 (px)
            const auto& d = t->det_tlwh;
            // 마지막 검출 위치 → 지금 예측 위치로 움직인 방향 (물체 자체 속도 + 카메라 움직임 보정 포함)
            const float dx = (b[0] + b[2] / 2) - (d[0] + d[2] / 2);
            const float dy = (b[1] + b[3] / 2) - (d[1] + d[3] / 2);
            const bool exiting = (d[0] <= kEdge && dx < 0.0f) ||
                                 (d[0] + d[2] >= image_width_ - kEdge && dx > 0.0f) ||
                                 (d[1] <= kEdge && dy < 0.0f) ||
                                 (d[1] + d[3] >= image_height_ - kEdge && dy > 0.0f);
            if (exiting)
                continue;
        }
        Track out;
        out.track_id = t->track_id;
        out.x = b[0];
        out.y = b[1];
        out.width = b[2];
        out.height = b[3];
        out.class_id = t->class_id;
        out.confidence = t->score;
        out.predicted = true;
        tracks.push_back(out);
    }
    return tracks;
}


// ByteTrack이 유지하는 내부 Tracking 상태를 초기화한다.
void ByteTrackTracker::reset()
{
    tracked_stracks_.clear();
    lost_stracks_.clear();
    frame_id_ = 0;
    camera_motion_ = CameraMotion{};
    next_track_id_ = 0;

    Logger::info(
        "[Tracking] ByteTrackTracker 상태 초기화 완료"
    );
}
