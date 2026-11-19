#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

struct Box {
    double x1, y1, x2, y2, cx, cy, height, area;
};

bool parse_boxes(const double* xyxy, std::vector<Box>& boxes) {
    for (auto& box : boxes) {
        const double x1 = xyxy[0], y1 = xyxy[1], x2 = xyxy[2], y2 = xyxy[3];
        const double h = y2 - y1, w = x2 - x1;
        box = {x1, y1, x2, y2, (x1 + x2) / 2, (y1 + y2) / 2, h, w * h};
        if (!std::isfinite(box.cx) || !std::isfinite(box.cy) ||
            w < 0 || h < 0 || !std::isfinite(box.area)) return false;
        xyxy += 4;
    }
    return true;
}

double overlap(const Box& a, const Box& b) {
    const double w = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    const double h = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
    if (w <= 0 || h <= 0) return 0;
    const double inter = w * h;
    return inter / (a.area + b.area - inter);
}

bool nearby(const Box& a, const Box& b, double radius, std::uint8_t& fallback) {
    const double dx = a.cx - b.cx, dy = a.cy - b.cy;
    const double dist2 = dx * dx + dy * dy, radius2 = radius * radius;
    // Near the threshold Python's math.hypot must decide; different
    // rounding of the squared comparison cannot change a feature column.
    const double margin = 16 * std::numeric_limits<double>::epsilon() *
                          std::max({1.0, dist2, radius2});
    if (!std::isfinite(dist2) || !std::isfinite(radius2) ||
        std::abs(dist2 - radius2) <= margin) {
        fallback = 1;
        return false;
    }
    return dist2 < radius2;
}

struct PoseBox {
    const float* xyxy;
    float area;
};

int best_pose(const double* xywh, const std::vector<PoseBox>& boxes) {
    const double x1 = xywh[0] - xywh[2] / 2, y1 = xywh[1] - xywh[3] / 2;
    const double x2 = x1 + xywh[2], y2 = y1 + xywh[3];
    const double area = xywh[2] * xywh[3];
    double best = 0;
    int selected = -1;
    for (std::size_t j = 0; j < boxes.size(); ++j) {
        const auto& b = boxes[j];
        const double w = std::max(0.0, std::min(double(b.xyxy[2]), x2) - std::max(double(b.xyxy[0]), x1));
        const double h = std::max(0.0, std::min(double(b.xyxy[3]), y2) - std::max(double(b.xyxy[1]), y1));
        const double iou = w * h / std::max(double(b.area) + area - w * h, 1e-6);
        if (iou >= .5 && iou > best) {
            best = iou;
            selected = static_cast<int>(j);
        }
    }
    return selected;
}

void pose_features(const double* state, const double* innovation, const double* previous,
                   const double* z, const float* pose, bool posetrack, bool edges, double* out) {
    const double h = std::max(z[3], 1.0), w = std::max(z[2], 1.0);
    out[0] = state[4] / h;
    out[1] = state[5] / h;
    std::copy_n(innovation, 4, out + 2);
    out[6] = (z[0] - previous[0]) / h;
    out[7] = (z[1] - previous[1]) / h;
    // The caller applies NumPy log to these ratios in one batch, preserving the
    // exact floating-point inputs supplied to the frozen recurrent weights.
    out[8] = w / std::max(previous[2], 1.0);
    out[9] = h / std::max(previous[3], 1.0);
    out[10] = h / 100;
    out[11] = w / h;
    out[12] = edges ? (z[0] - w / 2) / h : 5;
    out[13] = edges ? (640 - (z[0] + w / 2)) / h : 5;
    out[14] = edges ? (z[1] - h / 2 - 16) / h : 5;
    out[15] = edges ? (496 - (z[1] + h / 2)) / h : 5;
    for (int j = 0; j < 17; ++j) {
        const double seen = pose && pose[j * 3 + 2] >= .5f &&
                            !(posetrack && (j == 1 || j == 2));
        out[16 + j * 3] = pose ? (pose[j * 3] - z[0]) / h * seen : 0;
        out[17 + j * 3] = pose ? (pose[j * 3 + 1] - z[1]) / h * seen : 0;
        out[18 + j * 3] = seen;
    }
    out[67] = 0;
}

}  // namespace

extern "C" int lwir_keepid_geometry(const double* track_boxes, const std::int64_t* track_ids,
                                    const std::int32_t* detection_indices, int track_count,
                                    const double* det_boxes, int det_count, double near_heights,
                                    double* geometry, std::uint8_t* fallback) {
    std::vector<Box> tracks(track_count), detections(det_count);
    if (!parse_boxes(track_boxes, tracks) || !parse_boxes(det_boxes, detections)) return 0;
    for (int i = 0; i < track_count; ++i, geometry += 4) {
        const int detection = detection_indices[i];
        if (detection >= det_count) return 0;
        const auto& track = tracks[i];
        const auto& ref = detection >= 0 ? detections[detection] : track;
        fallback[i] = 0;
        double max_iou = 0;
        int overlapping = 0, near_dets = 0, near_tracks = 0;
        for (int j = 0; j < det_count; ++j) {
            if (j == detection) continue;
            const auto& other = detections[j];
            const double iou = overlap(ref, other);
            if (!std::isfinite(iou)) return 0;
            max_iou = std::max(max_iou, iou);
            overlapping += iou > 0;
            near_dets += nearby(ref, other, near_heights * ref.height, fallback[i]);
        }
        for (int j = 0; j < track_count; ++j) {
            if (track_ids[i] == track_ids[j]) continue;
            near_tracks += nearby(track, tracks[j], near_heights * track.height, fallback[i]);
        }
        geometry[0] = max_iou;
        geometry[1] = overlapping;
        geometry[2] = near_dets;
        geometry[3] = near_tracks;
    }
    return 1;
}

extern "C" int lwir_position_seen(const double* states, const double* innovations,
                                  const double* previous, const std::uint8_t* has_previous,
                                  int count, const float* det_boxes, const float* poses,
                                  int det_count, int posetrack, int edges,
                                  double* observed, double* features) {
    std::vector<PoseBox> boxes;
    boxes.reserve(det_count);
    for (int i = 0; i < det_count; ++i) {
        const float* b = det_boxes + i * 4;
        const float area = (b[2] - b[0]) * (b[3] - b[1]);
        if (!std::all_of(b, b + 4, [](float x) { return std::isfinite(x); }) || !std::isfinite(area))
            return 0;
        boxes.push_back({b, area});
    }
    for (int i = 0; i < count; ++i) {
        const double* state = states + i * 6;
        const double* innovation = innovations + i * 4;
        double* z = observed + i * 4;
        if (!std::all_of(state, state + 6, [](double x) { return std::isfinite(x); }) ||
            !std::all_of(innovation, innovation + 4, [](double x) { return std::isfinite(x); }))
            return 0;
        const int j = best_pose(state, boxes);
        const float* pose = nullptr;
        if (j >= 0) {
            const float* b = boxes[j].xyxy;
            z[0] = (b[0] + b[2]) / 2;
            z[1] = (b[1] + b[3]) / 2;
            z[2] = b[2] - b[0];
            z[3] = b[3] - b[1];
            pose = poses + j * 51;
            if (!std::all_of(pose, pose + 51, [](float x) { return std::isfinite(x); })) return 0;
        } else {
            std::copy_n(state, 4, z);
        }
        const double* prior = has_previous[i] ? previous + i * 4 : z;
        if (!std::all_of(prior, prior + 4, [](double x) { return std::isfinite(x); })) return 0;
        pose_features(state, innovation, prior, z, pose, posetrack, edges, features + i * 68);
    }
    return 1;
}

extern "C" int lwir_nested_people(const float* boxes, const float* poses,
                                 int count, std::uint8_t* keep) {
    struct Face {
        int index;
        double x, y;
    };
    std::vector<Face> strong;
    std::vector<int> weak;
    std::fill_n(keep, count, 1);
    for (int i = 0; i < count; ++i) {
        const float* box = boxes + i * 5;
        const float* pose = poses + i * 51;
        if (!std::all_of(box, box + 5, [](float x) { return std::isfinite(x); }) ||
            !std::all_of(pose, pose + 21, [](float x) { return std::isfinite(x); })) continue;
        if (box[2] <= box[0] || box[3] <= box[1]) continue;
        int witnesses = 0, visible_face = 0;
        float sum_x = 0, sum_y = 0;
        for (int j = 0; j < 7; ++j) {
            const float* kpt = pose + j * 3;
            witnesses += kpt[2] >= .35f;
            if (j < 5 && kpt[2] >= .50f) {
                sum_x += kpt[0];
                sum_y += kpt[1];
                ++visible_face;
            }
        }
        if (witnesses <= 1) weak.push_back(i);
        if (!std::isfinite(sum_x) || !std::isfinite(sum_y)) return 0;
        if (visible_face >= 3 && (pose[17] >= .50f || pose[20] >= .50f))
            strong.push_back({i, double(sum_x) / visible_face, double(sum_y) / visible_face});
    }
    for (const int i : weak) {
        const float* large = boxes + i * 5;
        const float height = large[3] - large[1], upper = large[1] + .45f * height;
        const float center = (large[0] + large[2]) * .5f;
        bool nested = false;
        int witnessed = 0;
        for (const auto& face : strong) {
            if (face.x < large[0] || face.x > large[2] || face.y < large[1] || face.y > upper)
                continue;
            if (++witnessed > 1) break;
            const float* small = boxes + face.index * 5;
            const float w = small[2] - small[0], h = small[3] - small[1];
            const float ratio = height / h;
            const float iw = std::max(0.0f, std::min(small[2], large[2]) - std::max(small[0], large[0]));
            const float ih = std::max(0.0f, std::min(small[3], large[3]) - std::max(small[1], large[1]));
            nested = h >= 48 && ratio >= 1.30f && ratio <= 2.15f && iw * ih / (w * h) >= .74f &&
                     std::abs((small[0] + small[2]) * .5f - center) <= .16f * w &&
                     std::abs(small[1] - large[1]) <= .15f * h && small[4] >= large[4];
        }
        if (witnessed == 1 && nested) keep[i] = 0;
    }
    return 1;
}

extern "C" int lwir_merge_heads(const float* const* raw, const std::int32_t* spatial,
                                const std::int32_t* planes, int groups, float* const* merged) {
    const int total = spatial[0] + spatial[1] + spatial[2];
    for (int g = 0; g < groups; ++g) {
        int offset = 0;
        for (int s = 0; s < 3; ++s) {
            const float* head = raw[g * 3 + s];
            const int width = spatial[s];
            std::uint32_t largest = 0;
            for (int j = 0; j < planes[g] * width; ++j) {
                std::uint32_t bits;
                std::memcpy(&bits, head + j, sizeof(bits));
                largest = std::max(largest, bits & 0x7fffffffU);
            }
            if (largest >= 0x7f800000U) return 0;
            for (int c = 0; c < planes[g]; ++c)
                std::copy_n(head + c * width, width, merged[g] + c * total + offset);
            offset += width;
        }
    }
    return 1;
}

extern "C" int lwir_pose_anchors(const float* decoded, int anchors, const float* rows,
                                 int count, std::int64_t* chosen) {
    struct Corner {
        int index;
        float x, y, score;
    };
    float min_score = std::numeric_limits<float>::infinity();
    for (int i = 0; i < count; ++i) {
        if (!std::isfinite(rows[i * 6 + 4])) return 0;
        min_score = std::min(min_score, rows[i * 6 + 4]);
    }
    std::vector<Corner> candidate;
    for (int i = 0; i < anchors; ++i) {
        const float score = decoded[4 * anchors + i];
        if (score <= min_score - 1e-5f) continue;
        const float x = decoded[i] - decoded[2 * anchors + i] * .5f;
        const float y = decoded[anchors + i] - decoded[3 * anchors + i] * .5f;
        if (!std::isfinite(score) || !std::isfinite(x) || !std::isfinite(y)) return 0;
        candidate.push_back({i, x, y, score});
    }
    for (int i = 0; i < count; ++i) {
        const float* row = rows + i * 6;
        float best = std::numeric_limits<float>::infinity();
        int index = -1;
        for (const auto& other : candidate) {
            if (std::abs(other.score - row[4]) >= 1e-6f) continue;
            const float d = std::abs(other.x - row[0]) + std::abs(other.y - row[1]);
            if (d < best) {
                index = other.index;
                best = d;
                if (d == 0) break;
            }
        }
        if (index < 0) return 0;
        chosen[i] = index;
    }
    return 1;
}

extern "C" int lwir_body_deltas(const double* table, const double* heights, int nh,
                                const double* ratios, int nr, const double* inputs,
                                int count, double* out) {
    for (int i = 0; i < count; ++i) {
        const double h = inputs[i * 2], r = inputs[i * 2 + 1];
        if (!std::isfinite(h) || !std::isfinite(r)) return 0;
        const int hh = std::clamp(int(std::upper_bound(heights, heights + nh, h) - heights) - 1, 0, nh - 2);
        const int rr = std::clamp(int(std::upper_bound(ratios, ratios + nr, r) - ratios) - 1, 0, nr - 2);
        const double t = std::clamp((h - heights[hh]) / (heights[hh + 1] - heights[hh]), 0.0, 1.0);
        const double u = std::clamp((r - ratios[rr]) / (ratios[rr + 1] - ratios[rr]), 0.0, 1.0);
        const double* low = table + (hh * nr + rr) * 4;
        const double* high = low + nr * 4;
        for (int c = 0; c < 4; ++c)
            out[i * 4 + c] = (1 - t) * (1 - u) * low[c] + (1 - t) * u * low[4 + c]
                           + t * (1 - u) * high[c] + t * u * high[4 + c];
    }
    return 1;
}

extern "C" int lwir_motion_chain(const double* history, int capacity, int first, int count,
                                 double begin, double end, double* out) {
    double time = begin;
    for (int step = 0; step < count; ++step) {
        const double* entry = nullptr;
        for (int i = 0; i < count; ++i) {
            const double* row = history + ((first + i) % capacity) * 8;
            if (row[0] == time) {
                entry = row;
                break;
            }
        }
        if (!entry) return 0;
        const double* warp = entry + 2;
        if (!std::all_of(warp, warp + 6, [](double x) { return std::isfinite(x); })) return 0;
        if (step == 0) {
            std::copy_n(warp, 6, out);
        } else {
            const std::array<double, 6> old{out[0], out[1], out[2], out[3], out[4], out[5]};
            for (int j = 0; j < 6; j += 3) {
                out[j] = warp[j] * old[0] + warp[j + 1] * old[3];
                out[j + 1] = warp[j] * old[1] + warp[j + 1] * old[4];
                out[j + 2] = warp[j] * old[2] + warp[j + 1] * old[5] + warp[j + 2];
            }
        }
        time = entry[1];
        if (time == end) return 1;
        if (time > end) return 0;
    }
    return 0;
}
