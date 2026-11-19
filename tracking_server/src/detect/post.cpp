#include <array>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdint>
#include <cstring>

static std::array<float, 4> decode_box(const float* box, int n, int j, int width, int stride) {
    const float ax = float(j % width) + .5f;
    const float ay = float(j / width) + .5f;
    const float l = ax - box[j], t = ay - box[n + j];
    const float r = ax + box[2 * n + j], b = ay + box[3 * n + j];
    return {((l + r) * .5f) * stride, ((t + b) * .5f) * stride,
            (r - l) * stride, (b - t) * stride};
}

extern "C" void decode_raw(const float* box,const float* scores,float* out,int height,int width){
 const int N=(height/8)*(width/8)+(height/16)*(width/16)+(height/32)*(width/32);int offset=0;
 for(int stride: {8,16,32}){int w=width/stride,h=height/stride;
  for(int j=0;j<h*w;++j){int i=offset+j;float ax=float(j%w)+.5f,ay=float(j/w)+.5f;
   float l=ax-box[i],t=ay-box[N+i],r=ax+box[2*N+i],b=ay+box[3*N+i];
   out[i]=((l+r)*.5f)*stride;out[N+i]=((t+b)*.5f)*stride;
   out[2*N+i]=(r-l)*stride;out[3*N+i]=(b-t)*stride;
  }offset+=h*w;
 }
 // Sigmoid is performed by OpenCV SIMD in the caller.
}

namespace {

constexpr float score_threshold = .001f;
constexpr int max_detections = 300;
struct Box { float x1, y1, x2, y2, score; int cls, anchor; };

Box corners(float x, float y, float w, float h, float score, int cls, int anchor) {
    return {x - w * .5f, y - h * .5f, x + w * .5f, y + h * .5f, score, cls, anchor};
}

template<int class_count>
int suppress(std::vector<Box>& candidates, float* out, int* anchors = nullptr) {
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Box& a, const Box& b) { return a.score > b.score; });
    if (candidates.size() > 30000) {
        candidates.resize(30000);
    }

    std::array<std::vector<size_t>, class_count> selected;
    int result_count = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const Box& z = candidates[i];
        auto& same_class = selected[z.cls];
        bool overlaps = false;
        for (size_t j : same_class) {
            const Box& u = candidates[j];
            float iw = std::max(0.f, std::min(u.x2, z.x2) - std::max(u.x1, z.x1));
            float ih = std::max(0.f, std::min(u.y2, z.y2) - std::max(u.y1, z.y1));
            float area = (u.x2 - u.x1) * (u.y2 - u.y1);
            float inter = iw * ih, other = (z.x2 - z.x1) * (z.y2 - z.y1);
            float ratio = inter / (area + other - inter);
            if (ratio > .7f) {
                overlaps = true;
                break;
            }
        }
        if (overlaps) {
            continue;
        }
        same_class.push_back(i);
        if (anchors) anchors[result_count] = z.anchor;
        float* row = out + 6 * result_count++;
        row[0] = z.x1; row[1] = z.y1; row[2] = z.x2;
        row[3] = z.y2; row[4] = z.score; row[5] = float(z.cls);
        // Lower-scored candidates cannot change any emitted row.
        if (result_count == max_detections) {
            break;
        }
    }
    return result_count;
}

template<int class_count>
int nms(const float* in, float* out, int anchors) {
    std::vector<Box> candidates;
    candidates.reserve(4000);
    for (int i = 0; i < anchors; ++i) {
        for (int c = 0; c < class_count; ++c) {
            const float score = in[(c + 4) * anchors + i];
            if (score > score_threshold) {
                candidates.push_back(corners(in[i], in[anchors + i], in[2 * anchors + i],
                                             in[3 * anchors + i], score, c, i));
            }
        }
    }
    return suppress<class_count>(candidates, out);
}

// The split asset's dimensions and head layout are checked by both SDK wrappers.
constexpr int tile_height = 480;
constexpr int tile_width = 384;
constexpr int right_offset = 256;
constexpr int seam = 320;
constexpr int points = 17;
constexpr int pose_channels = 3 * points;
constexpr std::array<int, 3> strides{8, 16, 32};
constexpr int level_size(int stride) {
    return (tile_height / stride) * (tile_width / stride);
}
constexpr int tile_anchors = level_size(8) + level_size(16) + level_size(32);
constexpr float pose_score_tolerance = 1e-6f;

bool all_finite(const float* data, int size) {
    // A max reduction over IEEE float magnitudes vectorizes on the device.
    std::uint32_t largest = 0;
    for (int i = 0; i < size; ++i) {
        std::uint32_t bits;
        std::memcpy(&bits, data + i, sizeof(bits));
        largest = std::max(largest, bits & 0x7fffffffU);
    }
    return largest < 0x7f800000U;
}

struct PoseSource {
    const float* data;
    int cell, width, stride, offset;
};

PoseSource pose_source(const float* const* heads, int anchor) {
    const int tile = anchor / tile_anchors;
    int cell = anchor % tile_anchors;
    for (int level = 0; level < 3; ++level) {
        const int stride = strides[level];
        if (cell < level_size(stride)) {
            return {heads[tile * 9 + level * 3 + 2] + cell * pose_channels,
                    cell, tile_width / stride, stride, tile * right_offset};
        }
        cell -= level_size(stride);
    }
    return {};
}

}  // namespace

extern "C" int nms_decoded(const float* in, float* out, int anchors) {
    return nms<16>(in, out, anchors);
}

extern "C" int nms_person(const float* in, float* out, int anchors) {
    return nms<1>(in, out, anchors);
}

// Heads are [box, score, pose] per stride, left tile first. Return the index
// of an invalid head, or -1 on success. No box or pose planes are copied.
extern "C" int prepare_split_heads(const float* const* heads, float* negative_logits) {
    for (int tile = 0; tile < 2; ++tile) {
        int offset = tile * tile_anchors;
        for (int level = 0; level < 3; ++level) {
            const int head = tile * 9 + level * 3;
            const int n = level_size(strides[level]);
            if (!all_finite(heads[head], 4 * n)) return head;
            const float* score = heads[head + 1];
            bool finite = true;
            for (int j = 0; j < n; ++j) {
                finite &= std::isfinite(score[j]);
                negative_logits[offset + j] = -score[j];
            }
            if (!finite) return head + 1;
            if (!all_finite(heads[head + 2], pose_channels * n)) return head + 2;
            offset += n;
        }
    }
    return -1;
}

extern "C" int decode_split_heads(const float* const* heads, const float* exp_scores,
                                  float y_offset, float* rows, float* poses) {
    std::vector<Box> candidates, pose_candidates;
    int anchor = 0;
    for (int tile = 0; tile < 2; ++tile) {
        for (int level = 0; level < 3; ++level) {
            const auto stride = strides[level];
            const auto n = level_size(stride);
            const auto* box = heads[tile * 9 + level * 3];
            for (int j = 0; j < n; ++j, ++anchor) {
                const float score = 1.f / (1.f + exp_scores[anchor]);
                // Include just-below-threshold anchors in the legacy pose matching rule.
                if (score < score_threshold - 2 * pose_score_tolerance) continue;
                auto xywh = decode_box(box, n, j, tile_width / stride, stride);
                xywh[0] += tile * right_offset;
                if (tile == 0 ? xywh[0] >= seam : xywh[0] < seam) continue;
                const auto candidate = corners(xywh[0], xywh[1], xywh[2], xywh[3], score, 0, anchor);
                if (!std::isfinite(candidate.x1) || !std::isfinite(candidate.y1)) return -1;
                pose_candidates.push_back(candidate);
                if (score > score_threshold) candidates.push_back(candidate);
            }
        }
    }
    std::array<int, max_detections> anchors;
    const int n = suppress<1>(candidates, rows, anchors.data());
    for (int i = 0; i < n; ++i) {
        float* row = rows + i * 6;
        int anchor = anchors[i];
        // NMS's source has zero x1/y1 distance. Legacy pose lookup resolves
        // zero-distance ties to the first anchor within the score tolerance.
        for (const auto& candidate : pose_candidates) {
            if (candidate.anchor >= anchor) break;
            if (std::abs(candidate.score - row[4]) < pose_score_tolerance &&
                candidate.x1 == row[0] && candidate.y1 == row[1]) {
                anchor = candidate.anchor;
                break;
            }
        }
        const auto source = pose_source(heads, anchor);
        const float ax = float(source.cell % source.width) + .5f;
        const float ay = float(source.cell / source.width) + .5f;
        float* pose = poses + i * pose_channels;
        for (int p = 0; p < pose_channels; p += 3) {
            pose[p] = ((source.data[p] + ax) * source.stride) + source.offset;
            pose[p + 1] = ((source.data[p + 1] + ay) * source.stride) + y_offset;
            pose[p + 2] = 1.f / (1.f + std::exp(-source.data[p + 2]));
        }
        row[1] += y_offset;
        row[3] += y_offset;
    }
    return n;
}
