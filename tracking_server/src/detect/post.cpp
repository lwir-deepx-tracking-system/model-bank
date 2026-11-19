#include <array>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdint>
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

struct Box { float x1, y1, x2, y2, score; int cls; };

template<int class_count>
static int nms(const float* in, float* out, int anchors) {
    std::vector<Box> candidates;
    candidates.reserve(4000);
    for (int i = 0; i < anchors; ++i) {
        for (int c = 0; c < class_count; ++c) {
            float score = in[(c + 4) * anchors + i];
            if (score > .001f) {
                float x = in[i], y = in[anchors + i], w = in[2 * anchors + i], h = in[3 * anchors + i];
                candidates.push_back({x - w * .5f, y - h * .5f, x + w * .5f, y + h * .5f, score, c});
            }
        }
    }
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
        float* row = out + 6 * result_count++;
        row[0] = z.x1; row[1] = z.y1; row[2] = z.x2;
        row[3] = z.y2; row[4] = z.score; row[5] = float(z.cls);
        // Lower-scored candidates cannot change any emitted row.
        if (result_count == 300) {
            break;
        }
    }
    return result_count;
}

extern "C" int nms_decoded(const float* in, float* out, int anchors) {
    return nms<16>(in, out, anchors);
}

extern "C" int nms_person(const float* in, float* out, int anchors) {
    return nms<1>(in, out, anchors);
}
