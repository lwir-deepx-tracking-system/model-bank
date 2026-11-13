// Multi-scale deformable attention, forward only, on the CPU (float32).
// Same arithmetic as the official CUDA kernel ms_deform_attn_im2col_bilinear (MOTIP models/ops/src/cuda) and as the
// pure PyTorch reference ms_deform_attn_core_pytorch (grid_sample bilinear, zero padding, align_corners=False):
//   x = loc_x * W - 0.5, y = loc_y * H - 0.5; the four neighbours outside the map contribute 0.
// value [N, S, M, D], spatial_shapes [L, 2] (H, W), level_start_index [L], sampling_locations [N, Lq, M, L, P, 2] (x, y in 0..1),
// attention_weights [N, Lq, M, L, P]  ->  output [N, Lq, M * D]. Work is split over (n, query) with at::parallel_for.
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cmath>

torch::Tensor msda_forward(torch::Tensor value, torch::Tensor spatial_shapes, torch::Tensor level_start_index,
                           torch::Tensor sampling_locations, torch::Tensor attention_weights) {
  TORCH_CHECK(value.scalar_type() == torch::kFloat && sampling_locations.scalar_type() == torch::kFloat &&
              attention_weights.scalar_type() == torch::kFloat, "msda_forward: float32 only");
  TORCH_CHECK(!value.is_cuda(), "msda_forward: CPU tensors only");
  value = value.contiguous();
  auto loc = sampling_locations.contiguous();
  auto attn = attention_weights.contiguous();
  auto shapes = spatial_shapes.to(torch::kLong).contiguous();
  auto starts = level_start_index.to(torch::kLong).contiguous();
  const int64_t N = value.size(0), S = value.size(1), M = value.size(2), D = value.size(3);
  const int64_t Lq = loc.size(1), L = loc.size(3), P = loc.size(4);
  TORCH_CHECK(loc.size(0) == N && loc.size(2) == M && loc.size(5) == 2, "msda_forward: sampling_locations shape");
  TORCH_CHECK(attn.size(0) == N && attn.size(1) == Lq && attn.size(2) == M && attn.size(3) == L && attn.size(4) == P,
              "msda_forward: attention_weights shape");
  TORCH_CHECK(shapes.size(0) == L && starts.size(0) == L, "msda_forward: level count");
  auto out = torch::zeros({N, Lq, M * D}, value.options());
  const float* v = value.data_ptr<float>();
  const float* lp = loc.data_ptr<float>();
  const float* ap = attn.data_ptr<float>();
  const int64_t* sh = shapes.data_ptr<int64_t>();
  const int64_t* st = starts.data_ptr<int64_t>();
  float* o = out.data_ptr<float>();
  const int64_t row = M * D;  // floats between two neighbouring positions of one level
  at::parallel_for(0, N * Lq, 8, [&](int64_t b0, int64_t b1) {
    for (int64_t nq = b0; nq < b1; ++nq) {
      const int64_t n = nq / Lq;
      for (int64_t m = 0; m < M; ++m) {
        float* op = o + (nq * M + m) * D;
        for (int64_t l = 0; l < L; ++l) {
          const int64_t H = sh[2 * l], W = sh[2 * l + 1];
          const float* base = v + ((n * S + st[l]) * M + m) * D;
          for (int64_t p = 0; p < P; ++p) {
            const int64_t idx = ((nq * M + m) * L + l) * P + p;
            const float x = lp[2 * idx] * (float)W - 0.5f;
            const float y = lp[2 * idx + 1] * (float)H - 0.5f;
            const float w = ap[idx];
            if (!(y > -1.f && x > -1.f && y < (float)H && x < (float)W)) continue;
            const float fy = std::floor(y), fx = std::floor(x);
            const int64_t y0 = (int64_t)fy, x0 = (int64_t)fx;
            const float ly = y - fy, lx = x - fx, hy = 1.f - ly, hx = 1.f - lx;
            const int64_t ys[2] = {y0, y0 + 1}, xs[2] = {x0, x0 + 1};
            const float wy[2] = {hy, ly}, wx[2] = {hx, lx};
            for (int a = 0; a < 2; ++a) {
              if (ys[a] < 0 || ys[a] >= H) continue;
              for (int b = 0; b < 2; ++b) {
                if (xs[b] < 0 || xs[b] >= W) continue;
                const float cw = w * wy[a] * wx[b];
                const float* vp = base + (ys[a] * W + xs[b]) * row;
                for (int64_t d = 0; d < D; ++d) op[d] += cw * vp[d];
              }
            }
          }
        }
      }
    }
  });
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &msda_forward, "MSDeformAttn forward (CPU, float32)"); }
