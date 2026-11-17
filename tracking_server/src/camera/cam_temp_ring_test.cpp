// cam_temp_ring_test (11-14): writes a temperature ring in the cam_pipe v2.11 layout with cam_pipe's own functions (temp_ring_open / temp_ring_put / temp_off), so that the
// layout can be checked from Python without a camera: CAM_TEMP_RING_FILE=<file> python -m unittest src.camera.test_cam_temp_ring (from tracking_server).
// It never opens the camera and never takes the camera lock: it only includes cam_pipe.cpp for those functions (cam_pipe's main is renamed and not called).
// Usage: ./cam_temp_ring_test <ring file> [stopped]
//   Writes 6 maps of 64 x 48 into the 4-slot ring at <ring file> as cam_pipe would with LWIR_CAM_TEMP=3 LWIR_CAM_TEMP_EMIS=0.98: frames 3, 6, ..., 18 (maps 5 and 6 overwrite
//   slots 0 and 1). Map k (1..6): t_mono_ns = k * 1e9 + 7, calc_us = 1000 + k, pixel i = 5000 + 100 * (20 + k) + i % 50, i.e. T_C = 20 + k + (i % 50) / 100.
//   "stopped": temp_off afterwards (ring state 2, one TEMP_OFF line on stderr). The file is kept (unmapped, not unlinked) for the reader.
// Build (on the board, next to cam_pipe.cpp): g++ -O2 cam_temp_ring_test.cpp -o cam_temp_ring_test -D_LINUX -I/usr/local/include/i3system -li3system_te_64 -li3system_usb_64 -li3system_imgproc_impl_64 -lpthread
#define main cam_pipe_main
#include "cam_pipe.cpp"
#undef main

int main(int argc, char** argv)
{
  if(argc < 2) { std::fprintf(stderr, "usage: %s <ring file> [stopped]\n", argv[0]); return 2; }
  const int w = 64, h = 48;
  g_temp.every = 3; g_temp.emis = 0.98f; g_temp.emis_set = true; g_temp.max_ms = 50;
  if(!temp_ring_open(argv[1], w, h)) return 1;
  g_temp.on = g_temp.started = true;
  std::vector<unsigned short> map(static_cast<size_t>(w) * h);
  for(uint32_t k = 1; k <= 6; ++k)
  {
    for(size_t i = 0; i < map.size(); ++i) map[i] = static_cast<unsigned short>(5000u + 100u * (20u + k) + i % 50u);
    temp_ring_put(map.data(), w, h, 3ull * k, 1000000000ull * k + 7ull, 1000u + k, TEMP_FLAG_AGC_READ);
  }
  if(argc > 2 && std::strcmp(argv[2], "stopped") == 0) temp_off("test");
  std::printf("WROTE %s maps %u stride %u payload %u bytes %zu id %u\n", argv[1], g_tring.maps, g_tring.stride, g_tring.payload, g_tring.bytes, g_tring.id);
  munmap(g_tring.base, g_tring.bytes); g_tring.base = nullptr;   // keep the file for the reader (temp_ring_close would unlink it)
  return 0;
}
