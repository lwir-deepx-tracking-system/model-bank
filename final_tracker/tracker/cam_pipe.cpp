// Camera -> stdout pipe for the live tracker (DX-M1 host, read-only). Writes every received 16-bit frame (w*h*2 bytes, native endian) to stdout and prints stats to stderr.
// Opens the camera like the team runtime (runtime/src/camera/camera.cpp). Optional frame-rate setting: ./cam_pipe [fps_code] [seconds]   fps_code: 0 = 30 fps, 1 = 25 fps, -1 = leave as is (default)
// Build: g++ -O2 cam_pipe.cpp -o cam_pipe -D_LINUX -I/usr/local/include/i3system $(pkg-config --cflags --libs opencv4) -li3system_te_64 -li3system_usb_64 -li3system_imgproc_impl_64 -lpthread
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <thread>
#include <vector>
#include <csignal>

#include <i3system_TE.h>
#include <i3_types.h>

static volatile std::sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

int main(int argc, char** argv)
{
  const int fps_code = argc > 1 ? std::atoi(argv[1]) : -1;
  const double seconds = argc > 2 ? std::atof(argv[2]) : 1e9;
  std::signal(SIGINT, on_sig); std::signal(SIGTERM, on_sig); std::signal(SIGPIPE, SIG_IGN);
  int idx = -1;
  for(int attempt = 0; attempt < 10 && idx < 0; ++attempt)
  {
    i3::SCANINFO devs[MAX_USB_NUM];
    if(ScanTE(devs) == 1) for(int i = 0; i < MAX_USB_NUM; ++i) if(devs[i].bDevCon == 1) { idx = i; break; }
    if(idx < 0) std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  if(idx < 0) { std::fprintf(stderr, "NO_CAMERA_FOUND\n"); return 2; }
  auto* te = i3::OpenTE_A(idx);
  if(!te) { std::fprintf(stderr, "OPEN_FAILED\n"); return 3; }
  const int w = te->GetImageWidth(), h = te->GetImageHeight();
  std::fprintf(stderr, "CAMERA %d %d\n", w, h);
  te->SetShutterMode(I3_TEMP_SHUTTER); te->SetShutterTemperature(1); te->ShutterCalibrationOn();

  std::vector<unsigned short> buf(static_cast<size_t>(w) * h);
  std::map<int, long> codes; long ok = 0;
  auto t0 = std::chrono::steady_clock::now(), tl = t0; long okl = 0; double maxgap = 0; auto tprev = t0;
  while(!g_stop && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds)
  {
    int r = te->RecvImage(buf.data(), true);
    ++codes[r];
    if(r == 1)
    {
      auto now = std::chrono::steady_clock::now();
      maxgap = std::max(maxgap, std::chrono::duration<double>(now - tprev).count()); tprev = now;
      if(std::fwrite(buf.data(), sizeof(unsigned short), buf.size(), stdout) != buf.size()) break;
      std::fflush(stdout); ++ok; ++okl;
      double dt = std::chrono::duration<double>(now - tl).count();
      if(dt >= 2.0) { std::fprintf(stderr, "CAMFPS %.1f\n", okl / dt); okl = 0; tl = now; }
    }
    else std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::fprintf(stderr, "FRAMES %ld in %.2f s = %.2f fps, longest gap %.0f ms, return codes:", ok, secs, ok / secs, maxgap * 1000);
  for(auto& kv : codes) std::fprintf(stderr, " %d x%ld", kv.first, kv.second);
  std::fprintf(stderr, "\n");
  te->CloseTE();
  return 0;
}
