// cam_temp_probe (2026-11-14): one-off test of the i3system TE-EV1 temperature functions, separate from cam_pipe (cam_pipe is not changed).
// Question (user): can the camera give real temperatures (a thermal-radiation-only cue: skin 30-37 C) in real time?
// It takes the same lock as cam_pipe (~/.cam_pipe.lock), so it never touches the camera while cam_pipe runs (exit 7), and cam_pipe started
// while this runs exits 7 and is retried by the server. It opens the camera like cam_pipe (scan, OpenTE_A, shutter d = TEMP 1 + calibration),
// runs for at most ~80 s, and always closes the camera (CloseTE) at the end or on SIGINT/SIGTERM.
// Phases (each read paced at one frame period like cam_pipe v2.7):
//   P1  60 reads with the SDK's AGC OFF  RecvImage(buf, false): call times, success rate, distinct values, saves 3 frames (raw16 *.u16)
//   P2  5 x [AGC-off read + CalcTemp(whole frame)]: CalcTemp time, saves temperature maps (*.u16, T_C = (v - 5000) / 100) + CalcTemp(320,240)
//   P3  5 x [AGC-ON read + CalcTemp]: does CalcTemp still give temperatures after an AGC read? saves both
//   P4  SetEmissivity(0.98) then 3 x [AGC-off read + CalcTemp]: saves maps (skin emissivity)
//   P5  10 s stream with AGC ON every frame and CalcTemp every 10th frame: fps, longest gap (does CalcTemp stall the stream?)
//   P6  only with PROBE_TELEMETRY=1: GetFpaTemp + GetShutterPt100Temp once (known to stop the stream ~2 s on 11-09)
// Output: <dir> (default /tmp/cam_temp_probe): *.u16 frames (640x480 little endian) and summary lines on stdout (KEY value ...).
// Build: g++ -O2 cam_temp_probe.cpp -o cam_temp_probe -D_LINUX -I/usr/local/include/i3system -li3system_te_64 -li3system_usb_64 -li3system_imgproc_impl_64 -lpthread
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <i3system_TE.h>
#include <i3_types.h>

static volatile std::sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }
using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

static std::string g_dir = "/tmp/cam_temp_probe";
static void save(const std::string& name, const std::vector<unsigned short>& b)
{
  const std::string p = g_dir + "/" + name;
  FILE* f = std::fopen(p.c_str(), "wb");
  if(!f) { std::printf("SAVE_FAILED %s\n", p.c_str()); return; }
  std::fwrite(b.data(), 2, b.size(), f); std::fclose(f);
}
static void stats(const char* tag, const std::vector<unsigned short>& b)
{
  std::vector<unsigned short> s(b); std::sort(s.begin(), s.end());
  std::set<unsigned short> d(b.begin(), b.end());
  const size_t n = s.size();
  std::printf("%s min %u p01 %u p50 %u p99 %u max %u distinct %zu\n", tag, s[0], s[n / 100], s[n / 2], s[n * 99 / 100], s[n - 1], d.size());
}

int main(int argc, char** argv)
{
  if(argc > 1) g_dir = argv[1];
  mkdir(g_dir.c_str(), 0755);
  struct sigaction sa; sigemptyset(&sa.sa_mask); sa.sa_handler = on_sig; sa.sa_flags = 0;
  sigaction(SIGINT, &sa, nullptr); sigaction(SIGTERM, &sa, nullptr);
  const std::string lock_path = std::string(std::getenv("HOME") ? std::getenv("HOME") : "/tmp") + "/.cam_pipe.lock";
  const int lock_fd = open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0644);
  if(lock_fd >= 0 && flock(lock_fd, LOCK_EX | LOCK_NB) != 0) { std::printf("CAMERA_BUSY cam_pipe holds the camera (exit 7)\n"); return 7; }

  int idx = -1;
  for(int a = 0; a < 10 && idx < 0 && !g_stop; ++a)
  {
    i3::SCANINFO devs[MAX_USB_NUM];
    if(ScanTE(devs) == 1) for(int i = 0; i < MAX_USB_NUM; ++i) if(devs[i].bDevCon == 1) { idx = i; std::printf("ENGINE product %u core %u\n", devs[i].nProdVer, devs[i].nCoreID); break; }
    if(idx < 0) std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  if(idx < 0) { std::printf("NO_CAMERA_FOUND\n"); return 2; }
  i3::TE_A* te = i3::OpenTE_A(idx);
  if(!te) { std::printf("OPEN_FAILED\n"); return 3; }
  const int w = te->GetImageWidth(), h = te->GetImageHeight();
  std::printf("CAMERA %d %d\n", w, h);
  { i3::TE_SETTING st{}; te->GetSetting(&st); std::printf("ENGINE_SETTING frameRate %u shutterMode %u shutterTime %u shutterTemp %u highTempMode %u\n", st.frameRate, st.shutterMode, st.shutterTime, st.shutterTemp, st.highTempMode); }
  const int sm = te->SetShutterMode(I3_TEMP_SHUTTER) ? 1 : 0, sv = te->SetShutterTemperature(1) ? 1 : 0, sc = te->ShutterCalibrationOn() ? 1 : 0;
  std::printf("SHUTTER d set_mode %d set_value %d calibration %d\n", sm, sv, sc);

  std::vector<unsigned short> buf(static_cast<size_t>(w) * h), tmp(static_cast<size_t>(w) * h);
  const auto t_all = clk::now();
  auto last_start = clk::now(); bool last_ok = false;
  auto read = [&](bool agc, double* call_ms) -> int {
    if(last_ok) std::this_thread::sleep_until(last_start + std::chrono::microseconds(33333));
    const auto t = clk::now(); const int r = te->RecvImage(buf.data(), agc); *call_ms = ms_since(t);
    last_ok = (r == 1); if(last_ok) last_start = t; return r;
  };
  auto read_ok = [&](bool agc, double* call_ms) -> bool {   // up to 40 tries for one frame
    for(int k = 0; k < 40 && !g_stop; ++k) if(read(agc, call_ms) == 1) return true;
    return false;
  };
  auto done = [&]() { return g_stop || ms_since(t_all) > 75000.0; };

  // skip the shutter calibration at the open (~1.3 s of repeated frames)
  { double c; const auto t = clk::now(); while(ms_since(t) < 2500 && !done()) read(false, &c); }

  // P1: AGC off
  { int ok = 0, fail = 0, saved = 0; double cmax = 0, csum = 0; double c;
    for(int k = 0; k < 60 && !done(); ++k)
    { const int r = read(false, &c); if(r == 1) { ++ok; csum += c; cmax = std::max(cmax, c); if(k % 25 == 5 && saved < 3) { save("p1_agcoff_" + std::to_string(saved++) + ".u16", buf); if(saved == 1) stats("P1_FRAME", buf); } } else ++fail; }
    std::printf("P1_AGC_OFF ok %d fail %d call_mean_ms %.1f call_max_ms %.1f\n", ok, fail, ok ? csum / ok : 0.0, cmax); }

  // P2: AGC off + CalcTemp
  for(int k = 0; k < 5 && !done(); ++k)
  { double c; if(!read_ok(false, &c)) { std::printf("P2_READ_FAILED %d\n", k); continue; }
    const auto t = clk::now(); te->CalcTemp(tmp.data()); const double ct = ms_since(t);
    const auto t2 = clk::now(); const unsigned short cpix = te->CalcTemp(static_cast<unsigned short>(w / 2), static_cast<unsigned short>(h / 2)); const double ct2 = ms_since(t2);
    std::printf("P2_CALCTEMP %d frame_ms %.1f pixel_ms %.2f centre_raw %u centre_C %.2f\n", k, ct, ct2, cpix, (cpix - 5000) / 100.0);
    if(k == 0 || k == 4) { save("p2_raw_" + std::to_string(k) + ".u16", buf); save("p2_temp_" + std::to_string(k) + ".u16", tmp); stats("P2_TEMP", tmp); } }

  // P3: AGC on + CalcTemp
  for(int k = 0; k < 5 && !done(); ++k)
  { double c; if(!read_ok(true, &c)) { std::printf("P3_READ_FAILED %d\n", k); continue; }
    const auto t = clk::now(); te->CalcTemp(tmp.data()); const double ct = ms_since(t);
    std::printf("P3_AGC_ON_CALCTEMP %d read_ms %.1f frame_ms %.1f\n", k, c, ct);
    if(k == 0 || k == 4) { save("p3_agc_" + std::to_string(k) + ".u16", buf); save("p3_temp_" + std::to_string(k) + ".u16", tmp); stats("P3_TEMP", tmp); } }

  // P4: emissivity 0.98
  te->SetEmissivity(0.98f); std::printf("P4_EMISSIVITY 0.98 set\n");
  for(int k = 0; k < 3 && !done(); ++k)
  { double c; if(!read_ok(false, &c)) { std::printf("P4_READ_FAILED %d\n", k); continue; }
    te->CalcTemp(tmp.data()); if(k == 0) { save("p4_raw_0.u16", buf); save("p4_temp_e098_0.u16", tmp); stats("P4_TEMP", tmp); } }

  // P5: AGC on every frame, CalcTemp every 10th, 10 s
  { int ok = 0, ct_n = 0; double gap_max = 0, ct_max = 0; auto t5 = clk::now(), last = clk::now(); double c;
    while(ms_since(t5) < 10000 && !done())
    { if(read(true, &c) == 1) { ++ok; gap_max = std::max(gap_max, ms_since(last)); last = clk::now();
        if(ok % 10 == 0) { const auto t = clk::now(); te->CalcTemp(tmp.data()); ct_max = std::max(ct_max, ms_since(t)); ++ct_n; } } }
    std::printf("P5_STREAM frames %d fps %.1f longest_gap_ms %.1f calctemp_calls %d calctemp_max_ms %.1f\n", ok, ok / (ms_since(t5) / 1000.0), gap_max, ct_n, ct_max); }

  if(std::getenv("PROBE_TELEMETRY") && !done())
  { const auto t = clk::now(); const float fpa = te->GetFpaTemp(); const double a = ms_since(t);
    const auto t2 = clk::now(); const float pt = te->GetShutterPt100Temp(); const double b = ms_since(t2);
    std::printf("P6_TELEMETRY fpa_C %.2f fpa_ms %.0f pt100_C %.2f pt100_ms %.0f\n", fpa, a, pt, b); }

  te->CloseTE();
  std::printf("CLOSED total_s %.1f stopped_by_signal %d\n", ms_since(t_all) / 1000.0, (int)g_stop);
  return 0;
}
