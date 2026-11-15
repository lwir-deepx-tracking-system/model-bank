// fake_i3 (11-15): a stand-in for the i3system camera SDK, so that cam_pipe.cpp's stall / reopen / exit decisions can be run without a camera (test_cam_pipe_frozen.py).
// Build (Linux, from tracking_server/src/camera):
//   g++ -O2 -Wall -Wextra -D_LINUX -Ifake_i3 cam_pipe.cpp fake_i3/fake_i3.cpp -o cam_pipe_fake -lpthread
// Run it with HOME set to an empty folder (cam_pipe takes $HOME/.cam_pipe.lock; the real cam_pipe on the board must not be blocked) and stdout to /dev/null.
// Environment:
//   FAKE_I3_PLAN  what happens at each OpenTE_A, in order, separated by ';' (after the last step the camera is gone):
//                   ok,<frames>,<op>,<result>,<reset>   the open works; RecvImage gives <frames> different 64 x 48 frames (-1 = no end), then only "no frame" (4);
//                                                       from the first empty read on GetLastUsbDiagnostic says operation <op>, result <result>;
//                                                       ResetMainBoard returns <reset> (0 or 1)
//                   ok,<frames>,<op>,<result>,<reset>,<blank_from>,<blank_to>   (11-15, cam_pipe v2.13) the same, and frames number blank_from (counted from 1
//                                                       in this open) up to before blank_to (-1 = to the end) have no scene: the middle half of the picture is one value
//                   fail,<op>,<result>                  OpenTE_A returns nullptr; the diagnostic says <op>, <result>
//                   gone                                ScanTE finds no camera (cam_pipe: NO_CAMERA_FOUND after its 10 scans)
//   FAKE_I3_LOG   file that gets one line per SDK event: "open N ok", "open N fail", "stop N", "reset N sent=S", "close N", "scan none"
//   FAKE_I3_LOG_CAL=1  also "cal N" for every ShutterCalibrationOn (11-15, cam_pipe v2.13 tests; off so that the v2.12 tests see the same lines as before)
//   FAKE_I3_FRAME_US=<n>  (11-15, cam_pipe v2.14, test_cam_pipe_pacing.py) the camera gets its own frame clock with a period of n microseconds, as measured on the board
//                   on 11-15 with every read logged. A read with a frame takes FAKE_I3_CALL_US (default 17500), an empty read 25 ms. The clock starts at the first read
//                   after an open; every read has a place = (its start - clock start) modulo n, between -n/2 and n/2. A read that follows a read with a frame:
//                   place -11 ms .. +7.2 ms = a new picture; up to 2.5 ms later than that (on the camera this band is under 1 ms) (and at least 10 frames since the last empty read) = the same picture again
//                   for 5 reads, then an empty read; any other place = an empty read. A read that follows an empty read always gets a new picture (cam_pipe retries
//                   about 26 ms after the empty read began, 7 ms earlier in the frame). Lines "late" / "early" / "miss" go to FAKE_I3_LOG.
//                   Unset: no clock, every read returns at once (the tests before v2.14 are unchanged).
// Operation codes: 2 open, 3 control, 4 send, 5 receive. Results: -7 libusb_timeout, -4 libusb_no_device, 1 success, 7 device_not_connected.
#include "i3system_TE.h"

#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
struct Step { std::string kind = "gone"; long frames = -1; int op = 0, result = 1; bool reset = false; long blank_from = 0, blank_to = -1; };
std::vector<Step> g_plan;
bool g_parsed = false;
size_t g_next = 0;                         // index of the step the next OpenTE_A takes
i3::USB_DIAGNOSTIC g_diag = {0, 1, -1};

void parse()
{
  if(g_parsed) return;
  g_parsed = true;
  const char* p = std::getenv("FAKE_I3_PLAN");
  std::string s = p ? p : "";
  size_t a = 0;
  while(a <= s.size() && !s.empty())
  {
    size_t b = s.find(';', a); if(b == std::string::npos) b = s.size();
    const std::string part = s.substr(a, b - a);
    Step st; long f = -1, bf = 0, bt = -1; int op = 0, res = 1, rs = 0;   // anything else than ok,... or fail,... = gone
    if(std::sscanf(part.c_str(), "ok,%ld,%d,%d,%d,%ld,%ld", &f, &op, &res, &rs, &bf, &bt) >= 4) { st.kind = "ok"; st.frames = f; st.op = op; st.result = res; st.reset = rs != 0; st.blank_from = bf; st.blank_to = bt; }
    else if(std::sscanf(part.c_str(), "fail,%d,%d", &op, &res) == 2) { st.kind = "fail"; st.op = op; st.result = res; }
    g_plan.push_back(st);
    a = b + 1;
  }
}

const Step& step(size_t i) { static const Step gone; parse(); return i < g_plan.size() ? g_plan[i] : gone; }

// the camera's own frame clock (FAKE_I3_FRAME_US), see the top of this file
struct FrameClock { int on = -1; double frame_us = 0, call_us = 17500; bool started = false, last_frame = false; std::chrono::steady_clock::time_point ref; long since_empty = 0; int bad = 0; };
FrameClock g_clock;
const double LATE_US = 7200.0, EARLY_US = 11000.0, LATE_ZONE_US = 2500.0; const int LATE_REPEATS = 5; const long LATE_MIN_FRAMES = 10;
bool clock_on()
{
  if(g_clock.on < 0)
  {
    const char* f = std::getenv("FAKE_I3_FRAME_US"); g_clock.on = f && std::atof(f) > 0 ? 1 : 0;
    if(g_clock.on) { g_clock.frame_us = std::atof(f); const char* c = std::getenv("FAKE_I3_CALL_US"); if(c && std::atof(c) >= 0) g_clock.call_us = std::atof(c); }
  }
  return g_clock.on == 1;
}

void logf(const char* fmt, ...)
{
  const char* path = std::getenv("FAKE_I3_LOG");
  if(!path || !*path) return;
  FILE* f = std::fopen(path, "a");
  if(!f) return;
  va_list ap; va_start(ap, fmt); std::vfprintf(f, fmt, ap); va_end(ap);
  std::fputc('\n', f); std::fclose(f);
}
}

namespace i3 {
TE_A::TE_A(int s) : step_(s) {}
void TE_A::CloseTE() { if(!closed_) { closed_ = true; logf("close %d", step_ + 1); } }
int TE_A::RecvImage(unsigned short* px, bool)
{
  const Step& st = step(static_cast<size_t>(step_));
  if(!closed_ && (st.frames < 0 || sent_ < st.frames))
  {
    bool again = false;   // the frame clock says: the same picture once more
    if(clock_on())
    {
      FrameClock& c = g_clock; const auto t = std::chrono::steady_clock::now();
      const auto empty = [&c](const char* word) { if(word) logf("%s", word); c.last_frame = false; c.since_empty = 0; c.bad = 0; std::this_thread::sleep_for(std::chrono::milliseconds(25)); return 4; };
      if(!c.started) { c.started = true; c.ref = t; }
      else if(c.last_frame)
      {
        double place = std::fmod(std::chrono::duration<double, std::micro>(t - c.ref).count(), c.frame_us);
        if(place > c.frame_us / 2) place -= c.frame_us;
        if(c.bad > 0)
        {
          if(c.bad >= LATE_REPEATS) return empty(nullptr);
          ++c.bad; again = true;
        }
        else if(place > LATE_US && place <= LATE_US + LATE_ZONE_US && c.since_empty >= LATE_MIN_FRAMES) { logf("late"); c.bad = 1; again = true; }
        else if(place < -EARLY_US || place > LATE_US) return empty(place < -EARLY_US && place >= -EARLY_US - LATE_ZONE_US ? "early" : "miss");
      }
      c.last_frame = true; ++c.since_empty;
      std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long>(c.call_us)));
    }
    if(!again) ++value_;
    ++sent_;
    const bool blank = st.blank_from > 0 && sent_ >= st.blank_from && (st.blank_to < 0 || sent_ < st.blank_to);
    for(int i = 0; i < 64 * 48; ++i) px[i] = static_cast<unsigned short>(value_ + i);   // every frame differs from the one before: the duplicate filter keeps none back
    if(blank) for(int y = 12; y < 36; ++y) for(int x = 16; x < 48; ++x) px[y * 64 + x] = 30000;   // no scene: the middle half one value, the rim still changes
    return 1;
  }
  if(!stopped_) { stopped_ = true; g_diag.operation = st.op; g_diag.result = st.result; g_diag.deviceIndex = 0; logf("stop %d", step_ + 1); }
  std::this_thread::sleep_for(std::chrono::milliseconds(26));   // the SDK waits about 26 ms for a frame before it returns 4
  return 4;
}
int TE_A::GetImageWidth() { return 64; }
int TE_A::GetImageHeight() { return 48; }
void TE_A::CalcTemp(unsigned short* map) { for(int i = 0; i < 64 * 48; ++i) map[i] = 7500; }
bool TE_A::ResetMainBoard() { const bool r = step(static_cast<size_t>(step_)).reset; logf("reset %d sent=%d", step_ + 1, r ? 1 : 0); return r; }
void TE_A::SetEmissivity(float) {}
void TE_A::GetSetting(TE_SETTING* s) { std::memset(s, 0, sizeof *s); }
bool TE_A::SetShutterMode(unsigned short) { return true; }
bool TE_A::SetShutterTime(unsigned short) { return true; }
bool TE_A::SetShutterTemperature(unsigned short) { return true; }
bool TE_A::ShutterCalibrationOn() { const char* c = std::getenv("FAKE_I3_LOG_CAL"); if(c && c[0] == '1') logf("cal %d", step_ + 1); return true; }
float TE_A::GetFpaTemp() { return 30.0f; }
float TE_A::GetShutterPt100Temp() { return 30.0f; }

extern "C" int ScanTE(SCANINFO* devs)
{
  for(int i = 0; i < MAX_USB_NUM; ++i) devs[i] = SCANINFO();
  if(step(g_next).kind == "gone") { g_diag.operation = I3_USB_OPERATION_SCAN; g_diag.result = 1; g_diag.deviceIndex = -1; logf("scan none"); return 1; }
  devs[0].bDevCon = 1; devs[0].nProdVer = 4; devs[0].nCoreID = 321;
  return 1;
}
extern "C" bool GetLastUsbDiagnostic(USB_DIAGNOSTIC* d) { *d = g_diag; return true; }
extern "C" const char* GetUsbOperationName(int op)
{
  static const char* names[] = {"none", "scan", "open", "control", "send", "receive", "initialize"};
  return op >= 0 && op <= 6 ? names[op] : "unknown";
}
extern "C" const char* GetUsbResultName(int r)
{
  return r == -7 ? "libusb_timeout" : r == -4 ? "libusb_no_device" : r == 1 ? "success" : r == 7 ? "device_not_connected" : "other";
}
extern "C" TE_A* OpenTE_A(int idx)
{
  const size_t i = g_next++;
  const Step& st = step(i);
  if(st.kind != "ok")
  {
    g_diag.operation = st.kind == "fail" ? st.op : I3_USB_OPERATION_OPEN; g_diag.result = st.kind == "fail" ? st.result : 7; g_diag.deviceIndex = idx;
    logf("open %zu fail", i + 1); return nullptr;
  }
  g_diag.operation = I3_USB_OPERATION_OPEN; g_diag.result = 1; g_diag.deviceIndex = idx;
  logf("open %zu ok", i + 1);
  g_clock.started = false; g_clock.last_frame = false; g_clock.bad = 0; g_clock.since_empty = 0;   // a new open: the frame clock starts at its first read
  return new TE_A(static_cast<int>(i));
}
}
