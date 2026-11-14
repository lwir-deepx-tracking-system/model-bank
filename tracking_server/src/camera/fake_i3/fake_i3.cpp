// fake_i3 (11-15): a stand-in for the i3system camera SDK, so that cam_pipe.cpp's stall / reopen / exit decisions can be run without a camera (test_cam_pipe_frozen.py).
// Build (Linux, from tracking_server/src/camera):
//   g++ -O2 -Wall -Wextra -D_LINUX -Ifake_i3 cam_pipe.cpp fake_i3/fake_i3.cpp -o cam_pipe_fake -lpthread
// Run it with HOME set to an empty folder (cam_pipe takes $HOME/.cam_pipe.lock; the real cam_pipe on the board must not be blocked) and stdout to /dev/null.
// Environment:
//   FAKE_I3_PLAN  what happens at each OpenTE_A, in order, separated by ';' (after the last step the camera is gone):
//                   ok,<frames>,<op>,<result>,<reset>   the open works; RecvImage gives <frames> different 64 x 48 frames (-1 = no end), then only "no frame" (4);
//                                                       from the first empty read on GetLastUsbDiagnostic says operation <op>, result <result>;
//                                                       ResetMainBoard returns <reset> (0 or 1)
//                   fail,<op>,<result>                  OpenTE_A returns nullptr; the diagnostic says <op>, <result>
//                   gone                                ScanTE finds no camera (cam_pipe: NO_CAMERA_FOUND after its 10 scans)
//   FAKE_I3_LOG   file that gets one line per SDK event: "open N ok", "open N fail", "stop N", "reset N sent=S", "close N", "scan none"
// Operation codes: 2 open, 3 control, 4 send, 5 receive. Results: -7 libusb_timeout, -4 libusb_no_device, 1 success, 7 device_not_connected.
#include "i3system_TE.h"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
struct Step { std::string kind = "gone"; long frames = -1; int op = 0, result = 1; bool reset = false; };
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
    Step st; long f = -1; int op = 0, res = 1, rs = 0;   // anything else than ok,... or fail,... = gone
    if(std::sscanf(part.c_str(), "ok,%ld,%d,%d,%d", &f, &op, &res, &rs) == 4) { st.kind = "ok"; st.frames = f; st.op = op; st.result = res; st.reset = rs != 0; }
    else if(std::sscanf(part.c_str(), "fail,%d,%d", &op, &res) == 2) { st.kind = "fail"; st.op = op; st.result = res; }
    g_plan.push_back(st);
    a = b + 1;
  }
}

const Step& step(size_t i) { static const Step gone; parse(); return i < g_plan.size() ? g_plan[i] : gone; }

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
    ++value_; ++sent_;
    for(int i = 0; i < 64 * 48; ++i) px[i] = static_cast<unsigned short>(value_ + i);   // every frame differs from the one before: the duplicate filter keeps none back
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
bool TE_A::ShutterCalibrationOn() { return true; }
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
  return new TE_A(static_cast<int>(i));
}
}
