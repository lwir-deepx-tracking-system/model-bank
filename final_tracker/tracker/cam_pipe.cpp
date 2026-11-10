// Camera -> stdout pipe for the live tracker (DX-M1 host). Writes every received 16-bit frame (w*h*2 bytes, native endian) to stdout and prints stats to stderr.
// Opens the camera like the team runtime (runtime/src/camera/camera.cpp). Optional frame-rate setting: ./cam_pipe [fps_code] [seconds]   fps_code: 0 = 30 fps, 1 = 25 fps, -1 = leave as is (default)
// v2.2 (11-09): single-instance lock (~/.cam_pipe.lock, exit 7 when another cam_pipe runs); experiment switches CAM_SHUTTER / CAM_NO_RECOVER / CAM_TELEMETRY (telemetry OFF by default:
// every SDK temperature getter blocks the stream for about 2 s, measured 11-09: 24.5 fps without, 0.5 fps with one getter pair per 2 s).
// v2.1 (11-09): stdout is non-blocking and written with poll() in 100 ms steps, and SIGTERM/SIGINT are installed without SA_RESTART, so a stop signal ends the program
// even while the tracker is not reading the pipe (a blocking fwrite of a 614 KB frame into a full 64 KB pipe could not be interrupted: the tracker then had to SIGKILL it with the camera open).
// v2 (11-09): SELF-HEALING. The camera sometimes stops sending frames after a few minutes (RecvImage keeps returning 4/5 even after a clean reopen) and then only a board reboot
// brought it back. The engine can be rebooted from software: when no frame arrives for STALL_S seconds (FIRST_STALL_S before the first frame after an open),
// this program calls ResetMainBoard(), closes the handle, waits RESET_WAIT_S seconds, scans again and reopens (tested 11-09: frames came back about 25 s later, no board reboot).
// After MAX_FAILED_RECOVERIES recoveries in a row without a single frame it ends with exit code 6 (the tracker may start a new process).
// Build: g++ -O2 cam_pipe.cpp -o cam_pipe -D_LINUX -I/usr/local/include/i3system $(pkg-config --cflags --libs opencv4) -li3system_te_64 -li3system_usb_64 -li3system_imgproc_impl_64 -lpthread
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include <csignal>
#include <sys/prctl.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/file.h>
#include <unistd.h>

#include <i3system_TE.h>
#include <i3_types.h>

static volatile std::sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

using clk = std::chrono::steady_clock;
static double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }
static void nap(double seconds)   // sleeps in 50 ms steps so that SIGTERM ends the wait at once
{
  auto t = clk::now();
  while(!g_stop && since(t) < seconds) std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

static bool write_all(const void* data, size_t n)   // false: write error (reader gone) or stop requested
{
  const char* p = static_cast<const char*>(data); size_t left = n;
  while(left > 0)
  {
    if(g_stop) return false;
    pollfd pfd; pfd.fd = STDOUT_FILENO; pfd.events = POLLOUT; pfd.revents = 0;
    const int pr = poll(&pfd, 1, 100);
    if(pr < 0) { if(errno == EINTR) continue; return false; }
    if(pr == 0) continue;
    if(pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) return false;
    const ssize_t w = write(STDOUT_FILENO, p, left);
    if(w < 0) { if(errno == EINTR || errno == EAGAIN) continue; return false; }
    p += w; left -= static_cast<size_t>(w);
  }
  return true;
}

static const double STALL_S = 6.0;          // no frame for this long while the stream was running: engine reset
static const double FIRST_STALL_S = 30.0;   // no first frame this long after an open: engine reset (a healthy open gives its first frame within a few seconds)
static const double RESET_WAIT_S = 20.0;    // time given to the engine to reboot before the next scan/open
static const int MAX_FAILED_RECOVERIES = 4;

static void diag(const char* where)
{
  i3::USB_DIAGNOSTIC d{};
  if(i3::GetLastUsbDiagnostic(&d))
    std::fprintf(stderr, "DIAG %s: op=%d(%s) result=%d(%s)\n", where, d.operation, i3::GetUsbOperationName(d.operation), d.result, i3::GetUsbResultName(d.result));
}

static int scan_device(int attempts)
{
  for(int attempt = 0; attempt < attempts && !g_stop; ++attempt)
  {
    i3::SCANINFO devs[MAX_USB_NUM];
    if(ScanTE(devs) == 1) for(int i = 0; i < MAX_USB_NUM; ++i) if(devs[i].bDevCon == 1) return i;
    nap(1.0);
  }
  return -1;
}

// Shutter setup. Env CAM_SHUTTER (experiment, 11-09): d = TEMP shutter limit 1 + calibration (the team runtime's setting, default), tN = TEMP limit N (1..10) + calibration,
// TN = TIME shutter every N*30 s (1..120) + calibration, n = no shutter call at all (camera defaults).
static void apply_shutter(i3::TE_A* te)
{
  const char* env = std::getenv("CAM_SHUTTER");
  const std::string s = env ? env : "d";
  if(s == "n") return;
  if(s[0] == 't') { te->SetShutterMode(I3_TEMP_SHUTTER); te->SetShutterTemperature(static_cast<unsigned short>(std::atoi(s.c_str() + 1))); te->ShutterCalibrationOn(); return; }
  if(s[0] == 'T') { te->SetShutterMode(I3_TIME_SHUTTER); te->SetShutterTime(static_cast<unsigned short>(std::atoi(s.c_str() + 1))); te->ShutterCalibrationOn(); return; }
  te->SetShutterMode(I3_TEMP_SHUTTER); te->SetShutterTemperature(1); te->ShutterCalibrationOn();
}

static i3::TE_A* open_camera(int* w, int* h)
{
  const int idx = scan_device(10);
  if(idx < 0) { std::fprintf(stderr, "NO_CAMERA_FOUND\n"); return nullptr; }
  auto* te = i3::OpenTE_A(idx);
  if(!te) { diag("open"); std::fprintf(stderr, "OPEN_FAILED\n"); return nullptr; }
  *w = te->GetImageWidth(); *h = te->GetImageHeight();
  std::fprintf(stderr, "CAMERA %d %d\n", *w, *h);
  apply_shutter(te);
  return te;
}

int main(int argc, char** argv)
{
  const int fps_code = argc > 1 ? std::atoi(argv[1]) : -1; (void)fps_code;   // kept for the old command line; SetFrameRate is not in our header
  const double seconds = argc > 2 ? std::atof(argv[2]) : 1e9;
  struct sigaction sa; sigemptyset(&sa.sa_mask); sa.sa_handler = on_sig; sa.sa_flags = 0;   // no SA_RESTART: blocking calls return EINTR
  sigaction(SIGINT, &sa, nullptr); sigaction(SIGTERM, &sa, nullptr); std::signal(SIGPIPE, SIG_IGN);
  fcntl(STDOUT_FILENO, F_SETFL, fcntl(STDOUT_FILENO, F_GETFL) | O_NONBLOCK);
  prctl(PR_SET_PDEATHSIG, SIGTERM);   // the tracker ended in any way: release the camera, also when no frame is arriving and no write can fail
  // v2.2: one camera reader per board. A second instance (old run script, manual test, orphan after a server crash) would scan/open/reset the camera under the first one's
  // stream. The lock lives in the home directory (not /tmp: survives nothing it should not, and other users cannot block it). Exit code 7 = another cam_pipe holds the camera.
  const std::string lock_path = std::string(std::getenv("HOME") ? std::getenv("HOME") : "/tmp") + "/.cam_pipe.lock";
  const int lock_fd = open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0644);
  if(lock_fd >= 0 && flock(lock_fd, LOCK_EX | LOCK_NB) != 0)
  {
    std::fprintf(stderr, "CAMERA_BUSY another cam_pipe holds %s; not touching the camera (exit 7)\n", lock_path.c_str());
    return 7;
  }
  int w = 0, h = 0;
  auto* te = open_camera(&w, &h);
  if(!te) return te == nullptr && g_stop ? 0 : 2;
  std::vector<unsigned short> buf(static_cast<size_t>(w) * h);
  std::map<int, long> codes; long ok = 0;
  auto t0 = clk::now(), tl = t0; long okl = 0; double maxgap = 0; auto tprev = t0;
  auto t_open = t0, t_frame = t0; bool frame_since_open = false; int failed_recoveries = 0, recoveries = 0; int exit_code = 0;
  const bool no_recover = std::getenv("CAM_NO_RECOVER") != nullptr, telemetry = std::getenv("CAM_TELEMETRY") != nullptr;   // experiment switches (11-09): no_recover = log the stall and exit with code 7; telemetry = FPA and shutter temperature in the CAMFPS line (read once per 2 s)
  const double test_stall_at = std::getenv("CAM_PIPE_TEST_STALL_AT") ? std::atof(std::getenv("CAM_PIPE_TEST_STALL_AT")) : 0; bool test_done = false;   // test hook: pretend the stream stopped once at this time (tests ResetMainBoard + reopen on a healthy camera)
  while(!g_stop && since(t0) < seconds)
  {
    int r = te->RecvImage(buf.data(), true);
    if(test_stall_at > 0 && !test_done && since(t0) > test_stall_at) { test_done = true; std::fputs("TEST: simulated stall", stderr); std::fputc(10, stderr); nap(STALL_S + 1.0); r = 4; }
    ++codes[r];
    if(r == 1)
    {
      auto now = clk::now();
      maxgap = std::max(maxgap, std::chrono::duration<double>(now - tprev).count()); tprev = now;
      t_frame = now; frame_since_open = true; failed_recoveries = 0;
      if(!write_all(buf.data(), buf.size() * sizeof(unsigned short))) break;
      ++ok; ++okl;
      double dt = std::chrono::duration<double>(now - tl).count();
      if(dt >= 2.0)
      {
        if(telemetry) std::fprintf(stderr, "CAMFPS %.1f t %.0f fpa %.2f shutter %.2f", okl / dt, since(t0), te->GetFpaTemp(), te->GetShutterPt100Temp());
        else std::fprintf(stderr, "CAMFPS %.1f", okl / dt);
        std::fputc(10, stderr); okl = 0; tl = now;
      }
      continue;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const double idle = since(frame_since_open ? t_frame : t_open);
    if(idle < (frame_since_open ? STALL_S : FIRST_STALL_S)) continue;
    if(no_recover)   // experiment: measure how long the camera lives, do not repair it
    {
      std::fprintf(stderr, "CAMERA_STALL_NO_RECOVER after %ld frames, no frame for %.1f s, run time %.1f s, shutter=%s", ok, idle, since(t0), std::getenv("CAM_SHUTTER") ? std::getenv("CAM_SHUTTER") : "d");
      std::fputc(10, stderr); diag("stall"); exit_code = 7; break;
    }
    // stalled: reboot the camera engine from software, then reopen
    ++recoveries;
    if(++failed_recoveries > MAX_FAILED_RECOVERIES) { std::fprintf(stderr, "CAMERA_UNRECOVERABLE after %d resets without a frame: power-cycle the camera or reboot the board\n", MAX_FAILED_RECOVERIES); exit_code = 6; break; }
    std::fprintf(stderr, "CAMERA_STALL no frame for %.1f s after %ld frames (%s): ResetMainBoard #%d (failed in a row: %d)\n", idle, ok, frame_since_open ? "stream stopped" : "no first frame", recoveries, failed_recoveries);
    diag("stall");
    const bool sent = te->ResetMainBoard();
    std::fprintf(stderr, "ResetMainBoard sent=%d\n", (int)sent);
    te->CloseTE(); te = nullptr;
    nap(RESET_WAIT_S);
    if(g_stop) break;
    te = open_camera(&w, &h);
    if(!te) { std::fprintf(stderr, "REOPEN_FAILED\n"); exit_code = 3; break; }
    if(static_cast<size_t>(w) * h != buf.size()) buf.assign(static_cast<size_t>(w) * h, 0);
    t_open = clk::now(); frame_since_open = false; tprev = t_open; tl = t_open; okl = 0;
  }
  double secs = since(t0);
  std::fprintf(stderr, "FRAMES %ld in %.2f s = %.2f fps, longest gap %.0f ms, resets %d, return codes:", ok, secs, ok / secs, maxgap * 1000, recoveries);
  for(auto& kv : codes) std::fprintf(stderr, " %d x%ld", kv.first, kv.second);
  std::fprintf(stderr, "\n");
  if(te) te->CloseTE();
  return exit_code;
}
