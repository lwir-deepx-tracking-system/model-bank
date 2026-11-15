// Camera -> stdout pipe for the live tracker (DX-M1 host). Writes every received 16-bit frame (w*h*2 bytes, native endian) to stdout and prints stats to stderr.
// Opens the camera like the team runtime (runtime/src/camera/camera.cpp). Optional frame-rate setting: ./cam_pipe [fps_code] [seconds]   fps_code: 0 = 30 fps, 1 = 25 fps, -1 = leave as is (default)
// v2.14 (11-15): READS ON A GRID THAT FOLLOWS THE CAMERA. Since the CPU governor went to "performance" (11-15 19:32) the picture stood still for 0.16-0.30 s every
//   3.7 s (445 REPEAT_RUN lines in 2486 s; while a person was followed the server sent "no people" for the repeated frames and the gimbal stopped and started
//   again). Measured on the board the same evening with every read in a file (LWIR_CAM_READ_LOG below; the camera stayed open, settings changed while running):
//   - The v2.7 pacing started a read 33.333 ms after the ACTUAL start of the read before, so each wake-up delay (64 us) added up: a read every 33.400 ms.
//     The camera makes a picture every 33.3341 ms (30.00 a second), so the reads slid later by 66 us a read.
//   - 7.2 ms later than the place where the stream was picked up, the camera gives the same picture 4-5 times and then no frame (return 4 after 25 ms): 0.23 s
//     without a new picture. The read after the empty one starts at once, 7 ms earlier in the camera's frame, and all is well for the next 7.2 ms of sliding:
//     108 reads = 3.6 s. With the slower governor the delay was about 117 us a read: the same thing every 2 s, with 2-3 repeats that the duplicate filter hid.
//     So v2.8's "the engine itself repeats frames, it is not the read timing" was wrong: it is the read timing (every repeat was at this late edge).
//     The repeats come only when the reads creep over that edge (under 1 ms past it); a read that jumps further past it (1 ms steps, or one late start)
//     gets no frame at once, without repeats, and the retry is back inside: one gap of 59 ms.
//   - 11 ms earlier than the pick-up place a read gets no frame at once (no repeats); cam_pipe's retries then give frame, empty, frame, empty, frame (three
//     gaps of 59 ms, never a repeated picture) and the stream is steady again. So the reads may start anywhere in a window of about 18 ms of the 33.3 ms frame.
//   - A read exactly one period late (66.667 ms after the one before) gets a new picture like any other: only the place inside the frame matters.
//     A read 22 ms late got a picture, but the read 18.5 ms after it got none.
//   - The call takes 17.5 ms wherever in the window it starts, so the place cannot be seen from the call time; only reaching an edge shows it.
//   - With the reads on a fixed grid of 33.333 ms nothing was repeated or empty for 8 minutes (30.00 new pictures a second instead of 28.5); the grid then
//     reached the EARLY edge, because the camera's frame is 0.77 us longer than 33.333 ms of the board's clock (23 ppm).
//   What v2.14 does (LWIR_CAM_PACE_MODE=grid, the default; "after" = the v2.7 pacing exactly as before):
//   (a) GRID. A read starts one period after the PLANNED start of the read before; waking up late does not move the next read. After an empty read the next
//       read starts at once (as before) and begins a new grid. A read that would start more than 6 ms after its place (the server was slow to take the frame,
//       or no CPU) waits for the next place: one picture is lost and the place in the camera's frame is kept.
//   (b) THE PERIOD FOLLOWS THE CAMERA (LWIR_CAM_PACE_TRIM=1, the default). A grid that began at a pick-up knows where it is (7.2 ms before the late edge, 11 ms
//       after the early one). When it ends in an empty read after N reads (600 or more), the period was wrong by the distance travelled / N: too long when
//       repeated pictures came before the empty read (late edge), too short when none came (early edge). An empty read whose own start, or the start of
//       the read before it, was more than 250 us late (LWIR_CAM_PACE_ON_TIME_US; the board wakes 64 us late, 111 us at the 99th percentile) teaches nothing:
//       measured, a read 0.5 ms late near the late edge got a picture and the next read none, with no repeat to tell late from early. 80 % is corrected, at most 5 us at a time and
//       10 us in all (300 ppm). The correction is kept in ~/.cam_pipe_state (pace_trim_ns, with the period it belongs to) and the next start begins with it.
//       A grid that began at an open (or within 5 s of it: the shutter calibration there ends in an empty read of its own) or after a stall (3 or more
//       empty reads in a row), or was moved by hand, teaches nothing and is not moved to the resting place. Each lesson is a PACE_RESYNC line.
//   (c) RESTING PLACE (LWIR_CAM_PACE_CENTRE_US, default -4000). 30 reads after a pick-up the grid moves 4 ms earlier: 11.2 ms to the late edge (the costly
//       one: 0.23 s and repeated pictures) and 7 ms to the early edge (three gaps of 59 ms). If an empty read comes within 100 reads of that move, the move is
//       switched off for the rest of the process (PACE_CENTRE_OFF).
//   (d) LINES. READ_PACING at the start (mode, period in use). READ_STATS every 5 minutes and at the end: reads, new pictures a second, repeats, empty reads,
//       the longest time without a new picture, how often the reads came too early / too late, grid places skipped, the period in use.
//   (e) TEST HOOKS, off unless set. LWIR_CAM_READ_LOG=<file>: one line per read (start ms, call ms, return code, kind 0 new / 1 repeat held back / 2 repeat
//       passed on / - none, us late, period, mode). LWIR_CAM_PACE_FILE=<file>: settings read from that file whenever it changes (period_us= mode= shift_us=
//       trim= centre_us= rephase_us= log=off), so that periods can be compared without opening the camera again. LWIR_CAM_PACE_REPHASE_US (0 = off): an
//       experiment not yet seen working on the camera, moves the grid at the first repeated picture instead of waiting for the empty read.
//   The duplicate filter, REPEAT_RUN, INPUT_GAP, the stall repair and everything else are unchanged. Without a camera: fake_i3/ has the frame clock as
//   measured (FAKE_I3_FRAME_US) and test_cam_pipe_pacing.py runs old and new pacing against it.
// v2.13 (11-15): FEWER SHUTTER MOVES, A PICTURE WITHOUT A SCENE, NO CLOSE WHILE THE SHUTTER IS SHUT. On 11-15 the camera was opened about 25 times in 2.5 hours
//   (server restarts, plus a gimbal-speed tool and a temperature probe that open it themselves) and every open moved the shutter (ShutterCalibrationOn). Between
//   16:35:25 and 16:38:04 the picture became a flat field while frames still came at 30 fps: the middle of the picture one single value, only a noisy rim in the
//   shape of the shutter blade. Moving the gimbal changed nothing and the calibrations of the next opens did not bring the scene back; a hand replug on site did.
//   (a) CALIBRATION REUSE. The engine keeps its correction while it is powered, so the first open of a process skips ShutterCalibrationOn when the state file
//       ~/.cam_pipe_state says the last calibration of ours was less than LWIR_CAM_CAL_REUSE_S seconds ago (default 1800; 0 = calibrate at every open, the
//       behaviour up to v2.12) on the same camera power-on (same boot, same USB entry and device number: a replug or a power cut changes the number), with the
//       same shutter setting and no ResetMainBoard sent since. A calibration counts only once frames have run CAL_SETTLE_S after it (a process ended during
//       its calibration leaves none). The shutter mode and time are still set at every open; a reopen after a stall always calibrates. A CALIBRATION line at
//       every open says what was done and why; the SHUTTER line says calibration_at_open=-2 when it was reused.
//   (b) OPEN COUNT. Every open goes into the state file and an OPENS line says how many opens this camera power-on had in the last 10 min and hour and which
//       program started this cam_pipe. Programs other than the server should use the server's frames instead of opening the camera (HANDOFF.md, section 7).
//   (c) NO SCENE. Once a second the middle half of the picture just passed on (every 2nd pixel) is checked: centre_same = the share of its most common value.
//       Real pictures of this camera: at most 0.10 (8 recordings 11-09..11-13, 350 frames, median about 0.02); the 11-15 16:38 flat field: 0.985.
//       centre_same >= 0.5 at every check for LWIR_CAM_BLANK_S seconds (default 20; 0 = no check) -> SCENE_LOST line and ONE engine reset (ResetMainBoard, close,
//       20 s, reopen with a calibration: the steps of the stall repair) per camera power-on, kept in the state file so that a restarted cam_pipe does not repeat
//       it (LWIR_CAM_BLANK_RESET=0: no reset, lines only). After that only the SCENE_LOST line (again every 10 min while it lasts): replug the camera. Once the
//       picture has a scene again (centre_same < 0.2 for 5 s) a SCENE_BACK line follows. A SCENE_CHECK line every 10 min gives the largest centre_same seen.
//       The server puts SCENE_LOST on the page (src/camera/cam_scene.py).
//   (d) A stop signal within CAL_SETTLE_S of a calibration waits until that time has passed before CloseTE (CLOSE_WAIT line): the camera is never closed
//       while the shutter may be shut for its calibration.
//   State file: "key value" lines, times in seconds since boot (CLOCK_BOOTTIME), written to <file>.tmp and renamed; one cam_pipe at a time (the lock).
//   LWIR_CAM_SYSFS points the camera's USB lookup at another folder (tests). Test without a camera: test_cam_pipe_scene.py (fake_i3/).
// v2.12 (11-15): FROZEN CAMERA, EXIT 8. Since 11-12 every hard camera loss looked the same: the camera's own USB chip (0547:0080) stops answering while it stays on the bus.
//   The stall's DIAG says op=3(control) or op=4(send) result=-7(libusb_timeout), ResetMainBoard() is refused (sent=0) because the engine cannot be reached, and every open
//   after it fails with op=4(send) libusb_timeout. Only cutting its power cleared it (a hand replug: its board root port has no software 5 V switch); the 20 s reopen loop,
//   usbreset, the 5 V step (it switches only the board's hub, not that root port) and board reboots never did. cam_pipe now stops trying at once in two cases:
//     (a) a stall whose DIAG is control or send + libusb_timeout and whose ResetMainBoard() returns false;
//     (b) a reopen after a stall that fails in OpenTE_A with control or send + libusb_timeout, when this process had passed frames on (the camera was streaming).
//   Then it closes the handle, prints one line "CAMERA_FROZEN <reason>: the camera's USB chip does not answer; ..." (the reason has no colon) and exits with code 8:
//   no 20 s wait, no further ResetMainBoard. The server waits for the camera's USB presence to change (replug) instead of starting cam_pipe again every few seconds.
//   Unchanged: libusb_no_device stalls (the kernel dropped the device; it comes back by itself and the reopen 20 s later works), stalls whose ResetMainBoard is accepted,
//   a first open that fails (exit 2, nothing streamed yet), a reopen that finds no camera or fails otherwise (exit 3), CAM_NO_RECOVER (exit 7), exit 6.
//   Test hook: CAM_PIPE_TEST_FROZEN=1 together with CAM_PIPE_TEST_STALL_AT=<s> makes that one simulated stall take the frozen path (its DIAG is taken as control
//   libusb_timeout and ResetMainBoard is not called), so the server's handling of exit 8 can be tried with a healthy camera.
//   Test without a camera: fake_i3/ is a stand-in for the SDK (plan of opens and stalls in FAKE_I3_PLAN); test_cam_pipe_frozen.py builds cam_pipe against it and runs
//   the frozen and the unchanged repair paths (python3 -m unittest src.camera.test_cam_pipe_frozen, Linux with g++).
// v2.11 (11-14): OPTIONAL TEMPERATURE MAPS IN SHARED MEMORY, OFF UNLESS LWIR_CAM_TEMP IS SET. With LWIR_CAM_TEMP unset, empty or 0, the bytes on stdout and the lines on stderr are
//   exactly those of v2.10 (LWIR_CAM_TEMP_EMIS and LWIR_CAM_TEMP_MAX_MS are then not even read). Purpose: a single-frame person cue from radiation alone (skin about 28-38 C).
//   LWIR_CAM_TEMP=N (whole number 1..300): after every N-th frame that is passed on (frame numbers N, 2N, 3N, ...) te->CalcTemp(map) is called for the whole picture. CalcTemp
//     works on the most recent RecvImage, so it runs right after that frame's read: after the picture has been handed on (the picture is not delayed) and before the next read.
//     Its time comes out of the read pacing (v2.7), so a slow CalcTemp can cost frames. The map (one uint16 per pixel, T_C = (v - 5000) / 100) is copied into its own ring
//     /dev/shm/lwir_temp_ring_<uid> (4 slots) with the same rule as the picture ring: the slot's frame number is 0 while the slot is written and the frame number after it.
//     The pictures do not change: still RecvImage(buf, true), the SDK's AGC.
//   OPEN QUESTION, answered by the cam_temp_probe run with the camera (phase P3): does CalcTemp give temperatures after an AGC-on read? If it does not, the fix is a later change,
//     not made here: every N-th read becomes an AGC-off read RecvImage(raw, false) followed by CalcTemp, and that frame's picture is either not passed on (the tracker loses
//     1 frame in N) or made from the raw frame by an AGC of our own. The slot flags (bit 0 = the read before CalcTemp had the AGC on) tell a reader which kind of map it got.
//   LWIR_CAM_TEMP_EMIS=e (0.01..1.0; unset = SetEmissivity is not called, the camera keeps its own setting): te->SetEmissivity(e) after every open (skin: about 0.98).
//   LWIR_CAM_TEMP_MAX_MS=m (whole number 1..10000, default 50): when CalcTemp takes longer than m ms more than 3 times in a row, the maps stop for the rest of this run (TEMP_OFF
//     line, ring state 2). This protects the stream in case CalcTemp blocks like the SDK's temperature getters do (about 2 s each, v2.2).
//   Log: one TEMP_RING line at the start (or "TEMP_RING failed" / "TEMP_RING not used"), a TEMP_STATS line every 30 s from the CAMFPS block (not every 2 s: camera_extra.py keeps
//     every cam_pipe line except CAMFPS in camera_events.log), TEMP_OFF, TEMP_EMIS after a reopen, TEMP_TOTAL at the end, and the INPUT_GAP line names the CalcTemp time.
//   Ring layout (native byte order = little endian on the board; the Python struct format of each header in brackets; the server's reader is src/camera/temp_ring.py):
//     file header, 64 bytes at offset 0 ['<4s7IQ2If2I4x']: magic "LWT1", u32 version 1, u32 ring_id, u32 nslots 4, u32 stride (bytes per slot), u32 payload (= width*height*2),
//       u32 width, u32 height, u64 pid, u32 every (N), u32 state (1 = maps are written, 2 = stopped for this run: TEMP_OFF or the picture size changed at a reopen, 3 = cam_pipe
//       ended normally, the file is unlinked right after), f32 emissivity (0 = SetEmissivity not called), u32 max_ms, u32 maps (maps written so far; map k is in slot
//       (k - 1) % nslots; stored after that slot is complete), 4 bytes unused. The first 40 bytes have the same fields as the picture ring header (LWR1).
//     slot i at offset 64 + i * stride: 64-byte slot header ['<Q4sHHQIHHfIIIHH12x']: u64 frame (cam_pipe frame number of the picture the map belongs to = the seq of its stamp
//       header; 0 while the slot is written), magic "LWTS", u16 version 1, u16 flags (bit 0: the read before CalcTemp had the AGC on), u64 t_mono_ns (CLOCK_MONOTONIC ns when that
//       frame's RecvImage returned = its stamp header's t_mono_ns), u32 calc_us (time CalcTemp took), u16 width, u16 height, f32 emissivity, u32 map_seq (k), u32 payload,
//       u32 ring_id, u16 t_offset 5000, u16 t_div 100 (T_C = (v - t_offset) / t_div), 12 bytes unused; then the map: height rows of width uint16 (row-major).
//     Writing a slot: frame := 0 (atomic, then a full fence), the slot header without the frame and the map, frame := the frame number (atomic release), then maps := k.
//     Reader: k = maps (0 = no map yet); slot = (k - 1) % nslots; f1 = that slot's frame (0 = being written: try again or take map k - 1); copy the slot header and the map;
//       f2 = the slot's frame again; use the copy only if f1 == f2 and the copied header's frame == f1 (and its map_seq == k, ring_id == the file's ring_id).
//   Frame numbers count the frames passed on since this cam_pipe started, from 1 (repeats held back by the duplicate filter are not counted); with LWIR_CAM_STAMP=1 they are
//   the stamp header's seq, without it a reader counts the pictures it took from the pipe. A ring left by a killed cam_pipe is replaced at the start; unlinked on a normal end.
//   Layout check without a camera: cam_temp_ring_test.cpp writes a ring with the functions of this file, test_cam_temp_ring.py reads it (also with temp_ring.py).
// v2.6 (11-11): TWO OPTIONAL THINGS, BOTH OFF UNLESS THE ENVIRONMENT TURNS THEM ON. With neither variable set, the bytes on stdout and the lines on stderr are exactly those of v2.5.
//   LWIR_CAM_STAMP=1 (exactly "1"): a 48-byte header goes before every frame: magic "LWF1", version, flags, frame number (from 1), CLOCK_MONOTONIC time in ns of the moment
//     RecvImage returned, microseconds since the previous frame, ring slot, payload bytes, ring id, width, height, check word (xor of the 11 words before it and 0x5A17C0DE).
//     Native byte order (little endian on the board). The tracker (src/camera/camera_extra.py) reads the same variable and passes it on when it starts this program,
//     so the two always agree; a tracker that does not find the header starts this program again with LWIR_CAM_STAMP=0.
//   LWIR_CAM_SHM=1 (needs LWIR_CAM_STAMP=1): the picture is not sent through the 64 KB pipe (one 614 KB frame = about ten writes, and a busy reader made this program wait).
//     It is copied into a ring of 4 slots in shared memory, the file /dev/shm/lwir_cam_ring_<uid> (opened directly, so no extra library is needed at link time), and only the
//     48-byte header (flags bit 0 set, slot number, ring id) goes through the pipe. Each slot starts with the frame number: it is set to 0 before the copy and to the frame
//     number after it, so a reader that sees another number before or after its own copy knows the slot was overwritten and drops that frame. A ring left by a killed cam_pipe
//     is replaced at the start; the ring is unlinked on a normal end. When the ring cannot be made (no /dev/shm, no room), a SHM_RING failed line is printed and the pictures go
//     through the pipe behind the header, as with LWIR_CAM_STAMP=1 alone.
// v2.5 (11-11): SHUTTER SETTING THAT CAN BE CHANGED WITHOUT RESTARTING THE TRACKER, AND EVIDENCE IN THE LOG. Today's stalls come at fixed moments after an open (about 1.4 s, 4.4 s and
// every ~180 s), which points at shutter operations. The shutter setting is now read at every open from env CAM_SHUTTER or, when that is not set, from the first word of
// cam_shutter.txt next to this program (no file = d, the old behaviour). Change the file, then end cam_pipe with SIGTERM: it closes the camera and exits 0, and the tracker starts
// it again at once. New settings: m = MANUAL shutter mode (the engine never moves the shutter by itself; one calibration at the open), mN = MANUAL + our own calibration every
// N*30 s (logged as SHUTTER_CAL, so a stall right after one is visible), a trailing x on any setting = no ShutterCalibrationOn() at the open (dx, T120x, mx). Every open logs a
// SHUTTER line with what was asked and whether the engine accepted it, every stall gets a STALL_INFO line saying how long this open lived, and lines carry the board clock. An
// optional second word "info" in cam_shutter.txt reads the engine's saved settings once (GetSetting) before the first frame. The repair logic (6 s, ResetMainBoard,
// 20 s) is unchanged.
// v2.2 (11-09): single-instance lock (~/.cam_pipe.lock, exit 7 when another cam_pipe runs); experiment switches CAM_SHUTTER / CAM_NO_RECOVER / CAM_TELEMETRY (telemetry OFF by default:
// every SDK temperature getter blocks the stream for about 2 s, measured 11-09: 24.5 fps without, 0.5 fps with one getter pair per 2 s).
// v2.1 (11-09): stdout is non-blocking and written with poll() in 100 ms steps, and SIGTERM/SIGINT are installed without SA_RESTART, so a stop signal ends the program
// even while the tracker is not reading the pipe (a blocking fwrite of a 614 KB frame into a full 64 KB pipe could not be interrupted: the tracker then had to SIGKILL it with the camera open).
// v2 (11-09): SELF-HEALING. The camera sometimes stops sending frames after a few minutes (RecvImage keeps returning 4/5 even after a clean reopen) and then only a board reboot
// brought it back. The engine can be rebooted from software: when no frame arrives for STALL_S seconds (FIRST_STALL_S before the first frame after an open),
// this program calls ResetMainBoard(), closes the handle, waits RESET_WAIT_S seconds, scans again and reopens (tested 11-09: frames came back about 25 s later, no board reboot).
// After MAX_FAILED_RECOVERIES recoveries in a row without a single frame it ends with exit code 6 (the tracker may start a new process).
// Build: g++ -O2 cam_pipe.cpp -o cam_pipe -D_LINUX -I/usr/local/include/i3system $(pkg-config --cflags --libs opencv4) -li3system_te_64 -li3system_usb_64 -li3system_imgproc_impl_64 -lpthread
//   (v2.11: the same line; nothing new to link. Checked 11-14 with g++ 11.4 on the board, also with -Wall -Wextra: no warning from the v2.11 code.)
//   (v2.12: the same line. Checked 11-15 with g++ 11.4 on the board, also with -Wall -Wextra: no warning.)
//   (v2.13: the same line; nothing new to link.)
//   (v2.14: the same line; nothing new to link. Checked 11-15 with g++ 11.4 on the board, also with -Wall -Wextra: no warning.)
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include <csignal>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <sys/prctl.h>
#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
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

// v2.6: optional frame header (LWIR_CAM_STAMP=1) and shared-memory ring (LWIR_CAM_SHM=1). Nothing below runs unless g_stamp is true. The layouts are repeated in src/camera/camera_extra.py.
static bool g_stamp = false;
struct FrameHead { char magic[4]; uint16_t version, flags; uint64_t seq, t_mono_ns; uint32_t dt_us, slot, payload, ring_id; uint16_t width, height; uint32_t check; };
static_assert(sizeof(FrameHead) == 48, "frame header must be 48 bytes");
struct RingHead { char magic[4]; uint32_t version, ring_id, nslots, stride, payload, width, height; uint64_t pid; unsigned char pad[24]; };
static_assert(sizeof(RingHead) == 64, "ring header must be 64 bytes");
static const uint32_t RING_SLOTS = 4, SLOT_HEAD = 16;   // a slot = 8 bytes frame number (0 while it is being written), 8 bytes unused, then the picture
struct Ring { unsigned char* base = nullptr; size_t bytes = 0; uint32_t id = 0, stride = 0, payload = 0, next = 0; std::string path; };
static Ring g_ring;

static bool env_is_1(const char* name) { const char* v = std::getenv(name); return v && v[0] == '1' && v[1] == 0; }
static uint64_t mono_ns() { timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec); }   // the clock of Python's time.monotonic()

static void ring_close()   // unmaps and unlinks our ring (normal end)
{
  if(g_ring.base) { munmap(g_ring.base, g_ring.bytes); g_ring.base = nullptr; }
  if(!g_ring.path.empty()) { unlink(g_ring.path.c_str()); g_ring.path.clear(); }
}

static bool ring_open(int w, int h)   // false = no ring: the pictures go through the pipe behind the header
{
  const uint32_t payload = static_cast<uint32_t>(w) * static_cast<uint32_t>(h) * 2u, stride = (SLOT_HEAD + payload + 63u) & ~63u;
  const size_t bytes = sizeof(RingHead) + static_cast<size_t>(RING_SLOTS) * stride;
  char name[64]; std::snprintf(name, sizeof name, "/dev/shm/lwir_cam_ring_%u", static_cast<unsigned>(getuid()));
  unlink(name);   // a ring left by a cam_pipe that was killed (only one cam_pipe runs: the lock in main)
  const int fd = open(name, O_CREAT | O_EXCL | O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0600);
  if(fd < 0) { std::fprintf(stderr, "SHM_RING failed (open %s: %s): pictures go through the pipe\n", name, std::strerror(errno)); return false; }
  const int fe = posix_fallocate(fd, 0, static_cast<off_t>(bytes));   // takes the memory now: a full /dev/shm fails here and not later in the middle of a copy
  if(fe != 0) { std::fprintf(stderr, "SHM_RING failed (no room for %zu bytes in %s: %s): pictures go through the pipe\n", bytes, name, std::strerror(fe)); close(fd); unlink(name); return false; }
  void* m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if(m == MAP_FAILED) { std::fprintf(stderr, "SHM_RING failed (mmap %s: %s): pictures go through the pipe\n", name, std::strerror(errno)); unlink(name); return false; }
  std::memset(m, 0, bytes);
  g_ring.base = static_cast<unsigned char*>(m); g_ring.bytes = bytes; g_ring.stride = stride; g_ring.payload = payload; g_ring.next = 0; g_ring.path = name;
  g_ring.id = static_cast<uint32_t>(mono_ns() / 1000u) ^ (static_cast<uint32_t>(getpid()) << 16); if(g_ring.id == 0) g_ring.id = 1;
  RingHead rh; std::memset(&rh, 0, sizeof rh); std::memcpy(rh.magic, "LWR1", 4); rh.version = 1; rh.ring_id = g_ring.id; rh.nslots = RING_SLOTS; rh.stride = stride; rh.payload = payload;
  rh.width = static_cast<uint32_t>(w); rh.height = static_cast<uint32_t>(h); rh.pid = static_cast<uint64_t>(getpid());
  std::memcpy(g_ring.base, &rh, sizeof rh);
  std::fprintf(stderr, "SHM_RING on %s slots=%u picture=%u bytes id=%u\n", name, RING_SLOTS, payload, g_ring.id);
  return true;
}

static bool send_stamped(const unsigned short* px, int w, int h, uint64_t t_ns)   // header (+ picture, when it is not in the ring). false: write error or stop requested
{
  static uint64_t seq = 0, t_prev = 0;
  FrameHead hd; std::memset(&hd, 0, sizeof hd); std::memcpy(hd.magic, "LWF1", 4); hd.version = 1; hd.seq = ++seq; hd.t_mono_ns = t_ns;
  const uint64_t d = t_prev ? (t_ns - t_prev) / 1000u : 0; hd.dt_us = d > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<uint32_t>(d); t_prev = t_ns;
  hd.payload = static_cast<uint32_t>(w) * static_cast<uint32_t>(h) * 2u; hd.width = static_cast<uint16_t>(w); hd.height = static_cast<uint16_t>(h); hd.slot = 0xFFFFFFFFu;
  const bool in_ring = g_ring.base != nullptr && g_ring.payload == hd.payload;
  if(in_ring)
  {
    unsigned char* s = g_ring.base + sizeof(RingHead) + static_cast<size_t>(g_ring.next) * g_ring.stride;
    uint64_t* sq = reinterpret_cast<uint64_t*>(s);
    __atomic_store_n(sq, static_cast<uint64_t>(0), __ATOMIC_SEQ_CST);   // "being written": a reader in the middle of this slot sees its number gone
    std::memcpy(s + SLOT_HEAD, px, hd.payload);
    __atomic_store_n(sq, hd.seq, __ATOMIC_SEQ_CST);
    hd.flags = 1; hd.slot = g_ring.next; hd.ring_id = g_ring.id; g_ring.next = (g_ring.next + 1) % RING_SLOTS;
  }
  uint32_t wd[11]; std::memcpy(wd, &hd, sizeof wd); uint32_t x = 0x5A17C0DEu; for(const uint32_t v : wd) x ^= v; hd.check = x;
  if(!write_all(&hd, sizeof hd)) return false;
  return in_ring || write_all(px, hd.payload);
}

static const double STALL_S = 6.0;          // no frame for this long while the stream was running: engine reset
static const double FIRST_STALL_S = 30.0;   // no first frame this long after an open: engine reset (a healthy open gives its first frame within a few seconds)
static const double RESET_WAIT_S = 20.0;    // time given to the engine to reboot before the next scan/open
static const int MAX_FAILED_RECOVERIES = 4;

struct UsbDiag { bool have = false; int op = 0, result = 0; };   // v2.12: the last USB diagnostic, kept for the frozen-camera rule
static UsbDiag diag(const char* where)
{
  UsbDiag r; i3::USB_DIAGNOSTIC d{};
  if(i3::GetLastUsbDiagnostic(&d))
  {
    r.have = true; r.op = d.operation; r.result = d.result;
    std::fprintf(stderr, "DIAG %s: op=%d(%s) result=%d(%s)\n", where, d.operation, i3::GetUsbOperationName(d.operation), d.result, i3::GetUsbResultName(d.result));
  }
  return r;
}

static std::string hms()   // board wall clock, to match this log with the tracker and kernel logs
{
  char b[16]; std::time_t t = std::time(nullptr); std::tm tmv; localtime_r(&t, &tmv); std::strftime(b, sizeof b, "%H:%M:%S", &tmv); return b;
}

// v2.12 (11-15): the camera's USB chip does not answer = a control or send transfer timed out (libusb_timeout = -7, the raw libusb code the SDK passes on).
// A receive timeout (op 5) is the old soft stall that ResetMainBoard repaired; libusb_no_device (-4) is the kernel dropping the device, which comes back by itself.
static const int USB_TIMEOUT = -7;
static bool chip_mute(const UsbDiag& d)
{
  return d.have && d.result == USB_TIMEOUT && (d.op == i3::I3_USB_OPERATION_CONTROL || d.op == i3::I3_USB_OPERATION_SEND);
}
static const char* op_word(int op) { return op == i3::I3_USB_OPERATION_CONTROL ? "control" : op == i3::I3_USB_OPERATION_SEND ? "send" : "other"; }
static const int EXIT_FROZEN = 8;
static int camera_frozen(i3::TE_A*& te, const std::string& reason)   // closes the handle, prints the CAMERA_FROZEN line, returns the exit code
{
  if(te) { te->CloseTE(); te = nullptr; }
  std::fprintf(stderr, "CAMERA_FROZEN %s: the camera's USB chip does not answer; only a power cut (unplug ~10 s, or the 5 V step on a switchable port) clears it, at %s\n",
               reason.c_str(), hms().c_str());
  return EXIT_FROZEN;
}

static std::string exe_dir()
{
  char b[1024]; const ssize_t n = readlink("/proc/self/exe", b, sizeof b - 1);
  if(n <= 0) return ".";
  b[n] = 0; const std::string s(b); const size_t p = s.rfind('/'); return p == std::string::npos ? "." : s.substr(0, p);
}

// v2.13 (11-15): the state file (calibration reuse, open count, one blank-picture reset per camera power-on). See the top of this file.
static std::string home_dir() { return std::getenv("HOME") ? std::getenv("HOME") : "/tmp"; }
static double boot_s() { timespec ts; clock_gettime(CLOCK_BOOTTIME, &ts); return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9; }
static std::string read_word(const std::string& path)   // the first word of a small file, "" = cannot be read
{
  FILE* f = std::fopen(path.c_str(), "r"); if(!f) return "";
  char b[160] = {0}; const int n = std::fscanf(f, "%159s", b); std::fclose(f); return n == 1 ? b : "";
}
static std::string boot_id() { return read_word("/proc/sys/kernel/random/boot_id"); }
static std::string usb_entry()   // "<sysfs name>:<busnum>:<devnum>" of the camera (0547:0080; the first by name), "" = not found. Reads only idVendor/idProduct/busnum/devnum
{
  const char* e = std::getenv("LWIR_CAM_SYSFS"); const std::string root = e && *e ? e : "/sys/bus/usb/devices";
  DIR* d = opendir(root.c_str()); if(!d) return "";
  std::vector<std::string> hits;
  while(dirent* de = readdir(d))
  {
    const std::string n = de->d_name; if(n.empty() || n[0] == '.') continue;
    const std::string p = root + "/" + n + "/";
    if(read_word(p + "idVendor") != "0547" || read_word(p + "idProduct") != "0080") continue;
    const std::string b = read_word(p + "busnum"), v = read_word(p + "devnum");
    if(!b.empty() && !v.empty()) hits.push_back(n + ":" + b + ":" + v);
  }
  closedir(d);
  if(hits.empty()) return "";
  std::sort(hits.begin(), hits.end()); return hits[0];
}
static long env_long(const char* name, long dflt, long lo, long hi)   // a whole number lo..hi from the environment, else dflt
{
  const char* e = std::getenv(name); if(!e || !*e) return dflt;
  char* end = nullptr; const long v = std::strtol(e, &end, 10);
  return end != e && *end == 0 && v >= lo && v <= hi ? v : dflt;
}
typedef std::map<std::string, std::string> KV;
static std::string state_path() { return home_dir() + "/.cam_pipe_state"; }
static KV state_read()
{
  KV kv; FILE* f = std::fopen(state_path().c_str(), "r"); if(!f) return kv;
  char line[2048];
  while(std::fgets(line, sizeof line, f))
  {
    std::string s(line); while(!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    const size_t sp = s.find(' '); if(sp == std::string::npos || sp == 0) continue;
    kv[s.substr(0, sp)] = s.substr(sp + 1);
  }
  std::fclose(f); return kv;
}
static void state_write(const KV& kv)   // never fails loudly: without the file every open just calibrates (the old behaviour)
{
  const std::string p = state_path(), t = p + ".tmp";
  FILE* f = std::fopen(t.c_str(), "w"); if(!f) return;
  for(const auto& x : kv) std::fprintf(f, "%s %s\n", x.first.c_str(), x.second.c_str());
  const bool ok = std::fflush(f) == 0; std::fclose(f);
  if(ok) std::rename(t.c_str(), p.c_str()); else unlink(t.c_str());
}
static double kv_num(const KV& kv, const char* key, double dflt) { const auto it = kv.find(key); if(it == kv.end() || it->second.empty()) return dflt; char* end = nullptr; const double v = std::strtod(it->second.c_str(), &end); return end != it->second.c_str() ? v : dflt; }
static std::string kv_str(const KV& kv, const char* key) { const auto it = kv.find(key); return it == kv.end() ? "" : it->second; }
static std::string num_s(double v) { char b[32]; std::snprintf(b, sizeof b, "%.1f", v); return b; }
static std::string parent_text()   // "<name> (pid N): <command line>" of the program that started this cam_pipe
{
  const pid_t pp = getppid(); char p[64]; std::snprintf(p, sizeof p, "/proc/%d/", static_cast<int>(pp));
  std::string name = read_word(std::string(p) + "comm"), cmd;
  FILE* f = std::fopen((std::string(p) + "cmdline").c_str(), "r");
  if(f) { char b[240] = {0}; const size_t n = std::fread(b, 1, sizeof b - 1, f); std::fclose(f); for(size_t i = 0; i < n; ++i) cmd += b[i] ? b[i] : ' '; }
  while(!cmd.empty() && cmd.back() == ' ') cmd.pop_back();
  char b[64]; std::snprintf(b, sizeof b, " (pid %d)", static_cast<int>(pp));
  return (name.empty() ? std::string("?") : name) + b + (cmd.empty() ? "" : ": " + cmd);
}
static KV g_state;                 // read at the first open, written at every change
static std::string g_boot, g_usb;  // this boot, the camera's USB entry at the last open
static int g_opens_here = 0;       // opens in this process
static const double CAL_SETTLE_S = 5.0;   // a calibration counts (state file) once frames ran this long after it
static const double CLOSE_WAIT_S = 2.5;   // no CloseTE sooner than this after a calibration of ours (the shutter is shut about 1.4 s, 40 repeated frames)

static unsigned g_prod = 0, g_core = 0;   // from the last scan: product (1 Q1, 2 V1, 3 EQ1, 4 EV1, 9 M1, 12 EQ2, 13 EV2) and core id
static int scan_device(int attempts)
{
  for(int attempt = 0; attempt < attempts && !g_stop; ++attempt)
  {
    i3::SCANINFO devs[MAX_USB_NUM];
    if(ScanTE(devs) == 1) for(int i = 0; i < MAX_USB_NUM; ++i) if(devs[i].bDevCon == 1) { g_prod = devs[i].nProdVer; g_core = devs[i].nCoreID; return i; }
    nap(1.0);
  }
  return -1;
}

// Shutter setting, read at every open: env CAM_SHUTTER, else the first word of cam_shutter.txt next to this program, else d.
//   d = TEMP shutter limit 1 + calibration at the open (the team runtime's setting, the behaviour up to v2.2), tN = TEMP limit N (1..10), TN = TIME shutter every N*30 s (1..120),
//   m = MANUAL mode (the engine never moves the shutter by itself), mN = MANUAL + our own ShutterCalibrationOn() every N*30 s (1..120), n = no shutter call at all (engine defaults).
//   A trailing x (dx, t5x, T120x, mx, m20x) = no ShutterCalibrationOn() at the open. Anything not understood = d.
struct ShutterSet { std::string text, source; int mode = I3_TEMP_SHUTTER, value = 1; bool cal_at_open = true, info = false; double own_period_s = 0; };
static ShutterSet read_shutter_setting()
{
  ShutterSet r; std::string s = "d"; r.source = "default";
  const char* env = std::getenv("CAM_SHUTTER");
  if(env && *env) { s = env; r.source = "env"; }
  else
  {
    FILE* f = std::fopen((exe_dir() + "/cam_shutter.txt").c_str(), "r");
    if(f)
    {
      char a[64] = {0}, b[64] = {0}; const int n = std::fscanf(f, "%63s %63s", a, b); std::fclose(f);
      if(n >= 1 && a[0] != '#') { s = a; r.source = "file"; r.info = n >= 2 && std::string(b) == "info"; }
    }
  }
  r.text = s;
  if(s.size() > 1 && s[s.size() - 1] == 'x') { r.cal_at_open = false; s.erase(s.size() - 1); }
  bool digits = s.size() > 1; for(size_t i = 1; i < s.size(); ++i) if(!std::isdigit(static_cast<unsigned char>(s[i]))) digits = false;
  const int v = digits ? std::atoi(s.c_str() + 1) : -1;
  if(s == "n") { r.mode = -2; r.value = -1; r.cal_at_open = false; }
  else if(s == "d") { r.mode = I3_TEMP_SHUTTER; r.value = 1; }
  else if(s[0] == 't' && v >= I3_SHUTTER_TEMP_MIN && v <= I3_SHUTTER_TEMP_MAX) { r.mode = I3_TEMP_SHUTTER; r.value = v; }
  else if(s[0] == 'T' && v >= I3_SHUTTER_TIME_MIN && v <= I3_SHUTTER_TIME_MAX) { r.mode = I3_TIME_SHUTTER; r.value = v; }
  else if(s == "m") { r.mode = I3_MANUAL_SHUTTER; r.value = 0; }
  else if(s[0] == 'm' && v >= 1 && v <= 120) { r.mode = I3_MANUAL_SHUTTER; r.value = v; r.own_period_s = v * 30.0; }
  else { r.source += " (\"" + r.text + "\" not understood: using d)"; r.text = "d"; r.mode = I3_TEMP_SHUTTER; r.value = 1; r.cal_at_open = true; }
  return r;
}

static ShutterSet g_shutter;                 // the setting used at the last open
static clk::time_point g_t_cal = clk::now();  // last ShutterCalibrationOn() of ours (or the open, when none was done)
static bool g_cal_done = false, g_info_done = false; static int g_own_cals = 0;
static bool g_cal_pending = false;            // v2.13: a calibration of ours was accepted and is not yet in the state file (it counts after CAL_SETTLE_S of frames)
static double g_cal_pending_up = 0;           // its time since boot

// v2.13 (a): may this open skip ShutterCalibrationOn? true = reuse; *why = the reason either way (for the CALIBRATION line)
static bool cal_reuse(const ShutterSet& s, std::string* why)
{
  const long reuse_s = env_long("LWIR_CAM_CAL_REUSE_S", 1800, 0, 86400);
  if(!s.cal_at_open || s.mode == -2) { *why = "the shutter setting asks for no calibration at the open"; return false; }
  if(reuse_s == 0) { *why = "LWIR_CAM_CAL_REUSE_S=0: calibration at every open"; return false; }
  if(g_opens_here > 1) { *why = "reopen after an engine reset"; return false; }
  if(g_usb.empty()) { *why = "the camera's USB entry cannot be read, so the same power-on cannot be told"; return false; }
  const double cal = kv_num(g_state, "cal_up", -1), now = boot_s(), age = now - cal;
  if(g_boot.empty() || kv_str(g_state, "cal_boot") != g_boot || cal < 0) { *why = "first calibration of ours since the board started"; return false; }
  if(kv_str(g_state, "cal_usb") != g_usb) { *why = "the camera was plugged in or powered again since the last calibration (USB " + kv_str(g_state, "cal_usb") + " -> " + g_usb + ")"; return false; }
  if(kv_str(g_state, "cal_setting") != s.text) { *why = "the shutter setting changed (" + kv_str(g_state, "cal_setting") + " -> " + s.text + ")"; return false; }
  if(kv_str(g_state, "reset_boot") == g_boot && kv_num(g_state, "reset_up", -1) >= cal) { *why = "an engine reset (ResetMainBoard) was sent after the last calibration"; return false; }
  if(age < 0 || age >= static_cast<double>(reuse_s)) { *why = "the last calibration of ours was " + num_s(age) + " s ago (reuse limit " + std::to_string(reuse_s) + " s)"; return false; }
  *why = "the last calibration of ours was " + num_s(age) + " s ago on this camera power-on (USB " + g_usb + "), setting " + s.text + ", no engine reset since (reuse limit "
         + std::to_string(reuse_s) + " s, LWIR_CAM_CAL_REUSE_S)";
  return true;
}

static void cal_commit()   // v2.13: the pending calibration ran CAL_SETTLE_S with frames: it goes into the state file
{
  if(!g_cal_pending) return;
  g_cal_pending = false;
  g_state["cal_up"] = num_s(g_cal_pending_up); g_state["cal_usb"] = g_usb; g_state["cal_setting"] = g_shutter.text; g_state["cal_boot"] = g_boot;
  state_write(g_state);
}

static void note_reset(bool sent)   // v2.13: an engine reset that was sent makes the calibration on record useless for reuse
{
  g_cal_pending = false;
  if(!sent) return;
  g_state["reset_up"] = num_s(boot_s()); g_state["reset_boot"] = g_boot; state_write(g_state);
}

static void apply_shutter(i3::TE_A* te)
{
  const ShutterSet& s = g_shutter; int rm = -1, rv = -1, rc = -1;   // -1 = not called, 1 = the engine accepted it, 0 = refused, -2 = reused (v2.13)
  std::string why; const bool reuse = cal_reuse(s, &why);
  if(s.mode != -2)
  {
    rm = te->SetShutterMode(static_cast<unsigned short>(s.mode)) ? 1 : 0;
    if(s.mode == I3_TEMP_SHUTTER) rv = te->SetShutterTemperature(static_cast<unsigned short>(s.value)) ? 1 : 0;
    else if(s.mode == I3_TIME_SHUTTER) rv = te->SetShutterTime(static_cast<unsigned short>(s.value)) ? 1 : 0;
    if(s.cal_at_open && !reuse) rc = te->ShutterCalibrationOn() ? 1 : 0;
    else if(reuse) rc = -2;
  }
  g_t_cal = clk::now(); g_cal_done = rc >= 0;
  if(rc == 1) { g_cal_pending = true; g_cal_pending_up = boot_s(); }
  std::fprintf(stderr, "CALIBRATION %s: %s\n", reuse ? "REUSED (no ShutterCalibrationOn at this open)" : rc == 1 ? "at this open" : rc == 0 ? "at this open REFUSED by the engine" : "none at this open", why.c_str());
  std::fprintf(stderr, "SHUTTER setting=%s (%s) mode=%s value=%d set_mode=%d set_value=%d calibration_at_open=%d own_timer=%.0f s at %s\n", s.text.c_str(), s.source.c_str(),
               s.mode == -2 ? "untouched" : s.mode == I3_TEMP_SHUTTER ? "TEMP" : s.mode == I3_TIME_SHUTTER ? "TIME" : "MANUAL", s.value, rm, rv, rc, s.own_period_s, hms().c_str());
}

static void note_open()   // v2.13 (b): this open into the state file (opens of this camera power-on, last hour) and the OPENS line
{
  if(g_opens_here == 1) { g_state = state_read(); g_boot = boot_id(); }
  g_usb = usb_entry();
  const double now = boot_s();
  const bool same = kv_str(g_state, "boot") == g_boot && !g_usb.empty() && kv_str(g_state, "usb") == g_usb;   // same camera power-on as the opens on record
  std::vector<double> t; std::string keep;
  if(same)
  {
    const std::string s = kv_str(g_state, "opens"); size_t a = 0;
    while(a < s.size()) { size_t b = s.find(',', a); if(b == std::string::npos) b = s.size(); const double v = std::atof(s.substr(a, b - a).c_str()); if(v > 0 && now - v < 3600 && now >= v) t.push_back(v); a = b + 1; }
  }
  t.push_back(now);
  int n10 = 0; for(const double v : t) { if(now - v < 600) ++n10; if(!keep.empty()) keep += ","; keep += num_s(v); }
  g_state["boot"] = g_boot; g_state["usb"] = g_usb; g_state["opens"] = keep;
  state_write(g_state);
  std::fprintf(stderr, "OPENS this camera power-on (USB %s): %d in the last 10 min, %zu in the last hour; open %d of this cam_pipe, started by %s\n",
               g_usb.empty() ? "not found" : g_usb.c_str(), n10, t.size(), g_opens_here, parent_text().c_str());
}

static UsbDiag g_open_diag;   // v2.12: the DIAG of the last OpenTE_A that failed (have = false: the last open_camera did not fail in OpenTE_A)
static i3::TE_A* open_camera(int* w, int* h)
{
  g_open_diag = UsbDiag();
  const int idx = scan_device(10);
  if(idx < 0) { std::fprintf(stderr, "NO_CAMERA_FOUND\n"); return nullptr; }
  auto* te = i3::OpenTE_A(idx);
  if(!te) { g_open_diag = diag("open"); std::fprintf(stderr, "OPEN_FAILED\n"); return nullptr; }
  *w = te->GetImageWidth(); *h = te->GetImageHeight();
  std::fprintf(stderr, "CAMERA %d %d\n", *w, *h);
  std::fprintf(stderr, "ENGINE product=%u core=%u at %s\n", g_prod, g_core, hms().c_str());
  ++g_opens_here; note_open();   // v2.13 (b): state file, OPENS line
  g_shutter = read_shutter_setting();
  if(g_shutter.info && !g_info_done)   // asked for in cam_shutter.txt: what the engine itself has stored (once per process, before the first frame)
  {
    g_info_done = true; i3::TE_SETTING st{}; te->GetSetting(&st);
    std::fprintf(stderr, "ENGINE_SETTING frameRate=%u exportable=%u shutterMode=%u shutterTime=%u shutterTemp=%u highTempMode=%u\n", st.frameRate, st.exportable, st.shutterMode, st.shutterTime, st.shutterTemp, st.highTempMode);
  }
  apply_shutter(te);
  return te;
}

// v2.11 (11-14): optional temperature maps (LWIR_CAM_TEMP=N, see the top of this file for the layout). Nothing below runs unless temp_setup found LWIR_CAM_TEMP set to 1..300.
struct TempRingHead { char magic[4]; uint32_t version, ring_id, nslots, stride, payload, width, height; uint64_t pid; uint32_t every, state; float emissivity; uint32_t max_ms, maps, unused; };
static_assert(sizeof(TempRingHead) == 64 && offsetof(TempRingHead, pid) == 32 && offsetof(TempRingHead, every) == 40 && offsetof(TempRingHead, emissivity) == 48
              && offsetof(TempRingHead, maps) == 56, "temperature ring header layout (src/camera/temp_ring.py RING_HDR)");
struct TempSlotHead { uint64_t frame; char magic[4]; uint16_t version, flags; uint64_t t_mono_ns; uint32_t calc_us; uint16_t width, height; float emissivity; uint32_t map_seq, payload, ring_id; uint16_t t_offset, t_div; unsigned char unused[12]; };
static_assert(sizeof(TempSlotHead) == 64 && offsetof(TempSlotHead, t_mono_ns) == 16 && offsetof(TempSlotHead, calc_us) == 24 && offsetof(TempSlotHead, emissivity) == 32
              && offsetof(TempSlotHead, ring_id) == 44 && offsetof(TempSlotHead, t_offset) == 48, "temperature slot header layout (src/camera/temp_ring.py SLOT_HDR)");
static const uint32_t TEMP_SLOTS = 4, TEMP_STATE_ON = 1, TEMP_STATE_STOPPED = 2, TEMP_STATE_ENDED = 3;
static const uint16_t TEMP_FLAG_AGC_READ = 1;   // slot flags bit 0: the read before CalcTemp had the SDK's AGC on (the only kind in v2.11)
static const double TEMP_STATS_S = 30.0;        // TEMP_STATS period (each line goes into camera_events.log)
struct TempRing { unsigned char* base = nullptr; size_t bytes = 0; uint32_t id = 0, stride = 0, payload = 0, maps = 0; std::string path; };
static TempRing g_tring;
struct TempOpt
{
  bool on = false, started = false, emis_set = false;   // on = maps are being made now; started = the ring was made in this run (TEMP_TOTAL, INPUT_GAP note)
  long every = 0, max_ms = 50, calls = 0, over = 0, win_calls = 0; int slow_in_row = 0; float emis = 0;
  double sum_ms = 0, top_ms = 0, win_sum = 0, win_top = 0; clk::time_point t_win = clk::now(); std::vector<unsigned short> map;
};
static TempOpt g_temp;

static TempRingHead* temp_head() { return reinterpret_cast<TempRingHead*>(g_tring.base); }

static void temp_ring_close()   // state 3 (for a reader that still has it mapped), unmap, unlink (normal end)
{
  if(g_tring.base) { __atomic_store_n(&temp_head()->state, TEMP_STATE_ENDED, __ATOMIC_SEQ_CST); munmap(g_tring.base, g_tring.bytes); g_tring.base = nullptr; }
  if(!g_tring.path.empty()) { unlink(g_tring.path.c_str()); g_tring.path.clear(); }
}

static bool temp_ring_open(const std::string& name, int w, int h)   // false = no maps in this run (the line saying why is printed here)
{
  const uint32_t payload = static_cast<uint32_t>(w) * static_cast<uint32_t>(h) * 2u, stride = (static_cast<uint32_t>(sizeof(TempSlotHead)) + payload + 63u) & ~63u;
  const size_t bytes = sizeof(TempRingHead) + static_cast<size_t>(TEMP_SLOTS) * stride;
  unlink(name.c_str());   // a ring left by a cam_pipe that was killed (only one cam_pipe runs: the lock in main)
  const int fd = open(name.c_str(), O_CREAT | O_EXCL | O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0600);
  if(fd < 0) { std::fprintf(stderr, "TEMP_RING failed (open %s: %s): no temperature maps\n", name.c_str(), std::strerror(errno)); return false; }
  const int fe = posix_fallocate(fd, 0, static_cast<off_t>(bytes));
  if(fe != 0) { std::fprintf(stderr, "TEMP_RING failed (no room for %zu bytes in %s: %s): no temperature maps\n", bytes, name.c_str(), std::strerror(fe)); close(fd); unlink(name.c_str()); return false; }
  void* m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if(m == MAP_FAILED) { std::fprintf(stderr, "TEMP_RING failed (mmap %s: %s): no temperature maps\n", name.c_str(), std::strerror(errno)); unlink(name.c_str()); return false; }
  std::memset(m, 0, bytes);
  g_tring.base = static_cast<unsigned char*>(m); g_tring.bytes = bytes; g_tring.stride = stride; g_tring.payload = payload; g_tring.maps = 0; g_tring.path = name;
  g_tring.id = (static_cast<uint32_t>(mono_ns() / 1000u) ^ (static_cast<uint32_t>(getpid()) << 16)) + 1u; if(g_tring.id == 0) g_tring.id = 1;
  TempRingHead rh; std::memset(&rh, 0, sizeof rh); std::memcpy(rh.magic, "LWT1", 4); rh.version = 1; rh.ring_id = g_tring.id; rh.nslots = TEMP_SLOTS; rh.stride = stride; rh.payload = payload;
  rh.width = static_cast<uint32_t>(w); rh.height = static_cast<uint32_t>(h); rh.pid = static_cast<uint64_t>(getpid()); rh.every = static_cast<uint32_t>(g_temp.every); rh.state = TEMP_STATE_ON;
  rh.emissivity = g_temp.emis_set ? g_temp.emis : 0.0f; rh.max_ms = static_cast<uint32_t>(g_temp.max_ms);
  std::memcpy(g_tring.base, &rh, sizeof rh);
  return true;
}

static void temp_ring_put(const unsigned short* map, int w, int h, uint64_t frame, uint64_t t_ns, uint32_t calc_us, uint16_t flags)   // map k = maps + 1 into slot maps % TEMP_SLOTS
{
  const uint32_t payload = static_cast<uint32_t>(w) * static_cast<uint32_t>(h) * 2u;
  if(!g_tring.base || payload != g_tring.payload || frame == 0) return;
  unsigned char* s = g_tring.base + sizeof(TempRingHead) + static_cast<size_t>(g_tring.maps % TEMP_SLOTS) * g_tring.stride;
  uint64_t* fr = reinterpret_cast<uint64_t*>(s);
  __atomic_store_n(fr, static_cast<uint64_t>(0), __ATOMIC_SEQ_CST);   // "being written"
  __atomic_thread_fence(__ATOMIC_SEQ_CST);                            // the 0 is visible before any byte below (on arm64 a store alone does not hold LATER plain stores back)
  TempSlotHead sh; std::memset(&sh, 0, sizeof sh); std::memcpy(sh.magic, "LWTS", 4); sh.version = 1; sh.flags = flags; sh.t_mono_ns = t_ns; sh.calc_us = calc_us;
  sh.width = static_cast<uint16_t>(w); sh.height = static_cast<uint16_t>(h); sh.emissivity = g_temp.emis_set ? g_temp.emis : 0.0f; sh.map_seq = g_tring.maps + 1u;
  sh.payload = payload; sh.ring_id = g_tring.id; sh.t_offset = 5000; sh.t_div = 100;
  std::memcpy(s + sizeof(uint64_t), reinterpret_cast<const unsigned char*>(&sh) + sizeof(uint64_t), sizeof sh - sizeof(uint64_t));   // all of the slot header but the frame number
  std::memcpy(s + sizeof sh, map, payload);
  __atomic_store_n(fr, frame, __ATOMIC_RELEASE);                      // after the map: everything above is visible before the number
  ++g_tring.maps; __atomic_store_n(&temp_head()->maps, g_tring.maps, __ATOMIC_RELEASE);
}

static void temp_off(const char* why)   // the maps stop for the rest of this run; the pictures go on
{
  g_temp.on = false;
  if(g_tring.base) __atomic_store_n(&temp_head()->state, TEMP_STATE_STOPPED, __ATOMIC_SEQ_CST);
  std::fprintf(stderr, "TEMP_OFF %s: no more temperature maps in this run (the pictures go on), after %ld CalcTemp calls, at %s\n", why, g_temp.calls, hms().c_str());
}

static void temp_setup(i3::TE_A* te, int w, int h)   // reads LWIR_CAM_TEMP*. Unset, empty or 0: returns at once and prints nothing
{
  const char* e = std::getenv("LWIR_CAM_TEMP");
  if(!e || !*e || std::strcmp(e, "0") == 0) return;
  char* end = nullptr; const long n = std::strtol(e, &end, 10);
  if(end == e || *end != 0 || n < 1 || n > 300) { std::fprintf(stderr, "TEMP_RING not used: LWIR_CAM_TEMP=\"%s\" is not a whole number 1..300 (0 or unset = off)\n", e); return; }
  g_temp.every = n; std::string notes;
  if(const char* m = std::getenv("LWIR_CAM_TEMP_MAX_MS"))
  {
    char* me = nullptr; const long v = std::strtol(m, &me, 10);
    if(me != m && *me == 0 && v >= 1 && v <= 10000) g_temp.max_ms = v; else notes += std::string("; LWIR_CAM_TEMP_MAX_MS=\"") + m + "\" not understood (1..10000): 50 ms";
  }
  if(const char* em = std::getenv("LWIR_CAM_TEMP_EMIS"))
  {
    char* ee = nullptr; const double v = std::strtod(em, &ee);
    if(ee != em && *ee == 0 && v >= 0.01 && v <= 1.0) { g_temp.emis = static_cast<float>(v); g_temp.emis_set = true; }
    else notes += std::string("; LWIR_CAM_TEMP_EMIS=\"") + em + "\" not understood (0.01..1.0): SetEmissivity not called";
  }
  char name[64]; std::snprintf(name, sizeof name, "/dev/shm/lwir_temp_ring_%u", static_cast<unsigned>(getuid()));
  if(!temp_ring_open(name, w, h)) return;
  if(g_temp.emis_set) te->SetEmissivity(g_temp.emis);
  g_temp.map.assign(static_cast<size_t>(w) * static_cast<size_t>(h), 0); g_temp.on = g_temp.started = true; g_temp.t_win = clk::now();
  char eb[64];
  if(g_temp.emis_set) std::snprintf(eb, sizeof eb, "%.3f (SetEmissivity called)", static_cast<double>(g_temp.emis)); else std::snprintf(eb, sizeof eb, "camera default (SetEmissivity not called)");
  std::fprintf(stderr, "TEMP_RING on %s slots=%u map=%u bytes every=%ld frames emissivity=%s max_ms=%ld (off after more than 3 slower calls in a row) id=%u; CalcTemp after the AGC read, T_C = (v - 5000) / 100%s\n",
               name, TEMP_SLOTS, g_tring.payload, g_temp.every, eb, g_temp.max_ms, g_tring.id, notes.c_str());
}

static void temp_after_reopen(i3::TE_A* te, int w, int h)   // a new camera handle after the stall repair
{
  if(!g_temp.on) return;
  g_temp.slow_in_row = 0;
  if(static_cast<uint32_t>(w) * static_cast<uint32_t>(h) * 2u != g_tring.payload) { temp_off("the picture size changed at the reopen"); return; }
  if(g_temp.emis_set) { te->SetEmissivity(g_temp.emis); std::fprintf(stderr, "TEMP_EMIS %.3f set again after the reopen\n", static_cast<double>(g_temp.emis)); }
}

static double temp_take(i3::TE_A* te, int w, int h, uint64_t frame, uint64_t t_ns)   // CalcTemp for the frame just read and passed on, map into the ring. Returns the CalcTemp time in ms
{
  if(g_temp.map.size() != static_cast<size_t>(w) * static_cast<size_t>(h)) { temp_off("the map buffer does not fit the picture"); return 0; }
  const auto t = clk::now(); te->CalcTemp(g_temp.map.data()); const double ms = std::chrono::duration<double, std::milli>(clk::now() - t).count();
  const double us = ms * 1000.0 + 0.5;
  temp_ring_put(g_temp.map.data(), w, h, frame, t_ns, us >= 4294967295.0 ? 0xFFFFFFFFu : static_cast<uint32_t>(us), TEMP_FLAG_AGC_READ);
  ++g_temp.calls; ++g_temp.win_calls; g_temp.sum_ms += ms; g_temp.win_sum += ms; g_temp.top_ms = std::max(g_temp.top_ms, ms); g_temp.win_top = std::max(g_temp.win_top, ms);
  if(ms > static_cast<double>(g_temp.max_ms))
  {
    ++g_temp.over;
    if(++g_temp.slow_in_row > 3) { char b[128]; std::snprintf(b, sizeof b, "CalcTemp took more than %ld ms %d times in a row (last %.0f ms)", g_temp.max_ms, g_temp.slow_in_row, ms); temp_off(b); }
  }
  else g_temp.slow_in_row = 0;
  return ms;
}

static void temp_stats()   // from the CAMFPS block: one TEMP_STATS line every TEMP_STATS_S while the maps are on
{
  if(!g_temp.on) return;
  const double s = since(g_temp.t_win);
  if(s < TEMP_STATS_S) return;
  std::fprintf(stderr, "TEMP_STATS %ld CalcTemp calls in %.0f s, calc_ms mean %.1f max %.1f (whole run: %ld calls, mean %.1f max %.1f, %ld over %ld ms; maps in the ring %u), at %s\n",
               g_temp.win_calls, s, g_temp.win_calls ? g_temp.win_sum / static_cast<double>(g_temp.win_calls) : 0.0, g_temp.win_top, g_temp.calls,
               g_temp.calls ? g_temp.sum_ms / static_cast<double>(g_temp.calls) : 0.0, g_temp.top_ms, g_temp.over, g_temp.max_ms, g_tring.maps, hms().c_str());
  g_temp.win_calls = 0; g_temp.win_sum = 0; g_temp.win_top = 0; g_temp.t_win = clk::now();
}

// v2.13 (c): NO SCENE (see the top of this file). centre_same = share of the most common value in the middle half of the picture, every 2nd pixel.
static double centre_same(const unsigned short* px, int w, int h)
{
  static std::vector<unsigned short> v; v.clear();
  for(int y = h / 4; y < 3 * h / 4; y += 2) for(int x = w / 4; x < 3 * w / 4; x += 2) v.push_back(px[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)]);
  if(v.size() < 2) return 0;
  std::sort(v.begin(), v.end());
  size_t best = 1, run = 1;
  for(size_t i = 1; i < v.size(); ++i) { run = v[i] == v[i - 1] ? run + 1 : 1; if(run > best) best = run; }
  return static_cast<double>(best) / static_cast<double>(v.size());
}
static const double SCENE_LOST_AT = 0.5, SCENE_BACK_BELOW = 0.2, SCENE_BACK_S = 5.0, SCENE_LINE_S = 600.0, SCENE_CHECK_S = 1.0;
struct SceneWatch
{
  long hold_s = 20; bool may_reset = true;
  bool bad = false, good = false, lost = false, announced = false;   // bad / good = a run of checks above SCENE_LOST_AT / below SCENE_BACK_BELOW is going on
  clk::time_point bad_since = clk::now(), good_since = clk::now(), t_check = clk::now(), t_line = clk::now(), t_lost_line = clk::now(), lost_at = clk::now();
  double win_max = 0, last = 0; long checks = 0;
};
static SceneWatch g_scene;
static void scene_restart()   // after a reopen: a new run of checks (an announced loss stays announced until SCENE_BACK)
{
  g_scene.bad = g_scene.good = g_scene.lost = false; g_scene.t_check = clk::now();
}
// One check (every SCENE_CHECK_S, on the frame just passed on). Returns true = do the one engine reset now (the caller runs the stall repair steps).
static bool scene_check(const unsigned short* px, int w, int h, bool no_recover)
{
  const auto now = clk::now(); g_scene.t_check = now;
  const double s = centre_same(px, w, h); g_scene.last = s; ++g_scene.checks; g_scene.win_max = std::max(g_scene.win_max, s);
  if(std::chrono::duration<double>(now - g_scene.t_line).count() >= SCENE_LINE_S)
  {
    std::fprintf(stderr, "SCENE_CHECK %ld checks in %.0f s, centre_same largest %.3f, now %.3f (no scene at %.2f for %ld s), at %s\n", g_scene.checks,
                 std::chrono::duration<double>(now - g_scene.t_line).count(), g_scene.win_max, s, SCENE_LOST_AT, g_scene.hold_s, hms().c_str());
    g_scene.checks = 0; g_scene.win_max = 0; g_scene.t_line = now;
  }
  if(s >= SCENE_LOST_AT) { g_scene.good = false; if(!g_scene.bad) { g_scene.bad = true; g_scene.bad_since = now; } }
  else
  {
    g_scene.bad = false;
    if(s < SCENE_BACK_BELOW) { if(!g_scene.good) { g_scene.good = true; g_scene.good_since = now; } }
    else g_scene.good = false;
  }
  if(g_scene.good && std::chrono::duration<double>(now - g_scene.good_since).count() >= SCENE_BACK_S)
  {
    g_scene.lost = false;
    if(g_scene.announced)
    {
      g_scene.announced = false;
      std::fprintf(stderr, "SCENE_BACK the picture has a scene again (centre_same %.3f for %.0f s), %.0f s after SCENE_LOST, at %s\n", s, SCENE_BACK_S,
                   std::chrono::duration<double>(now - g_scene.lost_at).count(), hms().c_str());
    }
    return false;
  }
  if(g_scene.lost)
  {
    if(std::chrono::duration<double>(now - g_scene.t_lost_line).count() >= SCENE_LINE_S)
    {
      g_scene.t_lost_line = now;
      std::fprintf(stderr, "SCENE_LOST still no scene (centre_same %.3f), %.0f s after it began; no automatic repair left: unplug the camera USB cable for 10 s and plug it back, at %s\n",
                   s, std::chrono::duration<double>(now - g_scene.lost_at).count(), hms().c_str());
    }
    return false;
  }
  if(!g_scene.bad || std::chrono::duration<double>(now - g_scene.bad_since).count() < static_cast<double>(g_scene.hold_s)) return false;
  g_scene.lost = true; g_scene.t_lost_line = now; if(!g_scene.announced) g_scene.lost_at = g_scene.bad_since;
  g_scene.announced = true;
  // one engine reset per camera power-on: the same boot and USB entry as the last one on record means it was already tried
  const bool tried = kv_str(g_state, "blank_boot") == g_boot && kv_str(g_state, "blank_usb") == g_usb;
  char head[200]; std::snprintf(head, sizeof head, "SCENE_LOST no scene in the picture for %ld s (centre_same %.3f: the middle of the picture is one value; seen 11-15 16:38 when the shutter stayed shut)",
                                g_scene.hold_s, s);
  if(no_recover || !g_scene.may_reset || tried)
  {
    std::fprintf(stderr, "%s; no automatic repair %s: unplug the camera USB cable for 10 s and plug it back, at %s\n", head,
                 no_recover ? "(CAM_NO_RECOVER)" : !g_scene.may_reset ? "(LWIR_CAM_BLANK_RESET=0)" : ("left (the one engine reset of this camera power-on was at " + num_s(kv_num(g_state, "blank_up", 0)) + " s after boot)").c_str(),
                 hms().c_str());
    return false;
  }
  g_state["blank_up"] = num_s(boot_s()); g_state["blank_boot"] = g_boot; g_state["blank_usb"] = g_usb; state_write(g_state);   // before the reset: a cam_pipe killed in the middle does not repeat it
  std::fprintf(stderr, "%s; one engine reset now (ResetMainBoard, close, %.0f s, reopen with a calibration), the only one on this camera power-on (USB %s), at %s\n",
               head, RESET_WAIT_S, g_usb.empty() ? "not found" : g_usb.c_str(), hms().c_str());
  return true;
}

// v2.14 (11-15): READ PACING ON A GRID THAT FOLLOWS THE CAMERA. See the top of this file for what was measured.
static const double PACE_LATE_US = 7200.0, PACE_EARLY_US = 11000.0;   // measured 11-15: how far a read may start after / before the place where the stream was picked up (the first read after an empty one)
static const long PACE_SKIP_US = 6000;          // a read that would start more than this after its grid place (a slow send, no CPU) waits for the next place instead: one picture lost, the place in the camera's frame kept
static const long PACE_CENTRE_CHECK_READS = 100; // an empty read sooner than this after the move to the resting place: the move was wrong, no more moves in this process
static const long PACE_SETTLE_READS = 30;       // reads with a frame after a pick-up before the grid is moved to its resting place (LWIR_CAM_PACE_CENTRE_US); an empty read sooner than this belongs to the same pick-up
static const long PACE_LEARN_MIN_READS = 600;   // an empty read sooner than this after the pick-up (20 s) teaches nothing about the period
static const double PACE_LEARN_GAIN = 0.8, PACE_LEARN_STEP_NS = 5000.0, PACE_TRIM_MAX_NS = 10000.0;   // one lesson changes the period by at most 5 us, all lessons together by at most 10 us (300 ppm)
static const double PACE_OPEN_S = 5.0;          // an empty read sooner than this after an open belongs to the shutter calibration there: the place after it is not taken as known
static const double READ_STATS_S = 300.0;       // READ_STATS period (each line goes into camera_events.log)
static const long READ_LOG_MAX_LINES = 4000000; // LWIR_CAM_READ_LOG closes itself after this many reads (about 37 hours, about 170 MB)
struct Pace
{
  long period_us = 33333;      // LWIR_CAM_READ_PERIOD_US; 0 = no pacing
  int mode = 1;                // LWIR_CAM_PACE_MODE: 1 = grid (v2.14), 0 = after (v2.7 to v2.13)
  bool learn = true;           // LWIR_CAM_PACE_TRIM: the period follows the camera's own frame period
  long centre_us = -4000;      // LWIR_CAM_PACE_CENTRE_US: where the grid rests relative to the pick-up place (negative = earlier)
  long rephase_us = 0;         // LWIR_CAM_PACE_REPHASE_US (experiment, 0 = off): at the first repeated frame move the grid by this much at once
  long on_time_us = 250;       // LWIR_CAM_PACE_ON_TIME_US: an empty read teaches only when it and the read before it started within this of their plan
  double trim_ns = 0;          // learned: period in use = period_us + trim_ns
  clk::time_point plan;        // grid: the planned start of the last paced read
  bool have_plan = false;
  bool anchored = false;       // this grid began at a pick-up (the first read with a frame after 1 or 2 empty reads), so pos_us is known
  double pos_us = 0;           // where the grid is now relative to the pick-up place, not counting the slow drift
  long grid_reads = 0;         // reads with a frame on this grid
  bool centred = false, rephased = false;
  int empties = 0;             // empty reads in a row just now
  long rep_run = 0;            // identical pictures in a row just now
  long prev_late_us = -1;      // how late the read before started (-1 = not paced)
  bool pend = false, pend_late = false, pend_on_time = false, pend_anchored = false; long pend_reads = 0; double pend_pos = 0;   // the empty read that ended the last grid: what it can teach
  // READ_STATS
  clk::time_point st_t0 = clk::now(), st_last_new = clk::now(); bool st_have_new = false;
  long st_reads = 0, st_new = 0, st_rep = 0, st_empty = 0, st_early = 0, st_late = 0, st_late_n = 0, st_late_max = 0, st_regrid = 0; double st_gap_ms = 0, st_late_sum = 0;
  // test hooks
  std::string ctl_path; time_t ctl_sec = 0; long ctl_nsec = 0; clk::time_point ctl_checked = clk::now();   // LWIR_CAM_PACE_FILE
  FILE* log = nullptr; long log_lines = 0; clk::time_point log_t0 = clk::now();                             // LWIR_CAM_READ_LOG
};
static Pace g_pace;

static double pace_period_us() { return static_cast<double>(g_pace.period_us) + g_pace.trim_ns / 1000.0; }
static void pace_line(const std::string& why)
{
  if(g_pace.mode == 1)
    std::fprintf(stderr, "READ_PACING period=%ld us (0 = off) mode=grid, period in use %.3f us (follows the camera: %s), rests %ld us from the pick-up place, move at a repeat %ld us%s%s\n", g_pace.period_us,
                 pace_period_us(), g_pace.learn ? "yes" : "no", g_pace.centre_us, g_pace.rephase_us, why.empty() ? "" : "; ", why.c_str());
  else std::fprintf(stderr, "READ_PACING period=%ld us (0 = off) mode=after (v2.7: counted from the actual start of the last read)%s%s\n", g_pace.period_us, why.empty() ? "" : "; ", why.c_str());
}
static void pace_new_grid() { g_pace.have_plan = false; g_pace.anchored = false; g_pace.pos_us = 0; g_pace.grid_reads = 0; g_pace.centred = false; g_pace.rephased = false; }
// LWIR_CAM_PACE_FILE (a test hook, off unless set): a small text file with any of "period_us=<n> mode=<after|grid> shift_us=<n> trim=<0|1> centre_us=<n> rephase_us=<n>",
// read again when it changes (looked at once a second), so that settings can be compared on ONE camera open. shift_us moves the grid once by that much; log=off closes LWIR_CAM_READ_LOG.
static void pace_poll()
{
  if(g_pace.ctl_path.empty() || since(g_pace.ctl_checked) < 1.0) return;
  g_pace.ctl_checked = clk::now();
  struct stat st;
  if(stat(g_pace.ctl_path.c_str(), &st) != 0 || (st.st_mtim.tv_sec == g_pace.ctl_sec && st.st_mtim.tv_nsec == g_pace.ctl_nsec)) return;
  g_pace.ctl_sec = st.st_mtim.tv_sec; g_pace.ctl_nsec = st.st_mtim.tv_nsec;
  FILE* f = std::fopen(g_pace.ctl_path.c_str(), "r"); if(!f) return;
  char text[256]; const size_t n = std::fread(text, 1, sizeof text - 1, f); std::fclose(f); text[n] = 0;
  long shift = 0; bool any = false;
  const auto num = [](const char* tok, const char* key, long lo, long hi, long* out) -> bool
  {
    const size_t k = std::strlen(key); if(std::strncmp(tok, key, k)) return false;
    char* end = nullptr; const long v = std::strtol(tok + k, &end, 10);
    if(end == tok + k || *end != 0 || v < lo || v > hi) return false;
    *out = v; return true;
  };
  for(char* tok = std::strtok(text, " \t\r\n"); tok; tok = std::strtok(nullptr, " \t\r\n"))
  {
    long v = 0;
    if(num(tok, "period_us=", 0, 1000000, &v)) { g_pace.period_us = v; g_pace.trim_ns = 0; pace_new_grid(); any = true; }
    else if(!std::strcmp(tok, "mode=after")) { g_pace.mode = 0; pace_new_grid(); any = true; }
    else if(!std::strcmp(tok, "mode=grid")) { g_pace.mode = 1; pace_new_grid(); any = true; }
    else if(num(tok, "shift_us=", -1000000, 1000000, &v)) { shift = v; any = true; }
    else if(num(tok, "trim=", 0, 1, &v)) { g_pace.learn = v == 1; if(!g_pace.learn) g_pace.trim_ns = 0; any = true; }
    else if(num(tok, "centre_us=", -10000, 6000, &v)) { g_pace.centre_us = v; any = true; }
    else if(num(tok, "rephase_us=", -30000, 0, &v)) { g_pace.rephase_us = v; any = true; }
    else if(!std::strcmp(tok, "log=off")) { if(g_pace.log) { std::fclose(g_pace.log); g_pace.log = nullptr; std::fprintf(stderr, "READ_LOG closed after %ld lines (settings file), at %s%c", g_pace.log_lines, hms().c_str(), 10); } }
  }
  if(!any) return;
  if(shift && g_pace.mode == 1 && g_pace.have_plan) { g_pace.plan += std::chrono::microseconds(shift); g_pace.anchored = false; }   // moved by hand: the place is no longer known, nothing is learned from this grid
  pace_line("from " + g_pace.ctl_path + " (shift " + std::to_string(shift) + " us) at " + hms());
}
// when the next read should start, after a read with a frame that started at last_start
static clk::time_point pace_target(clk::time_point last_start)
{
  if(g_pace.mode != 1) return last_start + std::chrono::microseconds(g_pace.period_us);
  const auto per = std::chrono::nanoseconds(static_cast<long long>(g_pace.period_us) * 1000 + static_cast<long long>(g_pace.trim_ns));
  if(!g_pace.have_plan) { g_pace.plan = last_start; g_pace.have_plan = true; }
  g_pace.plan += per;
  const auto now = clk::now();   // measured 11-15: a read exactly one period late gets a new picture like any other, so a place is skipped, not moved
  while(now - g_pace.plan > std::chrono::microseconds(PACE_SKIP_US)) { g_pace.plan += per; ++g_pace.st_regrid; }
  return g_pace.plan;
}
// after every RecvImage. frame = it returned a picture, repeat = that picture is the one before again, late_us = how late it started (-1 = not paced)
static void pace_after(bool frame, bool repeat, double open_age_s, long late_us)
{
  Pace& p = g_pace;
  if(p.period_us <= 0) return;
  if(!frame)
  {
    if(p.empties == 0)   // the read that ended a grid. Repeats right before it = the reads came too late (the period too long); none = too early (too short)
    {
      p.pend = p.grid_reads >= PACE_SETTLE_READS; p.pend_late = p.rep_run > 0; p.pend_on_time = late_us <= p.on_time_us && p.prev_late_us <= p.on_time_us; p.pend_anchored = p.anchored; p.pend_reads = p.grid_reads; p.pend_pos = p.pos_us;
      if(p.centred && p.centre_us != 0 && p.grid_reads < PACE_SETTLE_READS + PACE_CENTRE_CHECK_READS)   // the resting place is not where it was measured to be
      {
        std::fprintf(stderr, "PACE_CENTRE_OFF an empty read %ld reads after the grid moved %ld us to its resting place: no such move any more in this process, at %s\n",
                     p.grid_reads - PACE_SETTLE_READS, p.centre_us, hms().c_str());
        p.centre_us = 0;
      }
    }
    ++p.empties; p.rep_run = 0; p.prev_late_us = late_us; pace_new_grid();
    return;
  }
  if(p.empties > 0)   // the stream was picked up again by this read
  {
    const bool clean = p.empties <= 2;   // more empty reads than that is a camera stall, not a read that came too early or too late
    if(p.pend && clean)
    {
      if(p.pend_late) ++p.st_late; else if(p.pend_on_time) ++p.st_early;
      const double before = pace_period_us();
      const bool lesson = p.mode == 1 && p.learn && p.pend_anchored && p.pend_on_time && p.pend_reads >= PACE_LEARN_MIN_READS;
      if(lesson)
      {
        const double travelled_us = p.pend_late ? PACE_LATE_US - p.pend_pos : -(PACE_EARLY_US + p.pend_pos);   // + = the reads slid later
        double step_ns = -PACE_LEARN_GAIN * travelled_us * 1000.0 / static_cast<double>(p.pend_reads);
        step_ns = std::max(-PACE_LEARN_STEP_NS, std::min(PACE_LEARN_STEP_NS, step_ns));
        p.trim_ns = std::max(-PACE_TRIM_MAX_NS, std::min(PACE_TRIM_MAX_NS, p.trim_ns + step_ns));
        g_state["pace_trim_ns"] = num_s(p.trim_ns); g_state["pace_period_us"] = std::to_string(p.period_us); state_write(g_state);   // the next start begins with it
      }
      if(p.mode == 1) std::fprintf(stderr, "PACE_RESYNC %s after %ld reads on this grid; read period %.3f -> %.3f us%s, at %s\n",
                   p.pend_late ? "reads came too late (repeated pictures, then an empty read)" : p.pend_on_time ? "reads came too early (an empty read)" : "an empty read after a late start",
                   p.pend_reads, before, pace_period_us(),
                   lesson ? "" : !p.learn ? " (not following the camera)" : !p.pend_on_time ? " (nothing learned: a late start can land anywhere)" : " (nothing learned: the grid was too young or its place unknown)", hms().c_str());
    }
    p.pend = false; p.anchored = clean && p.mode == 1 && open_age_s > PACE_OPEN_S; p.pos_us = 0; p.grid_reads = 0; p.centred = false; p.rephased = false; p.empties = 0;
  }
  ++p.grid_reads;
  const long rep_before = p.rep_run;
  if(repeat) ++p.rep_run; else { p.rep_run = 0; p.rephased = false; }
  if(p.anchored && !p.centred && p.grid_reads >= PACE_SETTLE_READS && !repeat && p.have_plan)   // the stream is steady again: go to the resting place
  {
    p.plan += std::chrono::microseconds(p.centre_us); p.pos_us += static_cast<double>(p.centre_us); p.centred = true;
  }
  p.prev_late_us = late_us;
  if(repeat && p.rephase_us != 0 && !p.rephased && p.have_plan && rep_before == 0 && p.grid_reads > PACE_SETTLE_READS && open_age_s > PACE_OPEN_S)   // experiment: do not wait for the empty read
  {
    p.plan += std::chrono::microseconds(p.rephase_us); p.rephased = true; p.anchored = false;
    std::fprintf(stderr, "PACE_REPHASE a repeated picture after %ld reads on this grid: the grid moves %ld us, at %s\n", p.grid_reads, p.rephase_us, hms().c_str());
  }
}
// LWIR_CAM_READ_LOG (off unless set): one line per RecvImage call: start in ms since the log began, call time in ms, return code, kind (0 new, 1 repeat held back,
// 2 repeat passed on, - no frame), how late the read started after its plan in us (-1 = not paced), the period in use in us and the mode (1 grid, 0 after).
static void read_note(clk::time_point start, clk::time_point end, int r, int kind, long late_us)
{
  Pace& p = g_pace;
  ++p.st_reads;
  if(kind == 0)
  {
    ++p.st_new;
    if(p.st_have_new) p.st_gap_ms = std::max(p.st_gap_ms, std::chrono::duration<double, std::milli>(end - p.st_last_new).count());
    p.st_last_new = end; p.st_have_new = true;
  }
  else if(kind > 0) ++p.st_rep; else ++p.st_empty;
  if(late_us >= 0) { ++p.st_late_n; p.st_late_sum += static_cast<double>(late_us); p.st_late_max = std::max(p.st_late_max, late_us); }
  if(!p.log) return;
  std::fprintf(p.log, "%.3f %.3f %d %c %ld %.3f %d\n", std::chrono::duration<double, std::milli>(start - p.log_t0).count(),
               std::chrono::duration<double, std::milli>(end - start).count(), r, kind < 0 ? '-' : static_cast<char>('0' + kind), late_us, pace_period_us(), p.mode);
  if(++p.log_lines % 30 == 0) std::fflush(p.log);
  if(p.log_lines >= READ_LOG_MAX_LINES) { std::fclose(p.log); p.log = nullptr; std::fprintf(stderr, "READ_LOG full after %ld lines: closed\n", p.log_lines); }
}
static void read_stats(bool at_end)   // one READ_STATS line every READ_STATS_S, and one at the end
{
  Pace& p = g_pace;
  const double dt = since(p.st_t0);
  if((!at_end && dt < READ_STATS_S) || p.st_reads == 0) return;
  std::fprintf(stderr, "READ_STATS %.0f s: reads %ld, new pictures %ld (%.2f a second), repeats %ld, empty reads %ld, longest time without a new picture %.0f ms, reads too early %ld, too late %ld, "
               "grid places skipped after a late start %ld, period in use %.3f us (mode %s), reads started %.0f us after their plan on average (largest %ld), at %s\n", dt, p.st_reads, p.st_new, p.st_new / std::max(dt, 1e-9),
               p.st_rep, p.st_empty, p.st_gap_ms, p.st_early, p.st_late, p.st_regrid, p.mode == 1 ? pace_period_us() : static_cast<double>(p.period_us), p.mode == 1 ? "grid" : "after",
               p.st_late_n ? p.st_late_sum / static_cast<double>(p.st_late_n) : 0.0, p.st_late_max, hms().c_str());
  p.st_t0 = clk::now(); p.st_reads = p.st_new = p.st_rep = p.st_empty = p.st_early = p.st_late = p.st_late_n = p.st_late_max = p.st_regrid = 0; p.st_gap_ms = 0; p.st_late_sum = 0;
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
  g_stamp = env_is_1("LWIR_CAM_STAMP");   // v2.6: off unless set to exactly 1 (see the top of this file)
  if(g_stamp) { std::fprintf(stderr, "STAMP on: a 48-byte header goes before every frame\n"); if(env_is_1("LWIR_CAM_SHM")) ring_open(w, h); }
  else if(env_is_1("LWIR_CAM_SHM")) std::fprintf(stderr, "SHM_RING not used: LWIR_CAM_SHM=1 needs LWIR_CAM_STAMP=1\n");
  temp_setup(te, w, h);   // v2.11: off (and silent) unless LWIR_CAM_TEMP is set (see the top of this file)
  uint64_t t_rx = 0;
  std::map<int, long> codes; long ok = 0;
  auto t0 = clk::now(), tl = t0; long okl = 0; double maxgap = 0; auto tprev = t0;
  auto t_open = t0, t_frame = t0; bool frame_since_open = false; int failed_recoveries = 0, recoveries = 0; int exit_code = 0; long ok_open = 0;
  const bool no_recover = std::getenv("CAM_NO_RECOVER") != nullptr, telemetry = std::getenv("CAM_TELEMETRY") != nullptr;   // experiment switches (11-09): no_recover = log the stall and exit with code 7; telemetry = FPA and shutter temperature in the CAMFPS line (read once per 2 s)
  const double test_stall_at = std::getenv("CAM_PIPE_TEST_STALL_AT") ? std::atof(std::getenv("CAM_PIPE_TEST_STALL_AT")) : 0; bool test_done = false;   // test hook: pretend the stream stopped once at this time (tests ResetMainBoard + reopen on a healthy camera)
  const bool test_frozen = test_stall_at > 0 && env_is_1("CAM_PIPE_TEST_FROZEN");   // v2.12 test hook: that simulated stall takes the frozen path (exit 8), see the top of this file
  g_scene.hold_s = env_long("LWIR_CAM_BLANK_S", 20, 0, 3600); g_scene.may_reset = env_long("LWIR_CAM_BLANK_RESET", 1, 0, 1) == 1;   // v2.13 (c)
  bool scene_reset_now = false;   // v2.13 (c): the no-scene check asks for its one engine reset (the stall repair steps below)
  std::fprintf(stderr, "SCENE_WATCH %s\n", g_scene.hold_s > 0 ? ("on: no scene = centre_same >= " + num_s(SCENE_LOST_AT) + " for " + std::to_string(g_scene.hold_s) + " s, then "
               + (g_scene.may_reset ? std::string("one engine reset per camera power-on") : std::string("lines only (LWIR_CAM_BLANK_RESET=0)"))).c_str() : "off (LWIR_CAM_BLANK_S=0)");
  // v2.7 (11-13): READ PACING. The engine sends 30 frames a second (GetSetting frameRate 0). RecvImage waits at most about 26 ms
  // for a frame and returns 4 when none completed in that time, and a read that takes a frame returns after 17 ms (no AGC) to
  // 27 ms (AGC). Calling it again at once therefore starts a second read in the middle of the frame period: that read times
  // out just before the next frame is complete, the frame is lost, and the loop falls into a take-one / miss-one rhythm
  // (gaps of about 25 and 52 ms, 22.8 fps measured over 30 s: 684 frames, 486 empty reads). Starting each read one frame
  // period (33.333 ms) after the START of the last successful read keeps the read window on the frame: 29.0 fps over 30 s
  // (871 frames, 22 empty reads), same pictures (AGC unchanged). After an empty read the loop does not wait, so it finds the
  // phase again by itself. LWIR_CAM_READ_PERIOD_US changes the period in microseconds (40000 for an engine set to 25 fps);
  // 0 = no pacing, the behaviour up to v2.6.
  // v2.14 (11-15): "after the START of the last successful read" let every wake-up delay add up (a read every 33.400 ms); the reads are now on a grid
  // (see the top of this file). LWIR_CAM_PACE_MODE=after gives this v2.7 pacing back.
  if(const char* e = std::getenv("LWIR_CAM_READ_PERIOD_US")) { char* end = nullptr; const long v = std::strtol(e, &end, 10); if(end != e && v >= 0 && v <= 1000000) g_pace.period_us = v; }
  {   // v2.14: the grid (see the top of this file)
    if(const char* e = std::getenv("LWIR_CAM_PACE_MODE")) g_pace.mode = !std::strcmp(e, "after") ? 0 : 1;
    g_pace.learn = env_long("LWIR_CAM_PACE_TRIM", 1, 0, 1) == 1;
    g_pace.centre_us = env_long("LWIR_CAM_PACE_CENTRE_US", -4000, -10000, 6000);
    g_pace.rephase_us = env_long("LWIR_CAM_PACE_REPHASE_US", 0, -30000, 0);
    g_pace.on_time_us = env_long("LWIR_CAM_PACE_ON_TIME_US", 250, 50, 5000);
    if(const char* e = std::getenv("LWIR_CAM_PACE_FILE")) g_pace.ctl_path = e;
    std::string why;
    if(g_pace.mode == 1 && g_pace.learn && kv_str(g_state, "pace_period_us") == std::to_string(g_pace.period_us))   // what earlier runs learned about this camera's frame period
    {
      g_pace.trim_ns = std::max(-PACE_TRIM_MAX_NS, std::min(PACE_TRIM_MAX_NS, kv_num(g_state, "pace_trim_ns", 0)));
      why = "period from the state file";
    }
    if(!g_pace.ctl_path.empty()) why += std::string(why.empty() ? "" : ", ") + "settings file on (LWIR_CAM_PACE_FILE)";
    pace_line(why);
    if(const char* e = std::getenv("LWIR_CAM_READ_LOG")) if(*e)
    {
      g_pace.log = std::fopen(e, "w"); g_pace.log_t0 = clk::now();
      std::fprintf(stderr, "READ_LOG %s %s\n", e, g_pace.log ? "on: one line per read (start ms, call ms, code, kind, late us, period us, mode)" : "cannot be written: off");
    }
  }
  bool last_read_ok = false; auto last_read_start = clk::now();
  // v2.8 (11-13): REPEATED FRAMES. The engine itself sends the same picture again, byte for byte, for 1 to 3 frames (seldom up to
  // 8) about once a second, and for about 40 frames (1.3 s) during the shutter calibration at an open. Measured with every read
  // logged (paced, 25-30 s each): repeats arrive on schedule exactly like new frames (same 33.5 ms spacing, same call time), with
  // the SDK's AGC on (9.7 %) and off (9.6 %), and in shutter modes TEMP, MANUAL (11.4 %) and TIME (7.8 %); without the long run
  // at the open it is 2-6 %. So it is not the read timing and cannot be removed here. The server takes an identical frame for a
  // frozen camera, logs it and gives the gimbal no people for that frame. Short repeats are therefore not sent on: the first
  // drop_dup_max repeats in a row are held back and the pacing just continues. A picture that stays the same longer is passed on
  // from then, so the server's frozen-picture handling and the stall repair see a real freeze as before.
  // LWIR_CAM_DROP_DUP sets the limit (0 = send every frame, the behaviour up to v2.7).
  // v2.14 (11-15): "not the read timing" was wrong. With every read logged over 30 minutes, each of these repeats came when the reads had slid 7.2 ms late in
  // the camera's frame (see the top of this file); on the grid they are gone. The filter stays for the repeats that are left: the shutter calibration at an
  // open, and a grid that reaches the late edge.
  long drop_dup_max = 3;
  if(const char* e = std::getenv("LWIR_CAM_DROP_DUP")) { char* end = nullptr; const long v = std::strtol(e, &end, 10); if(end != e && v >= 0 && v <= 1000) drop_dup_max = v; }
  std::vector<unsigned short> prev_frame; long dup_run = 0, dups_dropped = 0, dups_passed = 0;
  // v2.9 (11-13): WHEN THE PICTURE STOOD STILL, IN THE LOG. A run of repeats longer than the limit (the shutter is closed for a
  // calibration, about 40 frames) gets a REPEAT_RUN line when it ends, and a stall gets a STALL_REPEATS line saying where the
  // last such run was relative to the last frame. Two stalls so far (11-12 15:49, 11-13 14:45) came 279 s and 238 s after an
  // open with nothing in the log to tell whether the shutter had just closed; the next one will show it. Nothing else changes.
  bool rep_seen = false; long rep_len = 0; auto rep_start = clk::now(), rep_end = clk::now();
  // v2.10 (11-13): TIMES WITHOUT ANY FRAME, IN THE LOG. Now and then the camera input drops for a moment (15.7 fps in one
  // second at 15:09:33, longest gaps of 0.49 and 0.56 s in earlier runs) and nothing said why. Every time GAP_LOG_MS or more
  // passes between two frames taken from the engine, an INPUT_GAP line says how the time was spent: how many reads came back
  // empty (the engine had no frame: camera or USB), the longest single read (a read that hung), the longest wait between two
  // reads (this program was not given the CPU), how long the read that ended the gap took and how long sending the frame before it to the server took (a full pipe = the server was slow to take it), whether the frame after the gap is a repeat, and where the last long run of
  // repeats (shutter) was. Only log lines are added; frames, pacing and the duplicate filter are as in v2.8.
  const double GAP_LOG_MS = 150.0;
  bool gp_have = false; auto gp_last_ok = clk::now(), gp_prev_end = clk::now(); long gp_fail = 0, gp_empty = 0; double gp_max_call = 0, gp_max_wait = 0, gp_call = 0, gp_wait = 0, gp_write = 0;
  double gp_temp = 0;   // v2.11: CalcTemp time after the last frame passed on (0 = none), named in INPUT_GAP when the maps were started
  std::fprintf(stderr, "DUPLICATE_FILTER holds back up to %ld repeated frames in a row (0 = off)\n", drop_dup_max);
  while(!g_stop && since(t0) < seconds)
  {
    // v2.7 (11-13): start each read one frame period after the START of the last successful read (see the note above the loop).
    // v2.14: mode "after" is this same wait; mode "grid" counts from the PLANNED start, so waking up late does not add up (pace_target)
    pace_poll();
    long late_us = -1;
    if(last_read_ok && g_pace.period_us > 0)
    {
      const auto target = pace_target(last_read_start);
      std::this_thread::sleep_until(target);
      late_us = std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - target).count();
    }
    const auto read_start = clk::now();
    int r = te->RecvImage(buf.data(), true);
    const auto read_end = clk::now();
    {   // v2.10: what happens between two frames
      const double call_ms = std::chrono::duration<double, std::milli>(read_end - read_start).count();
      gp_call = call_ms; gp_wait = std::chrono::duration<double, std::milli>(read_start - gp_prev_end).count();   // this read, and the time since the read before it ended (sending that frame on + the pacing wait)
      if(gp_fail > 0) gp_max_wait = std::max(gp_max_wait, gp_wait);
      gp_prev_end = read_end;
      if(r != 1) { ++gp_fail; if(r == 4) ++gp_empty; gp_max_call = std::max(gp_max_call, call_ms); }
    }
    last_read_ok = (r == 1); if(last_read_ok) last_read_start = read_start;
    if(g_stamp || g_temp.on) t_rx = mono_ns();
    bool test_stall_now = false;   // v2.12: this read is the simulated stall
    if(test_stall_at > 0 && !test_done && since(t0) > test_stall_at) { test_done = true; test_stall_now = true; std::fputs("TEST: simulated stall", stderr); std::fputc(10, stderr); nap(STALL_S + 1.0); r = 4; }
    ++codes[r];
    if(r != 1) { read_note(read_start, read_end, r, -1, late_us); pace_after(false, false, since(t_open), late_us); }   // v2.14
    if(r == 1)
    {
      const bool identical = prev_frame.size() == buf.size() && std::memcmp(prev_frame.data(), buf.data(), buf.size() * sizeof(unsigned short)) == 0;   // v2.14: known also with the filter off (the grid needs it)
      const bool same = drop_dup_max > 0 && identical;
      read_note(read_start, read_end, r, identical ? (drop_dup_max > 0 && dup_run + 1 <= drop_dup_max ? 1 : 2) : 0, late_us);   // v2.14
      pace_after(true, identical, since(t_open), late_us);
      if(gp_have)   // v2.10
      {
        const double gap_ms = std::chrono::duration<double, std::milli>(read_end - gp_last_ok).count();
        if(gap_ms >= GAP_LOG_MS)
        {
          char repb[96] = "no long run of repeats yet";
          if(dup_run > 0) std::snprintf(repb, sizeof repb, "inside a run of repeats (%ld before the gap)", dup_run);
          else if(rep_seen) std::snprintf(repb, sizeof repb, "last long run of repeats ended %.1f s before", std::chrono::duration<double>(gp_last_ok - rep_end).count());
          char tmpb[48] = "";   // v2.11: empty unless the temperature maps were started
          if(g_temp.started) std::snprintf(tmpb, sizeof tmpb, ", CalcTemp after it %.0f ms", gp_temp);
          std::fprintf(stderr, "INPUT_GAP %.0f ms between frames: the read that brought this frame took %.0f ms and began %.0f ms after the read before it ended (sending the frame before to the server took %.0f ms%s); "
                       "reads without a frame in between %ld (%ld empty), longest of them %.0f ms, longest wait between them %.1f ms; frame after the gap is %s, %s, %.1f s after this open, at %s\n",
                       gap_ms, gp_call, gp_wait, gp_write, tmpb, gp_fail, gp_empty, gp_max_call, gp_max_wait, drop_dup_max > 0 ? (same ? "a repeat" : "new") : "not compared", repb, since(t_open), hms().c_str());
        }
      }
      gp_have = true; gp_last_ok = read_end; gp_fail = 0; gp_empty = 0; gp_max_call = 0; gp_max_wait = 0; gp_write = 0;
      gp_temp = 0;
      if(drop_dup_max > 0)
      {
        if(same) { ++dup_run; if(dup_run == 1) rep_start = clk::now(); if(dup_run <= drop_dup_max) { ++dups_dropped; continue; } ++dups_passed; }
        else
        {
          if(dup_run > drop_dup_max)
          {
            rep_seen = true; rep_len = dup_run; rep_end = clk::now();
            std::fprintf(stderr, "REPEAT_RUN %ld identical frames in a row (%.2f s), ended %.1f s after this open, at %s\n", dup_run, std::chrono::duration<double>(rep_end - rep_start).count(), since(t_open), hms().c_str());
          }
          dup_run = 0; prev_frame = buf;
        }
      }
      else if(g_pace.mode == 1 && !identical) prev_frame = buf;   // v2.14: filter off, grid on: the picture to compare the next one with
      auto now = clk::now();
      maxgap = std::max(maxgap, std::chrono::duration<double>(now - tprev).count()); tprev = now;
      t_frame = now; frame_since_open = true; failed_recoveries = 0;
      const auto gp_w0 = clk::now();
      if(g_stamp) { if(!send_stamped(buf.data(), w, h, t_rx)) break; }
      else
      if(!write_all(buf.data(), buf.size() * sizeof(unsigned short))) break;
      gp_write = std::chrono::duration<double, std::milli>(clk::now() - gp_w0).count();
      ++ok; ++okl;
      if(g_temp.on && ok % g_temp.every == 0) gp_temp = temp_take(te, w, h, static_cast<uint64_t>(ok), t_rx);   // v2.11: after the picture went out, before the next read (CalcTemp uses the last RecvImage)
      if(g_shutter.own_period_s > 0 && since(g_t_cal) >= g_shutter.own_period_s)   // v2.5 (setting mN): the engine is in MANUAL mode, the calibration is ours and it is in the log
      {
        const auto tc = clk::now(); const bool okc = te->ShutterCalibrationOn(); g_t_cal = clk::now(); g_cal_done = true; ++g_own_cals;
        if(okc) { g_cal_pending = true; g_cal_pending_up = boot_s(); }   // v2.13 (a): counts like a calibration at the open
        std::fprintf(stderr, "SHUTTER_CAL own timer #%d ok=%d took %.2f s, %ld frames and %.1f s after this open, at %s\n", g_own_cals, (int)okc, since(tc), ok - ok_open, since(t_open), hms().c_str());
      }
      if(g_cal_pending && since(g_t_cal) >= CAL_SETTLE_S) cal_commit();   // v2.13 (a): frames ran CAL_SETTLE_S after the calibration: it counts
      if(g_scene.hold_s > 0 && since(g_scene.t_check) >= SCENE_CHECK_S) scene_reset_now = scene_check(buf.data(), w, h, no_recover);   // v2.13 (c)
      double dt = std::chrono::duration<double>(now - tl).count();
      if(dt >= 2.0)
      {
        if(telemetry) std::fprintf(stderr, "CAMFPS %.1f t %.0f fpa %.2f shutter %.2f", okl / dt, since(t0), te->GetFpaTemp(), te->GetShutterPt100Temp());
        else std::fprintf(stderr, "CAMFPS %.1f", okl / dt);
        std::fputc(10, stderr); okl = 0; tl = now;
        temp_stats();   // v2.11: every 30 s while the maps are on, nothing otherwise
        read_stats(false);   // v2.14: every 5 min
      }
      if(!scene_reset_now) continue;
    }
    if(!scene_reset_now)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      const double idle = since(frame_since_open ? t_frame : t_open);
      if(idle < (frame_since_open ? STALL_S : FIRST_STALL_S)) continue;
      if(no_recover)   // experiment: measure how long the camera lives, do not repair it
      {
        std::fprintf(stderr, "CAMERA_STALL_NO_RECOVER after %ld frames, no frame for %.1f s, run time %.1f s, shutter=%s", ok, idle, since(t0), std::getenv("CAM_SHUTTER") ? std::getenv("CAM_SHUTTER") : "d");
        std::fputc(10, stderr); diag("stall"); exit_code = 7; break;
      }
    }
    // stalled (or v2.13: no scene): reboot the camera engine from software, then reopen
    ++recoveries;
    if(++failed_recoveries > MAX_FAILED_RECOVERIES) { std::fprintf(stderr, "CAMERA_UNRECOVERABLE after %d resets without a frame: power-cycle the camera or reboot the board\n", MAX_FAILED_RECOVERIES); exit_code = 6; break; }
    if(scene_reset_now) std::fprintf(stderr, "CAMERA_RESET_FOR_SCENE frames still come (%ld so far) but show no scene: ResetMainBoard #%d\n", ok, recoveries);
    else
    {
      const double idle = since(frame_since_open ? t_frame : t_open);
      std::fprintf(stderr, "CAMERA_STALL no frame for %.1f s after %ld frames (%s): ResetMainBoard #%d (failed in a row: %d)\n", idle, ok, frame_since_open ? "stream stopped" : "no first frame", recoveries, failed_recoveries);
      {   // v2.5: how long this open lived and where the last shutter call of ours was
        const double lived = frame_since_open ? std::chrono::duration<double>(t_frame - t_open).count() : 0.0;
        char calb[96] = "none"; const double cd = std::chrono::duration<double>(t_frame - g_t_cal).count();
        if(g_cal_done) { if(cd >= 0) std::snprintf(calb, sizeof calb, "%.1f s before the last frame", cd); else std::snprintf(calb, sizeof calb, "%.1f s after the last frame (no frame since it)", -cd); }
        std::fprintf(stderr, "STALL_INFO this open: %ld frames, last frame %.1f s after the open, shutter=%s, own calibrations %d, last shutter call of ours: %s, at %s\n", ok - ok_open, lived, g_shutter.text.c_str(), g_own_cals, frame_since_open ? calb : "n/a", hms().c_str());
        // v2.9: was the picture standing still (shutter closed) when the stream stopped?
        if(dup_run > 0) std::fprintf(stderr, "STALL_REPEATS the stream stopped DURING a run of identical frames: %ld so far, begun %.2f s before the last frame read\n", dup_run, std::chrono::duration<double>(last_read_start - rep_start).count());
        else if(rep_seen) std::fprintf(stderr, "STALL_REPEATS last long run of identical frames: %ld frames, ended %.1f s before the last frame\n", rep_len, std::chrono::duration<double>(t_frame - rep_end).count());
        else std::fprintf(stderr, "STALL_REPEATS no long run of identical frames since this program started\n");
      }
    }
    UsbDiag sd = diag(scene_reset_now ? "scene" : "stall");
    bool sent = false;
    if(test_stall_now && test_frozen)   // v2.12 test hook (CAM_PIPE_TEST_FROZEN=1): this stall is taken as frozen
    {
      sd.have = true; sd.op = i3::I3_USB_OPERATION_CONTROL; sd.result = USB_TIMEOUT;
      std::fprintf(stderr, "TEST: the simulated stall is taken as frozen (DIAG control libusb_timeout, ResetMainBoard not called)\n");
    }
    else
    {
      sent = te->ResetMainBoard();
      std::fprintf(stderr, "ResetMainBoard sent=%d\n", (int)sent);
      note_reset(sent);   // v2.13 (a): no calibration reuse across an engine reset
    }
    if(!sent && chip_mute(sd))   // v2.12 (a): the engine cannot be reached and the chip does not answer: a reopen would only time out again
    {
      exit_code = camera_frozen(te, "at a stall after " + std::to_string(ok) + " frames (" + op_word(sd.op) + " libusb_timeout, ResetMainBoard refused)");
      break;
    }
    te->CloseTE(); te = nullptr;
    nap(RESET_WAIT_S);
    if(g_stop) break;
    te = open_camera(&w, &h);
    if(!te)
    {
      if(ok > 0 && chip_mute(g_open_diag))   // v2.12 (b): the camera streamed in this process and now its chip does not answer the open
      {
        exit_code = camera_frozen(te, "at the reopen after a stall, after " + std::to_string(ok) + " frames (" + op_word(g_open_diag.op) + " libusb_timeout)");
        break;
      }
      std::fprintf(stderr, "REOPEN_FAILED\n"); exit_code = 3; break;
    }
    if(static_cast<size_t>(w) * h != buf.size()) buf.assign(static_cast<size_t>(w) * h, 0);
    temp_after_reopen(te, w, h);   // v2.11: emissivity again on the new handle; maps stop if the picture size changed
    t_open = clk::now(); frame_since_open = false; tprev = t_open; tl = t_open; okl = 0; ok_open = ok; gp_have = false; gp_fail = 0; gp_empty = 0; gp_max_call = 0; gp_max_wait = 0;
    scene_reset_now = false; scene_restart();   // v2.13 (c): checks start again on the new handle
    pace_new_grid(); g_pace.pend = false; g_pace.empties = std::max(g_pace.empties, 3);   // v2.14: a new handle: a new grid at its first frame, its place unknown, nothing learned from the stall
  }
  double secs = since(t0);
  std::fprintf(stderr, "DUPLICATE_FRAMES held back %ld, passed on %ld\n", dups_dropped, dups_passed);
  std::fprintf(stderr, "FRAMES %ld in %.2f s = %.2f fps, longest gap %.0f ms, resets %d, return codes:", ok, secs, ok / secs, maxgap * 1000, recoveries);
  for(auto& kv : codes) std::fprintf(stderr, " %d x%ld", kv.first, kv.second);
  std::fprintf(stderr, "\n");
  if(g_temp.started)   // v2.11
    std::fprintf(stderr, "TEMP_TOTAL %ld CalcTemp calls, calc_ms mean %.1f max %.1f, %ld over %ld ms, maps %u, %s\n", g_temp.calls, g_temp.calls ? g_temp.sum_ms / static_cast<double>(g_temp.calls) : 0.0,
                 g_temp.top_ms, g_temp.over, g_temp.max_ms, g_tring.maps, g_temp.on ? "on to the end" : "stopped earlier (TEMP_OFF)");
  read_stats(true);   // v2.14
  if(g_pace.log) { std::fclose(g_pace.log); g_pace.log = nullptr; }
  ring_close();
  temp_ring_close();
  if(te && g_cal_pending && since(g_t_cal) < CLOSE_WAIT_S)   // v2.13 (d): a stop signal right after a calibration of ours: let the shutter open again first
  {
    const double wait = CLOSE_WAIT_S - since(g_t_cal);
    std::fprintf(stderr, "CLOSE_WAIT the shutter calibration of this open began %.1f s ago: closing the camera in %.1f s, after it\n", since(g_t_cal), wait);
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<long>(wait * 1000.0) + 1));
  }
  if(te) te->CloseTE();
  return exit_code;
}
