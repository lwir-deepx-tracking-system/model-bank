// Connects the tracker to the team's gimbal and PC GUI code (team/, unchanged copy of the runtime repository, see team/SOURCE.txt).
// All control decisions, safety stops and serial output stay in the team's GimbalCommandExecutor; the PC wire format stays in the team's
// GuiSender / GuiReceiver / TrackingCommandHandler. This program only feeds them and reports their state.
//
// stdin, one line per message:
//   F <frame id> <capture time, CLOCK_MONOTONIC ns> <n> then n times: <id> <x> <y> <w> <h> <confidence>     people in camera image pixels (640x480), x y = top left
//   G <frame id> <capture time ns> <width> <height> <n> then n times: <id> <x> <y> <w> <h> <confidence>     same, followed (after the newline) by width*height*2 bytes:
//                  the 16-bit camera frame, row major, native byte order. It goes to the gimbal executor AND to the PC GUI (video + metadata)
//   P              heartbeat of the web page. The first one opens the session. Without one for --heartbeat-ms the executor stops the gimbal and drops the selection
//   A <id>         follow this ID
//   M              stop following, keep an ongoing manual move
//   V <x> <y>      manual move, each -1, 0 or 1 (+x right, +y up). V 0 0 = key released
//   H              stop following and stop moving
// stdout, one line per message:
//   READY
//   R <A|M> <OK|INVALID|NOT_READY>            answer to A or M
//   S <auto> <lost> <moving> <error> <id>     the executor's state, on every change
//   T <gui client 0|1> <frames given> <frames sent> <frames encoded> <rtp bytes> <send failures>   once a second when --gui-host is set
// stdin closed, SIGTERM or SIGINT: the executor sends STOP, then the program ends.
//
// PC GUI (the team's pc-application, branch feat/gui-video): --gui-host <PC address> turns it on. Video H.264/RTP to UDP --gui-video-port (5000),
// track metadata to UDP --gui-metadata-port (5001), the PC's commands on TCP --gui-bind:--gui-command-port (5002, 0 = no command listener).
// The GUI's AUTO/MANUAL/MOVE/HOME reach the same executor as the web page, so both show the same state. A GUI connection or disconnection
// drops the current selection (team behaviour); the web page re-opens its session with its next heartbeat.
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <future>
#include <map>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <poll.h>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <opencv2/core.hpp>

#include "common/logger.hpp"
#include "control/gimbal_command_executor.hpp"
#include "control/tracking_command_handler.hpp"
#include "app/pipeline_requests.hpp"
#include "gui/gui_command_handler.hpp"
#include "gui/gui_receiver.hpp"
#include "gui/gui_sender.hpp"

namespace {
volatile std::sig_atomic_t stop_signal = 0;
void on_signal(int) { stop_signal = 1; }

int protocol_fd = 1;
void say(const std::string& line)
{
    const std::string text = line + "\n";
    std::size_t sent = 0;
    while (sent < text.size()) {
        const auto count = ::write(protocol_fd, text.data() + sent, text.size() - sent);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return;   // the tracker is gone; stdin will close too
        sent += static_cast<std::size_t>(count);
    }
}

const char* answer(ControlResult result)
{
    return result == ControlResult::Ok ? "OK" : result == ControlResult::InvalidTarget ? "INVALID" : "NOT_READY";
}

std::string state_line(const ControlSnapshot& s)
{
    std::ostringstream out;
    out << "S " << int(s.automatic) << ' ' << int(s.lost) << ' ' << int(s.moving) << ' ' << int(s.error) << ' ' << s.selected_id;
    return out.str();
}

struct GuiOptions {
    bool enabled = false;
    GuiVideoConfig video;
    GuiMetadataConfig metadata;
    GuiCommandConfig command;
    std::uint16_t clip_min = 0, clip_max = 65535;   // the team's model settings: camera_input clip_min / clip_max (fixed 16 -> 8 bit mapping)
};

struct Options {
    ControlConfig control;
    GuiOptions gui;
};

// Aim point: fraction of the box height from its top that the gimbal brings to the image centre (0.5 = the team's box centre).
static double g_aim_y = 0.5;

// The executor gets boxes moved so that their centre is the aim point; height and width stay, so the team's checks see the same box size.
static TrackingResultPtr aimed(const TrackingResultPtr& result)
{
    if (!result || g_aim_y == 0.5) return result;
    auto copy = std::make_shared<TrackingResult>(*result);
    for (auto& t : copy->tracks) t.y += static_cast<float>((g_aim_y - 0.5) * t.height);
    return copy;
}

// The team's pipeline boundary for the PC GUI: model list size, model change. The tracker (Python) owns the models: a request goes to it as a line.
static std::uint32_t g_model_count = 0;
class ModelRequests final : public IPipelineRequests {
    std::mutex mutex_;
    std::map<unsigned, std::promise<PipelineResult>> waiting_;
    unsigned next_ = 1;
    bool busy_ = false;
public:
    std::uint32_t model_count() const override { return g_model_count; }
    std::uint32_t tracker_count() const override { return 1; }   // one tracker (ByteTrack + our gate/judge); nothing to switch
    PipelineChangeTicket request_model_change(std::uint32_t id) override
    {
        PipelineChangeTicket ticket;
        if (g_model_count == 0) { ticket.result = PipelineResult::NotReady; return ticket; }
        if (id < 1 || id > g_model_count) { ticket.result = PipelineResult::InvalidId; return ticket; }
        std::promise<PipelineResult> promise;
        auto future = promise.get_future().share();
        unsigned request = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (busy_) { ticket.result = PipelineResult::Busy; return ticket; }
            busy_ = true; request = next_++; waiting_.emplace(request, std::move(promise));
        }
        say("M " + std::to_string(request) + " " + std::to_string(id));
        ticket.result = PipelineResult::Ok; ticket.completion = future;
        return ticket;
    }
    PipelineChangeTicket request_tracker_change(std::uint32_t id) override
    {
        PipelineChangeTicket ticket;
        if (id != 1) { ticket.result = PipelineResult::InvalidId; return ticket; }
        std::promise<PipelineResult> promise; promise.set_value(PipelineResult::Ok);
        ticket.result = PipelineResult::Ok; ticket.completion = promise.get_future().share();
        return ticket;
    }
    void answer(unsigned request, int result)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = waiting_.find(request);
        if (it == waiting_.end()) return;
        it->second.set_value(static_cast<PipelineResult>(result < 0 || result > 6 ? 6 : result));
        waiting_.erase(it); busy_ = false;
    }
};
static ModelRequests g_models;

Options parse(int argc, char** argv)
{
    Options o;
    ControlConfig& c = o.control;
    GuiOptions& g = o.gui;
    c.enabled = true;
    c.gimbal.speed = 30;                    // the values in the team's board settings before tuning on the real gimbal
    c.automatic.speed_max = 30;
    c.automatic.min_confidence = 0.1;       // this tracker keeps a track on detections down to 0.1; the team default is 0.5
    g.video.port = 5000; g.video.codec = GuiVideoCodec::H264; g.video.encoder = "mpph264enc";   // config/runtime.yaml of the team
    g.video.bitrate_kbps = 3000; g.video.jpeg_quality = 80; g.video.keyframe_interval = 30; g.video.fps = 30; g.video.rtp_mtu = 1200;
    g.metadata.enabled = true; g.metadata.port = 5001; g.metadata.max_packet_bytes = 1200;
    g.command.handler = GuiCommandHandlerType::Tracking; g.command.enabled = true; g.command.bind_address = "0.0.0.0"; g.command.port = 5002; g.command.receive_timeout_ms = 1000;
    for (int i = 1; i < argc; ++i) {
        const std::string name = argv[i];
        if (i + 1 >= argc) throw std::invalid_argument("missing value of " + name);
        const std::string value = argv[++i];
        if (name == "--device") c.gimbal.device = value;
        else if (name == "--baud") c.gimbal.baud_rate = std::stoi(value);
        else if (name == "--manual-speed") c.gimbal.speed = std::stoi(value);
        else if (name == "--speed-max") c.automatic.speed_max = std::stoi(value);
        else if (name == "--speed-min") c.automatic.speed_min = std::stoi(value);
        else if (name == "--speed-rate") c.automatic.speed_rate = std::stod(value);
        else if (name == "--x-sign") c.gimbal.x_sign = std::stoi(value);
        else if (name == "--y-sign") c.gimbal.y_sign = std::stoi(value);
        else if (name == "--kp") c.automatic.pid_x.kp = c.automatic.pid_y.kp = std::stod(value);
        // v25: the team's PidController already has I and D terms (anti-windup, filtered derivative); the default stays P, as before
        else if (name == "--controller") {
            if (value == "p") c.automatic.controller = AutoControllerType::P;
            else if (value == "pi") c.automatic.controller = AutoControllerType::PI;
            else if (value == "pd") c.automatic.controller = AutoControllerType::PD;
            else if (value == "pid") c.automatic.controller = AutoControllerType::PID;
            else throw std::invalid_argument("--controller is p, pi, pd or pid");
        }
        else if (name == "--ki") { c.automatic.pid_x.ki = c.automatic.pid_y.ki = std::stod(value); if (!(c.automatic.pid_x.ki >= 0.0)) throw std::invalid_argument("--ki must be >= 0"); }
        else if (name == "--kd") { c.automatic.pid_x.kd = c.automatic.pid_y.kd = std::stod(value); if (!(c.automatic.pid_x.kd >= 0.0)) throw std::invalid_argument("--kd must be >= 0"); }
        else if (name == "--integral-limit") { c.automatic.pid_x.integral_limit = c.automatic.pid_y.integral_limit = std::stod(value); if (!(c.automatic.pid_x.integral_limit >= 0.0)) throw std::invalid_argument("--integral-limit must be >= 0"); }
        else if (name == "--derivative-filter-ms") { c.automatic.pid_x.derivative_filter_ms = c.automatic.pid_y.derivative_filter_ms = std::stod(value); if (!(c.automatic.pid_x.derivative_filter_ms > 0.0)) throw std::invalid_argument("--derivative-filter-ms must be > 0"); }
        else if (name == "--deadband") c.automatic.deadband_x = c.automatic.deadband_y = std::stod(value);
        else if (name == "--model-count") g_model_count = static_cast<std::uint32_t>(std::stoul(value));
        else if (name == "--min-confidence") c.automatic.min_confidence = std::stod(value);
        else if (name == "--aim-y") { g_aim_y = std::stod(value); if (!(g_aim_y >= 0.0 && g_aim_y <= 1.0)) throw std::invalid_argument("--aim-y must be within 0..1"); }
        else if (name == "--heartbeat-ms") c.heartbeat_timeout_ms = std::stoi(value);
        else if (name == "--input-timeout-ms") c.automatic.input_timeout_ms = std::stoi(value);
        else if (name == "--reacquire-ms") c.automatic.reacquire_timeout_ms = std::stoi(value);
        else if (name == "--gui-host") { g.enabled = true; g.video.host = value; g.metadata.host = value; }
        else if (name == "--gui-bind") g.command.bind_address = value;
        else if (name == "--gui-video-port") g.video.port = static_cast<std::uint16_t>(std::stoi(value));
        else if (name == "--gui-metadata-port") g.metadata.port = static_cast<std::uint16_t>(std::stoi(value));
        else if (name == "--gui-command-port") { g.command.port = static_cast<std::uint16_t>(std::stoi(value)); g.command.enabled = g.command.port != 0; }
        else if (name == "--gui-codec") {
            if (value == "h264") g.video.codec = GuiVideoCodec::H264;
            else if (value == "jpeg") g.video.codec = GuiVideoCodec::JPEG;
            else throw std::invalid_argument("--gui-codec is h264 or jpeg");
        }
        else if (name == "--gui-encoder") g.video.encoder = value;
        else if (name == "--gui-bitrate") g.video.bitrate_kbps = std::stoi(value);
        else if (name == "--gui-jpeg-quality") g.video.jpeg_quality = std::stoi(value);
        else if (name == "--gui-keyframe") g.video.keyframe_interval = std::stoi(value);
        else if (name == "--gui-fps") g.video.fps = std::stoi(value);
        else if (name == "--gui-mtu") g.video.rtp_mtu = std::stoi(value);
        else if (name == "--gui-max-packet") g.metadata.max_packet_bytes = static_cast<std::size_t>(std::stoul(value));
        else if (name == "--gui-clip") {
            const auto comma = value.find(',');
            if (comma == std::string::npos) throw std::invalid_argument("--gui-clip is <min>,<max>");
            g.clip_min = static_cast<std::uint16_t>(std::stoul(value.substr(0, comma)));
            g.clip_max = static_cast<std::uint16_t>(std::stoul(value.substr(comma + 1)));
        }
        else throw std::invalid_argument("unknown option " + name);
    }
    if (c.gimbal.device.empty()) throw std::invalid_argument("--device is required");
    if (c.gimbal.speed < 1 || c.gimbal.speed > c.gimbal.speed_limit || c.automatic.speed_max < c.automatic.speed_min || c.automatic.speed_max > c.gimbal.speed_limit)
        throw std::invalid_argument("speed outside the device range");
    if ((c.gimbal.x_sign != 1 && c.gimbal.x_sign != -1) || (c.gimbal.y_sign != 1 && c.gimbal.y_sign != -1))
        throw std::invalid_argument("axis sign must be 1 or -1");
    if (g.enabled) {
        if (g.video.fps <= 0 || g.video.port == 0 || g.metadata.port == 0) throw std::invalid_argument("GUI ports and fps must be positive");
        if (g.clip_min >= g.clip_max) throw std::invalid_argument("--gui-clip min must be below max");
        if (g.video.codec == GuiVideoCodec::H264 && g.video.bitrate_kbps <= 0) throw std::invalid_argument("--gui-bitrate must be positive");
    }
    return o;
}

// Sends frames + tracks to the PC on its own thread: the newest result only, so a slow encoder never delays the executor.
class GuiLink {
    GuiSender sender_;
    std::mutex mutex_;
    std::condition_variable wake_;
    TrackingResultPtr latest_;
    bool closing_ = false;
    std::thread worker_;
    std::atomic<std::uint64_t> given_{0}, sent_{0}, failed_{0};
    void run()
    {
        for (;;) {
            TrackingResultPtr result;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [this] { return closing_ || latest_; });
                if (closing_) return;
                result.swap(latest_);
            }
            const auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            if (sender_.send(*result, static_cast<std::uint64_t>(now_us))) ++sent_; else ++failed_;
        }
    }
public:
    explicit GuiLink(const GuiOptions& g) : sender_(g.video, g.metadata, g.clip_min, g.clip_max), worker_(&GuiLink::run, this) {}
    ~GuiLink() { close(); }
    void publish(TrackingResultPtr result)
    {
        ++given_;
        { std::lock_guard<std::mutex> lock(mutex_); latest_ = std::move(result); }
        wake_.notify_one();
    }
    void close()
    {
        { std::lock_guard<std::mutex> lock(mutex_); if (closing_) return; closing_ = true; }
        wake_.notify_one();
        worker_.join();
        sender_.stop();
    }
    std::string status(bool gui_client) const
    {
        const auto s = sender_.stats();
        std::ostringstream out;
        out << "T " << int(gui_client) << ' ' << given_.load() << ' ' << sent_.load() << ' ' << s.encoded_frames << ' ' << s.rtp_bytes << ' ' << failed_.load();
        return out.str();
    }
};

// The team's TCP command strategy, observed: this program needs to know when the GUI connects and leaves (the executor keeps one session).
class ObservedHandler final : public IGuiCommandHandler {
    std::unique_ptr<IGuiCommandHandler> inner_;
    std::atomic<bool>& client_;
    std::atomic<std::uint64_t>& changes_;
public:
    ObservedHandler(std::unique_ptr<IGuiCommandHandler> inner, std::atomic<bool>& client, std::atomic<std::uint64_t>& changes)
        : inner_(std::move(inner)), client_(client), changes_(changes) {}
    GuiCommand initial_state() const override { return inner_->initial_state(); }
    bool handle(const GuiCommand& c, std::vector<GuiCommand>& replies) override { return inner_->handle(c, replies); }
    void connected() override { inner_->connected(); client_ = true; ++changes_; }
    void disconnected() override { inner_->disconnected(); client_ = false; ++changes_; }
    void tick(std::vector<GuiCommand>& replies) override { inner_->tick(replies); }
    int poll_timeout_ms() const override { return inner_->poll_timeout_ms(); }
};

class Bridge {
    GimbalCommandExecutor& executor_;
    GuiLink* gui_;
    const std::atomic<std::uint64_t>& session_changes_;
    cv::Mat image_;              // frames without a picture: the executor reads only the type and the size of the frame
    bool session_ = false;
    std::uint64_t seen_changes_ = 0;
    // a G line waits for its picture
    std::shared_ptr<TrackingResult> waiting_;
    int wait_w_ = 0, wait_h_ = 0;
public:
    Bridge(GimbalCommandExecutor& executor, GuiLink* gui, const std::atomic<std::uint64_t>& session_changes)
        : executor_(executor), gui_(gui), session_changes_(session_changes), image_(cv::Mat::zeros(480, 640, CV_16UC1)) {}

    std::size_t picture_bytes() const { return waiting_ ? static_cast<std::size_t>(wait_w_) * wait_h_ * 2 : 0; }

    std::shared_ptr<TrackingResult> tracks(std::istringstream& in, std::uint64_t id, std::uint64_t captured_ns, std::size_t count)
    {
        auto context = std::make_shared<FrameContext>();
        context->metadata.frame_id = id;
        context->metadata.captured_at = std::chrono::steady_clock::time_point(std::chrono::nanoseconds(captured_ns));
        context->image = image_;
        auto result = std::make_shared<TrackingResult>();
        result->tracks.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            Track t;
            if (!(in >> t.track_id >> t.x >> t.y >> t.width >> t.height >> t.confidence)) return nullptr;   // a damaged line is dropped; a missing frame stops the gimbal by the input timeout
            t.class_id = 0;
            result->tracks.push_back(t);
        }
        result->frame = context;
        result->enqueued_at = std::chrono::steady_clock::now();
        return result;
    }

    void frame(std::istringstream& in)
    {
        std::uint64_t id = 0, captured_ns = 0;
        std::size_t count = 0;
        if (!(in >> id >> captured_ns >> count) || id == 0) return;
        if (auto result = tracks(in, id, captured_ns, count)) executor_.publish(aimed(result));
    }

    void frame_with_picture(std::istringstream& in)
    {
        std::uint64_t id = 0, captured_ns = 0;
        int w = 0, h = 0;
        std::size_t count = 0;
        if (!(in >> id >> captured_ns >> w >> h >> count) || id == 0 || w <= 0 || h <= 0 || w > 4096 || h > 4096) return;
        waiting_ = tracks(in, id, captured_ns, count);   // null: the picture bytes are still consumed, nothing is published
        wait_w_ = w; wait_h_ = h;
        if (!waiting_) { waiting_ = std::make_shared<TrackingResult>(); waiting_->frame = nullptr; }
    }

    void picture(const char* bytes)
    {
        auto result = std::move(waiting_);
        waiting_.reset();
        if (!result || !result->frame) return;
        auto context = std::make_shared<FrameContext>(*result->frame);
        context->image = cv::Mat(wait_h_, wait_w_, CV_16UC1);
        std::memcpy(context->image.data, bytes, static_cast<std::size_t>(wait_w_) * wait_h_ * 2);
        result->frame = context;
        executor_.publish(aimed(result));   // aim point for the gimbal only
        if (gui_) gui_->publish(result);    // the GUI draws the real boxes
    }

    void heartbeat()
    {
        const auto changes = session_changes_.load();
        if (!session_ || changes != seen_changes_) { executor_.connected(); session_ = true; seen_changes_ = changes; }   // first heartbeat, or the GUI came or left since: open the web session again
        else executor_.ping();
    }

    void message(const std::string& line)
    {
        std::istringstream in(line);
        std::string kind;
        in >> kind;
        if (kind == "F") frame(in);
        else if (kind == "G") frame_with_picture(in);
        else if (kind == "P") heartbeat();
        else if (kind == "A") {
            long long id = -1;
            in >> id;
            say(std::string("R A ") + answer(id > 0 && id <= 0xffffffffLL ? executor_.select(static_cast<std::uint32_t>(id)) : ControlResult::InvalidTarget));
        }
        else if (kind == "M") say(std::string("R M ") + answer(executor_.manual()));
        else if (kind == "N") {   // the tracker finished a model change the PC GUI asked for
            unsigned request = 0; int result = 6;
            if (in >> request >> result) g_models.answer(request, result);
        }
        else if (kind == "V") {
            int x = 0, y = 0;
            if (in >> x >> y) executor_.move(x, y);
        }
        else if (kind == "H") {
            executor_.halt();
            std::uint64_t ticket = 0;
            if (in >> ticket) say("HALTED " + std::to_string(ticket));
        }
    }
};
}

int main(int argc, char** argv) try {
    // The team's logger writes to stdout; stdout of this program is the protocol. Its log goes to stderr.
    protocol_fd = ::dup(1);
    ::dup2(2, 1);
    std::signal(SIGPIPE, SIG_IGN);
    std::signal(SIGTERM, on_signal);
    std::signal(SIGINT, on_signal);
    const auto options = parse(argc, argv);
    const auto& config = options.control;
    RuntimeControl runtime;
    GimbalCommandExecutor executor(config, GimbalDeviceFactory::create(config.gimbal), runtime);   // sends one STOP when it opens the device
    std::unique_ptr<GuiLink> gui;
    std::atomic<bool> gui_client{false};
    std::atomic<std::uint64_t> session_changes{0};
    std::unique_ptr<GuiReceiver> receiver;
    std::thread receiver_thread;
    if (options.gui.enabled) {
        gui = std::make_unique<GuiLink>(options.gui);
        if (options.gui.command.enabled) {
            auto handler = std::make_unique<ObservedHandler>(
                GuiCommandHandlerFactory::create(GuiCommandHandlerType::Tracking, config, &executor, &g_models), gui_client, session_changes);
            receiver = std::make_unique<GuiReceiver>(options.gui.command, std::move(handler));
            receiver_thread = std::thread([&receiver] {
                GuiCommand command;
                try { while (receiver->receive(command)) {} }
                catch (const std::exception& error) { std::cerr << "gimbal_bridge: GUI command session: " << error.what() << std::endl; }
                receiver->finish_session();
            });
        }
    }
    Bridge bridge(executor, gui.get(), session_changes);
    say("READY");
    say("TRACK_CONTROL 1");
    std::string pending, reported;
    std::vector<char> buffer(262144);
    auto next_status = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    bool input_open = true;
    while (!stop_signal && input_open) {
        pollfd input{0, POLLIN, 0};
        const int ready = ::poll(&input, 1, 50);
        if (ready < 0 && errno != EINTR) break;
        if (ready > 0) {
            const auto count = ::read(0, buffer.data(), buffer.size());
            if (count == 0) input_open = false;
            else if (count < 0 && errno != EINTR && errno != EAGAIN) break;
            else if (count > 0) pending.append(buffer.data(), static_cast<std::size_t>(count));
        }
        for (;;) {
            const auto need = bridge.picture_bytes();
            if (need) {
                if (pending.size() < need) break;
                bridge.picture(pending.data());
                pending.erase(0, need);
                continue;
            }
            const auto end = pending.find('\n');
            if (end == std::string::npos) break;
            bridge.message(pending.substr(0, end));
            pending.erase(0, end + 1);
        }
        const auto state = state_line(executor.snapshot());
        if (state != reported) { say(state); reported = state; }
        if (gui && std::chrono::steady_clock::now() >= next_status) { say(gui->status(gui_client.load())); next_status += std::chrono::seconds(1); }
    }
    if (receiver) { receiver->stop(); receiver_thread.join(); }
    if (gui) gui->close();
    executor.shutdown();                                  // cancels following and sends STOP when it was moving
    say(state_line(executor.snapshot()));
    return 0;
} catch (const std::exception& error) {
    std::cerr << "gimbal_bridge: " << error.what() << std::endl;
    return 2;
}
