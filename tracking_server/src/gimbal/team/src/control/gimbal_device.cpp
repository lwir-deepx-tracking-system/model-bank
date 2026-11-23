#include "control/gimbal_device.hpp"
#include "control/waveshare_pose.hpp"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <termios.h>
#include <unistd.h>

namespace {
class SerialPort {
    int fd_ = -1;
    int timeout_ms_;
    std::string feedback_line_;
    bool discard_line_ = false;
    std::optional<GimbalPose> pose_, pending_pose_;
public:
    explicit SerialPort(const GimbalConfig& config) : timeout_ms_(config.write_timeout_ms)
    {
        // Nonblocking open/write and an overall deadline keep STOP and TCP alive.
        const int fd = ::open(config.device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
        if (fd < 0) throw std::runtime_error("Gimbal open " + config.device + ": " + std::strerror(errno));
        try {
            termios tty{};
            if (::tcgetattr(fd, &tty)) throw std::runtime_error("Gimbal tcgetattr failed");
            ::cfmakeraw(&tty);
            speed_t baud;
            switch (config.baud_rate) {
            case 9600: baud = B9600; break;
            case 57600: baud = B57600; break;
            case 115200: baud = B115200; break;
            default: throw std::runtime_error("Unsupported gimbal baud rate");
            }
            if (::cfsetispeed(&tty, baud) || ::cfsetospeed(&tty, baud))
                throw std::runtime_error("Gimbal baud configuration failed");
            tty.c_cflag |= CLOCAL | CREAD;
            tty.c_cflag &= ~(PARENB | CSTOPB | CSIZE | CRTSCTS);
            tty.c_cflag |= CS8;
            if (::tcsetattr(fd, TCSANOW, &tty)) throw std::runtime_error("Gimbal tcsetattr failed");
            fd_ = fd;
        } catch (...) { ::close(fd); throw; }
    }
    ~SerialPort() { if (fd_ >= 0) ::close(fd_); }
    SerialPort(const SerialPort&) = delete;
    SerialPort& operator=(const SerialPort&) = delete;

    void reset_feedback() {
        ::tcflush(fd_, TCIFLUSH);
        feedback_line_.clear(); discard_line_ = false;
        pose_.reset(); pending_pose_.reset();
    }

    std::optional<GimbalPose> receive_pose()
    {
        // Bounded nonblocking reads on the existing owner FD. No response wait
        // or per-frame query may hold up tracking/control output.
        char bytes[1024];
        for (int batch = 0; batch < 4; ++batch) {
            const auto count = ::read(fd_, bytes, sizeof bytes);
            if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                if (pending_pose_) {
                    pending_pose_->received_at = std::chrono::steady_clock::now();
                    pose_ = pending_pose_; pending_pose_.reset();
                }
                return pose_;
            }
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) { reset_feedback(); return std::nullopt; }
            for (ssize_t i = 0; i < count; ++i) {
                if (bytes[i] == '\n') {
                    if (!discard_line_) {
                        const auto parsed = parse_waveshare_pose(feedback_line_);
                        if (parsed) pending_pose_ = parsed;
                    }
                    feedback_line_.clear(); discard_line_ = false;
                } else if (!discard_line_) {
                    if (feedback_line_.size() >= 4096) { feedback_line_.clear(); discard_line_ = true; }
                    else feedback_line_.push_back(bytes[i]);
                }
            }
        }
        return pose_; // finish draining next poll before timestamping new samples
    }

    void send(const std::string& json)
    {
        const std::string message = json + "\n";
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms_);
        std::size_t sent = 0;
        while (sent < message.size()) {
            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - std::chrono::steady_clock::now()).count();
            if (remaining <= 0) throw std::runtime_error("Gimbal serial write timeout");
            pollfd descriptor{fd_, POLLOUT, 0};
            const int ready = ::poll(&descriptor, 1, static_cast<int>(remaining));
            if (ready < 0 && errno == EINTR) continue;
            if (ready <= 0 || (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)))
                throw std::runtime_error("Gimbal serial unavailable or write timeout");
            const auto count = ::write(fd_, message.data() + sent, message.size() - sent);
            if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
            if (count <= 0) throw std::runtime_error("Gimbal serial write failed");
            sent += static_cast<std::size_t>(count);
        }
        // tcdrain is deliberately avoided: it has no deadline. Kernel write
        // success means submission, never device acknowledgement/completion.
    }
};
class WaveshareGimbal final : public IGimbalDevice {
    GimbalConfig config_;
    SerialPort serial_;
public:
    explicit WaveshareGimbal(const GimbalConfig& config) : config_(config), serial_(config) {}
    void move(int x, int y) override { serial_.send(WaveshareCommands::move(config_, x, y)); }
    void move_with_speed(int x, int y, int speed) override { serial_.send(WaveshareCommands::move(config_, x, y, speed)); }
    void stop() override { serial_.send(WaveshareCommands::stop()); }
    void home() override { serial_.send(WaveshareCommands::home(config_)); }
    void enable_pose_feedback() override {
        serial_.reset_feedback();
        serial_.send("{\"T\":142,\"cmd\":100}"); // 10 Hz, feedback only
        serial_.send("{\"T\":131,\"cmd\":1}");
    }
    std::optional<GimbalPose> read_pose() override { return serial_.receive_pose(); }
};
}
std::string WaveshareCommands::move(const GimbalConfig& config, int x, int y)
{
    return move(config, x, y, config.speed);
}
std::string WaveshareCommands::move(const GimbalConfig& config, int x, int y, int speed)
{
    if (speed < 1 || speed > config.speed_limit) throw std::invalid_argument("Gimbal speed exceeds configured range");
    if (x < -1 || x > 1 || y < -1 || y > 1) throw std::invalid_argument("Invalid gimbal direction");
    std::ostringstream command;
    command << "{\"T\":141,\"X\":" << x * config.x_sign << ",\"Y\":" << y * config.y_sign
            << ",\"SPD\":" << speed << '}';
    return command.str();
}
std::string WaveshareCommands::stop() { return "{\"T\":135}"; }
std::string WaveshareCommands::home(const GimbalConfig& config)
{
    std::ostringstream command;
    command << "{\"T\":133,\"X\":" << config.home_x_deg << ",\"Y\":" << config.home_y_deg
            << ",\"SPD\":" << config.speed << ",\"ACC\":" << config.acceleration << '}';
    return command.str();
}
std::unique_ptr<IGimbalDevice> GimbalDeviceFactory::create(const GimbalConfig& config)
{
    return std::make_unique<WaveshareGimbal>(config);
}
