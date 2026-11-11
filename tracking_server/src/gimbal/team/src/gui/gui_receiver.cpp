#include "gui/gui_receiver.hpp"

#include <mutex>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include "common/logger.hpp"

namespace {
void close_socket(int fd)
{
    if (fd < 0) return;
    // shutdown()이 다른 thread에서 대기 중인 poll/recv/accept를 깨우고,
    // close()가 process의 FD 소유권을 최종 해제한다.
    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
}
}

GuiReceiver::GuiReceiver(GuiCommandConfig config, std::unique_ptr<IGuiCommandHandler> handler)
    : config_(std::move(config)),
      protocol_(std::make_unique<PcGuiProtocol>()),
      handler_(handler ? std::move(handler) : GuiCommandHandlerFactory::create(config_.handler))
{
}

GuiReceiver::~GuiReceiver()
{
    stop();
}

bool GuiReceiver::stopped()
{
    // stop()은 다른 thread에서 호출되므로 socket 상태와 같은 mutex로 보호한다.
    std::lock_guard<std::mutex> lock(socket_mutex_);
    return stopped_;
}

// 설정 주소에 listener를 열어 GUI command TCP 연결을 받을 준비를 한다.
bool GuiReceiver::open_listener()
{
    // 이미 열렸거나 종료된 상태를 먼저 확인해 중복 listener 생성을 막는다.
    {
        std::lock_guard<std::mutex> lock(socket_mutex_);
        if (listen_fd_ >= 0) return true;
        if (stopped_) return false;
    }

    // 설정된 bind 주소를 IPv4/IPv6 socket 후보로 변환한다.
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_PASSIVE;

    addrinfo* addresses = nullptr;
    const std::string port = std::to_string(config_.port);
    const int lookup = ::getaddrinfo(
        config_.bind_address.c_str(), port.c_str(), &hints, &addresses);
    if (lookup != 0)
    {
        Logger::error(
            "[GuiReceiver] bind 주소 해석 실패: " +
            std::string(gai_strerror(lookup)));
        return false;
    }

    // 주소 후보를 순회하며 실제 bind/listen 가능한 첫 socket을 선택한다.
    int listener = -1;
    for (addrinfo* address = addresses; address; address = address->ai_next)
    // socket 생성 중 stop()이 호출됐을 수 있으므로 공유 FD로 넘기기 전에
    // 종료 상태를 다시 확인한다.
    {
        const int fd = ::socket(
            address->ai_family, address->ai_socktype, address->ai_protocol);
        if (fd < 0) continue;

        const int reuse = 1;
        ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        if (::bind(fd, address->ai_addr, address->ai_addrlen) == 0 &&
            ::listen(fd, 1) == 0)
        {
            listener = fd;
            break;
        }
        ::close(fd);
    }
    ::freeaddrinfo(addresses);

    if (listener < 0)
    {
        Logger::error(
            "[GuiReceiver] TCP listen 실패: " + config_.bind_address + ":" +
            port + " (" + std::strerror(errno) + ")");
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(socket_mutex_);
        if (stopped_)
        {
            close_socket(listener);
            return false;
        }
        listen_fd_ = listener;
    }
    Logger::info(
        "[GuiReceiver] TCP 명령 대기: " + config_.bind_address + ":" + port);
    return true;
}

// 종료 가능하도록 timeout poll을 반복하며 한 GUI client 연결을 수립한다.
bool GuiReceiver::accept_client()
{
    // blocking accept 대신 poll timeout을 사용해 stop 요청을 주기적으로 확인한다.
    while (!stopped())
    {
        int listener = -1;
        {
            std::lock_guard<std::mutex> lock(socket_mutex_);
            if (client_fd_ >= 0) return true;
            listener = listen_fd_;
        }
        if (listener < 0) return false;

        pollfd descriptor{};
        descriptor.fd = listener;
        descriptor.events = POLLIN;
        const int ready = ::poll(
            &descriptor, 1, poll_timeout_ms());
        if (ready == 0) continue;
        if (ready < 0)
        {
            if (errno == EINTR) continue;
            if (!stopped())
                Logger::error("[GuiReceiver] TCP accept 대기 실패");
            return false;
        }
        if ((descriptor.revents & POLLIN) == 0)
            return false;

        // POLLIN은 pending client가 있다는 뜻이므로 이때만 accept를 호출한다.
        const int client = ::accept(listener, nullptr, nullptr);
        if (client < 0)
        {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
                continue;
            if (!stopped())
                Logger::error("[GuiReceiver] TCP client accept 실패");
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(socket_mutex_);
            if (stopped_)
            {
                close_socket(client);
                return false;
            }
            client_fd_ = client;
        }
        handler_->connected();
        if (!send_message(client, handler_->initial_state()))
        {
            release_client(client);
            continue;
        }
        Logger::info("[GuiReceiver] GUI 명령 연결 수립");
        return true;
    }
    return false;
}

// TCP stream에서 header 또는 body의 요청 byte 수를 모두 모은다.
bool GuiReceiver::receive_exact(
    int fd,
    std::uint8_t* data,
    std::size_t size)
{
    // TCP 한 번의 recv가 command 전체를 반환한다는 보장이 없으므로
    // 요청한 크기가 찰 때까지 조각을 누적한다.
    std::size_t received = 0;
    while (received < size && !stopped())
    {
        if (!service_client(fd)) return false;
        // Timer processing also continues while a partial command is pending.
        pollfd descriptor{};
        descriptor.fd = fd;
        descriptor.events = POLLIN;
        const int ready = ::poll(
            &descriptor, 1, poll_timeout_ms());
        if (ready == 0) continue;
        if (ready < 0)
        {
            if (errno == EINTR) continue;
            return false;
        }

        if ((descriptor.revents & POLLIN) != 0)
        {
            ssize_t count;
            {
                std::lock_guard<std::mutex> lock(socket_mutex_);
                if (client_fd_ != fd || stopped_) return false;
                count = ::recv(fd, data + received, size - received, MSG_DONTWAIT);
            }
            if (count > 0)
            {
                received += static_cast<std::size_t>(count);
                continue;
            }
            if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
            return false;
        }

        if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
            return false;
    }
    return received == size;
}

// 끊어지거나 잘못된 client 연결을 반납해 다음 receive에서 재접속을 받게 한다.
void GuiReceiver::release_client(int fd)
{
    // receive thread와 stop thread 중 한쪽만 같은 FD를 닫도록 소유권을 회수한다.
    bool owns_fd = false;
    {
        std::lock_guard<std::mutex> lock(socket_mutex_);
        if (client_fd_ == fd)
        {
            client_fd_ = -1;
            owns_fd = true;
        }
    }
    // stop()이 이미 FD를 가져가 닫았다면 여기서 다시 닫지 않는다.
    if (owns_fd) close_socket(fd);
    handler_->disconnected();
}

int GuiReceiver::poll_timeout_ms() const
{
    return std::min(config_.receive_timeout_ms, handler_->poll_timeout_ms());
}
bool GuiReceiver::service_client(int fd)
{
    std::vector<GuiCommand> replies;
    handler_->tick(replies);
    for (const auto& reply : replies)
        if (!send_message(fd, reply)) return false;
    return true;
}

// Small replies use nonblocking sends and bounded poll waits; stop() can interrupt.
bool GuiReceiver::send_message(int fd, const GuiCommand& message)
{
    const auto packet = protocol_->encode_command(message);
    if (packet.empty()) return false;
    std::size_t sent = 0;
    while (sent < packet.size() && !stopped()) {
        ssize_t count;
        {
            std::lock_guard<std::mutex> lock(socket_mutex_);
            if (client_fd_ != fd || stopped_) return false;
            count = ::send(fd, packet.data() + sent, packet.size() - sent,
                           MSG_NOSIGNAL | MSG_DONTWAIT);
        }
        if (count > 0) { sent += static_cast<std::size_t>(count); continue; }
        if (count < 0 && errno == EINTR) continue;
        if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            pollfd descriptor{fd, POLLOUT, 0};
            const int ready = ::poll(&descriptor, 1, poll_timeout_ms());
            if (ready < 0 && errno == EINTR) continue;
            if (ready > 0 && (descriptor.revents & POLLOUT)) continue;
        }
        return false;
    }
    return sent == packet.size();
}

bool GuiReceiver::receive(GuiCommand& command)
{
    if (!config_.enabled || stopped() || !open_listener()) { finish_session(); return false; }
    while (!stopped()) {
        if (!accept_client()) { finish_session(); return false; }
        int client;
        {
            std::lock_guard<std::mutex> lock(socket_mutex_);
            client = client_fd_;
        }
        if (client < 0) continue;
        std::vector<std::uint8_t> packet(protocol_->command_header_size());
        if (!receive_exact(client, packet.data(), packet.size())) {
            release_client(client);
            continue;
        }
        const auto total = protocol_->command_packet_size(packet.data(), packet.size());
        if (!total) {
            Logger::warn("[GuiReceiver] Invalid PC command header");
            release_client(client);
            continue;
        }
        const auto header = packet.size();
        packet.resize(total);
        if (!receive_exact(client, packet.data() + header, total - header) ||
            !protocol_->decode_command(packet.data(), packet.size(), command)) {
            release_client(client);
            continue;
        }
        std::vector<GuiCommand> replies;
        if (!handler_->handle(command, replies)) {
            release_client(client);
            continue;
        }
        bool sent = true;
        for (const auto& reply : replies) {
            if (!send_message(client, reply)) { sent = false; break; }
        }
        if (!sent) { release_client(client); continue; }
        return true;
    }
    finish_session();
    return false;
}

// 공유 FD 소유권을 회수한 뒤 socket 대기를 깨워 receive loop를 종료시킨다.
void GuiReceiver::stop()
{
    // mutex 안에서는 공유 상태와 FD 소유권만 가져오고 실제 shutdown/close는
    // lock 밖에서 수행해 대기 중인 receive thread와의 교착 가능성을 줄인다.
    int listener = -1;
    int client = -1;
    {
        std::lock_guard<std::mutex> lock(socket_mutex_);
        if (stopped_) return;
        stopped_ = true;
        listener = listen_fd_;
        client = client_fd_;
        listen_fd_ = -1;
        client_fd_ = -1;
    }
    close_socket(client);
    close_socket(listener);
}
