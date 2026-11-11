#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>

#include "config/app_config.hpp"
#include "gui/gui_protocol.hpp"
#include "gui/gui_command_handler.hpp"

// PC TCP session: framing/transport, with replaceable wire and command strategies.
class GuiReceiver
{
private:
    GuiCommandConfig config_;
    std::unique_ptr<IGuiProtocol> protocol_;
    std::unique_ptr<IGuiCommandHandler> handler_;
    bool stopped_ = false;

    // receive()를 실행하는 worker가 socket을 사용하고, 다른 thread의 stop()이
    // 이를 닫아 blocking poll/recv를 깨운다. mutex는 FD 소유권 인계와
    // double-close를 방지한다.
    std::mutex socket_mutex_;
    int listen_fd_ = -1;
    int client_fd_ = -1;

    // 설정 주소에 TCP listen socket을 열어 GUI 재접속을 받을 준비를 한다.
    bool open_listener();

    // poll timeout마다 종료 상태를 확인하면서 GUI client 연결을 기다린다.
    bool accept_client();

    // TCP stream에서 header/body의 요청 byte 수를 누적 수신한다.
    bool receive_exact(int fd, std::uint8_t* data, std::size_t size);

    bool send_message(int fd, const GuiCommand& message);
    bool service_client(int fd);
    int poll_timeout_ms() const;

    // 현재 receive 경로가 FD 소유권을 가진 경우에만 client socket을 닫는다.
    void release_client(int fd);

    // receive worker와 stop 호출자 사이에서 종료 상태를 안전하게 조회한다.
    bool stopped();

public:
    explicit GuiReceiver(GuiCommandConfig config,
        std::unique_ptr<IGuiCommandHandler> handler = nullptr);
    ~GuiReceiver();

    GuiReceiver(const GuiReceiver&) = delete;
    GuiReceiver& operator=(const GuiReceiver&) = delete;

    // 접속 시 상태를 송신하고, 검증된 명령을 처리/응답한 뒤 관찰용으로 반환한다.
    bool receive(GuiCommand& command);

    // listen/client socket을 shutdown/close해 다른 thread의 poll/recv를 깨운다.
    void stop();

    bool stop_requested() { return stopped(); }
    // Call on the worker after errors/exit, before destroying the receiver.
    void finish_session() { handler_->disconnected(); }
};
