#pragma once

#include "control/gimbal_command_executor.hpp"
#include "gui/gui_command_handler.hpp"

// TCP adapter only; state and device ownership remain in the executor.
class TrackingCommandHandler final : public IGuiCommandHandler {
    GimbalCommandExecutor& executor_;
    int poll_ms_;
    std::vector<std::uint8_t> reported_;
    void report(std::vector<GuiCommand>&);
public:
    TrackingCommandHandler(GimbalCommandExecutor& executor, int poll_ms)
        : executor_(executor), poll_ms_(poll_ms) {}
    GuiCommand initial_state() const override;
    bool handle(const GuiCommand&, std::vector<GuiCommand>&) override;
    void connected() override;
    void disconnected() override;
    void tick(std::vector<GuiCommand>&) override;
    int poll_timeout_ms() const override { return poll_ms_; }
};
