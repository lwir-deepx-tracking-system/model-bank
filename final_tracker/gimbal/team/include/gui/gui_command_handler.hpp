#pragma once

#include <limits>
#include "config/app_config.hpp"
#include "gui/gui_protocol.hpp"

// Application policy strategy. It has no socket, YAML or tracking dependencies.
// Future target-following implementations replace this strategy via the factory.
class IGuiCommandHandler {
public:
    virtual ~IGuiCommandHandler() = default;
    virtual GuiCommand initial_state() const = 0;
    // Input must first pass the configured wire codec validation.
    // false: invalid direction (e.g. a PC sending board-only STATUS); disconnect.
    virtual bool handle(const GuiCommand&, std::vector<GuiCommand>& replies) = 0;
    // Lifecycle/timer callbacks run on the session worker, never the stop caller.
    virtual void connected() {}
    virtual void disconnected() {}
    virtual void tick(std::vector<GuiCommand>& replies) { replies.clear(); }
    virtual int poll_timeout_ms() const { return std::numeric_limits<int>::max(); }
};

class GimbalCommandExecutor;
class IPipelineRequests;

class GuiCommandHandlerFactory {
public:
    static std::unique_ptr<IGuiCommandHandler> create(
        GuiCommandHandlerType type, const ControlConfig& control = {},
        GimbalCommandExecutor* executor = nullptr,
        IPipelineRequests* pipeline_requests = nullptr);
};
