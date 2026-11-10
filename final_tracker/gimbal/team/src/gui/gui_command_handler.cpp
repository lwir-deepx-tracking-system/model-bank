#include "gui/gui_command_handler.hpp"
#include "control/tracking_command_handler.hpp"
#include "app/pipeline_requests.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace {
GuiCommand ack(const GuiCommand& command, GuiResult result)
{
    return {GuiCategory::System, static_cast<std::uint8_t>(GuiSystem::Ack),
            {static_cast<std::uint8_t>(command.category), command.command,
             static_cast<std::uint8_t>(result)}};
}

// Communication-only implementation: never selects a target or drives a gimbal.
class DeferredCommandHandler final : public IGuiCommandHandler {
public:
    GuiCommand initial_state() const override
    {
        // Manual / None / Idle / no error / target ID 0. No operation is active.
        return {GuiCategory::System, static_cast<std::uint8_t>(GuiSystem::Status),
                std::vector<std::uint8_t>(8, 0)};
    }
    bool handle(const GuiCommand& command, std::vector<GuiCommand>& replies) override
    {
        replies.clear();
        if (command.category == GuiCategory::System) {
            if (command.command != static_cast<std::uint8_t>(GuiSystem::Ping)) return false;
            replies.push_back(ack(command, GuiResult::Ok));
        } else if (command.category == GuiCategory::Mode) {
            if (command.command == static_cast<std::uint8_t>(GuiModeCommand::Manual)) {
                // Idempotent: already Manual with no selected target.
                replies.push_back(ack(command, GuiResult::Ok));
            } else {
                // Preserve the received uint32 ID in GuiCommand, but do not apply it
                // to TargetSelector. Target validation/following is future work.
                bool zero = true;
                for (auto byte : command.payload) zero = zero && byte == 0;
                replies.push_back(ack(command, zero ? GuiResult::InvalidTarget : GuiResult::NotReady));
            }
        } else if (command.category == GuiCategory::Gimbal) {
            // MOVE has no ACK in the PC protocol; it has no effect until a real
            // control strategy exists. HOME must not report fake completion.
            if (command.command == static_cast<std::uint8_t>(GuiGimbalCommand::Home))
                replies.push_back(ack(command, GuiResult::NotReady));
        } else {
            return false;
        }
        return true;
    }
};

// 기존 제어 전략을 감싸 Pipeline 명령과 현재 TCP 세션의 완료 응답을 처리한다.
class PipelineCommandHandler final : public IGuiCommandHandler {
    struct Pending {
        std::uint32_t request_id;
        std::uint8_t command;
        std::shared_future<GuiPipelineResult> completion;
    };
    std::unique_ptr<IGuiCommandHandler> control_;
    IPipelineRequests* requests_;
    std::vector<Pending> pending_;

    static std::uint32_t read32(const std::vector<std::uint8_t>& p, std::size_t offset) {
        std::uint32_t value = 0;
        for (std::size_t i = offset; i < offset + 4; ++i) value = (value << 8) | p[i];
        return value;
    }
    static void write32(std::vector<std::uint8_t>& p, std::uint32_t value) {
        for (int shift : {24, 16, 8, 0}) p.push_back(static_cast<std::uint8_t>(value >> shift));
    }
    static GuiCommand response(std::uint8_t command, std::uint32_t request_id,
        GuiPipelinePhase phase, GuiPipelineResult result) {
        GuiCommand reply{GuiCategory::Pipeline, std::uint8_t(command | 0x80), {}};
        write32(reply.payload, request_id);
        reply.payload.push_back(static_cast<std::uint8_t>(phase));
        reply.payload.push_back(static_cast<std::uint8_t>(result));
        return reply;
    }
public:
    PipelineCommandHandler(std::unique_ptr<IGuiCommandHandler> control, IPipelineRequests* requests)
        : control_(std::move(control)), requests_(requests) {}
    GuiCommand initial_state() const override { return control_->initial_state(); }
    void connected() override { pending_.clear(); control_->connected(); }
    void disconnected() override { pending_.clear(); control_->disconnected(); }
    int poll_timeout_ms() const override {
        return pending_.empty() ? control_->poll_timeout_ms() : std::min(25, control_->poll_timeout_ms());
    }

    bool handle(const GuiCommand& command, std::vector<GuiCommand>& replies) override {
        if (command.category != GuiCategory::Pipeline) return control_->handle(command, replies);
        replies.clear();
        const auto code = command.command;
        if (code < 1 || code > 4 || command.payload.size() != (code <= 2 ? 4u : 8u))
            return false; // 보드 응답을 PC가 보내는 경우도 거부한다.
        const auto request_id = read32(command.payload, 0);
        if (code <= 2) {
            GuiCommand reply{GuiCategory::Pipeline, std::uint8_t(code | 0x80), {}};
            write32(reply.payload, request_id);
            write32(reply.payload, requests_ ? (code == 1 ? requests_->model_count() : requests_->tracker_count()) : 0);
            replies.push_back(std::move(reply));
            return true;
        }
        const bool duplicate = std::any_of(pending_.begin(), pending_.end(),
            [request_id](const Pending& p) { return p.request_id == request_id; });
        if (duplicate || pending_.size() >= 2) {
            replies.push_back(response(code, request_id, GuiPipelinePhase::Completed, GuiPipelineResult::Busy));
            return true;
        }
        PipelineChangeTicket ticket;
        if (requests_) {
            const auto id = read32(command.payload, 4);
            ticket = code == 3 ? requests_->request_model_change(id) : requests_->request_tracker_change(id);
        }
        if (ticket.result != GuiPipelineResult::Ok || !ticket.completion.valid()) {
            replies.push_back(response(code, request_id, GuiPipelinePhase::Completed,
                ticket.result == GuiPipelineResult::Ok ? GuiPipelineResult::NotReady : ticket.result));
        } else {
            pending_.push_back({request_id, code, std::move(ticket.completion)});
            replies.push_back(response(code, request_id, GuiPipelinePhase::Accepted, GuiPipelineResult::Ok));
        }
        return true;
    }

    // PING과 분할 수신을 계속 처리하면서 현재 세션의 요청에만 완료 응답을 보낸다.
    void tick(std::vector<GuiCommand>& replies) override {
        control_->tick(replies);
        for (auto it = pending_.begin(); it != pending_.end();) {
            if (it->completion.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                ++it; continue;
            }
            auto result = GuiPipelineResult::NotReady;
            try { result = it->completion.get(); } catch (const std::future_error&) {}
            replies.push_back(response(it->command, it->request_id, GuiPipelinePhase::Completed, result));
            it = pending_.erase(it);
        }
    }
};
}

std::unique_ptr<IGuiCommandHandler> GuiCommandHandlerFactory::create(GuiCommandHandlerType type, const ControlConfig& control, GimbalCommandExecutor* executor, IPipelineRequests* pipeline_requests)
{
    std::unique_ptr<IGuiCommandHandler> handler;
    if (type == GuiCommandHandlerType::Deferred) handler = std::make_unique<DeferredCommandHandler>();
    else if (type == GuiCommandHandlerType::Tracking) {
        if (!control.enabled || !executor) throw std::runtime_error("tracking handler requires an enabled executor");
        handler = std::make_unique<TrackingCommandHandler>(*executor, control.poll_interval_ms);
    } else throw std::runtime_error("Unsupported GUI command handler");
    return std::make_unique<PipelineCommandHandler>(std::move(handler), pipeline_requests);
}
