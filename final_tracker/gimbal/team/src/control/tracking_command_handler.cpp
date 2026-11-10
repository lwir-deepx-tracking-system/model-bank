#include "control/tracking_command_handler.hpp"

namespace {
GuiCommand ack(const GuiCommand& c, GuiResult result)
{
    return {GuiCategory::System, static_cast<std::uint8_t>(GuiSystem::Ack),
        {static_cast<std::uint8_t>(c.category), c.command, static_cast<std::uint8_t>(result)}};
}
GuiResult result_code(ControlResult r)
{
    if (r == ControlResult::Ok) return GuiResult::Ok;
    if (r == ControlResult::InvalidTarget) return GuiResult::InvalidTarget;
    return GuiResult::NotReady;
}
}
GuiCommand TrackingCommandHandler::initial_state() const
{
    const auto s = executor_.snapshot();
    // PC ControlProtocol.h: None/Tracking/Lost; Idle/ManualMoving/AutoTracking/Hold/Error.
    const std::uint8_t target = s.automatic ? (s.lost ? 2 : 1) : 0;
    const std::uint8_t gimbal = s.error ? 5 : s.automatic ? (s.moving ? 2 : 4) : s.moving ? 1 : 0;
    return {GuiCategory::System, static_cast<std::uint8_t>(GuiSystem::Status),
        {std::uint8_t(s.automatic), target, gimbal, std::uint8_t(s.error),
         std::uint8_t(s.selected_id >> 24), std::uint8_t(s.selected_id >> 16),
         std::uint8_t(s.selected_id >> 8), std::uint8_t(s.selected_id)}};
}
void TrackingCommandHandler::report(std::vector<GuiCommand>& replies)
{
    auto status = initial_state();
    if (status.payload != reported_) {
        reported_ = status.payload;
        replies.push_back(std::move(status));
    }
}
void TrackingCommandHandler::connected() { executor_.connected(); reported_ = initial_state().payload; }
void TrackingCommandHandler::disconnected() { executor_.disconnected(); reported_.clear(); }
void TrackingCommandHandler::tick(std::vector<GuiCommand>& replies) { replies.clear(); report(replies); }
bool TrackingCommandHandler::handle(const GuiCommand& c, std::vector<GuiCommand>& replies)
{
    replies.clear();
    if (c.category == GuiCategory::System && c.command == 4 && c.payload.empty()) {
        executor_.ping(); replies.push_back(ack(c, GuiResult::Ok));
    } else if (c.category == GuiCategory::Mode && c.command == 1 && c.payload.size() == 4) {
        std::uint32_t id = 0;
        for (auto b : c.payload) id = (id << 8) | b;
        replies.push_back(ack(c, result_code(executor_.select(id))));
    } else if (c.category == GuiCategory::Mode && c.command == 2 && c.payload.empty()) {
        replies.push_back(ack(c, result_code(executor_.manual())));
    } else if (c.category == GuiCategory::Gimbal && c.command == 1 && c.payload.size() == 2) {
        const auto direction = [](std::uint8_t b) { return b < 128 ? int(b) : int(b) - 256; };
        executor_.move(direction(c.payload[0]), direction(c.payload[1]));
    } else if (c.category == GuiCategory::Gimbal && c.command == 3 && c.payload.empty()) {
        // PC clears keys for HOME. Release manual motion as well as AUTO.
        executor_.halt();
        replies.push_back(ack(c, GuiResult::NotReady));
    } else return false;
    report(replies);
    return true;
}
