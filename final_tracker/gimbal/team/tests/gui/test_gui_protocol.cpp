#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "gui/gui_protocol.hpp"
#include "gui/gui_command_handler.hpp"

namespace {
std::vector<std::uint8_t> hex(const std::string& text)
{
    std::vector<std::uint8_t> bytes;
    for (std::size_t i = 0; i < text.size(); i += 2)
        bytes.push_back(static_cast<std::uint8_t>(std::stoul(text.substr(i, 2), nullptr, 16)));
    return bytes;
}
}

int main()
{
    auto protocol = std::make_unique<PcGuiProtocol>();
    // Independent fixtures taken from PC ControlTest, including the full uint32 ID range.
    const std::vector<std::vector<std::uint8_t>> commands = {
        hex("4c57495202010004ffffffff"), hex("4c57495202020000"),
        hex("4c57495203010002ff01"), hex("4c57495203030000"),
        hex("4c57495201040000"), hex("4c57495201010003020105"),
        hex("4c574952010200080000000000000000"),
        hex("4c574952010300080000000000000000")
    };
    for (const auto& fixture : commands) {
        GuiCommand command;
        if (protocol->command_packet_size(fixture.data(), 8) != fixture.size() ||
            !protocol->decode_command(fixture.data(), fixture.size(), command) ||
            protocol->encode_command(command) != fixture) return 1;
        for (std::size_t n = 0; n < fixture.size(); ++n)
            if (protocol->decode_command(fixture.data(), n, command)) return 2;
        auto extra = fixture;
        extra.push_back(0);
        if (protocol->decode_command(extra.data(), extra.size(), command)) return 3;
    }
    for (const auto& bad : {hex("4c57495203020000"), hex("4c5749520201ffff"),
                            hex("4c57495200010000"), hex("4c57494d01040000")})
        if (protocol->command_packet_size(bad.data(), bad.size())) return 4;

    // Exact fixture from PC VideoReceiverTest::decodeRuntimeMetadata.
    const auto fixture = hex("4c574952000000000000002a00002328000000000001e24000000001"
                             "0000000742c80000424800004348000042c80000013f400000");
    GuiTrackingMetadata metadata;
    metadata.frame_id = 42;
    metadata.rtp_timestamp = 9000;
    metadata.gui_started_us = 123456;
    GuiTrackMetadata track;
    track.track_id = 7; track.class_id = 1; track.confidence = .75F;
    track.x = 100; track.y = 50; track.width = 200; track.height = 100;
    metadata.tracks.push_back(track);
    std::vector<std::uint8_t> packet;
    if (!protocol->encode_metadata(metadata, 1200, packet) || packet != fixture) return 5;
    GuiTrackingMetadata decoded;
    if (!decode_tracking_metadata(fixture.data(), fixture.size(), decoded) ||
        decoded.frame_id != 42 || decoded.gui_started_us != 123456 ||
        decoded.rtp_timestamp != 9000 || decoded.width != 640 || decoded.height != 480 ||
        decoded.tracks.size() != 1 || decoded.tracks[0].x != 100 ||
        decoded.tracks[0].class_id != 1 || decoded.tracks[0].confidence != .75F) return 6;
    for (std::size_t n = 0; n < fixture.size(); ++n)
        if (decode_tracking_metadata(fixture.data(), n, decoded)) return 7;
    auto bad = fixture;
    bad[0] = 0;
    if (decode_tracking_metadata(bad.data(), bad.size(), decoded) || decoded.frame_id != 42) return 8;
    metadata.tracks[0].class_id = 255;
    if (!protocol->encode_metadata(metadata, 1200, packet) || packet[48] != 255) return 9;
    metadata.tracks.resize(46, metadata.tracks[0]);
    if (!protocol->encode_metadata(metadata, 1200, packet) || packet.size() != 1178) return 10;
    metadata.tracks.push_back(track);
    if (!protocol->encode_metadata(metadata, 1200, packet) || packet.size() != 1178 ||
        packet[27] != 46) return 11;
    metadata.tracks.clear();
    if (!protocol->encode_metadata(metadata, 1200, packet) || packet.size() != 28) return 12;
    metadata.frame_id = 0;
    if (protocol->encode_metadata(metadata, 1200, packet)) return 13;
    metadata.frame_id = 42; metadata.width = 320;
    if (protocol->encode_metadata(metadata, 1200, packet) || protocol->accepts_frame(320, 240)) return 14;
    metadata.width = 640;
    track.track_id = 0xffffffffU;
    metadata.tracks.push_back(track);
    if (!protocol->encode_metadata(metadata, 1200, packet) ||
        !decode_tracking_metadata(packet.data(), packet.size(), decoded) ||
        decoded.tracks.size() != 1 || decoded.tracks[0].track_id != 0xffffffffU) return 15;

    // Bounds/filtering and selected-target priority follow the adopted design.
    metadata.tracks.clear();
    for (std::uint32_t id = 1; id <= 47; ++id) {
        track.track_id = id;
        metadata.tracks.push_back(track);
    }
    metadata.selected_track_id = 47;
    if (!protocol->encode_metadata(metadata, 1200, packet) ||
        !decode_tracking_metadata(packet.data(), packet.size(), decoded) ||
        decoded.tracks.size() != 46 || decoded.tracks[0].track_id != 47 ||
        decoded.tracks[1].track_id != 1 || decoded.tracks.back().track_id != 45) return 21;
    metadata.selected_track_id = 0;
    track.track_id = 1; track.x = -10; track.y = -20; track.width = 30; track.height = 40;
    metadata.tracks = {track};
    track.track_id = 0; metadata.tracks.push_back(track);
    track.track_id = 3; track.confidence = std::numeric_limits<float>::quiet_NaN();
    metadata.tracks.push_back(track);
    if (!protocol->encode_metadata(metadata, 1200, packet) || packet[27] != 1 ||
        !decode_tracking_metadata(packet.data(), packet.size(), decoded) ||
        decoded.tracks.size() != 1 || decoded.tracks[0].x != 0 || decoded.tracks[0].y != 0 ||
        decoded.tracks[0].width != 20 || decoded.tracks[0].height != 20) return 22;
    // Even a zero-record datagram preserves a valid frame header.
    if (!protocol->encode_metadata(metadata, 28, packet) || packet.size() != 28 || packet[27]) return 23;
    if (protocol->encode_metadata(metadata, 27, packet)) return 24;

    auto handler = GuiCommandHandlerFactory::create(GuiCommandHandlerType::Deferred);
    if (protocol->encode_command(handler->initial_state()) != commands.back()) return 16;
    std::vector<GuiCommand> replies;
    GuiCommand command;
    protocol->decode_command(commands[0].data(), commands[0].size(), command);
    if (!handler->handle(command, replies) || replies.size() != 1 ||
        protocol->encode_command(replies[0]) != hex("4c57495201010003020105")) return 17;
    // A PC must not publish board state to the runtime.
    if (handler->handle(handler->initial_state(), replies)) return 18;
    // Unknown command policies must fail instead of falling back.
    try { GuiCommandHandlerFactory::create(static_cast<GuiCommandHandlerType>(99)); return 20; }
    catch (const std::runtime_error&) {}
    return 0;
}
