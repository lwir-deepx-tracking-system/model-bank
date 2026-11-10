#include "gui/gui_protocol.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace {
void put(std::vector<std::uint8_t>& out, std::uint64_t value, unsigned bytes)
{
    for (unsigned i = bytes; i > 0; --i)
        out.push_back(static_cast<std::uint8_t>(value >> ((i - 1) * 8)));
}
std::uint64_t get(const std::uint8_t* data, unsigned bytes)
{
    std::uint64_t value = 0;
    for (unsigned i = 0; i < bytes; ++i) value = (value << 8) | data[i];
    return value;
}
void put_float(std::vector<std::uint8_t>& out, float value)
{
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    std::uint32_t bits;
    std::memcpy(&bits, &value, 4);
    put(out, bits, 4);
}
float get_float(const std::uint8_t* data)
{
    const auto bits = static_cast<std::uint32_t>(get(data, 4));
    float value;
    std::memcpy(&value, &bits, 4);
    return value;
}
int payload_size(GuiCategory category, std::uint8_t command)
{
    switch (category) {
    case GuiCategory::System:
        switch (static_cast<GuiSystem>(command)) {
        case GuiSystem::Ack: return 3;
        case GuiSystem::Init: case GuiSystem::Status: return 8;
        case GuiSystem::Ping: return 0;
        }
        break;
    case GuiCategory::Mode:
        if (command == static_cast<std::uint8_t>(GuiModeCommand::Auto)) return 4;
        if (command == static_cast<std::uint8_t>(GuiModeCommand::Manual)) return 0;
        break;
    case GuiCategory::Gimbal:
        if (command == static_cast<std::uint8_t>(GuiGimbalCommand::Move)) return 2;
        if (command == static_cast<std::uint8_t>(GuiGimbalCommand::Home)) return 0;
        break;
    case GuiCategory::Pipeline:
        if (command == 1 || command == 2) return 4;
        if (command == 3 || command == 4) return 8;
        if (command == 0x81 || command == 0x82) return 8;
        if (command == 0x83 || command == 0x84) return 6;
        break;
    }
    return -1;
}
std::size_t packet_size(const std::uint8_t* data, std::size_t size)
{
    if (!data || size < kGuiCommandHeaderSize || get(data, 4) != kGuiCommandMagic)
        return 0;
    const int payload = payload_size(static_cast<GuiCategory>(data[4]), data[5]);
    if (payload < 0 || get(data + 6, 2) != static_cast<unsigned>(payload)) return 0;
    return kGuiCommandHeaderSize + payload;
}
}

std::size_t PcGuiProtocol::command_header_size() const { return kGuiCommandHeaderSize; }
std::size_t PcGuiProtocol::command_packet_size(const std::uint8_t* p, std::size_t n) const
{ return packet_size(p, n); }
bool PcGuiProtocol::decode_command(const std::uint8_t* p, std::size_t n, GuiCommand& out) const
{ return decode_gui_command(p, n, out); }
std::vector<std::uint8_t> PcGuiProtocol::encode_command(const GuiCommand& command) const
{ return encode_gui_command(command); }
bool PcGuiProtocol::accepts_frame(std::uint32_t width, std::uint32_t height) const
{ return width == kGuiFrameWidth && height == kGuiFrameHeight; }
bool PcGuiProtocol::encode_metadata(const GuiTrackingMetadata& metadata, std::size_t limit,
                                    std::vector<std::uint8_t>& out) const
{ return encode_tracking_metadata(metadata, limit, out); }

std::vector<std::uint8_t> encode_gui_command(const GuiCommand& command)
{
    const int size = payload_size(command.category, command.command);
    if (size < 0 || command.payload.size() != static_cast<std::size_t>(size)) return {};
    std::vector<std::uint8_t> packet;
    put(packet, kGuiCommandMagic, 4);
    put(packet, static_cast<std::uint8_t>(command.category), 1);
    put(packet, command.command, 1);
    put(packet, size, 2);
    packet.insert(packet.end(), command.payload.begin(), command.payload.end());
    return packet;
}
bool decode_gui_command(const std::uint8_t* data, std::size_t size, GuiCommand& command)
{
    const auto expected = packet_size(data, size);
    if (!expected || size != expected) return false;
    command = {static_cast<GuiCategory>(data[4]), data[5],
               std::vector<std::uint8_t>(data + kGuiCommandHeaderSize, data + size)};
    return true;
}
namespace {
bool display_track(const GuiTrackMetadata& source, GuiTrackMetadata& track)
{
    if (!source.track_id || !std::isfinite(source.confidence) ||
        source.confidence < 0 || source.confidence > 1 ||
        !std::isfinite(source.x) || !std::isfinite(source.y) ||
        !std::isfinite(source.width) || !std::isfinite(source.height) ||
        source.width <= 0 || source.height <= 0) return false;
    // Use double for addition so finite float inputs cannot overflow while clipping.
    const double left = std::max(0.0, static_cast<double>(source.x));
    const double top = std::max(0.0, static_cast<double>(source.y));
    const double right = std::min(640.0, static_cast<double>(source.x) + source.width);
    const double bottom = std::min(480.0, static_cast<double>(source.y) + source.height);
    if (right <= left || bottom <= top) return false;
    track = source;
    track.x = static_cast<float>(left); track.y = static_cast<float>(top);
    track.width = static_cast<float>(right - left);
    track.height = static_cast<float>(bottom - top);
    return true;
}
}

bool encode_tracking_metadata(const GuiTrackingMetadata& metadata,
                              std::size_t limit, std::vector<std::uint8_t>& packet)
{
    if (!metadata.frame_id || metadata.width != 640 || metadata.height != 480 ||
        limit < kGuiMetadataHeaderSize) return false;
    const auto capacity = (std::min(limit, std::size_t{65507}) - kGuiMetadataHeaderSize)
                          / kGuiTrackMetadataSize;
    std::vector<GuiTrackMetadata> selected;
    selected.reserve(std::min(capacity, metadata.tracks.size()));
    // Approved overflow policy: valid selected target first, then stable input
    // order. Exclusion here affects display only, never target/tracker state.
    if (capacity && metadata.selected_track_id) {
        for (const auto& source : metadata.tracks) {
            GuiTrackMetadata track;
            if (source.track_id == metadata.selected_track_id && display_track(source, track)) {
                selected.push_back(track);
                break;
            }
        }
    }
    const bool prioritized = !selected.empty();
    for (const auto& source : metadata.tracks) {
        if (selected.size() == capacity) break;
        if (prioritized && source.track_id == metadata.selected_track_id) continue;
        GuiTrackMetadata track;
        if (display_track(source, track)) selected.push_back(track);
    }
    std::vector<std::uint8_t> encoded;
    encoded.reserve(kGuiMetadataHeaderSize + selected.size() * kGuiTrackMetadataSize);
    put(encoded, kGuiMetadataMagic, 4);
    put(encoded, metadata.frame_id, 8);
    put(encoded, metadata.rtp_timestamp, 4);
    put(encoded, metadata.gui_started_us, 8);
    put(encoded, selected.size(), 4);
    for (const auto& track : selected) {
        put(encoded, track.track_id, 4);
        put_float(encoded, track.x);
        put_float(encoded, track.y);
        put_float(encoded, track.width);
        put_float(encoded, track.height);
        put(encoded, track.class_id, 1);
        put_float(encoded, track.confidence);
    }
    packet = std::move(encoded);
    return true;
}
bool decode_tracking_metadata(const std::uint8_t* data, std::size_t size,
                              GuiTrackingMetadata& metadata)
{
    if (!data || size < kGuiMetadataHeaderSize || size > 65507 ||
        get(data, 4) != kGuiMetadataMagic || !get(data + 4, 8)) return false;
    const auto count = get(data + 24, 4);
    if (count > (size - kGuiMetadataHeaderSize) / kGuiTrackMetadataSize ||
        size != kGuiMetadataHeaderSize + count * kGuiTrackMetadataSize) return false;
    GuiTrackingMetadata decoded;
    decoded.frame_id = get(data + 4, 8);
    decoded.rtp_timestamp = get(data + 12, 4);
    decoded.gui_started_us = get(data + 16, 8);
    for (std::size_t i = 0; i < count; ++i) {
        const auto* p = data + kGuiMetadataHeaderSize + i * kGuiTrackMetadataSize;
        GuiTrackMetadata source, track;
        source.track_id = static_cast<std::uint32_t>(get(p, 4));
        source.x = get_float(p + 4); source.y = get_float(p + 8);
        source.width = get_float(p + 12); source.height = get_float(p + 16);
        source.class_id = p[20]; source.confidence = get_float(p + 21);
        if (display_track(source, track)) decoded.tracks.push_back(track);
    }
    metadata = std::move(decoded);
    return true;
}
