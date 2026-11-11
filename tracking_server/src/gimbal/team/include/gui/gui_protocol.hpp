#pragma once

#include "app/pipeline_result.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// PC application contract. Multibyte values (including float bits) are big endian.
constexpr std::uint32_t kGuiCommandMagic = 0x4c574952U; // LWIR
constexpr std::size_t kGuiCommandHeaderSize = 8;
constexpr std::uint32_t kGuiMetadataMagic = 0x4c574952U;
constexpr std::size_t kGuiMetadataHeaderSize = 28;
constexpr std::size_t kGuiTrackMetadataSize = 25;
constexpr std::uint32_t kGuiFrameWidth = 640;
constexpr std::uint32_t kGuiFrameHeight = 480;

enum class GuiCategory : std::uint8_t { System = 1, Mode = 2, Gimbal = 3, Pipeline = 4 };
// Pipeline 요청은 1..4, 보드 응답은 해당 요청 번호에 0x80을 더한다.
enum class GuiPipelineCommand : std::uint8_t {
    ModelCount = 1, TrackerCount = 2, ChangeModel = 3, ChangeTracker = 4
};
enum class GuiPipelinePhase : std::uint8_t { Accepted = 1, Completed = 2 };
// 기존 PC 프로토콜의 결과 코드 값을 공용 요청 결과와 동일하게 유지한다.
using GuiPipelineResult = PipelineResult;
enum class GuiSystem : std::uint8_t { Ack = 1, Init = 2, Status = 3, Ping = 4 };
enum class GuiModeCommand : std::uint8_t { Auto = 1, Manual = 2 };
enum class GuiGimbalCommand : std::uint8_t { Move = 1, Home = 3 };
enum class GuiResult : std::uint8_t {
    Ok, InvalidCommand, InvalidPayload, InvalidState, InvalidTarget, NotReady
};

struct GuiCommand {
    GuiCategory category = GuiCategory::System;
    std::uint8_t command = 0;
    std::vector<std::uint8_t> payload;
};

// Communication boundary keeps the full PC uint32 ID range independently of
// the tracker library's signed int IDs. Unknown class is 255.
struct GuiTrackMetadata {
    std::uint32_t track_id = 0;
    float x = 0, y = 0, width = 0, height = 0;
    std::uint8_t class_id = 255;
    float confidence = 0;
};

struct GuiTrackingMetadata {
    std::uint64_t frame_id = 0;
    std::uint32_t rtp_timestamp = 0;
    std::uint64_t gui_started_us = 0; // system_clock epoch microseconds
    // PC protocol has no dimensions on the wire; original pixels must be 640x480.
    std::uint32_t width = 640;
    std::uint32_t height = 480;
    std::vector<GuiTrackMetadata> tracks;
    // Local prioritization hint, never serialized. Zero means no selected target.
    // ID-based following is deferred, so the runtime currently leaves it zero.
    std::uint32_t selected_track_id = 0;
};

// Wire strategy: socket and pipeline classes never inspect protocol fields.
class IGuiProtocol {
public:
    virtual ~IGuiProtocol() = default;
    virtual std::size_t command_header_size() const = 0;
    // Zero means invalid header. Validate declared length before reading a body.
    virtual std::size_t command_packet_size(const std::uint8_t*, std::size_t) const = 0;
    virtual bool decode_command(const std::uint8_t*, std::size_t, GuiCommand&) const = 0;
    virtual std::vector<std::uint8_t> encode_command(const GuiCommand&) const = 0;
    virtual bool accepts_frame(std::uint32_t width, std::uint32_t height) const = 0;
    virtual bool encode_metadata(const GuiTrackingMetadata&, std::size_t,
                                 std::vector<std::uint8_t>&) const = 0;
};

// Single supported wire implementation; no configuration selector is needed.
class PcGuiProtocol final : public IGuiProtocol {
public:
    std::size_t command_header_size() const override;
    std::size_t command_packet_size(const std::uint8_t*, std::size_t) const override;
    bool decode_command(const std::uint8_t*, std::size_t, GuiCommand&) const override;
    std::vector<std::uint8_t> encode_command(const GuiCommand&) const override;
    bool accepts_frame(std::uint32_t width, std::uint32_t height) const override;
    bool encode_metadata(const GuiTrackingMetadata&, std::size_t,
                         std::vector<std::uint8_t>&) const override;
};

// PC codecs also exposed for independent wire fixtures and receiver tests.
std::vector<std::uint8_t> encode_gui_command(const GuiCommand&);
bool decode_gui_command(const std::uint8_t*, std::size_t, GuiCommand&);
bool encode_tracking_metadata(const GuiTrackingMetadata&, std::size_t,
                              std::vector<std::uint8_t>&);
bool decode_tracking_metadata(const std::uint8_t*, std::size_t, GuiTrackingMetadata&);
