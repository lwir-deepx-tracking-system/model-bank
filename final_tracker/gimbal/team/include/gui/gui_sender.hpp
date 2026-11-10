#pragma once

#include <atomic>
#include <mutex>
#include <map>
#include <cstdint>
#include <gst/gst.h>

#include <sys/socket.h>

#include "config/app_config.hpp"
#include "common/track.hpp"
#include "gui/gui_protocol.hpp"

struct _GstAppSrc;
struct _GstElement;

// Orange Pi -> PC GUI 방향의 영상과 Tracking metadata 송신을 담당한다.
//
// 입력:
//   TrackingResult(frame + tracks)
// 출력:
//   1) bbox가 없는 JPEG or H.264/RTP/UDP 영상
//   2) frame_id, RTP timestamp, bbox, track_id를 담은 metadata UDP packet
//
// 전체 흐름:
// TrackingThread -> gui_track_queue -> GuiSenderThread -> GuiSender
//   -> GStreamer 영상 송신
//   -> 별도 UDP socket을 통한 Tracking metadata 송신
//
// 영상과 metadata를 분리해 GUI가 원본 pixel 좌표의 bbox를 직접 그리도록 한다.
// 이 클래스가 GStreamer pipeline과 metadata socket의 수명을 모두 소유한다.
struct GuiVideoStats
{
    std::uint64_t submitted_frames;
    std::uint64_t encoded_frames;
    std::uint64_t preprocessing_ns;
    std::uint64_t processing_samples;
    std::uint64_t encode_rtp_ns;
    std::uint64_t rtp_bytes; // RTP header + payload; excludes UDP/IP/Ethernet headers.
};

class GuiSender
{
private:
    GuiVideoConfig video_config_;
    GuiMetadataConfig metadata_config_;
    std::unique_ptr<IGuiProtocol> protocol_;
    std::uint16_t clip_min_;
    std::uint16_t clip_max_;

    // GuiSender가 GStreamer pipeline 수명을 직접 소유한다. appsrc_는 pipeline
    // 내부 element를 가리키며 pipeline_을 해제할 때 함께 정리된다.
    _GstElement* pipeline_ = nullptr;
    _GstAppSrc* appsrc_ = nullptr;
    int frame_width_ = 0;
    int frame_height_ = 0;
    std::uint64_t frame_index_ = 0;
    std::uint64_t last_report_ns_ = 0;
    std::atomic<std::uint64_t> submitted_frames_{0}, encoded_frames_{0}, rtp_bytes_{0};
    std::atomic<std::uint64_t> preprocessing_ns_{0}, processing_samples_{0}, encode_rtp_ns_{0};
    std::mutex processing_mutex_;
    std::map<std::uint32_t, std::uint64_t> processing_started_;
    static GstPadProbeReturn observe_rtp(GstPad*, GstPadProbeInfo*, gpointer);

    // bbox metadata용 POSIX UDP socket과 목적지 주소도 GuiSender가 소유한다.
    int metadata_fd_ = -1;
    sockaddr_storage metadata_address_{};
    socklen_t metadata_address_size_ = 0;

    // 최초 frame 크기로 영상 pipeline을 만들고 PLAYING 상태로 시작한다.
    // 이미 시작된 경우에는 입력 크기가 기존 pipeline과 같은지만 확인한다.
    bool start_gstreamer_pipeline(int width, int height);

    // 변환된 BGR frame을 appsrc에 전달한다. encode, RTP packetize, UDP 송신은
    // downstream GStreamer element가 비동기로 수행한다.
    bool push_frame_to_gstreamer(const cv::Mat& image);

    // 영상 RTP 포트와 독립된 metadata UDP 목적지를 최초 1회 준비한다.
    bool open_metadata_socket();

    // 현재 frame의 Track 목록을 명시적 wire format으로 직렬화해 전송한다.
    bool send_tracking_metadata(
        const TrackingResult& result,
        std::uint64_t gui_started_us);

public:
    GuiSender(
        GuiVideoConfig video_config,
        GuiMetadataConfig metadata_config,
        std::uint16_t clip_min,
        std::uint16_t clip_max);
    ~GuiSender();

    GuiSender(const GuiSender&) = delete;
    GuiSender& operator=(const GuiSender&) = delete;

    // 한 TrackingResult를 두 GUI 채널로 전달한다.
    // CV_16UC1 원본을 clip 범위 기준 8-bit BGR로 변환하고, metadata를 별도
    // UDP 포트로 보낸 뒤 영상 frame을 GStreamer appsrc에 전달한다.
    bool send(
        const TrackingResult& result,
        std::uint64_t gui_started_us = 0,
        bool include_metadata = true);
    bool send_bypass(
        const FrameContextPtr& frame,
        std::uint64_t gui_started_us = 0);

    // 영상 pipeline과 metadata socket을 함께 닫아 송신 자원을 정리한다.
    bool finish(); // Send EOS and wait up to five seconds; keeps statistics.
    void stop();
    GuiVideoStats stats() const;
};
