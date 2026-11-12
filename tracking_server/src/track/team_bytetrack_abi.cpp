// Tracker-only ABI for offline playback. No camera, control, or deployment interfaces.
#include "tracking/bytetrack_tracker.hpp"
#include "common/logger.hpp"

#include <mutex>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

struct DetRow { float x1, y1, x2, y2, score; int cls; };
struct TrackRow { float x1, y1, x2, y2, score; int id, cls, source_index; };

EXPORT int lwirtk_abi() { return 2; }

EXPORT Tracker* lwirtk_create()
{
    try {
        static std::once_flag logger_init;
        std::call_once(logger_init, [] { Logger::set_level(LogLevel::WARN); });
        TrackingConfig cfg;
        cfg.track_threshold = 0.25F;
        cfg.new_track_threshold = 0.25F;
        cfg.fuse_score = true;
        cfg.kalman_format = "xywh";
        return new ByteTrackTracker(cfg);
    } catch (...) { return nullptr; }
}

EXPORT void lwirtk_close(Tracker* t) { delete t; }

EXPORT int lwirtk_reset(Tracker* t)
{
    if (!t) return -1;
    try { t->reset(); return 0; }
    catch (...) { return -2; }
}

EXPORT int lwirtk_step(Tracker* t, const DetRow* inputs, int n, TrackRow* out, int cap)
{
    if (!t || n < 0 || cap < n || (n && (!inputs || !out))) return -1;
    try {
        std::vector<Detection> detections;
        detections.reserve(n);
        for (int i = 0; i < n; ++i) {
            const auto& d = inputs[i];
            Detection row;
            row.x = d.x1; row.y = d.y1;
            row.width = d.x2 - d.x1; row.height = d.y2 - d.y1;
            row.confidence = d.score; row.class_id = d.cls; row.source_index = i;
            detections.push_back(row);
        }
        auto tracks = t->track(detections);
        // The selected presets only return current observations, never predictions.
        if (tracks.size() > static_cast<std::size_t>(cap)) return -3;
        for (std::size_t j = 0; j < tracks.size(); ++j) {
            const auto& r = tracks[j];
            if (r.predicted || r.source_index < 0 || r.source_index >= n) return -4;
            out[j] = {r.x, r.y, r.x + r.width, r.y + r.height, r.confidence,
                      r.track_id, r.class_id, r.source_index};
        }
        return static_cast<int>(tracks.size());
    } catch (...) { return -2; }
}
