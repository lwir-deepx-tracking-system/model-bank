"""Build an offline tracker-only library from a local, pinned team checkout."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import tempfile


REV = "0fe016f7edd069900457a746dc4cc787569c5ae6"
SOURCES = ("src/tracking/bytetrack_tracker.cpp", "src/tracking/ocsort_tracker.cpp",
           "src/tracking/kalman_filter.cpp", "src/common/logger.cpp")
HEADERS = (
    "include/common/detection.hpp", "include/common/track.hpp", "include/common/logger.hpp",
    "include/config/app_config.hpp", "include/tracking/bytetrack_tracker.hpp",
    "include/tracking/ocsort_tracker.hpp", "include/tracking/kalman_filter.hpp",
    "include/tracking/tracker.hpp", "include/tracking/camera_motion.hpp",
)


def adapt(path, data):
    def replace(before, after, count=1):
        nonlocal data
        if data.count(before) != count:
            raise ValueError(f"{path}: expected {count} occurrences of {before!r}")
        data = data.replace(before, after)

    if path in HEADERS[:2]:
        replace('#include "common/frame.hpp"',
                "#include <memory>\nstruct FrameContext;\nusing FrameContextPtr = std::shared_ptr<const FrameContext>;")
    if path == HEADERS[0]:
        replace("    std::string class_name;", "    std::string class_name;\n    int source_index = -1;")
    if path == HEADERS[1]:
        replace("    bool predicted = false;", "    bool predicted = false;\n    int source_index = -1;")
    if path == "include/tracking/bytetrack_tracker.hpp":
        replace("    int class_id = -1;", "    int class_id = -1;\n    int source_index = -1;")
    if path == SOURCES[0]:
        replace("    t.class_id = det.class_id;", "    t.class_id = det.class_id;\n    t.source_index = det.source_index;")
        replace("        det->class_id = d.class_id;", "        det->class_id = d.class_id;\n        det->source_index = d.source_index;")
        replace("        out.confidence = t->score;\n        tracks.push_back(out);",
                "        out.confidence = t->score;\n        out.source_index = t->frame_id == frame_id_ ? t->source_index : -1;\n        tracks.push_back(out);")
    if path == "include/tracking/ocsort_tracker.hpp":
        replace("    int class_id = 0;", "    int class_id = 0;\n    int source_index = -1;")
        replace("create_track(const ocsort::Box5& det, int class_id);",
                "create_track(const ocsort::Box5& det, int class_id, int source_index);")
        replace("update(ocsort::OCTrack& t, const ocsort::Box5* det, int class_id) const;",
                "update(ocsort::OCTrack& t, const ocsort::Box5* det, int class_id, int source_index = -1) const;")
    if path == SOURCES[1]:
        replace("create_track(const Box5& det, int class_id)", "create_track(const Box5& det, int class_id, int source_index)")
        replace("update(OCTrack& t, const Box5* det, int class_id) const",
                "update(OCTrack& t, const Box5* det, int class_id, int source_index) const")
        replace("    t.class_id = class_id;", "    t.class_id = class_id;\n    t.source_index = source_index;", count=2)
        replace("    std::vector<int> det_cls, second_cls;", "    std::vector<int> det_cls, second_cls, det_src, second_src;")
        for prefix in ("det", "second"):
            replace(f"            {prefix}_cls.push_back(d.class_id);",
                    f"            {prefix}_cls.push_back(d.class_id);\n            {prefix}_src.push_back(d.source_index);")
        replace("update(trackers_[t], &dets[d], det_cls[d]);",
                "update(trackers_[t], &dets[d], det_cls[d], det_src[d]);", count=2)
        replace("update(trackers_[t], &dets_second[d], second_cls[d]);",
                "update(trackers_[t], &dets_second[d], second_cls[d], second_src[d]);")
        replace("create_track(dets[d], det_cls[d])", "create_track(dets[d], det_cls[d], det_src[d])")
        replace("            r.class_id = t.class_id;", "            r.class_id = t.class_id;\n            r.source_index = t.source_index;")
    return data


def build(repo, output, cxx):
    root = Path(__file__).resolve().parent
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lwirtk-build-") as tmp:
        tmp = Path(tmp)
        for path in SOURCES + HEADERS:
            # Read Git objects, not dirty or sparse checkout files.
            data = subprocess.check_output(["git", "-C", str(repo), "show", f"{REV}:{path}"]).decode("utf-8").replace("\r\n", "\n")
            dest = tmp / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(adapt(path, data), encoding="utf-8")
        dll = tmp / output.name
        platform = ["-static-libstdc++", "-static-libgcc", "-static"] if os.name == "nt" else ["-fPIC", "-pthread"]
        subprocess.run([cxx, "-std=c++17", "-O3", "-shared", "-fvisibility=hidden", *platform, "-I", str(tmp / "include"),
                        str(root / "team_tracker_abi.cpp"), *(str(tmp / path) for path in SOURCES),
                        "-o", str(dll)], check=True)
        # Keep the last working library if compilation fails.
        shutil.copy2(dll, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team_checkout", type=Path)
    parser.add_argument("output_library", type=Path)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    args = parser.parse_args()
    print(build(args.team_checkout, args.output_library, args.cxx))
