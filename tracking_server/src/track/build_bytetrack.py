"""Build the checked-in ByteTrack I/O sources with pinned team dependencies."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import tempfile


REV = "0fe016f7edd069900457a746dc4cc787569c5ae6"
SOURCES = ("src/tracking/bytetrack_tracker.cpp",
           "src/tracking/kalman_filter.cpp", "src/common/logger.cpp")
HEADERS = (
    "include/common/detection.hpp", "include/common/track.hpp", "include/common/logger.hpp",
    "include/config/app_config.hpp", "include/tracking/bytetrack_tracker.hpp",
    "include/tracking/kalman_filter.hpp",
    "include/tracking/tracker.hpp", "include/tracking/camera_motion.hpp",
)


RUNTIME_IO_SOURCES = (
    "include/common/detection.hpp",
    "include/common/track.hpp",
    "include/tracking/bytetrack_tracker.hpp",
    "src/tracking/bytetrack_tracker.cpp",
)


def build(repo, output, cxx):
    root = Path(__file__).resolve().parent
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lwirtk-build-") as tmp:
        tmp = Path(tmp)
        for path in SOURCES + HEADERS:
            dest = tmp / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            if path in RUNTIME_IO_SOURCES:
                shutil.copyfile(root / "bytetrack_runtime" / path, dest)
            else:
                # Read the pinned Git objects even from sparse or dirty checkouts.
                data = subprocess.check_output(["git", "-C", str(repo), "show", f"{REV}:{path}"])
                dest.write_bytes(data)
        # This library passes rows only and does not need the OpenCV frame definition.
        (tmp / "include/common/frame.hpp").write_text(
            "#pragma once\n#include <memory>\nstruct FrameContext;\n"
            "using FrameContextPtr = std::shared_ptr<const FrameContext>;\n", encoding="utf-8")
        dll = tmp / output.name
        platform = ["-static-libstdc++", "-static-libgcc", "-static"] if os.name == "nt" else ["-fPIC", "-pthread"]
        subprocess.run([cxx, "-std=c++17", "-O3", "-shared", "-fvisibility=hidden", *platform, "-I", str(tmp / "include"),
                        str(root / "bytetrack.cpp"), *(str(tmp / path) for path in SOURCES),
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
