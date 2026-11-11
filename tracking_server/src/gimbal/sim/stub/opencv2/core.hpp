#pragma once
// Minimal stand-in for <opencv2/core.hpp>, ONLY for building sim.cpp and tests/test_v27_control.cpp on a machine
// without OpenCV (they need cv::Mat's size, type and emptiness, nothing else). On the device use the real OpenCV.
#define CV_16UC1 2
namespace cv {
struct Mat {
    int rows = 0, cols = 0, type_ = 0;
    Mat() = default;
    Mat(int r, int c, int t) : rows(r), cols(c), type_(t) {}
    int type() const { return type_; }
    bool empty() const { return rows <= 0 || cols <= 0; }
};
}
