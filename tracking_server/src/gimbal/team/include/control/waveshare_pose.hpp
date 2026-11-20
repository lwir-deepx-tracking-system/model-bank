#pragma once

#include "control/gimbal_device.hpp"
#include <cmath>
#include <cstdlib>
#include <string_view>

// Waveshare General_Driver baseInfoFeedback: flat numeric JSON, T=1001,
// pan and tilt in degrees. Ignore command echoes and other packet types.
inline std::optional<GimbalPose> parse_waveshare_pose(std::string_view line)
{
    std::size_t i = 0;
    const auto whitespace = [&] { while (i < line.size() &&
        (line[i] == ' ' || line[i] == '\t' || line[i] == '\r')) ++i; };
    const auto consume = [&](char c) {
        whitespace();
        if (i == line.size() || line[i] != c) return false;
        ++i; return true;
    };
    if (line.size() > 4096 || !consume('{')) return std::nullopt;
    bool type = false, pan = false, tilt = false;
    GimbalPose result;
    for (;;) {
        if (!consume('"')) return std::nullopt;
        const auto key_start = i;
        while (i < line.size() && line[i] != '"') {
            if (line[i] == '\\' || static_cast<unsigned char>(line[i]) < 32) return std::nullopt;
            ++i;
        }
        const auto key = line.substr(key_start, i - key_start);
        if (!consume('"') || !consume(':')) return std::nullopt;
        whitespace();
        const auto start = i;
        if (i < line.size() && line[i] == '-') ++i;
        if (i == line.size() || line[i] < '0' || line[i] > '9') return std::nullopt;
        if (line[i] == '0') ++i;
        else while (i < line.size() && line[i] >= '0' && line[i] <= '9') ++i;
        if (i < line.size() && line[i] == '.') {
            const auto first = ++i;
            while (i < line.size() && line[i] >= '0' && line[i] <= '9') ++i;
            if (i == first) return std::nullopt;
        }
        if (i < line.size() && (line[i] == 'e' || line[i] == 'E')) {
            ++i;
            if (i < line.size() && (line[i] == '+' || line[i] == '-')) ++i;
            const auto first = i;
            while (i < line.size() && line[i] >= '0' && line[i] <= '9') ++i;
            if (i == first) return std::nullopt;
        }
        const auto token = std::string(line.substr(start, i - start));
        char* end = nullptr;
        const double value = std::strtod(token.c_str(), &end);
        if (end != token.c_str() + token.size() || !std::isfinite(value)) return std::nullopt;
        if (key == "T") { if (type || value != 1001) return std::nullopt; type = true; }
        if (key == "pan") { if (pan || std::abs(value) > 360) return std::nullopt; pan = true; result.pan = value; }
        if (key == "tilt") { if (tilt || std::abs(value) > 360) return std::nullopt; tilt = true; result.tilt = value; }
        if (consume('}')) break;
        if (!consume(',')) return std::nullopt;
    }
    whitespace();
    if (i != line.size() || !type || !pan || !tilt) return std::nullopt;
    return result;
}
