/* Exact histogram tone mapping used by the optional E10 split-frame path. */
#include <stddef.h>
#include <stdint.h>

void percentile16(const uint16_t *src, size_t count, uint32_t *lo, uint32_t *hi)
{
    uint32_t hist[4096] = {0};
    for (size_t i = 0; i < count; ++i) ++hist[src[i] >> 4];
    const double lower = (double)count * 0.01;
    const double upper = (double)count * 0.99;
    uint32_t sum = 0, l = 0, h = 0;
    int found = 0;
    for (uint32_t i = 0; i < 4096; ++i) {
        sum += hist[i];
        if (!found && (double)sum >= lower) { l = i; found = 1; }
        if ((double)sum >= upper) { h = i; break; }
    }
    *lo = l;
    *hi = h;
}

void apply_tone(const uint16_t *restrict src, uint8_t *restrict dst, size_t count, float lo, float scale)
{
    for (size_t i = 0; i < count; ++i) {
        float v = ((float)src[i] - lo) * scale;
        if (v < 0.0f) v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        dst[i] = (uint8_t)v;
    }
}
