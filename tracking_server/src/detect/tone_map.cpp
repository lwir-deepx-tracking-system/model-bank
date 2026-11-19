/* Exact histogram tone mapping used by the optional E10 split-frame path. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

extern "C" void percentile16(const uint16_t *src, size_t count, uint32_t *lo, uint32_t *hi)
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

extern "C" void apply_tone(const uint16_t *__restrict src, uint8_t *__restrict dst,
                           size_t count, float lo, float scale)
{
    for (size_t i = 0; i < count; ++i) {
        float v = ((float)src[i] - lo) * scale;
        if (v < 0.0f) v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        dst[i] = (uint8_t)v;
    }
}

static void tile_rgb(const uint8_t *gray, uint8_t *rgb)
{
    constexpr int width = 384;
#ifdef __aarch64__
    for (int col = 0; col < width; col += 16) {
        const uint8x16_t row = vld1q_u8(gray + col);
        const uint8x16x3_t triplets = {{row, row, row}};
        vst3q_u8(rgb + 3 * col, triplets);
    }
#else
    for (int col = 0; col < width; ++col) {
        const uint8_t pixel = gray[col];
        rgb[3 * col] = rgb[3 * col + 1] = rgb[3 * col + 2] = pixel;
    }
#endif
}

extern "C" void prepare_split_u16(const uint16_t *raw, uint8_t *canvas,
                                  uint8_t *left, uint8_t *right, float lo, float scale)
{
    constexpr size_t width = 640, height = 480, padding = 16;
    constexpr size_t tile_bytes = 384 * 3, right_col = 256;
    uint8_t *gray = canvas + padding * width;
    for (size_t row = 0; row < height; ++row) {
        uint8_t *gray_row = gray + row * width;
        apply_tone(raw + row * width, gray_row, width, lo, scale);
        tile_rgb(gray_row, left + row * tile_bytes);
        tile_rgb(gray_row + right_col, right + row * tile_bytes);
    }
    for (size_t row = 0; row < padding; ++row) {
        memcpy(canvas + row * width, gray, width);
        memcpy(gray + (height + row) * width, gray + (height - 1) * width, width);
    }
}
