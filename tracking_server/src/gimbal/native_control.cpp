#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <new>
#include <optional>
#include <string>

namespace {
using Box = std::array<double, 4>;
using Point = std::array<double, 2>;
using Warp = std::array<double, 6>;

template <std::size_t N>
bool numbers(PyObject* sequence, std::array<double, N>& values, Py_ssize_t start = 0)
{
    PyObject* fast = PySequence_Fast(sequence, "expected coordinates");
    if (!fast) return false;
    if (PySequence_Fast_GET_SIZE(fast) < start + static_cast<Py_ssize_t>(N)) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "incomplete coordinates");
        return false;
    }
    for (std::size_t index = 0; index < N; ++index)
        values[index] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(fast, start + index));
    Py_DECREF(fast);
    return !PyErr_Occurred();
}

Box centre(const Box& box)
{
    return {(box[0] + box[2]) / 2., (box[1] + box[3]) / 2.,
            box[2] - box[0], box[3] - box[1]};
}

void to_camera(Box& box, double dx, double dy, double width, double height)
{
    if (dx == 0. && dy == 0.) return;
    box[0] = std::min(std::max(box[0] - dx, 0.), width - 1.);
    box[1] = std::min(std::max(box[1] - dy, 0.), height - 1.);
    box[2] = std::max(std::min(box[2] - dx, width), box[0] + 1.);
    box[3] = std::max(std::min(box[3] - dy, height), box[1] + 1.);
}

void limit_vertical(Box& box, double aim, double screen)
{
    const double offset = std::max(0., box[1] + aim * (box[3] - box[1]) - screen);
    box[1] -= offset;
    box[3] -= offset;
}

bool append_fixed(std::string& out, double value, int decimals, double* rounded = nullptr)
{
    char* digits = PyOS_double_to_string(value, 'f', decimals, 0, nullptr);
    if (!digits) return false;
    if (rounded) *rounded = PyOS_string_to_double(digits, nullptr, nullptr);
    out += ' ';
    out += digits;
    PyMem_Free(digits);
    return !PyErr_Occurred();
}

bool append_box(std::string& out, const Box& box, Box* rounded = nullptr)
{
    const Box fields{box[0], box[1], box[2] - box[0], box[3] - box[1]};
    for (std::size_t i = 0; i < fields.size(); ++i)
        if (!append_fixed(out, fields[i], 1, rounded ? &(*rounded)[i] : nullptr))
            return false;
    if (rounded) {
        (*rounded)[2] += (*rounded)[0];
        (*rounded)[3] += (*rounded)[1];
    }
    return true;
}

PyObject* prepare(PyObject*, PyObject* args)
{
    PyObject* people;
    PyObject* correction;
    long long hidden, selected;
    double width, height;
    if (!PyArg_ParseTuple(args, "OLOLdd", &people, &hidden, &correction, &selected, &width, &height))
        return nullptr;
    Point shift{};
    if (correction != Py_None && !numbers(correction, shift)) return nullptr;
    PyObject* rows = PySequence_Fast(people, "expected person rows");
    if (!rows) return nullptr;
    PyObject* real = PyList_New(0);
    std::optional<Box> seen;
    int matches = 0;
    const auto count = PySequence_Fast_GET_SIZE(rows);
    for (Py_ssize_t index = 0; real && index < count; ++index) {
        PyObject* row = PySequence_Fast(PySequence_Fast_GET_ITEM(rows, index), "expected person row");
        if (!row) { Py_CLEAR(real); break; }
        if (PySequence_Fast_GET_SIZE(row) < 6) {
            PyErr_SetString(PyExc_ValueError, "incomplete person row");
            Py_DECREF(row);
            Py_CLEAR(real);
            break;
        }
        PyObject* id_object = PySequence_Fast_GET_ITEM(row, 0);
        PyObject* integer = PyNumber_Long(id_object);
        const long long id = integer ? PyLong_AsLongLong(integer) : -1;
        Py_XDECREF(integer);
        if (PyErr_Occurred()) { Py_DECREF(row); Py_CLEAR(real); break; }
        if (hidden && id == hidden) { Py_DECREF(row); continue; }

        Box box;
        if (!numbers(row, box, 1)) { Py_DECREF(row); Py_CLEAR(real); break; }
        to_camera(box, shift[0], shift[1], width, height);
        if (id == selected && ++matches == 1) seen = centre(box);
        PyObject* corrected = PySequence_Fast_GET_ITEM(rows, index);
        if (shift[0] != 0. || shift[1] != 0.) {
            corrected = Py_BuildValue("(OddddO)", id_object, box[0], box[1], box[2], box[3],
                                      PySequence_Fast_GET_ITEM(row, 5));
        } else {
            Py_INCREF(corrected);
        }
        if (!corrected || PyList_Append(real, corrected) < 0) Py_CLEAR(real);
        Py_XDECREF(corrected);
        Py_DECREF(row);
    }
    Py_DECREF(rows);
    if (!real) return nullptr;
    PyObject* target = matches == 1 ? Py_BuildValue("(dddd)", (*seen)[0], (*seen)[1], (*seen)[2], (*seen)[3])
                                    : Py_NewRef(Py_None);
    if (!target) { Py_DECREF(real); return nullptr; }
    return Py_BuildValue("NN", real, target);
}

PyObject* observations(PyObject*, PyObject* args)
{
    PyObject* input;
    long long selected;
    int tilt;
    double aim, screen;
    if (!PyArg_ParseTuple(args, "OLpdd", &input, &selected, &tilt, &aim, &screen)) return nullptr;
    PyObject* rows = PySequence_Fast(input, "expected person rows");
    if (!rows) return nullptr;
    const auto count = PySequence_Fast_GET_SIZE(rows);
    std::string text;
    text.reserve(count * 42);
    Box selected_box{};
    int matches = 0;
    for (Py_ssize_t index = 0; index < count; ++index) {
        PyObject* row = PySequence_Fast_GET_ITEM(rows, index);
        std::array<double, 6> fields;
        if (!numbers(row, fields)) { Py_DECREF(rows); return nullptr; }
        // APP.gimbal_people passes integer IDs. Constrain conversion to the
        // exact integer range of a double; Python can handle other inputs.
        if (!std::isfinite(fields[0]) || std::abs(fields[0]) >= 0x1p53) {
            Py_DECREF(rows);
            PyErr_SetString(PyExc_OverflowError, "person ID outside native range");
            return nullptr;
        }
        const long long id = static_cast<long long>(fields[0]);
        Box box{fields[1], fields[2], fields[3], fields[4]};
        const bool target = id == selected;
        if (target) {
            ++matches;
            if (matches == 1) {
                std::string unused;
                if (!append_box(unused, box, &selected_box)) { Py_DECREF(rows); return nullptr; }
            }
            if (tilt) limit_vertical(box, aim, screen);
        }
        text += ' ';
        text += std::to_string(id);
        if (!append_box(text, box) || !append_fixed(text, fields[5], 3)) { Py_DECREF(rows); return nullptr; }
    }
    Py_DECREF(rows);
    PyObject* target = matches == 1 ? Py_BuildValue("(Ldddd)", selected, selected_box[0], selected_box[1],
                                                    selected_box[2], selected_box[3]) : Py_NewRef(Py_None);
    if (!target) return nullptr;
    return Py_BuildValue("sN", text.c_str(), target);
}

PyObject* target_suffix(PyObject*, PyObject* args)
{
    const char* kind;
    PyObject* coords;
    long long id;
    int tilt, xywh;
    double aim, screen;
    if (!PyArg_ParseTuple(args, "sLOppdd", &kind, &id, &coords, &xywh, &tilt, &aim, &screen))
        return nullptr;
    Box box;
    if (!numbers(coords, box)) return nullptr;
    if (!std::all_of(box.begin(), box.end(), [](double value) { return std::isfinite(value); }))
        return PyUnicode_FromString("");
    std::string text = std::string(" ") + kind + " " + std::to_string(id);
    if (xywh) {
        if (tilt && box[1] + aim * box[3] > screen + .05)
            return PyUnicode_FromString("");
        for (double value : box)
            if (!append_fixed(text, value, 1)) return nullptr;
    } else {
        if (tilt) limit_vertical(box, aim, screen);
        if (!append_box(text, box)) return nullptr;
    }
    return PyUnicode_FromStringAndSize(text.data(), text.size());
}

PyObject* velocity(PyObject*, PyObject* args)
{
    long long id;
    double vx, vy;
    int tilt;
    if (!PyArg_ParseTuple(args, "Lddp", &id, &vx, &vy, &tilt)) return nullptr;
    if (tilt) vy = std::min(0., vy);
    if (!std::isfinite(vx) || !std::isfinite(vy) || (vx == 0. && vy == 0.))
        return PyUnicode_FromString("");
    std::string text = " VEL " + std::to_string(id);
    if (!append_fixed(text, vx, 1) || !append_fixed(text, vy, 1)) return nullptr;
    return PyUnicode_FromStringAndSize(text.data(), text.size());
}

constexpr double canvas_y = 16.;
constexpr double drop_window = .18;
constexpr double recovery_time = .35;
constexpr double max_frame_gap = .25;

struct Observation {
    double time;
    Box box;
    Point correction;
};

std::optional<Point> project(const Observation& sample, double time, Point correction, PyObject* motion)
{
    PyObject* matrix = PyObject_CallFunction(motion, "dd", sample.time, time);
    if (!matrix) return {};
    if (matrix == Py_None) { Py_DECREF(matrix); return {}; }
    Py_buffer buffer;
    const bool acquired = PyObject_GetBuffer(matrix, &buffer, PyBUF_CONTIG_RO | PyBUF_FORMAT) == 0;
    Py_DECREF(matrix);
    if (!acquired) return {};
    if (buffer.len != static_cast<Py_ssize_t>(sizeof(Warp)) || buffer.itemsize != sizeof(double)
        || !buffer.format || buffer.format[0] != 'd' || buffer.format[1] != '\0') {
        PyBuffer_Release(&buffer);
        PyErr_SetString(PyExc_ValueError, "expected a contiguous double camera transform");
        return {};
    }
    Warp warp;
    std::memcpy(warp.data(), buffer.buf, sizeof(warp));
    PyBuffer_Release(&buffer);

    const double x = sample.box[0] + sample.correction[0];
    const double y = sample.box[1] - sample.box[3] / 2. + sample.correction[1] + canvas_y;
    double scale = std::sqrt(std::abs(warp[0] * warp[4] - warp[1] * warp[3]));
    if (scale == 0.) scale = 1.;
    const double top = warp[3] * x + warp[4] * y + warp[5] - canvas_y - correction[1];
    const double height = sample.box[3] * scale;
    if (!std::isfinite(top) || !std::isfinite(height) || height <= 0.) return {};
    return Point{top, height};
}

struct Reentry {
    std::deque<Observation> recent;
    std::optional<Observation> anchor;
    std::optional<double> visible_since;
    bool seen = false;

    bool step(double time, const std::optional<Box>& box, Point correction, PyObject* motion)
    {
        if (!box) {
            visible_since.reset();
            if (!anchor && !recent.empty()) anchor = recent.back();
            return seen;
        }
        seen = true;
        const double top = (*box)[1] - (*box)[3] / 2.;
        const double height = (*box)[3];
        const auto previous = recent.empty() ? std::nullopt : std::optional<Observation>(recent.back());
        if (!anchor) {
            while (!recent.empty() && time - recent.front().time > drop_window) recent.pop_front();
            const auto oldest = recent.empty() ? std::nullopt : std::optional<Observation>(recent.front());
            for (const auto& earlier : {oldest,
                     recent.size() == 1 ? std::nullopt : previous}) {
                if (!earlier || !(0. < time - earlier->time && time - earlier->time <= drop_window)) continue;
                const auto region = project(*earlier, time, correction, motion);
                if (PyErr_Occurred()) return false;
                if (region && top - (*region)[0] >= std::max(20., .18 * (*region)[1]) &&
                    (*box)[1] - ((*region)[0] + (*region)[1] / 2.) >= std::max(10., .09 * (*region)[1])) {
                    anchor = earlier;
                    visible_since.reset();
                    break;
                }
            }
        } else if (previous) {
            const auto region = project(*anchor, time, correction, motion);
            if (PyErr_Occurred()) return false;
            const auto last = project(*previous, time, correction, motion);
            if (PyErr_Occurred()) return false;
            const double last_top = last ? (*last)[0] : previous->box[1] - previous->box[3] / 2.;
            const double reference_height = region ? (*region)[1] : anchor->box[3];
            if (height < .8 * reference_height) {
                visible_since.reset();
            } else if (region && top <= (*region)[0] + .1 * reference_height) {
                anchor.reset();
            } else if (0. < time - previous->time && time - previous->time <= max_frame_gap
                       && std::abs(top - last_top) <= std::max(3., .05 * height)) {
                if (!visible_since) visible_since = time;
                if (time - *visible_since >= recovery_time) anchor.reset();
            } else {
                visible_since.reset();
            }
            if (!anchor) {
                recent.clear();
                visible_since.reset();
            }
        }
        if (recent.size() == 12) recent.pop_front();
        recent.push_back({time, *box, correction});
        return anchor.has_value();
    }
};

constexpr char reentry_capsule[] = "lwir.Reentry";
void destroy_reentry(PyObject* object)
{
    delete static_cast<Reentry*>(PyCapsule_GetPointer(object, reentry_capsule));
}

PyObject* new_reentry(PyObject*, PyObject*)
{
    auto* state = new Reentry{};
    PyObject* object = PyCapsule_New(state, reentry_capsule, destroy_reentry);
    if (!object) delete state;
    return object;
}

PyObject* reentry_step(PyObject*, PyObject* args)
{
    PyObject *object, *coords, *correction, *motion;
    double time;
    if (!PyArg_ParseTuple(args, "OdOOO", &object, &time, &coords, &correction, &motion)) return nullptr;
    auto* state = static_cast<Reentry*>(PyCapsule_GetPointer(object, reentry_capsule));
    if (!state) return nullptr;
    std::optional<Box> observed;
    if (coords != Py_None) {
        observed.emplace();
        if (!numbers(coords, *observed)) return nullptr;
    }
    Point shift;
    if (!numbers(correction, shift)) return nullptr;
    const bool held = state->step(time, observed, shift, motion);
    if (PyErr_Occurred()) return nullptr;
    return PyBool_FromLong(held);
}

double overlap(const Box& a, const Box& b)
{
    const double w = std::min(a[2], b[2]) - std::max(a[0], b[0]);
    const double h = std::min(a[3], b[3]) - std::max(a[1], b[1]);
    if (w <= 0. || h <= 0.) return 0.;
    const double common = w * h;
    const double area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - common;
    return area > 0. ? common / area : 0.;
}

PyObject* overlaps(PyObject*, PyObject* args)
{
    PyObject *input, *coords;
    if (!PyArg_ParseTuple(args, "OO", &coords, &input)) return nullptr;
    Box reference;
    if (!numbers(coords, reference, 1)) return nullptr;
    PyObject* rows = PySequence_Fast(input, "expected person rows");
    if (!rows) return nullptr;
    const auto count = PySequence_Fast_GET_SIZE(rows);
    PyObject* result = PyList_New(count);
    for (Py_ssize_t index = 0; result && index < count; ++index) {
        Box box;
        if (!numbers(PySequence_Fast_GET_ITEM(rows, index), box, 1)) { Py_CLEAR(result); break; }
        PyObject* value = PyFloat_FromDouble(overlap(reference, box));
        if (!value) { Py_CLEAR(result); break; }
        PyList_SET_ITEM(result, index, value);
    }
    Py_DECREF(rows);
    return result;
}

PyObject* search_crowd(PyObject*, PyObject* args)
{
    PyObject *input, *ignored;
    long long selected;
    double cx, cy, radius;
    if (!PyArg_ParseTuple(args, "OLdddO", &input, &selected, &cx, &cy, &radius, &ignored)) return nullptr;
    PyObject* rows = PySequence_Fast(input, "expected person rows");
    if (!rows) return nullptr;
    int near = 0, others = 0;
    const auto count = PySequence_Fast_GET_SIZE(rows);
    for (Py_ssize_t index = 0; index < count; ++index) {
        std::array<double, 5> fields;
        if (!numbers(PySequence_Fast_GET_ITEM(rows, index), fields)) { Py_DECREF(rows); return nullptr; }
        if (!std::isfinite(fields[0]) || std::abs(fields[0]) >= 0x1p53) {
            Py_DECREF(rows);
            PyErr_SetString(PyExc_OverflowError, "person ID outside native range");
            return nullptr;
        }
        const long long id = static_cast<long long>(fields[0]);
        if (id == selected) continue;
        if (ignored != Py_None) {
            PyObject* oid = PyLong_FromLongLong(id);
            if (!oid) { Py_DECREF(rows); return nullptr; }
            const int skip = PySequence_Contains(ignored, oid);
            Py_DECREF(oid);
            if (skip < 0) { Py_DECREF(rows); return nullptr; }
            if (skip) continue;
        }
        ++others;
        const double x = (fields[1] + fields[3]) / 2. - cx;
        const double y = (fields[2] + fields[4]) / 2. - cy;
        if (std::hypot(x, y) <= radius) ++near;
    }
    Py_DECREF(rows);
    return Py_BuildValue("ii", near, others);
}

PyObject* filter_velocity(PyObject*, PyObject* args)
{
    double dt, vx, vy, dx, dy, lead_ms;
    if (!PyArg_ParseTuple(args, "dddddd", &dt, &vx, &vy, &dx, &dy, &lead_ms)) return nullptr;
    const double alpha = -std::expm1(-dt / .120);
    vx = (1. - alpha) * vx + alpha * dx / dt;
    vy = (1. - alpha) * vy + alpha * dy / dt;
    const double speed = std::hypot(vx, vy);
    if (!std::isfinite(speed)) return Py_BuildValue("idddd", 0, vx, vy, 0., 0.);
    if (speed < 15.) return Py_BuildValue("idddd", 1, vx, vy, 0., 0.);
    const double scale = std::min(std::max(0., std::min(400., lead_ms)) / 1000., 60. / speed);
    return Py_BuildValue("idddd", 2, vx, vy, vx * scale, vy * scale);
}

PyObject* guard_geometry(PyObject*, PyObject* args)
{
    PyObject *observed, *prediction;
    double gate;
    if (!PyArg_ParseTuple(args, "OOd", &observed, &prediction, &gate)) return nullptr;
    Box box, pred;
    if (!numbers(observed, box) || !numbers(prediction, pred)) return nullptr;
    const double distance = std::hypot(box[0] - pred[0], box[1] - pred[1]) / std::max(1., pred[3]);
    const double hr = std::max(box[3], pred[3]) / std::max(1e-6, std::min(box[3], pred[3]));
    const double wr = std::max(box[2], pred[2]) / std::max(1e-6, std::min(box[2], pred[2]));
    return Py_BuildValue("dO", distance, distance <= gate && hr <= 1.5 && wr <= 1.5 ? Py_True : Py_False);
}

struct SearchStep {
    int phase = 0;
    int direction, leg, sweeps, skipped;
    double leg_time;
    bool swept, returned;
    std::optional<double> target;
    double lead = 0.;
};

PyObject* sweep_step(PyObject*, PyObject* args)
{
    double t, age, width, h0, w0, cx, aim_x, vx, sweep_end;
    int crowd, hidden, blind, phase, dir, leg, sweeps, skipped, swept, returned;
    double leg_time;
    if (!PyArg_ParseTuple(args, "dddddddddpppiiiiidpp", &t, &age, &width, &h0, &w0, &cx, &aim_x, &vx, &sweep_end,
                          &crowd, &hidden, &blind, &phase, &dir, &leg, &sweeps, &skipped,
                          &leg_time, &swept, &returned))
        return nullptr;
    const double radius = h0 * std::min(.5 + .05 * std::max(0., age) * 30., 4.);
    const double screen_room = width / 2. - .10 * width;
    const double amplitude = std::max(0., std::min(radius - screen_room, screen_room - w0 / 2.));
    const double pan = aim_x - cx;
    SearchStep s{phase, dir, leg, sweeps, skipped, leg_time, swept != 0, returned != 0, {}, 0.};
    if (age < .8) {
        s.phase = 1;  // predict
    } else if (crowd) {
        if (s.phase != 5) ++s.skipped;
        s.phase = 5;
    } else if (hidden) {
        s.phase = 8;
    } else if (blind) {
        s.phase = 6;
    } else if (age < sweep_end) {
        if (amplitude < 24. && !s.swept) {
            s.phase = 7;  // already in view
        } else {
            if (!s.swept) {
                s.swept = true;
                s.leg = 0;
                s.leg_time = t;
                ++s.sweeps;
                s.direction = std::abs(vx) >= .15 * h0 ? (vx > 0. ? 1 : -1) : (cx >= aim_x ? 1 : -1);
            }
            s.phase = 2;  // sweep
            for (int attempt = 0; attempt < 2; ++attempt) {
                const int side = s.leg % 2 ? -s.direction : s.direction;
                s.target = side * amplitude;
                s.lead = side * 120.;
                if (side * (*s.target - pan) > 30. && t - s.leg_time <= 2.5) break;
                ++s.leg;
                s.leg_time = t;
            }
        }
    } else if (s.swept && !s.returned && age < sweep_end + 1.2) {
        s.phase = 3;  // return
        s.target = 0.;
        if (std::abs(pan) <= 40.) {
            s.returned = true;
            s.phase = 4;
            s.target.reset();
        }
    } else {
        s.phase = 4;  // wait
    }
    PyObject* target = s.target ? PyFloat_FromDouble(*s.target) : Py_NewRef(Py_None);
    if (!target) return nullptr;
    return Py_BuildValue("iiiiidOOddNdd", s.phase, s.direction, s.leg, s.sweeps, s.skipped, s.leg_time,
                         s.swept ? Py_True : Py_False, s.returned ? Py_True : Py_False, radius,
                         amplitude, target, s.lead, pan);
}

PyObject* bounded_action(PyObject*, PyObject* args)
{
    double elapsed, top, aim_fraction;
    PyObject *center_arg, *origin_arg, *size_arg, *aim_arg;
    int axis, direction, measured_down;
    if (!PyArg_ParseTuple(args, "dOOOOiidpd", &elapsed, &center_arg, &origin_arg, &size_arg, &aim_arg,
                          &axis, &direction, &top, &measured_down, &aim_fraction))
        return nullptr;
    Point center, origin, size, aim;
    if (axis < 0 || axis >= 2) {
        PyErr_SetString(PyExc_ValueError, "invalid sweep axis");
        return nullptr;
    }
    if (!numbers(center_arg, center) || !numbers(origin_arg, origin) ||
        !numbers(size_arg, size) || !numbers(aim_arg, aim))
        return nullptr;
    const double span = size[axis];
    const double shift = center[axis] - origin[axis];
    if (elapsed >= 2.64) {
        if (std::abs(shift) <= .01 * span) return Py_BuildValue("OO", Py_True, Py_None);
        direction = shift > 0. ? 1 : -1;
    } else if (1.62 <= elapsed && elapsed < 2.50) {
        direction = -direction;
    } else if (elapsed >= 1.48) {
        return Py_BuildValue("OO", Py_False, Py_None);
    }
    if ((axis == 1 && direction > 0 && !measured_down && top < (.10 + .025) * size[1])
        || -direction * shift >= .025 * span)
        return Py_BuildValue("OO", Py_False, Py_None);
    aim[axis] += direction * .06 * span;
    const double x = aim[0] - 10., y = aim[1] - aim_fraction * 20.;
    if (!(0. <= x && x <= size[0] - 20. && 0. <= y && y <= size[1] - 20.))
        return Py_BuildValue("OO", Py_False, Py_None);
    return Py_BuildValue("O(dddd)", Py_False, x, y, 20., 20.);
}

PyObject* header(PyObject*, PyObject* args)
{
    long long frame, captured;
    int count, picture, width, height, observation;
    PyObject *rows, *suffixes;
    if (!PyArg_ParseTuple(args, "LLipiiOiO", &frame, &captured, &count, &picture, &width, &height,
                         &rows, &observation, &suffixes))
        return nullptr;
    std::string output = std::string(picture ? "G " : "F ") + std::to_string(frame) + " " + std::to_string(captured);
    if (picture) output += " " + std::to_string(width) + " " + std::to_string(height);
    output += " " + std::to_string(count);
    const char* text = PyUnicode_AsUTF8(rows);
    if (!text) return nullptr;
    output += text;
    if (observation >= 0) output += observation ? " OBS 1" : " OBS 0";
    PyObject* fast = PySequence_Fast(suffixes, "expected command suffixes");
    if (!fast) return nullptr;
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fast); ++i) {
        const char* suffix = PyUnicode_AsUTF8(PySequence_Fast_GET_ITEM(fast, i));
        if (!suffix) { Py_DECREF(fast); return nullptr; }
        output += suffix;
    }
    Py_DECREF(fast);
    return PyUnicode_FromStringAndSize(output.data(), output.size());
}

template <PyCFunction function>
PyObject* checked(PyObject* module, PyObject* args)
{
    try {
        return function(module, args);
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    }
}

PyMethodDef methods[] = {
    {"header", checked<header>, METH_VARARGS, nullptr},
    {"guard_geometry", checked<guard_geometry>, METH_VARARGS, nullptr},
    {"sweep_step", checked<sweep_step>, METH_VARARGS, nullptr},
    {"bounded_action", checked<bounded_action>, METH_VARARGS, nullptr},
    {"prepare", checked<prepare>, METH_VARARGS, nullptr},
    {"observations", checked<observations>, METH_VARARGS, nullptr},
    {"target_suffix", checked<target_suffix>, METH_VARARGS, nullptr},
    {"velocity", checked<velocity>, METH_VARARGS, nullptr},
    {"new_reentry", checked<new_reentry>, METH_NOARGS, nullptr},
    {"reentry_step", checked<reentry_step>, METH_VARARGS, nullptr},
    {"overlaps", checked<overlaps>, METH_VARARGS, nullptr},
    {"search_crowd", checked<search_crowd>, METH_VARARGS, nullptr},
    {"filter_velocity", checked<filter_velocity>, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native_control", nullptr, -1, methods};
}  // namespace

PyMODINIT_FUNC PyInit__native_control() { return PyModule_Create(&module); }
