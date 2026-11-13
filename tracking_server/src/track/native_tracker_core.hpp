// Shared C++ core of the CPython tracker extensions (sort.cpp, ocsort.cpp, botsort.cpp).
// Each .cpp sets the tracker kind and includes this file; scripts/build_native_trackers.sh
// compiles one module per .cpp. The Python classes in this folder stay the reference:
// every step below names the Python line it mirrors, including float32/float64 rounding.
#pragma once
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace lwirnt {

enum { ST_NEW = 0, ST_TRACKED = 1, ST_LOST = 2, ST_REMOVED = 3 };   // common/shared_values.py
enum Kind { K_SORT, K_OCSORT, K_BOTSORT };
static const long PERSON_CLS = 0;

struct Ref {   // owned reference
    PyObject* p;
    explicit Ref(PyObject* o = nullptr) : p(o) {}
    ~Ref() { Py_XDECREF(p); }
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    operator PyObject*() const { return p; }
    explicit operator bool() const { return p != nullptr; }
    PyObject* release() { PyObject* o = p; p = nullptr; return o; }
};

struct Globals {
    PyObject* np = nullptr;            // numpy module
    bool np_weak_scalars = true;       // numpy >= 2: a float32 value compared with a Python float stays float32
    PyObject* settings = nullptr;      // common.settings (EDGE_BIRTH, EDGE_PX)
    PyObject* posp = nullptr;          // position_memory.POSP
    PyObject* edge_ok = nullptr;       // tracker_common.edge_ok
    PyObject* pos_inn = nullptr;       // position_memory.pos_inn
    PyObject* pos_predict = nullptr;   // position_memory.pos_predict
    PyObject* KF = nullptr;            // kalman.KF
    PyTypeObject* track_type = nullptr;
    PyTypeObject* tracker_type = nullptr;
};
static Globals G;

// ---------------------------------------------------------------- numpy bridge (no numpy C API)

static PyObject* np_array(const double* p, Py_ssize_t n, bool f32 = false, int d0 = 0, int d1 = 0, int d2 = 0)
{
    Ref bytes;
    if (f32) {
        std::vector<float> tmp(p, p + n);
        bytes.p = PyBytes_FromStringAndSize(reinterpret_cast<const char*>(tmp.data()), n * 4);
    } else {
        bytes.p = PyBytes_FromStringAndSize(reinterpret_cast<const char*>(p), n * 8);
    }
    if (!bytes) return nullptr;
    Ref view(PyObject_CallMethod(G.np, "frombuffer", "Os", bytes.p, f32 ? "float32" : "float64"));
    if (!view) return nullptr;
    Ref copy(PyObject_CallMethod(view, "copy", nullptr));
    if (!copy || !d0) return copy.release();
    if (d2) return PyObject_CallMethod(copy, "reshape", "iii", d0, d1, d2);
    return PyObject_CallMethod(copy, "reshape", "ii", d0, d1);
}

static char buffer_code(const Py_buffer& b)
{
    const char* f = b.format ? b.format : "B";
    while (*f == '<' || *f == '>' || *f == '=' || *f == '@' || *f == '!') ++f;
    return *f;
}

// Any array-like -> doubles. *f32 tells whether the source was float32 (values are then exact float32).
static bool read_doubles(PyObject* o, std::vector<double>& out, bool* f32 = nullptr, std::vector<Py_ssize_t>* shape = nullptr)
{
    Ref a(PyObject_CallMethod(G.np, "ascontiguousarray", "O", o));
    if (!a) return false;
    Py_buffer b;
    if (PyObject_GetBuffer(a, &b, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) != 0) return false;
    char code = buffer_code(b);
    if (!((code == 'f' && b.itemsize == 4) || (code == 'd' && b.itemsize == 8))) {
        PyBuffer_Release(&b);
        Ref c(PyObject_CallMethod(G.np, "ascontiguousarray", "Os", o, "float64"));
        if (!c) return false;
        if (PyObject_GetBuffer(c, &b, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) != 0) return false;
        code = 'd';
    }
    Py_ssize_t n = b.len / b.itemsize;
    out.resize(static_cast<size_t>(n));
    if (code == 'f') {
        const float* s = static_cast<const float*>(b.buf);
        for (Py_ssize_t i = 0; i < n; ++i) out[i] = s[i];
    } else if (n) {
        std::memcpy(out.data(), b.buf, static_cast<size_t>(n) * 8);
    }
    if (f32) *f32 = code == 'f';
    if (shape) shape->assign(b.shape, b.shape + b.ndim);
    PyBuffer_Release(&b);
    return true;
}

// ---------------------------------------------------------------- track state

struct Feat {
    std::vector<double> v;
    bool f32 = false;
    bool has = false;
};

struct Obs { long frame; double b[4]; };

struct TrackData {
    double xywh0[4] = {0, 0, 0, 0};
    double score = 0;
    long cls = 0, idx = 0;
    Feat feat, det_feat;
    bool has_mean = false;
    double mean[8];
    double cov[64];
    int state = ST_NEW;
    bool is_activated = false;
    long track_id = 0, frame_id = 0, end_frame = 0, start_frame = 0;
    bool has_last = false;
    long last_frame = 0, last_idx = 0;
    double last_box[4];
    long feat_frame = -1;
    long hits = 1;
    // OC-SORT
    std::vector<Obs> obs;
    bool has_vdir = false;
    double vdir[2];
    bool has_kf_obs = false;
    double ko_mean[8];
    double ko_cov[64];
};

struct TrackObj {
    PyObject_HEAD
    TrackData* d;
    PyObject* dict;
};

typedef std::vector<TrackObj*> TV;

static inline const double* xywh_of(const TrackData* t) { return t->has_mean ? t->mean : t->xywh0; }
static inline void box_of(const double* q, double o[4])
{
    o[0] = q[0] - q[2] / 2; o[1] = q[1] - q[3] / 2; o[2] = q[0] + q[2] / 2; o[3] = q[1] + q[3] / 2;
}
static inline void xyxy_of(const TrackData* t, double o[4]) { box_of(xywh_of(t), o); }
static inline void det_box(const TrackData* t, double o[4]) { box_of(t->xywh0, o); }

static inline double iou_boxes(const double a[4], const double b[4])
{
    double iw = std::max(std::min(a[2], b[2]) - std::max(a[0], b[0]), 0.0);
    double ih = std::max(std::min(a[3], b[3]) - std::max(a[1], b[1]), 0.0);
    double inter = iw * ih;
    double aa = (a[2] - a[0]) * (a[3] - a[1]), ab = (b[2] - b[0]) * (b[3] - b[1]);
    return inter / std::max(aa + ab - inter, 1e-9);
}

static bool has(const TV& v, const TrackObj* t) { return std::find(v.begin(), v.end(), t) != v.end(); }

static void clear_tv(TV& v)
{
    TV old;
    old.swap(v);
    for (TrackObj* t : old) Py_DECREF(reinterpret_cast<PyObject*>(t));
}

static void push_ref(TV& v, TrackObj* t)
{
    Py_INCREF(reinterpret_cast<PyObject*>(t));
    v.push_back(t);
}

// ---------------------------------------------------------------- Kalman (kalman.py KF, gmc_warp)

static const double KF_SP = 1.0 / 20, KF_SV = 1.0 / 160;

static void kf_initiate(const double m[4], double mean[8], double cov[64])
{
    double s[8] = {2 * KF_SP * m[2], 2 * KF_SP * m[3], 2 * KF_SP * m[2], 2 * KF_SP * m[3],
                   10 * KF_SV * m[2], 10 * KF_SV * m[3], 10 * KF_SV * m[2], 10 * KF_SV * m[3]};
    for (int i = 0; i < 4; ++i) { mean[i] = m[i]; mean[i + 4] = 0; }
    std::fill(cov, cov + 64, 0.0);
    for (int i = 0; i < 8; ++i) cov[i * 9] = s[i] * s[i];
}

static void kf_predict(double mean[8], double cov[64])
{
    double w = mean[2], h = mean[3];
    double s[8] = {KF_SP * w, KF_SP * h, KF_SP * w, KF_SP * h, KF_SV * w, KF_SV * h, KF_SV * w, KF_SV * h};
    for (int i = 0; i < 4; ++i) mean[i] = mean[i] + mean[i + 4];
    double a[64];
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) a[i * 8 + j] = i < 4 ? cov[i * 8 + j] + cov[(i + 4) * 8 + j] : cov[i * 8 + j];
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) cov[i * 8 + j] = j < 4 ? a[i * 8 + j] + a[i * 8 + j + 4] : a[i * 8 + j];
    for (int i = 0; i < 8; ++i) cov[i * 9] += s[i] * s[i];
}

static void kf_update(double mean[8], double cov[64], const double z[4])
{
    double s2 = KF_SP * mean[2], s3 = KF_SP * mean[3];
    double r[4] = {s2 * s2, s3 * s3, s2 * s2, s3 * s3};
    double pc[16], lu[16], x[4][8];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) pc[i * 4 + j] = cov[i * 8 + j] + (i == j ? r[i] : 0.0);
    std::memcpy(lu, pc, sizeof(lu));
    for (int i = 0; i < 4; ++i)
        for (int c = 0; c < 8; ++c) x[i][c] = cov[c * 8 + i];   // right-hand side = cov[:, :4].T
    for (int k = 0; k < 4; ++k) {   // LU with partial pivoting, as numpy.linalg.solve
        int p = k;
        for (int i = k + 1; i < 4; ++i)
            if (std::fabs(lu[i * 4 + k]) > std::fabs(lu[p * 4 + k])) p = i;
        if (p != k) {
            for (int j = 0; j < 4; ++j) std::swap(lu[k * 4 + j], lu[p * 4 + j]);
            for (int c = 0; c < 8; ++c) std::swap(x[k][c], x[p][c]);
        }
        for (int i = k + 1; i < 4; ++i) {
            double f = lu[i * 4 + k] / lu[k * 4 + k];
            lu[i * 4 + k] = f;
            for (int j = k + 1; j < 4; ++j) lu[i * 4 + j] -= f * lu[k * 4 + j];
            for (int c = 0; c < 8; ++c) x[i][c] -= f * x[k][c];
        }
    }
    for (int i = 3; i >= 0; --i)
        for (int c = 0; c < 8; ++c) {
            double v = x[i][c];
            for (int j = i + 1; j < 4; ++j) v -= lu[i * 4 + j] * x[j][c];
            x[i][c] = v / lu[i * 4 + i];
        }
    double inn[4], kp[8][4], nm[8], nc[64];   // K[c][j] = x[j][c]
    for (int j = 0; j < 4; ++j) inn[j] = z[j] - mean[j];
    for (int i = 0; i < 8; ++i) {
        double v = 0;
        for (int j = 0; j < 4; ++j) v += inn[j] * x[j][i];
        nm[i] = mean[i] + v;
        for (int j = 0; j < 4; ++j) {
            double t = 0;
            for (int k = 0; k < 4; ++k) t += x[k][i] * pc[k * 4 + j];
            kp[i][j] = t;
        }
    }
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) {
            double v = 0;
            for (int k = 0; k < 4; ++k) v += kp[i][k] * x[k][j];
            nc[i * 8 + j] = cov[i * 8 + j] - v;
        }
    std::memcpy(mean, nm, sizeof(nm));
    std::memcpy(cov, nc, sizeof(nc));
}

struct Warp { double r[4]; double t[2]; };   // H[:2, :2] and H[:2, 2]

static void gmc_warp(double mean[8], double cov[64], const Warp& w)
{
    for (int k = 0; k < 4; ++k) {
        double a = mean[2 * k], b = mean[2 * k + 1];
        mean[2 * k] = w.r[0] * a + w.r[1] * b;
        mean[2 * k + 1] = w.r[2] * a + w.r[3] * b;
    }
    mean[0] += w.t[0]; mean[1] += w.t[1];
    double a[64];
    for (int k = 0; k < 4; ++k)   // R8 @ cov
        for (int j = 0; j < 8; ++j) {
            double u = cov[(2 * k) * 8 + j], v = cov[(2 * k + 1) * 8 + j];
            a[(2 * k) * 8 + j] = w.r[0] * u + w.r[1] * v;
            a[(2 * k + 1) * 8 + j] = w.r[2] * u + w.r[3] * v;
        }
    for (int i = 0; i < 8; ++i)   // (.) @ R8.T
        for (int k = 0; k < 4; ++k) {
            double u = a[i * 8 + 2 * k], v = a[i * 8 + 2 * k + 1];
            cov[i * 8 + 2 * k] = w.r[0] * u + w.r[1] * v;
            cov[i * 8 + 2 * k + 1] = w.r[2] * u + w.r[3] * v;
        }
}

// ---------------------------------------------------------------- assignment (tracker_common.linear_assignment)

struct Cost {
    int n = 0, m = 0;
    bool f32 = false;   // the Python matrix is float32: thresholds are compared in float32
    std::vector<double> v;
    double& at(int i, int j) { return v[static_cast<size_t>(i) * m + j]; }
    double at(int i, int j) const { return v[static_cast<size_t>(i) * m + j]; }
    bool empty() const { return n == 0 || m == 0; }
};

struct Assign {
    std::vector<std::pair<int, int>> m;
    std::vector<int> ua, ub;
};

// scipy.optimize.linear_sum_assignment (rectangular_lsap.cpp, Crouse 2016), same visiting order and tie rule.
static bool lsap(int nr, int nc, const double* input, std::vector<int>& rows, std::vector<int>& cols)
{
    const double INF = std::numeric_limits<double>::infinity();
    bool transpose = nc < nr;
    std::vector<double> temp;
    const double* cost = input;
    if (transpose) {
        temp.resize(static_cast<size_t>(nr) * nc);
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nc; ++j) temp[static_cast<size_t>(j) * nr + i] = input[static_cast<size_t>(i) * nc + j];
        std::swap(nr, nc);
        cost = temp.data();
    }
    std::vector<double> u(nr, 0.0), v(nc, 0.0), spc(nc);
    std::vector<int> path(nc, -1), col4row(nr, -1), row4col(nc, -1), remaining(nc);
    std::vector<char> SR(nr), SC(nc);
    for (int cur = 0; cur < nr; ++cur) {
        double min_val = 0;
        int i = cur, num_remaining = nc, sink = -1;
        for (int it = 0; it < nc; ++it) remaining[it] = nc - it - 1;
        std::fill(SR.begin(), SR.end(), 0);
        std::fill(SC.begin(), SC.end(), 0);
        std::fill(spc.begin(), spc.end(), INF);
        while (sink == -1) {
            int index = -1;
            double lowest = INF;
            SR[i] = 1;
            for (int it = 0; it < num_remaining; ++it) {
                int j = remaining[it];
                double r = min_val + cost[static_cast<size_t>(i) * nc + j] - u[i] - v[j];
                if (r < spc[j]) { path[j] = i; spc[j] = r; }
                if (spc[j] < lowest || (spc[j] == lowest && row4col[j] == -1)) { lowest = spc[j]; index = it; }
            }
            min_val = lowest;
            if (min_val == INF) return false;
            int j = remaining[index];
            if (row4col[j] == -1) sink = j; else i = row4col[j];
            SC[j] = 1;
            remaining[index] = remaining[--num_remaining];
        }
        u[cur] += min_val;
        for (int k = 0; k < nr; ++k)
            if (SR[k] && k != cur) u[k] += min_val - spc[col4row[k]];
        for (int j = 0; j < nc; ++j)
            if (SC[j]) v[j] -= min_val - spc[j];
        int j = sink;
        while (true) {
            int k = path[j];
            row4col[j] = k;
            std::swap(col4row[k], j);
            if (k == cur) break;
        }
    }
    rows.resize(nr); cols.resize(nr);
    if (transpose) {
        std::vector<int> order(nr);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return col4row[a] < col4row[b]; });
        for (int k = 0; k < nr; ++k) { rows[k] = col4row[order[k]]; cols[k] = order[k]; }
    } else {
        for (int k = 0; k < nr; ++k) { rows[k] = k; cols[k] = col4row[k]; }
    }
    return true;
}

static bool linear_assignment(const Cost& c, double thresh, Assign& out)
{
    out.m.clear(); out.ua.clear(); out.ub.clear();
    if (c.empty()) {
        for (int i = 0; i < c.n; ++i) out.ua.push_back(i);
        for (int j = 0; j < c.m; ++j) out.ub.push_back(j);
        return true;
    }
    // `cost > thresh` on a float32 array compares in float32. The single-value test `cost[i, j] <= thresh` does so
    // only from numpy 2 on; numpy 1 widens that one value to float64. Follow the numpy this process runs.
    double thr = c.f32 ? static_cast<double>(static_cast<float>(thresh)) : thresh;
    double accept = (c.f32 && G.np_weak_scalars) ? thr : thresh;
    if (c.n == 1 && c.m == 1) {
        if (c.v[0] <= accept) out.m.emplace_back(0, 0);
        else { out.ua.push_back(0); out.ub.push_back(0); }
        return true;
    }
    std::vector<double> capped(c.v.size());
    for (size_t k = 0; k < c.v.size(); ++k) capped[k] = c.v[k] > thr ? 1e6 : c.v[k];
    std::vector<int> rows, cols;
    if (!lsap(c.n, c.m, capped.data(), rows, cols)) {
        PyErr_SetString(PyExc_ValueError, "cost matrix is infeasible");
        return false;
    }
    std::vector<char> ma(c.n, 0), mb(c.m, 0);
    for (size_t k = 0; k < rows.size(); ++k)
        if (c.at(rows[k], cols[k]) <= accept) { out.m.emplace_back(rows[k], cols[k]); ma[rows[k]] = 1; mb[cols[k]] = 1; }
    for (int i = 0; i < c.n; ++i) if (!ma[i]) out.ua.push_back(i);
    for (int j = 0; j < c.m; ++j) if (!mb[j]) out.ub.push_back(j);
    return true;
}

// ---------------------------------------------------------------- appearance (tracker_common.Track.set_feat)

static double feat_dot(const Feat& a, const Feat& b)
{
    double s = 0;
    size_t n = std::min(a.v.size(), b.v.size());
    for (size_t i = 0; i < n; ++i) s += a.v[i] * b.v[i];
    return (a.f32 && b.f32) ? static_cast<double>(static_cast<float>(s)) : s;
}

static void feat_normalise(Feat& f)
{
    double n = std::sqrt(feat_dot(f, f));
    if (f.f32) n = static_cast<float>(n);
    n = std::max(n, 1e-12);
    for (double& x : f.v) x = f.f32 ? static_cast<double>(static_cast<float>(x) / static_cast<float>(n)) : x / n;
}

static void set_feat(TrackData* t, const Feat& in)
{
    Feat f = in;
    f.has = true;
    feat_normalise(f);
    if (!t->feat.has || t->feat.v.size() != f.v.size()) {
        t->feat = f;
    } else {
        bool f32 = t->feat.f32 && f.f32;
        for (size_t i = 0; i < f.v.size(); ++i) {
            if (f32) {
                float a = 0.9f * static_cast<float>(t->feat.v[i]), b = 0.1f * static_cast<float>(f.v[i]);
                t->feat.v[i] = static_cast<float>(a + b);
            } else {
                double a = t->feat.f32 ? static_cast<double>(0.9f * static_cast<float>(t->feat.v[i])) : 0.9 * t->feat.v[i];
                double b = f.f32 ? static_cast<double>(0.1f * static_cast<float>(f.v[i])) : 0.1 * f.v[i];
                t->feat.v[i] = a + b;
            }
        }
        t->feat.f32 = f32;
    }
    feat_normalise(t->feat);
}

// ---------------------------------------------------------------- tracker

struct TrackerData {
    Kind kind = K_SORT;
    std::string mode;
    long frame_id = 0, next_id = 0, n_init = 2, buffer = 30;
    bool class_gate = false;
    double high = 0.25, low = 0.10, new_ = 0.25, match = 0.8;
    long dt = 3;
    double vdc = 0.2, iou_min = 0.3;
    double app_thr = 0.25, prox_thr = 0.5;
    long app_used = 0;
    TV tracked, lost, new_tf, lost_tf;
};

struct TrackerObj {
    PyObject_HEAD
    TrackerData* d;
    PyObject* dict;
};

static TrackObj* new_track()
{
    TrackObj* t = PyObject_New(TrackObj, G.track_type);
    if (!t) return nullptr;
    t->dict = nullptr;
    t->d = new (std::nothrow) TrackData();
    if (!t->d) { Py_DECREF(reinterpret_cast<PyObject*>(t)); PyErr_NoMemory(); return nullptr; }
    return t;
}

static bool posp_active()
{
    if (!G.posp || !G.pos_predict || !PyDict_Check(G.posp)) return false;
    PyObject* model = PyDict_GetItemString(G.posp, "model");
    PyObject* show = PyDict_GetItemString(G.posp, "show");
    return (model && model != Py_None) || (show && PyObject_IsTrue(show) == 1);
}

static bool posp_model()
{
    if (!G.posp || !G.pos_inn || !PyDict_Check(G.posp)) return false;
    PyObject* model = PyDict_GetItemString(G.posp, "model");
    return model && model != Py_None;
}

static PyObject* tracker_kf(TrackerObj* self)   // borrowed, may be null
{
    return self->dict ? PyDict_GetItemString(self->dict, "kf") : nullptr;
}

// 1 = warp read, 0 = no pending warp, -1 = error
static int pending_warp(PyObject* kf, Warp& w)
{
    if (!kf) return 0;
    Ref a(PyObject_GetAttrString(kf, "warp"));
    if (!a) { PyErr_Clear(); return 0; }
    if (a.p == Py_None) return 0;
    std::vector<double> h;
    std::vector<Py_ssize_t> shape;
    if (!read_doubles(a, h, nullptr, &shape)) return -1;
    if (shape.size() != 2 || shape[0] < 2 || shape[1] < 3) {
        PyErr_SetString(PyExc_ValueError, "kf.warp must be a 2x3 or 3x3 matrix");
        return -1;
    }
    Py_ssize_t c = shape[1];
    w.r[0] = h[0]; w.r[1] = h[1]; w.t[0] = h[2];
    w.r[2] = h[c]; w.r[3] = h[c + 1]; w.t[1] = h[c + 2];
    return 1;
}

// Track._absorb
static bool absorb(TrackData* t, TrackObj* tobj, const TrackData* det, long frame_id)
{
    if (posp_model()) {   // v32: innovation against the prediction, kept as a Python attribute for pos_predict
        Ref z(np_array(det->xywh0, 4)), m(np_array(t->mean, 8));
        if (!z || !m) return false;
        Ref inn(PyObject_CallFunctionObjArgs(G.pos_inn, z.p, m.p, nullptr));
        if (!inn || PyObject_SetAttrString(reinterpret_cast<PyObject*>(tobj), "pp_inn", inn) != 0) return false;
    }
    kf_update(t->mean, t->cov, det->xywh0);
    t->state = ST_TRACKED; t->is_activated = true;
    t->frame_id = t->end_frame = frame_id;
    t->score = det->score; t->cls = det->cls; t->idx = det->idx;
    t->has_last = true; t->last_frame = frame_id; t->last_idx = det->idx;
    det_box(det, t->last_box);
    return true;
}

// Track.activate
static void activate(TrackData* t, long frame_id, long track_id)
{
    t->track_id = track_id;
    kf_initiate(t->xywh0, t->mean, t->cov);
    t->has_mean = true;
    t->state = ST_TRACKED;
    t->is_activated = frame_id == 1;
    t->frame_id = t->end_frame = t->start_frame = frame_id;
    t->has_last = true; t->last_frame = frame_id; t->last_idx = t->idx;
    xyxy_of(t, t->last_box);
}

// tracker_common.iou_dist (float32 result)
static Cost iou_dist(const TV& a, const TV& b)
{
    Cost c;
    c.n = static_cast<int>(a.size()); c.m = static_cast<int>(b.size()); c.f32 = true;
    if (c.empty()) return c;
    c.v.resize(static_cast<size_t>(c.n) * c.m);
    std::vector<double> bb(static_cast<size_t>(c.m) * 4);
    for (int j = 0; j < c.m; ++j) xyxy_of(b[j]->d, &bb[static_cast<size_t>(j) * 4]);
    for (int i = 0; i < c.n; ++i) {
        double ab[4];
        xyxy_of(a[i]->d, ab);
        for (int j = 0; j < c.m; ++j) c.at(i, j) = static_cast<float>(1 - iou_boxes(ab, &bb[static_cast<size_t>(j) * 4]));
    }
    return c;
}

// ByteTrackTracker._dists and BotSortTracker._dists
static Cost fused_dists(TrackerData* T, const TV& tr, const TV& de)
{
    Cost c = iou_dist(tr, de);
    if (c.empty()) return c;
    std::vector<double> d_iou = c.v;
    c.f32 = false;
    for (int a = 0; a < c.n; ++a)
        for (int b = 0; b < c.m; ++b) {
            float d = static_cast<float>(c.at(a, b));
            c.at(a, b) = 1 - static_cast<double>(1.0f - d) * de[b]->d->score;
        }
    if (T->kind != K_BOTSORT) return c;
    for (int a = 0; a < c.n; ++a) {
        const Feat& tf = tr[a]->d->feat;
        if (!tf.has) continue;
        for (int b = 0; b < c.m; ++b) {
            const Feat& df = de[b]->d->det_feat;
            if (!df.has || d_iou[static_cast<size_t>(a) * c.m + b] >= T->prox_thr) continue;
            double dc = 1.0 - feat_dot(tf, df);
            if (dc < T->app_thr && 0.5 * dc < c.at(a, b)) { c.at(a, b) = 0.5 * dc; T->app_used += 1; }
        }
    }
    return c;
}

// ByteTrackTracker._cg
static void class_gate(const TrackerData* T, Cost& c, const TV& a, const TV& b)
{
    if (!T->class_gate || c.empty()) return;
    for (int i = 0; i < c.n; ++i)
        for (int j = 0; j < c.m; ++j)
            if ((a[i]->d->cls == PERSON_CLS) != (b[j]->d->cls == PERSON_CLS)) c.at(i, j) = 1.0;
}

// OCSortTracker._prev
static const double* oc_prev(const TrackerData* T, const TrackData* t)
{
    for (long dt = T->dt; dt > 0; --dt)
        for (const Obs& o : t->obs)
            if (o.frame == T->frame_id - dt) return o.b;
    return t->obs.empty() ? nullptr : t->obs.back().b;
}

// OCSortTracker._note
static void oc_note(const TrackerData* T, TrackData* t, const TrackData* det)
{
    double b[4];
    det_box(det, b);
    const double* p = oc_prev(T, t);
    t->has_vdir = false;
    if (p) {
        double vx = (b[0] + b[2]) / 2 - (p[0] + p[2]) / 2, vy = (b[1] + b[3]) / 2 - (p[1] + p[3]) / 2;
        double n = std::hypot(vx, vy);
        if (n > 1e-6) { t->has_vdir = true; t->vdir[0] = vx / n; t->vdir[1] = vy / n; }
    }
    Obs o;
    o.frame = T->frame_id;
    std::memcpy(o.b, b, sizeof(b));
    t->obs.push_back(o);
    if (t->obs.size() > 8) t->obs.erase(t->obs.begin(), t->obs.end() - 8);
    t->has_kf_obs = true;
    std::memcpy(t->ko_mean, t->mean, sizeof(t->ko_mean));
    std::memcpy(t->ko_cov, t->cov, sizeof(t->ko_cov));
}

// AltTracker._take / OCSortTracker._take
static bool take(TrackerData* T, TrackObj* tobj, const TrackData* det)
{
    TrackData* t = tobj->d;
    if (T->kind == K_OCSORT) {
        long gap = T->frame_id - t->end_frame;
        if (gap > 1 && !t->obs.empty() && t->has_kf_obs) {   // ORU: replay the gap on a straight line between the two observations
            double m[8], c[64], z0[4];
            std::memcpy(m, t->ko_mean, sizeof(m));
            std::memcpy(c, t->ko_cov, sizeof(c));
            const double* b = t->obs.back().b;
            z0[0] = (b[0] + b[2]) / 2; z0[1] = (b[1] + b[3]) / 2; z0[2] = b[2] - b[0]; z0[3] = b[3] - b[1];
            for (long k = 1; k < gap; ++k) {
                kf_predict(m, c);
                double z[4];
                for (int q = 0; q < 4; ++q) z[q] = z0[q] + (det->xywh0[q] - z0[q]) * static_cast<double>(k) / static_cast<double>(gap);
                kf_update(m, c, z);
            }
            kf_predict(m, c);
            std::memcpy(t->mean, m, sizeof(m));
            std::memcpy(t->cov, c, sizeof(c));
        }
    }
    if (!absorb(t, tobj, det, T->frame_id)) return false;
    if (T->kind == K_OCSORT) oc_note(T, t, det);
    return true;
}

// OCSortTracker._ocm (float32 result)
static Cost oc_cost(const TrackerData* T, const TV& tr, const TV& de)
{
    static const double PI = 3.141592653589793;
    Cost c;
    c.n = static_cast<int>(tr.size()); c.m = static_cast<int>(de.size()); c.f32 = true;
    if (c.empty()) return c;
    c.v.resize(static_cast<size_t>(c.n) * c.m);
    std::vector<double> db(static_cast<size_t>(c.m) * 4);
    for (int j = 0; j < c.m; ++j) det_box(de[j]->d, &db[static_cast<size_t>(j) * 4]);
    for (int a = 0; a < c.n; ++a) {
        const TrackData* t = tr[a]->d;
        double tb[4];
        xyxy_of(t, tb);
        const double* p = oc_prev(T, t);
        bool directed = t->has_vdir && p;
        for (int j = 0; j < c.m; ++j) {
            const double* b = &db[static_cast<size_t>(j) * 4];
            double iou = iou_boxes(tb, b), ang = 0;
            if (directed) {
                double ux = (b[0] + b[2]) / 2 - (p[0] + p[2]) / 2, uy = (b[1] + b[3]) / 2 - (p[1] + p[3]) / 2;
                double n = std::hypot(ux, uy);
                double cosv = (ux * t->vdir[0] + uy * t->vdir[1]) / std::max(n, 1e-6);
                cosv = std::min(std::max(cosv, -1.0), 1.0);
                ang = (PI / 2 - std::acos(cosv)) / PI * (n > 1e-6 ? 1.0 : 0.0);
            }
            double cost = 1 - (iou + T->vdc * ang * de[j]->d->score);
            if (iou < T->iou_min) cost = 1.0;
            c.at(a, j) = static_cast<float>(cost);
        }
    }
    return c;
}

// OCSortTracker._prepare_prediction: stored observations follow the camera-motion warp
static bool oc_prepare(TrackerObj* self)
{
    Warp w;
    int got = pending_warp(tracker_kf(self), w);
    if (got <= 0) return got == 0;
    TrackerData* T = self->d;
    for (int pass = 0; pass < 2; ++pass)
        for (TrackObj* tobj : (pass ? T->lost : T->tracked)) {
            TrackData* t = tobj->d;
            if (t->obs.empty()) continue;
            for (Obs& o : t->obs) {
                double cx = (o.b[0] + o.b[2]) / 2, cy = (o.b[1] + o.b[3]) / 2;
                double nx = w.r[0] * cx + w.r[1] * cy + w.t[0], ny = w.r[2] * cx + w.r[3] * cy + w.t[1];
                double hw = (o.b[2] - o.b[0]) / 2, hh = (o.b[3] - o.b[1]) / 2;
                o.b[0] = nx - hw; o.b[1] = ny - hh; o.b[2] = nx + hw; o.b[3] = ny + hh;
            }
            if (t->has_kf_obs) gmc_warp(t->ko_mean, t->ko_cov, w);
        }
    return true;
}

// Kalman prediction of the tracked + lost pool (position_memory.pos_predict)
static bool predict_pool(TrackerObj* self, const TV& pool)
{
    if (pool.empty()) return true;
    for (TrackObj* t : pool)
        if (t->d->state != ST_TRACKED) { t->d->mean[6] = 0; t->d->mean[7] = 0; }
    PyObject* kf = tracker_kf(self);
    if (posp_active() && kf && PyObject_HasAttrString(kf, "multi_predict")) {
        // A learned position predictor or its overlay is switched on: run the Python step itself on this pool.
        size_t n = pool.size();
        std::vector<double> mean(n * 8), cov(n * 64);
        Ref list(PyList_New(static_cast<Py_ssize_t>(n)));
        if (!list) return false;
        for (size_t i = 0; i < n; ++i) {
            std::memcpy(&mean[i * 8], pool[i]->d->mean, 64);
            std::memcpy(&cov[i * 64], pool[i]->d->cov, 512);
            Py_INCREF(reinterpret_cast<PyObject*>(pool[i]));
            PyList_SET_ITEM(list.p, static_cast<Py_ssize_t>(i), reinterpret_cast<PyObject*>(pool[i]));
        }
        Ref m(np_array(mean.data(), static_cast<Py_ssize_t>(n * 8), false, static_cast<int>(n), 8));
        Ref c(np_array(cov.data(), static_cast<Py_ssize_t>(n * 64), false, static_cast<int>(n), 8, 8));
        if (!m || !c) return false;
        Ref out(PyObject_CallFunctionObjArgs(G.pos_predict, kf, list.p, m.p, c.p, nullptr));
        if (!out) return false;
        if (!PyTuple_Check(out.p) || PyTuple_GET_SIZE(out.p) != 2) {
            PyErr_SetString(PyExc_TypeError, "pos_predict must return (mean, cov)");
            return false;
        }
        std::vector<double> nm, nc;
        if (!read_doubles(PyTuple_GET_ITEM(out.p, 0), nm) || !read_doubles(PyTuple_GET_ITEM(out.p, 1), nc)) return false;
        if (nm.size() != n * 8 || nc.size() != n * 64) {
            PyErr_SetString(PyExc_ValueError, "pos_predict returned a different pool size");
            return false;
        }
        for (size_t i = 0; i < n; ++i) {
            std::memcpy(pool[i]->d->mean, &nm[i * 8], 64);
            std::memcpy(pool[i]->d->cov, &nc[i * 64], 512);
        }
        return true;
    }
    for (TrackObj* t : pool) kf_predict(t->d->mean, t->d->cov);
    Warp w;   // kalman.GKF.multi_predict: the pending camera-motion warp is applied once, then cleared
    int got = pending_warp(kf, w);
    if (got < 0) return false;
    if (got) {
        for (TrackObj* t : pool) gmc_warp(t->d->mean, t->d->cov, w);
        if (PyObject_SetAttrString(kf, "warp", Py_None) != 0) return false;
    }
    return true;
}

struct Frame {   // detections of one update call; owns its references
    TV hi, lo;
    ~Frame() { clear_tv(hi); clear_tv(lo); }
};

static bool read_feat_object(PyObject* o, Feat& f)
{
    std::vector<Py_ssize_t> shape;
    if (!read_doubles(o, f.v, &f.f32, &shape)) return false;
    f.has = true;
    return true;
}

static bool parse_detections(TrackerData* T, PyObject* arr_in, PyObject* det_feats, Frame& fr)
{
    Ref keep;
    Py_buffer b;
    bool ok = PyObject_GetBuffer(arr_in, &b, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) == 0;
    if (ok && !(buffer_code(b) == 'f' && b.itemsize == 4 && b.ndim == 2)) { PyBuffer_Release(&b); ok = false; }
    if (!ok) {
        PyErr_Clear();
        keep.p = PyObject_CallMethod(G.np, "ascontiguousarray", "Os", arr_in, "float32");
        if (!keep) return false;
        if (PyObject_GetBuffer(keep, &b, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) != 0) return false;
    }
    if (b.ndim != 2 || b.shape[1] < 6) {
        PyBuffer_Release(&b);
        PyErr_SetString(PyExc_ValueError, "expected detections [N, 6] = x1, y1, x2, y2, conf, cls");
        return false;
    }
    const float* a = static_cast<const float*>(b.buf);
    Py_ssize_t n = b.shape[0], cols = b.shape[1];
    float high = static_cast<float>(T->high), low = static_cast<float>(T->low);
    bool good = true;
    for (Py_ssize_t i = 0; i < n && good; ++i) {
        const float* r = a + i * cols;
        float w = r[2] - r[0], h = r[3] - r[1];   // float32 arithmetic, as numpy on the float32 input
        if (!(w > 0 && h > 0)) continue;
        bool is_hi = r[4] >= high, is_lo = r[4] > low && r[4] < high;
        if (!is_hi && !is_lo) continue;
        TrackObj* t = new_track();
        if (!t) { good = false; break; }
        (is_hi ? fr.hi : fr.lo).push_back(t);
        t->d->xywh0[0] = static_cast<float>((r[0] + r[2]) / 2); t->d->xywh0[1] = static_cast<float>((r[1] + r[3]) / 2);
        t->d->xywh0[2] = w; t->d->xywh0[3] = h;
        t->d->score = r[4]; t->d->cls = static_cast<long>(r[5]); t->d->idx = static_cast<long>(i);
        if (det_feats && det_feats != Py_None) {
            Ref key(PyLong_FromSsize_t(i));
            if (!key) { good = false; break; }
            Ref f(PyObject_GetItem(det_feats, key));
            if (!f) {
                if (PyErr_ExceptionMatches(PyExc_KeyError) || PyErr_ExceptionMatches(PyExc_IndexError)) PyErr_Clear();
                else good = false;
            } else if (f.p != Py_None) {
                good = read_feat_object(f, t->d->det_feat);
            }
        }
    }
    PyBuffer_Release(&b);
    return good;
}

static bool edge_ok(TrackObj* det, double& edge_birth)
{
    if (!G.settings || !G.edge_ok) return true;
    if (edge_birth == -2) {   // read once per frame
        Ref v(PyObject_GetAttrString(G.settings, "EDGE_BIRTH"));
        edge_birth = v ? PyFloat_AsDouble(v) : 0.0;
        if (PyErr_Occurred()) { PyErr_Clear(); edge_birth = 0.0; }
    }
    if (edge_birth <= 0) return true;
    Ref r(PyObject_CallFunctionObjArgs(G.edge_ok, reinterpret_cast<PyObject*>(det), nullptr));   // v24f rule incl. stabilizer margins
    if (!r) return false;
    return PyObject_IsTrue(r) == 1;
}

static PyObject* output_rows(const TrackerData* T)
{
    Ref rows(PyList_New(0));
    if (!rows) return nullptr;
    for (TrackObj* tobj : T->tracked) {
        const TrackData* t = tobj->d;
        if (!t->is_activated) continue;
        double b[4];
        xyxy_of(t, b);
        Ref row(Py_BuildValue("[ddddldll]", b[0], b[1], b[2], b[3], t->track_id, t->score, t->cls, t->idx));
        if (!row || PyList_Append(rows, row) != 0) return nullptr;
    }
    return rows.release();
}

// ByteTrackTracker.update (mode 'bytetrack', BoT-SORT cost) and AltTracker.update (SORT, OC-SORT)
static PyObject* tracker_update_impl(TrackerObj* self, PyObject* arr, PyObject* det_feats)
{
    TrackerData* T = self->d;
    T->frame_id += 1;
    clear_tv(T->new_tf);
    clear_tv(T->lost_tf);
    Frame fr;
    if (!parse_detections(T, arr, T->kind == K_BOTSORT ? det_feats : nullptr, fr)) return nullptr;
    TV& dets = fr.hi;
    TV& dets2 = fr.lo;
    TV act, refind, lost_now;
    if (T->kind == K_OCSORT && !oc_prepare(self)) return nullptr;

    TV unconfirmed, pool;
    for (TrackObj* t : T->tracked) (t->d->is_activated ? pool : unconfirmed).push_back(t);
    {
        size_t n_tracked = pool.size();
        for (TrackObj* t : T->lost)
            if (std::find(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(n_tracked), t) == pool.begin() + static_cast<std::ptrdiff_t>(n_tracked))
                pool.push_back(t);
    }
    if (!predict_pool(self, pool)) return nullptr;

    // first association
    Assign as;
    {
        Cost c;
        double thr;
        if (T->kind == K_BOTSORT) { c = fused_dists(T, pool, dets); thr = T->match; }
        else if (T->kind == K_SORT) { c = iou_dist(pool, dets); thr = 1 - T->iou_min; }
        else { c = oc_cost(T, pool, dets); thr = 0.95; }
        class_gate(T, c, pool, dets);
        if (!linear_assignment(c, thr, as)) return nullptr;
    }
    for (const auto& m : as.m) {
        TrackObj* t = pool[m.first];
        bool was_lost = t->d->state == ST_LOST;
        if (!take(T, t, dets[m.second]->d)) return nullptr;
        (was_lost ? refind : act).push_back(t);
        if (T->kind == K_BOTSORT && dets[m.second]->d->det_feat.has) {
            set_feat(t->d, dets[m.second]->d->det_feat);
            t->d->feat_frame = T->frame_id;
        }
    }
    std::vector<int> ut = as.ua, ud = as.ub;

    // second association: remaining tracked x low-score detections
    TV rt, done;
    for (int i : ut) if (pool[i]->d->state == ST_TRACKED) rt.push_back(pool[i]);
    if (!rt.empty() && !dets2.empty()) {
        Cost c = iou_dist(rt, dets2);
        class_gate(T, c, rt, dets2);
        Assign a2;
        if (!linear_assignment(c, 0.5, a2)) return nullptr;
        for (const auto& m : a2.m) {
            if (!take(T, rt[m.first], dets2[m.second]->d)) return nullptr;
            act.push_back(rt[m.first]);
            done.push_back(rt[m.first]);
        }
    }
    if (T->kind == K_OCSORT && !ud.empty()) {   // OCR: last observations x unused high-score detections
        TV tracks, remaining;
        for (int i : ut) if (!has(done, pool[i]) && !pool[i]->d->obs.empty()) tracks.push_back(pool[i]);
        if (!tracks.empty()) {
            for (int j : ud) remaining.push_back(dets[j]);
            Cost c;
            c.n = static_cast<int>(tracks.size()); c.m = static_cast<int>(remaining.size()); c.f32 = true;
            c.v.resize(static_cast<size_t>(c.n) * c.m);
            for (int a = 0; a < c.n; ++a)
                for (int b = 0; b < c.m; ++b) {
                    double db[4];
                    det_box(remaining[b]->d, db);
                    c.at(a, b) = static_cast<float>(1 - iou_boxes(tracks[a]->d->obs.back().b, db));
                }
            class_gate(T, c, tracks, remaining);
            Assign a3;
            if (!linear_assignment(c, 1 - T->iou_min, a3)) return nullptr;
            for (const auto& m : a3.m) {
                TrackObj* t = tracks[m.first];
                bool was_lost = t->d->state == ST_LOST;
                if (!take(T, t, remaining[m.second]->d)) return nullptr;
                (was_lost ? refind : act).push_back(t);
                done.push_back(t);
            }
            std::vector<int> left;
            for (int j : a3.ub) left.push_back(ud[j]);
            ud.swap(left);
        }
    }
    for (TrackObj* t : rt)
        if (!has(done, t) && t->d->state != ST_LOST) {
            t->d->state = ST_LOST;
            lost_now.push_back(t);
            push_ref(T->lost_tf, t);
        }

    // unconfirmed tracks x remaining high-score detections, then births
    TV rest;
    for (int j : ud) rest.push_back(dets[j]);
    std::vector<int> ud3;
    if (!unconfirmed.empty()) {
        Cost c = fused_dists(T, unconfirmed, rest);
        class_gate(T, c, unconfirmed, rest);
        Assign a4;
        if (!linear_assignment(c, 0.7, a4)) return nullptr;
        for (const auto& m : a4.m) {
            TrackObj* u = unconfirmed[m.first];
            if (!take(T, u, rest[m.second]->d)) return nullptr;
            act.push_back(u);
            if (T->n_init > 2) {   // ByteTrackTracker._tentative
                u->d->hits += 1;
                if (u->d->hits < T->n_init) u->d->is_activated = false;
            }
        }
        for (int i : a4.ua) unconfirmed[i]->d->state = ST_REMOVED;
        ud3 = a4.ub;
    } else {
        ud3.resize(rest.size());
        std::iota(ud3.begin(), ud3.end(), 0);
    }
    double edge_birth = -2;
    for (int j : ud3) {
        TrackObj* d = rest[j];
        if (!(d->d->score >= T->new_)) continue;
        bool ok = edge_ok(d, edge_birth);
        if (PyErr_Occurred()) return nullptr;
        if (!ok) continue;
        T->next_id += 1;
        activate(d->d, T->frame_id, T->next_id);
        act.push_back(d);
        push_ref(T->new_tf, d);
        if (T->kind == K_OCSORT) oc_note(T, d->d, d->d);
        if (T->kind == K_BOTSORT && d->d->det_feat.has) {
            set_feat(d->d, d->d->det_feat);
            d->d->feat_frame = T->frame_id;
        }
    }

    for (TrackObj* t : T->lost)
        if (T->frame_id - t->d->end_frame > T->buffer) t->d->state = ST_REMOVED;
    TV tracked;
    for (TrackObj* t : T->tracked) if (t->d->state == ST_TRACKED) push_ref(tracked, t);
    for (int pass = 0; pass < 2; ++pass)
        for (TrackObj* t : (pass ? refind : act)) if (!has(tracked, t)) push_ref(tracked, t);
    TV lost;
    for (TrackObj* t : T->lost) if (t->d->state == ST_LOST && !has(tracked, t)) push_ref(lost, t);
    for (TrackObj* t : lost_now) if (!has(tracked, t) && t->d->state != ST_REMOVED) push_ref(lost, t);
    T->tracked.swap(tracked);
    T->lost.swap(lost);
    clear_tv(tracked);
    clear_tv(lost);
    return output_rows(T);
}

// ---------------------------------------------------------------- Python type: Track

enum TrackField {
    F_XYWH0, F_SCORE, F_CLS, F_IDX, F_FEAT, F_DET_FEAT, F_MEAN, F_COV, F_STATE, F_ACTIVATED, F_TRACK_ID, F_FRAME_ID,
    F_END_FRAME, F_START_FRAME, F_LAST_DET, F_FEAT_FRAME, F_HITS, F_XYWH, F_XYXY, F_OBS, F_VDIR, F_KF_OBS
};

static PyObject* feat_object(const Feat& f)
{
    if (!f.has) Py_RETURN_NONE;
    return np_array(f.v.data(), static_cast<Py_ssize_t>(f.v.size()), f.f32);
}

static PyObject* track_get(PyObject* self, void* closure)
{
    TrackData* t = reinterpret_cast<TrackObj*>(self)->d;
    switch (static_cast<TrackField>(reinterpret_cast<intptr_t>(closure))) {
    case F_XYWH0: return np_array(t->xywh0, 4);
    case F_SCORE: return PyFloat_FromDouble(t->score);
    case F_CLS: return PyLong_FromLong(t->cls);
    case F_IDX: return PyLong_FromLong(t->idx);
    case F_FEAT: return feat_object(t->feat);
    case F_DET_FEAT: return feat_object(t->det_feat);
    case F_MEAN: if (!t->has_mean) Py_RETURN_NONE; return np_array(t->mean, 8);
    case F_COV: if (!t->has_mean) Py_RETURN_NONE; return np_array(t->cov, 64, false, 8, 8);
    case F_STATE: return PyLong_FromLong(t->state);
    case F_ACTIVATED: return PyBool_FromLong(t->is_activated);
    case F_TRACK_ID: return PyLong_FromLong(t->track_id);
    case F_FRAME_ID: return PyLong_FromLong(t->frame_id);
    case F_END_FRAME: return PyLong_FromLong(t->end_frame);
    case F_START_FRAME: return PyLong_FromLong(t->start_frame);
    case F_LAST_DET: {
        if (!t->has_last) Py_RETURN_NONE;
        Ref box(np_array(t->last_box, 4));
        if (!box) return nullptr;
        return Py_BuildValue("(llO)", t->last_frame, t->last_idx, box.p);
    }
    case F_FEAT_FRAME: return PyLong_FromLong(t->feat_frame);
    case F_HITS: return PyLong_FromLong(t->hits);
    case F_XYWH: return np_array(xywh_of(t), 4);
    case F_XYXY: { double b[4]; xyxy_of(t, b); return np_array(b, 4); }
    case F_OBS: {
        Ref list(PyList_New(0));
        if (!list) return nullptr;
        for (const Obs& o : t->obs) {
            Ref box(np_array(o.b, 4));
            if (!box) return nullptr;
            Ref item(Py_BuildValue("(lO)", o.frame, box.p));
            if (!item || PyList_Append(list, item) != 0) return nullptr;
        }
        return list.release();
    }
    case F_VDIR: if (!t->has_vdir) Py_RETURN_NONE; return np_array(t->vdir, 2);
    case F_KF_OBS: {
        if (!t->has_kf_obs) Py_RETURN_NONE;
        Ref m(np_array(t->ko_mean, 8)), c(np_array(t->ko_cov, 64, false, 8, 8));
        if (!m || !c) return nullptr;
        return Py_BuildValue("(OO)", m.p, c.p);
    }
    }
    Py_RETURN_NONE;
}

static int set_fixed(PyObject* value, double* dst, size_t n, const char* name)
{
    std::vector<double> v;
    if (!read_doubles(value, v)) return -1;
    if (v.size() != n) { PyErr_Format(PyExc_ValueError, "%s needs %zu numbers", name, n); return -1; }
    std::memcpy(dst, v.data(), n * 8);
    return 0;
}

static int track_set(PyObject* self, PyObject* value, void* closure)
{
    TrackData* t = reinterpret_cast<TrackObj*>(self)->d;
    TrackField f = static_cast<TrackField>(reinterpret_cast<intptr_t>(closure));
    if (!value) { PyErr_SetString(PyExc_AttributeError, "cannot delete a track field"); return -1; }
    long* lf = nullptr;
    switch (f) {
    case F_XYWH0: return set_fixed(value, t->xywh0, 4, "xywh0");
    case F_SCORE: { double v = PyFloat_AsDouble(value); if (PyErr_Occurred()) return -1; t->score = v; return 0; }
    case F_FEAT: case F_DET_FEAT: {
        Feat& dst = f == F_FEAT ? t->feat : t->det_feat;
        if (value == Py_None) { dst = Feat(); return 0; }
        Feat nf;
        if (!read_feat_object(value, nf)) return -1;
        dst = nf;
        return 0;
    }
    case F_MEAN: if (set_fixed(value, t->mean, 8, "mean") != 0) return -1; t->has_mean = true; return 0;
    case F_COV: return set_fixed(value, t->cov, 64, "cov");
    case F_STATE: { long v = PyLong_AsLong(value); if (PyErr_Occurred()) return -1; t->state = static_cast<int>(v); return 0; }
    case F_ACTIVATED: { int v = PyObject_IsTrue(value); if (v < 0) return -1; t->is_activated = v == 1; return 0; }
    case F_CLS: lf = &t->cls; break;
    case F_IDX: lf = &t->idx; break;
    case F_TRACK_ID: lf = &t->track_id; break;
    case F_FRAME_ID: lf = &t->frame_id; break;
    case F_END_FRAME: lf = &t->end_frame; break;
    case F_START_FRAME: lf = &t->start_frame; break;
    case F_FEAT_FRAME: lf = &t->feat_frame; break;
    case F_HITS: lf = &t->hits; break;
    default: PyErr_SetString(PyExc_AttributeError, "read-only track field"); return -1;
    }
    long v = PyLong_AsLong(value);
    if (PyErr_Occurred()) return -1;
    *lf = v;
    return 0;
}

#define LWIR_TF(name, field) {name, track_get, track_set, nullptr, reinterpret_cast<void*>(static_cast<intptr_t>(field))}
static PyGetSetDef track_getset[] = {
    LWIR_TF("xywh0", F_XYWH0), LWIR_TF("score", F_SCORE), LWIR_TF("cls", F_CLS), LWIR_TF("idx", F_IDX),
    LWIR_TF("feat", F_FEAT), LWIR_TF("det_feat", F_DET_FEAT), LWIR_TF("mean", F_MEAN), LWIR_TF("cov", F_COV),
    LWIR_TF("state", F_STATE), LWIR_TF("is_activated", F_ACTIVATED), LWIR_TF("track_id", F_TRACK_ID),
    LWIR_TF("frame_id", F_FRAME_ID), LWIR_TF("end_frame", F_END_FRAME), LWIR_TF("start_frame", F_START_FRAME),
    LWIR_TF("last_det", F_LAST_DET), LWIR_TF("feat_frame", F_FEAT_FRAME), LWIR_TF("hits", F_HITS),
    LWIR_TF("xywh", F_XYWH), LWIR_TF("xyxy", F_XYXY), LWIR_TF("obs", F_OBS), LWIR_TF("vdir", F_VDIR),
    LWIR_TF("kf_obs", F_KF_OBS),
    {"__dict__", PyObject_GenericGetDict, PyObject_GenericSetDict, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}
};

static PyObject* track_set_feat(PyObject* self, PyObject* arg)
{
    Feat f;
    if (!read_feat_object(arg, f)) return nullptr;
    set_feat(reinterpret_cast<TrackObj*>(self)->d, f);
    Py_RETURN_NONE;
}

static PyMethodDef track_methods[] = {
    {"set_feat", track_set_feat, METH_O, "Blend an L2-normalised appearance feature into the track (0.9 old + 0.1 new)."},
    {nullptr, nullptr, 0, nullptr}
};

static void track_dealloc(PyObject* self)
{
    TrackObj* t = reinterpret_cast<TrackObj*>(self);
    delete t->d;
    Py_XDECREF(t->dict);
    PyObject_Del(self);
}

// ---------------------------------------------------------------- Python type: Tracker

static PyObject* tv_list(const TV& v)
{
    PyObject* list = PyList_New(static_cast<Py_ssize_t>(v.size()));
    if (!list) return nullptr;
    for (size_t i = 0; i < v.size(); ++i) {
        Py_INCREF(reinterpret_cast<PyObject*>(v[i]));
        PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), reinterpret_cast<PyObject*>(v[i]));
    }
    return list;
}

static int tv_assign(TV& dst, PyObject* value)
{
    if (!value) { PyErr_SetString(PyExc_AttributeError, "cannot delete a track list"); return -1; }
    Ref seq(PySequence_Fast(value, "expected a list of tracks"));
    if (!seq) return -1;
    TV next;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.p);
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq.p, i);
        if (!PyObject_TypeCheck(item, G.track_type)) {
            clear_tv(next);
            PyErr_SetString(PyExc_TypeError, "track lists accept only this tracker's track objects");
            return -1;
        }
        push_ref(next, reinterpret_cast<TrackObj*>(item));
    }
    dst.swap(next);
    clear_tv(next);
    return 0;
}

enum TrackerField {
    T_TRACKED, T_LOST, T_NEW_TF, T_LOST_TF, T_FRAME_ID, T_NEXT_ID, T_N_INIT, T_BUFFER, T_CLASS_GATE, T_HIGH, T_LOW,
    T_NEW, T_MATCH, T_DT, T_VDC, T_IOU_MIN, T_APP_THR, T_PROX_THR, T_APP_USED, T_MODE, T_KIND
};

static const char* kind_name(Kind k) { return k == K_SORT ? "sort" : k == K_OCSORT ? "ocsort" : "botsort"; }

static PyObject* tracker_get(PyObject* self, void* closure)
{
    TrackerData* T = reinterpret_cast<TrackerObj*>(self)->d;
    switch (static_cast<TrackerField>(reinterpret_cast<intptr_t>(closure))) {
    case T_TRACKED: return tv_list(T->tracked);
    case T_LOST: return tv_list(T->lost);
    case T_NEW_TF: return tv_list(T->new_tf);
    case T_LOST_TF: return tv_list(T->lost_tf);
    case T_FRAME_ID: return PyLong_FromLong(T->frame_id);
    case T_NEXT_ID: return PyLong_FromLong(T->next_id);
    case T_N_INIT: return PyLong_FromLong(T->n_init);
    case T_BUFFER: return PyLong_FromLong(T->buffer);
    case T_CLASS_GATE: return PyBool_FromLong(T->class_gate);
    case T_HIGH: return PyFloat_FromDouble(T->high);
    case T_LOW: return PyFloat_FromDouble(T->low);
    case T_NEW: return PyFloat_FromDouble(T->new_);
    case T_MATCH: return PyFloat_FromDouble(T->match);
    case T_DT: return PyLong_FromLong(T->dt);
    case T_VDC: return PyFloat_FromDouble(T->vdc);
    case T_IOU_MIN: return PyFloat_FromDouble(T->iou_min);
    case T_APP_THR: return PyFloat_FromDouble(T->app_thr);
    case T_PROX_THR: return PyFloat_FromDouble(T->prox_thr);
    case T_APP_USED: return PyLong_FromLong(T->app_used);
    case T_MODE: return PyUnicode_FromString(T->mode.c_str());
    case T_KIND: return PyUnicode_FromString(kind_name(T->kind));
    }
    Py_RETURN_NONE;
}

static int tracker_set(PyObject* self, PyObject* value, void* closure)
{
    TrackerData* T = reinterpret_cast<TrackerObj*>(self)->d;
    TrackerField f = static_cast<TrackerField>(reinterpret_cast<intptr_t>(closure));
    if (!value) { PyErr_SetString(PyExc_AttributeError, "cannot delete a tracker field"); return -1; }
    long* lf = nullptr;
    double* df = nullptr;
    switch (f) {
    case T_TRACKED: return tv_assign(T->tracked, value);
    case T_LOST: return tv_assign(T->lost, value);
    case T_NEW_TF: return tv_assign(T->new_tf, value);
    case T_LOST_TF: return tv_assign(T->lost_tf, value);
    case T_CLASS_GATE: { int v = PyObject_IsTrue(value); if (v < 0) return -1; T->class_gate = v == 1; return 0; }
    case T_FRAME_ID: lf = &T->frame_id; break;
    case T_NEXT_ID: lf = &T->next_id; break;
    case T_N_INIT: lf = &T->n_init; break;
    case T_BUFFER: lf = &T->buffer; break;
    case T_DT: lf = &T->dt; break;
    case T_APP_USED: lf = &T->app_used; break;
    case T_HIGH: df = &T->high; break;
    case T_LOW: df = &T->low; break;
    case T_NEW: df = &T->new_; break;
    case T_MATCH: df = &T->match; break;
    case T_VDC: df = &T->vdc; break;
    case T_IOU_MIN: df = &T->iou_min; break;
    case T_APP_THR: df = &T->app_thr; break;
    case T_PROX_THR: df = &T->prox_thr; break;
    default: PyErr_SetString(PyExc_AttributeError, "read-only tracker field"); return -1;
    }
    if (lf) { long v = PyLong_AsLong(value); if (PyErr_Occurred()) return -1; *lf = v; return 0; }
    double v = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) return -1;
    *df = v;
    return 0;
}

#define LWIR_KF(name, field) {name, tracker_get, tracker_set, nullptr, reinterpret_cast<void*>(static_cast<intptr_t>(field))}
static PyGetSetDef tracker_getset[] = {
    LWIR_KF("tracked", T_TRACKED), LWIR_KF("lost", T_LOST), LWIR_KF("new_this_frame", T_NEW_TF),
    LWIR_KF("lost_this_frame", T_LOST_TF), LWIR_KF("frame_id", T_FRAME_ID), LWIR_KF("next_id", T_NEXT_ID),
    LWIR_KF("n_init", T_N_INIT), LWIR_KF("buffer", T_BUFFER), LWIR_KF("class_gate", T_CLASS_GATE),
    LWIR_KF("high", T_HIGH), LWIR_KF("low", T_LOW), LWIR_KF("new", T_NEW), LWIR_KF("match", T_MATCH),
    LWIR_KF("dt", T_DT), LWIR_KF("vdc", T_VDC), LWIR_KF("iou_min", T_IOU_MIN), LWIR_KF("APP_THR", T_APP_THR),
    LWIR_KF("PROX_THR", T_PROX_THR), LWIR_KF("app_used", T_APP_USED), LWIR_KF("mode", T_MODE), LWIR_KF("kind", T_KIND),
    {"__dict__", PyObject_GenericGetDict, PyObject_GenericSetDict, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}
};

static PyObject* tracker_update(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static const char* names[] = {"arr", "det_feats", "feat_fn", nullptr};
    PyObject *arr = nullptr, *det_feats = Py_None, *feat_fn = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OO", const_cast<char**>(names), &arr, &det_feats, &feat_fn)) return nullptr;
    return tracker_update_impl(reinterpret_cast<TrackerObj*>(self), arr, det_feats);   // feat_fn serves rebind only
}

static TrackObj* find_track(const TrackerData* T, long track_id)
{
    for (int pass = 0; pass < 2; ++pass)
        for (TrackObj* t : (pass ? T->lost : T->tracked))
            if (t->d->track_id == track_id && t->d->state != ST_REMOVED) return t;
    return nullptr;
}

static PyObject* tracker_find(PyObject* self, PyObject* arg)
{
    long id = PyLong_AsLong(arg);
    if (PyErr_Occurred()) return nullptr;
    TrackObj* t = find_track(reinterpret_cast<TrackerObj*>(self)->d, id);
    if (!t) Py_RETURN_NONE;
    Py_INCREF(reinterpret_cast<PyObject*>(t));
    return reinterpret_cast<PyObject*>(t);
}

// ByteTrackTracker.attach_embedding outside rebind mode: store the late feature, never merge tracks
static PyObject* tracker_attach_embedding(PyObject* self, PyObject* args)
{
    long track_id, frame_id;
    PyObject *feat, *kind;
    if (!PyArg_ParseTuple(args, "llOO", &track_id, &frame_id, &feat, &kind)) return nullptr;
    TrackObj* t = find_track(reinterpret_cast<TrackerObj*>(self)->d, track_id);
    if (!t) Py_RETURN_NONE;
    Feat f;
    if (!read_feat_object(feat, f)) return nullptr;
    set_feat(t->d, f);
    t->d->feat_frame = frame_id;
    Py_RETURN_NONE;
}

static PyMethodDef tracker_methods[] = {
    {"update", reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)(void)>(tracker_update)), METH_VARARGS | METH_KEYWORDS,
     "update(arr[N,6] x1,y1,x2,y2,conf,cls, det_feats=None, feat_fn=None) -> rows [x1,y1,x2,y2,id,conf,cls,det_idx]"},
    {"find", tracker_find, METH_O, "Track with this ID in the tracked or lost list, else None."},
    {"attach_embedding", tracker_attach_embedding, METH_VARARGS, "Store a late appearance feature on a track."},
    {nullptr, nullptr, 0, nullptr}
};

static void tracker_dealloc(PyObject* self)
{
    TrackerObj* t = reinterpret_cast<TrackerObj*>(self);
    if (t->d) {
        clear_tv(t->d->tracked); clear_tv(t->d->lost); clear_tv(t->d->new_tf); clear_tv(t->d->lost_tf);
        delete t->d;
    }
    Py_XDECREF(t->dict);
    PyObject_Del(self);
}

// ---------------------------------------------------------------- module

static bool ends_with(const std::string& s, const std::string& tail)
{
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

static PyObject* module_attr(PyObject* mods, const std::string& name, const char* attr)   // new reference or null
{
    PyObject* m = PyDict_GetItemString(mods, name.c_str());
    if (!m) return nullptr;
    PyObject* v = PyObject_GetAttrString(m, attr);
    if (!v) PyErr_Clear();
    return v;
}

// The server's runtime switches live in Python modules that are already imported next to native_backend.
static void resolve_runtime()
{
    if (G.edge_ok) return;
    PyObject* mods = PyImport_GetModuleDict();
    std::string prefix;
    bool found = false;
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    static const std::string tail = "track.tracker_common";
    while (PyDict_Next(mods, &pos, &key, &value)) {
        if (!PyUnicode_Check(key) || value == Py_None) continue;
        const char* raw = PyUnicode_AsUTF8(key);
        if (!raw) { PyErr_Clear(); continue; }
        std::string name(raw);
        if (!ends_with(name, tail)) continue;
        std::string p = name.substr(0, name.size() - std::strlen("tracker_common"));
        bool beside_loader = PyDict_GetItemString(mods, (p + "native_backend").c_str()) != nullptr;
        if (!found || beside_loader) { prefix = p; found = true; }
        if (beside_loader) break;
    }
    if (!found) return;
    std::string tc = prefix + "tracker_common";
    G.settings = module_attr(mods, tc, "settings");
    G.posp = module_attr(mods, tc, "POSP");
    G.pos_inn = module_attr(mods, tc, "pos_inn");
    G.pos_predict = module_attr(mods, prefix + "position_memory", "pos_predict");
    G.KF = module_attr(mods, prefix + "kalman", "KF");
    G.edge_ok = module_attr(mods, tc, "edge_ok");
}

static int dict_long(PyObject* kwargs, const char* key, long& out)
{
    PyObject* v = PyDict_GetItemString(kwargs, key);
    if (!v || v == Py_None) return 0;
    long x = PyLong_AsLong(v);
    if (PyErr_Occurred()) return -1;
    out = x;
    return 1;
}

static PyObject* create_tracker_for(Kind kind, PyObject* args)
{
    const char* mode;
    PyObject* kwargs;
    if (!PyArg_ParseTuple(args, "sO!", &mode, &PyDict_Type, &kwargs)) return nullptr;
    if (std::strcmp(mode, kind_name(kind)) != 0) {
        PyErr_Format(PyExc_ValueError, "this extension provides mode '%s', not '%s'", kind_name(kind), mode);
        return nullptr;
    }
    resolve_runtime();
    TrackerObj* self = PyObject_New(TrackerObj, G.tracker_type);
    if (!self) return nullptr;
    self->d = nullptr;
    self->dict = nullptr;
    Ref owner(reinterpret_cast<PyObject*>(self));
    self->d = new (std::nothrow) TrackerData();
    self->dict = PyDict_New();
    if (!self->d || !self->dict) return PyErr_NoMemory();
    TrackerData* T = self->d;
    T->kind = kind;
    T->mode = "bytetrack";   // every one of these Python classes runs the ByteTrack lifecycle in mode 'bytetrack'
    if (PyDict_Update(self->dict, kwargs) != 0) return nullptr;
    long gate = 0;
    if (dict_long(kwargs, "n_init", T->n_init) < 0 || dict_long(kwargs, "class_gate", gate) < 0) return nullptr;
    T->class_gate = gate != 0;
    if (kind == K_SORT) { T->low = T->high; T->buffer = 1; }   // SORT: no low-score association, max_age 1
    else if (dict_long(kwargs, "buffer", T->buffer) < 0) return nullptr;
    Ref kf;
    if (G.KF) kf.p = PyObject_CallObject(G.KF, nullptr);
    if (!kf) {
        PyErr_Clear();
        Ref types(PyImport_ImportModule("types"));
        if (!types) return nullptr;
        kf.p = PyObject_CallMethod(types, "SimpleNamespace", nullptr);
        if (!kf || PyObject_SetAttrString(kf, "warp", Py_None) != 0) return nullptr;
    }
    Ref zero(PyLong_FromLong(0));
    if (!zero || PyDict_SetItemString(self->dict, "kf", kf) != 0
        || PyDict_SetItemString(self->dict, "rebinds_motion", zero) != 0
        || PyDict_SetItemString(self->dict, "rebinds_app", zero) != 0
        || PyDict_SetItemString(self->dict, "rebinds_fallback", zero) != 0
        || PyDict_SetItemString(self->dict, "diag", Py_None) != 0) return nullptr;
    return owner.release();
}

// linear_assignment(cost[n, m] float32 or float64, thresh) -> (matches, unmatched_rows, unmatched_cols); for unit comparison with Python
static PyObject* py_linear_assignment(PyObject*, PyObject* args)
{
    PyObject* arr;
    double thresh;
    if (!PyArg_ParseTuple(args, "Od", &arr, &thresh)) return nullptr;
    Cost c;
    std::vector<Py_ssize_t> shape;
    if (!read_doubles(arr, c.v, &c.f32, &shape)) return nullptr;
    if (shape.size() != 2) { PyErr_SetString(PyExc_ValueError, "cost must be 2-D"); return nullptr; }
    c.n = static_cast<int>(shape[0]); c.m = static_cast<int>(shape[1]);
    Assign a;
    if (!linear_assignment(c, thresh, a)) return nullptr;
    Ref m(PyList_New(0)), ua(PyList_New(0)), ub(PyList_New(0));
    if (!m || !ua || !ub) return nullptr;
    for (const auto& p : a.m) { Ref t(Py_BuildValue("(ii)", p.first, p.second)); if (!t || PyList_Append(m, t) != 0) return nullptr; }
    for (int i : a.ua) { Ref t(PyLong_FromLong(i)); if (!t || PyList_Append(ua, t) != 0) return nullptr; }
    for (int j : a.ub) { Ref t(PyLong_FromLong(j)); if (!t || PyList_Append(ub, t) != 0) return nullptr; }
    return Py_BuildValue("(OOO)", m.p, ua.p, ub.p);
}

static PyTypeObject TrackType = {PyVarObject_HEAD_INIT(nullptr, 0)};
static PyTypeObject TrackerType = {PyVarObject_HEAD_INIT(nullptr, 0)};

static PyObject* module_init(PyModuleDef* def, const char* track_name, const char* tracker_name, Kind kind)
{
    G.np = PyImport_ImportModule("numpy");
    if (!G.np) return nullptr;
    {
        Ref version(PyObject_GetAttrString(G.np, "__version__"));
        const char* text = version && PyUnicode_Check(version.p) ? PyUnicode_AsUTF8(version) : nullptr;
        if (!text) return nullptr;
        G.np_weak_scalars = std::atoi(text) >= 2;
    }

    TrackType.tp_name = track_name;
    TrackType.tp_basicsize = sizeof(TrackObj);
    TrackType.tp_dealloc = track_dealloc;
    TrackType.tp_flags = Py_TPFLAGS_DEFAULT;
    TrackType.tp_doc = "One track or detection of a native tracker (same fields as tracker_common.Track).";
    TrackType.tp_methods = track_methods;
    TrackType.tp_getset = track_getset;
    TrackType.tp_dictoffset = offsetof(TrackObj, dict);
    TrackType.tp_getattro = PyObject_GenericGetAttr;
    TrackType.tp_setattro = PyObject_GenericSetAttr;

    TrackerType.tp_name = tracker_name;
    TrackerType.tp_basicsize = sizeof(TrackerObj);
    TrackerType.tp_dealloc = tracker_dealloc;
    TrackerType.tp_flags = Py_TPFLAGS_DEFAULT;
    TrackerType.tp_doc = "Native tracker with the state fields of the Python tracker classes.";
    TrackerType.tp_methods = tracker_methods;
    TrackerType.tp_getset = tracker_getset;
    TrackerType.tp_dictoffset = offsetof(TrackerObj, dict);
    TrackerType.tp_getattro = PyObject_GenericGetAttr;
    TrackerType.tp_setattro = PyObject_GenericSetAttr;

    if (PyType_Ready(&TrackType) < 0 || PyType_Ready(&TrackerType) < 0) return nullptr;
    G.track_type = &TrackType;
    G.tracker_type = &TrackerType;

    Ref m(PyModule_Create(def));
    if (!m) return nullptr;
    Ref modes(Py_BuildValue("(s)", kind_name(kind)));
    if (!modes || PyModule_AddIntConstant(m, "API_VERSION", 1) != 0
        || PyModule_AddObject(m, "SUPPORTED_MODES", modes.p) != 0) return nullptr;
    modes.release();   // reference taken by the module
    Py_INCREF(reinterpret_cast<PyObject*>(&TrackType));
    Py_INCREF(reinterpret_cast<PyObject*>(&TrackerType));
    if (PyModule_AddObject(m, "Track", reinterpret_cast<PyObject*>(&TrackType)) != 0
        || PyModule_AddObject(m, "Tracker", reinterpret_cast<PyObject*>(&TrackerType)) != 0) return nullptr;
    return m.release();
}

}   // namespace lwirnt

// One module per tracker kind: LWIR_NATIVE_TRACKER(sort, lwirnt::K_SORT) defines PyInit__native_sort.
#define LWIR_NATIVE_TRACKER(stem, kind)                                                                      \
    static PyObject* lwir_create_tracker(PyObject*, PyObject* args) { return lwirnt::create_tracker_for(kind, args); } \
    static PyMethodDef lwir_module_methods[] = {                                                             \
        {"create_tracker", lwir_create_tracker, METH_VARARGS, "create_tracker(mode, kwargs) -> tracker"},   \
        {"linear_assignment", lwirnt::py_linear_assignment, METH_VARARGS, "linear_assignment(cost, thresh)"},\
        {nullptr, nullptr, 0, nullptr}};                                                                     \
    static PyModuleDef lwir_module = {PyModuleDef_HEAD_INIT, "_native_" #stem,                               \
        "C++ " #stem " tracker for the LWIR tracking server (native extension ABI v1).", -1,                \
        lwir_module_methods, nullptr, nullptr, nullptr, nullptr};                                            \
    PyMODINIT_FUNC PyInit__native_##stem(void)                                                               \
    {                                                                                                        \
        return lwirnt::module_init(&lwir_module, "_native_" #stem ".Track", "_native_" #stem ".Tracker", kind); \
    }
