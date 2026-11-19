#pragma once

// v27 (gimbal_bridge_v27 only): motion-smoothing settings, kept apart from the team's
// config/app_config.hpp so that team/ stays unchanged. Every default reproduces the
// team's original behaviour (= gimbal_bridge_v25).
struct SmoothingConfig {
    static constexpr int max_catchup_speed = 7;
    // Deadband hysteresis: once an axis is inside the deadband it stays held until
    // |error| exceeds this exit threshold (same unit as deadband_x / deadband_y).
    // 0 or <= deadband = no hysteresis.
    double deadband_exit_x = 0.0;
    double deadband_exit_y = 0.0;
    // Deadband entry only: decelerate in the last direction at this rate
    // (SPD units/second), STOP once SPD would drop below speed_min.
    // 0 = immediate STOP. Safety stops are always immediate.
    double stop_rate = 0.0;
    // Direction reversal of one axis: false = STOP both axes; true = halt only that
    // axis, keep the other moving, and let the halted axis rejoin once the shared SPD
    // is at or below reversal_join_speed, or after reversal_hold_ms at the latest (so a
    // long fast move of the other axis cannot keep it waiting). SPD then follows only
    // the moving axes.
    bool axis_reversal = false;
    double reversal_join_speed = 5.0;
    int reversal_hold_ms = 500;
    // Maximum hold for one validated live detector miss. Zero disables it.
    int observation_grace_ms = 0;
    // Allow sustained measured motion to skip the slowest speed steps. Zero disables it.
    int catchup_speed = 0;
    // Predict a near-term horizontal deadband entry during a validated approach.
    bool predictive_brake = false;
    // Where on the screen the aim point of the followed box is held, as a fraction of the picture
    // height from the top. 0.5 = the centre (the team rule). 0.333 puts the head on the upper third,
    // so more of the body stays in the picture. Horizontal aiming stays on the centre.
    double aim_screen_y = 0.5;
    // Far gain and near brake (11-13, user: "fast when far, brake early when near"), per axis on the observed error in
    // pixels of a 640-wide picture. far_gain > 1 multiplies the PD speed demand by up to far_gain, growing linearly from
    // far_start_px to far_full_px of error. brake_px_per_spd > 0 caps the demand at (|error| - deadband) / brake_px_per_spd,
    // so the speed comes down while the target is still far enough away for the deceleration ramp and the coast after
    // STOP. 1 / 0 = off (the team rule).
    double far_gain = 1.0;
    double far_start_px = 64.0;
    double far_full_px = 160.0;
    double brake_px_per_spd = 0.0;
    // Search when lost (11-15, user: "sweep the surroundings so it can catch the person again"). false = the team rule:
    // STOP at once and wait for the same ID. true: after search_grace_ms without the followed ID, (1) keep turning in the
    // last horizontal following direction for at most search_follow_ms / search_follow_px, at the last following SPD
    // (at most search_follow_speed); (2) sweep left-right around the place where the person was lost, at search_speed,
    // first to +-search_sweep_px, each later turn search_widen times wider, never beyond +-search_max_px, with an optional
    // small tilt of +-search_tilt_px; (3) STOP and hold after search_timeout_ms (counted from the loss). The reacquire
    // timeout still drops the selection. Distances are pictures px of the 640-wide camera image, estimated from the
    // commanded SPD with the measured pan speed map (open loop; only used to bound and centre the pattern). The search
    // stops at once on a stale, unhealthy or missing frame, a lost heartbeat, a mode change or STOP, like following.
    // The follow-through uses the server's predictive-control velocity (VEL suffix, predictive_lead.py) when it is at
    // least search_min_velocity px/s and at most 300 ms older than the loss; otherwise the last following output.
    static constexpr double search_min_velocity = 15.0;
    bool search = false;
    int search_grace_ms = 300;
    int search_follow_ms = 1000;
    double search_follow_px = 320.0;
    int search_follow_speed = 10;
    int search_speed = 7;
    double search_sweep_px = 320.0;
    double search_widen = 1.5;
    double search_max_px = 640.0;
    double search_tilt_px = 0.0;
    int search_timeout_ms = 8000;
    // Coast when lost (11-15, user: "keep turning in the predicted direction for a short time while the followed id is
    // missing"). false = unchanged. true: while the followed ID is missing from a fresh healthy frame that carries the
    // server's prediction for it (PRED suffix: the last trusted box carried through the camera motion and moved on with the
    // person's velocity against the background, follow_guard.py), follow that predicted box with the normal controller for at
    // most coast_ms from the loss. Then as without coast: STOP and wait, or the search (its follow-through is skipped: the
    // coast already went where the person should be). Any frame without PRED, stale, unhealthy, heartbeat loss, mode change
    // or STOP ends it like following.
    bool coast = false;
    int coast_ms = 800;
    // Seek when lost (11-16, user: "while the prediction is shown the gimbal should search there too"). false = unchanged.
    // true: while the followed ID is missing and no coast applies, a fresh healthy frame may carry the server's SEEK box for
    // it (search_follow.py: a point of the displayed search region; the server walks it left-right inside that region and
    // measures the camera's real turning from the picture). The normal controller follows that box, with three limits that
    // hold whatever the server sends: the shared SPD never exceeds seek_speed, a change of direction always goes through a
    // full STOP, and nothing moves later than seek_ms after the loss. The state stays "lost" and the reacquire timeout
    // runs. Any frame without SEEK, stale, unhealthy, heartbeat loss, mode change or STOP ends it like following.
    // Not together with search (the open-loop pattern above).
    bool seek = false;
    int seek_ms = 4500;
    int seek_speed = 8;
    bool bounded_seek = false;
    // Control step at arrival (11-15, latency audit finding 2). false = the team rule: a new tracking result waits for the
    // worker's own period (period_ms, 25 ms; 0..25 ms, 12.5 ms on average). true: the control step runs as soon as a new
    // result is taken in, and the period timer restarts from there; the timer still runs the step every period_ms for
    // time-outs, ramps, coast, search and the heartbeat. The control law is the same: every rate (speed_rate, stop_rate)
    // is multiplied by the real time since the previous output (at most one period, as before), and every time limit
    // (reversal hold, coast, search, reacquire, input timeout) is measured on the clock, so ramps and time-outs take
    // the same real time. gimbal_bridge_v27 turns it on unless --arrival-control 0 is given.
    bool arrival_control = false;
};

// Measured picture speed (px/s of the 640x480 camera) at a pan SPD (PID_TUNING_RECORD_20261110.md sec. 5; above 20 not
// measured, held at 450). Only for the open-loop search bookkeeping.
inline double search_picture_px_per_s(int spd)
{
    static constexpr double table[] = {0, 156, 161, 160, 166, 172, 204, 237, 277, 316, 356, 367, 378, 388, 399, 410,
                                       418, 425, 433, 440, 448};
    if (spd <= 0) return 0.0;
    return spd < int(sizeof(table) / sizeof(table[0])) ? table[spd] : 450.0;
}
