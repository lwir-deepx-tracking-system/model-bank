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
};
