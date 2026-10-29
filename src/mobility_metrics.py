"""NumPy metrics at the rear axle, with explicit low-speed and settling guards."""
import numpy as np

SCENARIOS = ("hold", "straight", "left", "right", "reverse", "reverse_left",
             "reverse_right", "brake", "reverse_brake")


def mobility_metrics(log, names, brake_event=None, turn_window=(1.5, 3.5),
                     curvature_window=(2.5, 3.4), speed_window=(2.5, 2.9)):
    columns = {name: log[:, i] for i, name in enumerate(names)}
    c = columns
    t = c["time"]
    result = {"steering_rmse_deg": [], "rear_speed_rmse_rad_s": [], "curvature": None,
              "braking": None, "checks": {}}
    active = np.abs(c["v_command"]) > .01
    turn = (t >= turn_window[0]) & (t < turn_window[1]) & (np.abs(c["curvature_ref"]) > .01)
    tracking = turn if turn.any() else active
    if tracking.any():
        for suffix in ("l", "r"):
            result["steering_rmse_deg"].append(float(np.rad2deg(np.sqrt(np.mean(
                (c["steer_"+suffix][tracking]-c["steer_ref_"+suffix][tracking])**2)))))
            result["rear_speed_rmse_rad_s"].append(float(np.sqrt(np.mean(
                (c["rear_speed_"+suffix][tracking]-c["rear_ref_"+suffix][tracking])**2))))
        result["mean_speed_command_m_s"] = float(c["v_command"][tracking].mean())
        result["mean_speed_achieved_m_s"] = float(c["rear_v_long"][tracking].mean())
    speed_samples = active & (t >= speed_window[0]) & (t <= speed_window[1])
    if speed_samples.sum() >= 3:
        target_speed = float(c["v_command"][speed_samples].mean())
        achieved_speed = float(c["rear_v_long"][speed_samples].mean())
        result["steady_speed_ratio"] = achieved_speed/target_speed
        result["checks"]["steady_speed_error_below_20pct"] = bool(abs(achieved_speed/target_speed-1) < .2)
    if len(t) > 1:
        result["signed_longitudinal_travel_m"] = float(np.trapezoid(c["rear_v_long"], t))
    result["target_clipping_pct"] = float(100*np.mean(c["steer_target_clipped"]))
    result["checks"]["steering_targets_feasible"] = not bool(np.any(c["steer_target_clipped"]))
    # Use achieved signed longitudinal velocity at the same reference point as
    # Ackermann kinematics. Averaging pointwise omega/v near zero is avoided.
    steady = turn & (t >= curvature_window[0]) & (t <= curvature_window[1])
    result["measurement_windows_s"] = {"turn": list(turn_window), "curvature": list(curvature_window), "speed": list(speed_window)}
    if steady.sum() >= 3:
        time = t[steady]
        distance = float(np.trapezoid(c["rear_v_long"][steady], time))
        yaw = float(np.trapezoid(c["yaw_rate"][steady], time))
        target_yaw = float(np.trapezoid(c["curvature_ref"][steady]*c["rear_v_long"][steady], time))
        if abs(distance) > .03 and abs(target_yaw) > 1e-5:
            actual = yaw/distance
            desired = target_yaw/distance
            relative_error = abs(actual-desired)/abs(desired)
            result["curvature"] = {"window_s": [float(time[0]), float(time[-1])],
                                   "actual_per_m": actual, "target_per_m": desired,
                                   "ratio": actual/desired, "relative_error": relative_error,
                                   "signed_distance_m": distance,
                                   "measured_yaw_deg": float(np.rad2deg(yaw)),
                                   "target_yaw_at_achieved_speed_deg": float(np.rad2deg(target_yaw))}
            result["checks"]["curvature_error_below_20pct"] = bool(relative_error < .20)
        else:
            result["checks"]["curvature_error_below_20pct"] = False
    elif turn.any():
        result["checks"]["curvature_error_below_20pct"] = False
    # The first zero command after motion is the commanded brake event.
    moving = np.flatnonzero(active)
    stops = np.flatnonzero((np.arange(len(t)) > moving[0]) & ~active) if len(moving) else []
    if len(stops):
        start = int(stops[0])
        speed = np.hypot(c["rear_v_long"], c["rear_v_lat"])
        wheels = np.maximum(np.abs(c["rear_speed_l"]), np.abs(c["rear_speed_r"]))
        quiet = (speed < .01) & (wheels < .1) & (np.abs(c["yaw_rate"]) < np.deg2rad(1.))
        sample_dt = float(np.median(np.diff(t)))
        required = int(np.ceil(.2/sample_dt-1e-9)) + 1
        settled = next((i for i in range(start, len(t)-required+1) if quiet[i:i+required].all()), None)
        event_time = float(t[start]) if brake_event is None else brake_event["time_s"]
        positions = np.column_stack((c["rear_axle_x"], c["rear_axle_y"]))
        event_position = positions[start] if brake_event is None else brake_event["rear_axle_xy"]
        stop_time = None if settled is None else float(t[settled]-event_time)
        stop_distance = None if settled is None else float(np.linalg.norm(np.diff(
            np.vstack((event_position, positions[start:settled+1])), axis=0), axis=1).sum())
        tail = t >= t[-1]-.2
        residual_speed = float(speed[tail].max())
        residual_wheel = float(wheels[tail].max())
        before = (t < event_time) & (t >= event_time-.2)
        pre_brake_speed = float(np.mean(speed[before])) if before.any() else 0.
        result["braking"] = {"command_zero_time_s": event_time, "command_zero_sample_s": float(t[start]),
                             "measurement_interval_s": sample_dt,
                             "settling_duration_s": .2, "stopping_time_s": stop_time,
                             "stopping_distance_m": stop_distance,
                             "pre_brake_speed_m_s": pre_brake_speed,
                             "residual_body_speed_m_s": residual_speed,
                             "residual_rear_speed_rad_s": residual_wheel}
        result["checks"].update({"stop_within_1_s": stop_time is not None and stop_time < 1.,
                                  "stop_within_0_05_m": stop_distance is not None and stop_distance < .05,
                                  "residual_speed_below_0_005_m_s": residual_speed < .005,
                                  "residual_rear_speed_below_0_05_rad_s": residual_wheel < .05})
        result["checks"]["moving_before_brake"] = pre_brake_speed > .05
    return result
