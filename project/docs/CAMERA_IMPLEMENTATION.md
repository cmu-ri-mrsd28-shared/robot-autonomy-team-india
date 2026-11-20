# Camera tracking: implementation guide

## Requirements

Build a camera subsystem that tracks a flying table-tennis ball and supplies its **3D position and velocity** in the Franka base frame, with timestamps and a validity indicator. It must cover the incoming flight as well as the box; the 50 cm sphere limits the predicted landing point, not the ball's entire flight.

- Use a fixed, calibrated stereo/depth setup or synchronized cameras; choose the hardware with the team. A single ordinary RGB image does not directly provide metric 3D position.
- Provide position in meters and velocity in meters/second, each as `(x, y, z)`. Provide gravity/acceleration in meters/second² as described below.
- Deliver estimates to the controller at **50 Hz**, with capture and processing timing exposed so latency and stale frames are detectable. The camera may need a higher frame rate; determine it from measured tracking performance rather than assuming 50 Hz is sufficient.
- Calibrate the camera-to-robot-base transform and validate 3D error and latency over the intended flight volume, including near the box and under occlusion.
- Report lost, stale, ambiguous or low-confidence tracking explicitly. Never turn an invalid measurement into a believable zero position.
- Initially run offline or in observation-only mode. **This repository does not contain a hardware controller, camera tracker, or calibration routine.**

The current checkpoint receives more than camera data: it also needs robot, box, prediction and controller state. The camera subsystem must not fabricate those other inputs.

## Instructions for implementation

### 1. Read the existing contract

Use these files as the source of truth, rather than assuming this guide replaces the code:

- `catchbox/core.py`: `BallState` and `BallStateSource`. `BallState` has `time`, `position`, `velocity`, `valid`, and optional `acceleration`.
- `catchbox/learning/trajectory.py`: `Predictor.predict()` calculates a descending crossing from position, velocity, acceleration and the nominal opening height. It rejects invalid states, contact, non-finite acceleration, or acceleration magnitude above 30 m/s².
- `catchbox/learning/environment.py`: `CatchCore.observation()` encodes the checkpoint's 60-value input. `CatchCore.step()` defines the action smoothing, controller state and nominal motion. The checkpoint is `artifacts/original-model/box/final_model.zip`.

The simulation's “world” frame is fixed with gravity along negative z. For hardware, define the robot base as the fixed reference frame and explicitly transform **both** the camera estimates and robot/box states into one common frame with matching axes. Do not silently assume a camera or robot convention matches the simulation. Camera and robot calibration must establish the transform.

### 2. Define a timestamped tracker output

Keep camera/estimator data separate from policy observation encoding. A suitable adapter-facing record is:

```python
@dataclass
class TrackedBall:
    capture_time_s: float       # monotonic acquisition time in a documented clock domain
    estimate_time_s: float      # time represented by position and velocity
    published_time_s: float     # output time; exposes processing latency
    position_m: np.ndarray      # shape (3,), robot-base frame
    velocity_m_s: np.ndarray    # shape (3,), robot-base frame
    acceleration_m_s2: np.ndarray | None  # shape (3,), same frame; see below
    valid: bool
    confidence: float | None    # optional estimator-defined quality
```

Specify how camera and robot clocks are synchronized, how timestamps are mapped to a common monotonic time base, and whether the estimate has been propagated from image capture to `estimate_time_s`. Avoid double-predicting latency. Validate shapes and finite values; reject stale, out-of-order, missing, or low-confidence estimates. Do not treat confidence as an extra model input: the existing network has no such field.

For a freely flying ball, a sensible initial acceleration estimate is gravity `[0, 0, -9.81]` m/s² **only when the calibrated base frame has +z upward**. Rotate gravity appropriately otherwise. The simulator supplies actual MuJoCo acceleration, including contact effects; gravity-only acceleration is an approximation in flight, not a claim that post-contact acceleration is known. Do not numerically differentiate noisy position twice and present it as an exact measurement. Flag post-contact behavior separately and do not use the free-flight predictor after contact.

`BallState` has a narrower interface than the suggested record. Convert an explicitly valid, synchronized estimate into `BallState(time=..., position=..., velocity=..., valid=True, acceleration=...)`; retain capture/publish timestamps and confidence in the tracker or adapter. Clarify the clock/time convention at that boundary. The existing `BallStateSource.read()` is an interface seam, not a complete hardware integration.

### 3. Estimate and calibrate

1. Calibrate each camera's intrinsics and lens distortion; for multiple cameras, calibrate their relative pose and synchronize acquisition.
2. Find the rigid transform from camera coordinates to the Franka base frame using a calibration target observed at multiple known robot poses or independently surveyed points. Validate on held-out points spread throughout the intended volume, not just at the calibration target.
3. Detect the ball, reconstruct a metric 3D center, and associate detections over time. Estimate velocity with a timestamp-aware tracker, using gravity as a flight-model prior when appropriate. Document data association, filtering, initialization, track loss and reacquisition.
4. Measure acquisition-to-output latency, jitter, 3D position error and velocity error using representative throws, motion blur, different depths, lighting and partial occlusion. State measured performance and acceptable limits; do not invent thresholds or claim a particular camera will meet them.
5. Obtain the calibrated box-opening transform from the actual grasp and robot kinematics. The predictor needs the nominal opening **height** in the same frame; a real grasp can change this transform. Camera calibration by itself does not provide it.

The simulated box is visibly based on the STL but uses approximate collision geometry and a rigid, no-slip grasp. The ball tracker should report the ball center rather than attempting to infer the simulated collision metrics.

### 4. Preserve the checkpoint's exact 60-value observation

The model does **not** accept a camera record directly. The following is the existing zero-based ordering from `CatchCore.observation()`. Preserve the current scaling, ordering, clipping, rotation layout and state transitions for checkpoint compatibility; do not silently retrain or change the observation space. `q` contains seven arm joint coordinates followed by the two finger coordinates in the simulator's order. Finger positions are linear, even though the current encoder divides all nine position values by π.

| Indices | Source | Values before final clipping |
|---|---|---|
| 0–8 | Robot/gripper | Nine joint positions, `q / π` |
| 9–17 | Robot/gripper | Nine joint velocities, `qvel / 3` |
| 18–20 | Tracker | Ball position `(x,y,z)` in meters |
| 21–23 | Tracker | Ball velocity `(x,y,z) / 5` |
| 24–26 | Tracker | Ball acceleration clipped componentwise to ±60 m/s², then divided by 30 |
| 27–29 | Robot + calibrated tool transform | Box opening-center position in meters |
| 30–38 | Robot + calibrated tool transform | Box opening 3 × 3 rotation matrix, NumPy row-major flattening |
| 39–41 | Robot + calibrated tool transform | Opening-center linear velocity divided by 3 |
| 42–44 | Predictor | Predicted ball position at the nominal opening-height crossing |
| 45–47 | Predictor | Predicted ball velocity at crossing divided by 5 |
| 48 | Predictor | Time to crossing, clipped to `[0, 2]` seconds |
| 49 | Predictor | Current prediction-valid flag, 0 or 1 |
| 50 | Contact subsystem | Latched ball–box contact flag, 0 or 1 |
| 51–57 | Controller | Current seven smoothed residual values, in their existing order |
| 58 | Controller | Elapsed episode time divided by 3.5 seconds |
| 59 | Controller | `min(self.retention.maximum, 1)`; in the current **box** environment this stays zero |

The entire vector is clipped to `[-10, 10]` and converted to `float32`. Camera measurements supply indices 18–26; the predicted values at 42–49 are derived from them. All other values come from robot feedback, calibrated tool geometry, a contact subsystem, or controller state. The simulator latches contact on first ball–box contact and caches the last valid prediction; when the current prediction is invalid, index 49 is zero while the cached predicted fields can remain. Follow the existing code when implementing and testing this behavior, but **do not invoke a motion controller on stale/invalid tracking merely because cached fields exist**.

For the current original printed-box checkpoint, `soft_landing` and `velocity_match` are disabled. The simulation applies a seven-value action after smoothing (`residual = 0.8 * residual + 0.2 * clipped_action`) and combines it with its model-based controller. Do not substitute raw network outputs for robot joint commands.

### 5. Contact, invalid data and safety boundary

Simulation reads actual MuJoCo ball–box contacts. A real camera may lose sight of the ball at contact, and camera detection alone does not reproduce the simulator's contact signal. Define and validate a contact estimator or separate sensor with the robotics team; expose uncertainty outside the checkpoint. Until there is a reliable signal and hardware integration, use observation-only testing rather than inventing a contact value.

If data is invalid, stale, outside calibrated coverage, or timing cannot be aligned, signal the higher-level system to inhibit/cancel interception and follow its validated safe-state behavior. The 50 cm sphere bounds a predicted landing point and a commanded box-opening target in the simulation; it is **not** a collision check or a guarantee that the arm's swept volume remains clear. This camera guide does not authorize real robot motion or bypass any robot limits.

### 6. Deliverables and checks

- A documented camera/estimator interface, calibration artifacts and reproducible calibration procedure, with frame definitions and measured latency/error.
- Recorded-data replay tests for known positions and trajectories, invalid/missing frames, out-of-order timestamps, occlusion, and transform direction/units.
- Numerical tests comparing the hardware observation encoder with `CatchCore.observation()` on identical synthetic states: all 60 indices, dtype, scaling, row-major orientation, clipping, cached-prediction behavior, contact and residual state.
- A tracker-only dry run that logs image times, estimates, confidence, predicted intercept and ground-truth or surveyed comparison where available. Review error and coverage before connecting any motion path.
- A short report of limitations, uncertainty and unresolved contact estimation. Do not alter the checkpoint, run a training job, or command the robot as part of camera implementation without an explicit request.
