#pragma once
// controller.hpp — Joint-space PID + feedforward controller for SO101
//
// The controller receives a trajectory (from motion_planner) and runs a
// control loop at a fixed frequency, calling a user-supplied "send_command"
// callback with joint torques/positions.
//
// Supports two modes:
//   POSITION — servo-level position tracking (standard for SO101 Dynamixels)
//   TORQUE   — PID + gravity compensation (requires calibrated model)

#include "common.hpp"
#include "trajectory.hpp"
#include <functional>
#include <atomic>
#include <chrono>

namespace tinyvla {

// ---------------------------------------------------------------------------
// PID gains for one joint
// ---------------------------------------------------------------------------

struct PIDGains {
    double kp = 5.0;    // proportional
    double ki = 0.1;    // integral
    double kd = 0.5;    // derivative
    double i_clamp = 1.0;  // integral wind-up clamp (rad·s)
};

// ---------------------------------------------------------------------------
// Controller config
// ---------------------------------------------------------------------------

struct ControllerConfig {
    double loop_rate_hz      = 50.0;   // control loop frequency
    double position_tol_rad  = 0.01;   // convergence criterion
    double max_effort_nm     = 2.0;    // max torque output per joint (Nm)

    std::array<PIDGains, NUM_JOINTS> gains = {};  // default-constructed

    ControllerConfig() {
        for (auto& g : gains) g = PIDGains{};
    }
};

// ---------------------------------------------------------------------------
// ControllerMode
// ---------------------------------------------------------------------------

enum class ControllerMode { POSITION, TORQUE };

// ---------------------------------------------------------------------------
// JointCommand — output from the controller
// ---------------------------------------------------------------------------

struct JointCommand {
    std::array<double, NUM_JOINTS> position  = {};   // rad
    std::array<double, NUM_JOINTS> velocity  = {};   // rad/s
    std::array<double, NUM_JOINTS> effort    = {};   // Nm (torque mode)
    GripperCommand                 gripper   = GripperCommand::HOLD;
    ControllerMode                 mode      = ControllerMode::POSITION;
};

// ---------------------------------------------------------------------------
// Controller
// ---------------------------------------------------------------------------

class Controller {
public:
    using CommandCallback  = std::function<void(const JointCommand&)>;
    using StateCallback    = std::function<JointState()>;   // reads current joints
    using GripperCallback  = std::function<void(GripperCommand, double force)>;

    explicit Controller(
        const ControllerConfig& cfg            = {},
        ControllerMode          mode           = ControllerMode::POSITION
    );

    // ---- Execute a trajectory ----------------------------------------------

    // Synchronous execution: blocks until trajectory finishes or timeout.
    // Calls cmd_cb at cfg.loop_rate_hz, reads state via state_cb.
    // Returns Status::OK on success.
    Status execute(
        const Trajectory&  traj,
        const CommandCallback& cmd_cb,
        const StateCallback&   state_cb,
        const GripperCallback& gripper_cb,
        double timeout_s = 30.0
    );

    // ---- Stop / estop -------------------------------------------------------

    void request_stop();   // graceful deceleration
    void estop();          // immediate zero-velocity command

    bool is_running() const { return running_.load(); }

    // ---- Tuning -------------------------------------------------------------

    void set_gains(int joint, const PIDGains& g) { cfg_.gains[joint] = g; }
    void set_mode(ControllerMode m) { mode_ = m; }

    // ---- Single-step (used for testing) ------------------------------------

    JointCommand compute_command(
        const TrajectoryPoint& desired,
        const JointState&      actual,
        double                 dt_s
    );

private:
    ControllerConfig         cfg_;
    ControllerMode           mode_;
    std::atomic<bool>        running_  { false };
    std::atomic<bool>        stop_req_ { false };
    std::atomic<bool>        estop_    { false };

    // Per-joint PID state
    std::array<double, NUM_JOINTS> integral_   = {};
    std::array<double, NUM_JOINTS> prev_error_ = {};
    bool first_step_ = true;

    void reset_pid();

    JointCommand position_command(
        const TrajectoryPoint& desired,
        const JointState&      actual,
        double                 dt_s
    );

    JointCommand torque_command(
        const TrajectoryPoint& desired,
        const JointState&      actual,
        double                 dt_s
    );
};

} // namespace tinyvla
