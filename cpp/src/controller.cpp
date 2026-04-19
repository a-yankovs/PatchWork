// controller.cpp — PID controller implementation

#include "tinyvla/controller.hpp"
#include <cmath>
#include <thread>
#include <chrono>
#include <algorithm>
#include <stdexcept>

namespace tinyvla {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

Controller::Controller(const ControllerConfig& cfg, ControllerMode mode)
    : cfg_(cfg), mode_(mode)
{}

// ---------------------------------------------------------------------------
// PID reset
// ---------------------------------------------------------------------------

void Controller::reset_pid() {
    integral_.fill(0.0);
    prev_error_.fill(0.0);
    first_step_ = true;
}

// ---------------------------------------------------------------------------
// Position command (servo-level, typical for SO101 Dynamixels)
// ---------------------------------------------------------------------------

JointCommand Controller::position_command(
    const TrajectoryPoint& desired,
    const JointState&      actual,
    double                 dt_s
) {
    JointCommand cmd;
    cmd.mode       = ControllerMode::POSITION;
    cmd.gripper    = desired.gripper_cmd;

    for (int j = 0; j < NUM_JOINTS; ++j) {
        double error = desired.joints.position[j] - actual.position[j];
        error = wrap_angle(error);

        if (!first_step_) {
            integral_[j]   += error * dt_s;
            integral_[j]    = clamp(integral_[j],
                                   -cfg_.gains[j].i_clamp,
                                    cfg_.gains[j].i_clamp);
            double deriv    = (error - prev_error_[j]) / std::max(dt_s, 1e-6);
            double effort   = cfg_.gains[j].kp * error
                            + cfg_.gains[j].ki * integral_[j]
                            + cfg_.gains[j].kd * deriv;
            cmd.effort[j] = clamp(effort, -cfg_.max_effort_nm, cfg_.max_effort_nm);
        }

        // In POSITION mode the servo handles the actual tracking;
        // we output the desired position + feedforward velocity.
        cmd.position[j] = desired.joints.position[j];
        cmd.velocity[j] = desired.joints.velocity[j];
        prev_error_[j]  = error;
    }
    first_step_ = false;
    return cmd;
}

// ---------------------------------------------------------------------------
// Torque command (PID + gravity compensation placeholder)
// ---------------------------------------------------------------------------

JointCommand Controller::torque_command(
    const TrajectoryPoint& desired,
    const JointState&      actual,
    double                 dt_s
) {
    JointCommand cmd;
    cmd.mode    = ControllerMode::TORQUE;
    cmd.gripper = desired.gripper_cmd;

    for (int j = 0; j < NUM_JOINTS; ++j) {
        double e_pos = wrap_angle(desired.joints.position[j] - actual.position[j]);
        double e_vel = desired.joints.velocity[j] - actual.velocity[j];

        integral_[j] += e_pos * dt_s;
        integral_[j]  = clamp(integral_[j],
                              -cfg_.gains[j].i_clamp,
                               cfg_.gains[j].i_clamp);

        double effort = cfg_.gains[j].kp * e_pos
                      + cfg_.gains[j].ki * integral_[j]
                      + cfg_.gains[j].kd * e_vel;

        // Gravity compensation: very rough approximation for elbow/shoulder
        // A calibrated model would replace this with proper dynamics
        if (j == 1) effort += 0.3 * std::cos(actual.position[j]);  // shoulder
        if (j == 2) effort += 0.15 * std::cos(actual.position[j]); // elbow

        cmd.effort[j]   = clamp(effort, -cfg_.max_effort_nm, cfg_.max_effort_nm);
        cmd.position[j] = desired.joints.position[j];
        cmd.velocity[j] = desired.joints.velocity[j];
    }
    return cmd;
}

// ---------------------------------------------------------------------------
// compute_command (single step, public)
// ---------------------------------------------------------------------------

JointCommand Controller::compute_command(
    const TrajectoryPoint& desired,
    const JointState&      actual,
    double                 dt_s
) {
    if (mode_ == ControllerMode::TORQUE)
        return torque_command(desired, actual, dt_s);
    return position_command(desired, actual, dt_s);
}

// ---------------------------------------------------------------------------
// execute — synchronous trajectory execution
// ---------------------------------------------------------------------------

Status Controller::execute(
    const Trajectory&      traj,
    const CommandCallback& cmd_cb,
    const StateCallback&   state_cb,
    const GripperCallback& gripper_cb,
    double                 timeout_s
) {
    if (traj.empty()) return Status::TRAJECTORY_EMPTY;

    reset_pid();
    running_.store(true);
    stop_req_.store(false);
    estop_.store(false);

    using clock = std::chrono::steady_clock;
    auto dt_us = std::chrono::microseconds(
        (long long)(1e6 / cfg_.loop_rate_hz));
    auto deadline = clock::now() + std::chrono::duration<double>(timeout_s);

    double t_start = traj.front().time_s;
    double t_end   = traj.back().time_s;
    auto   wall_start = clock::now();

    GripperCommand last_gcmd = GripperCommand::OPEN;
    double prev_loop_t = 0.0;

    while (!estop_.load()) {
        auto loop_start = clock::now();
        if (loop_start >= deadline) {
            running_.store(false);
            return Status::TIMEOUT;
        }

        double elapsed = std::chrono::duration<double>(loop_start - wall_start).count();
        double traj_t  = t_start + elapsed;
        double dt_s    = elapsed - prev_loop_t;
        prev_loop_t    = elapsed;

        // Clamp to trajectory range
        if (traj_t > t_end) {
            // Send final point and finish
            auto final_pt = traj.back();
            auto state    = state_cb();
            auto cmd      = compute_command(final_pt, state, dt_s);
            cmd_cb(cmd);
            gripper_cb(final_pt.gripper_cmd, 0.6f);
            break;
        }

        TrajectoryPoint desired = traj.sample(traj_t);
        JointState      actual  = state_cb();
        JointCommand    cmd     = compute_command(desired, actual, dt_s);

        cmd_cb(cmd);

        // Issue gripper command if it changed
        if (desired.gripper_cmd != last_gcmd) {
            gripper_cb(desired.gripper_cmd, 0.6);
            last_gcmd = desired.gripper_cmd;
        }

        if (stop_req_.load()) break;

        // Sleep for remaining loop time
        auto elapsed_loop = clock::now() - loop_start;
        if (elapsed_loop < dt_us)
            std::this_thread::sleep_for(dt_us - elapsed_loop);
    }

    running_.store(false);
    if (estop_.load()) return Status::UNKNOWN_ERROR;  // E-stop is abnormal
    return Status::OK;
}

// ---------------------------------------------------------------------------
// Stop / estop
// ---------------------------------------------------------------------------

void Controller::request_stop() {
    stop_req_.store(true);
}

void Controller::estop() {
    estop_.store(true);
}

} // namespace tinyvla
