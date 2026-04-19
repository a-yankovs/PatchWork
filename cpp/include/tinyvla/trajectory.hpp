#pragma once
// trajectory.hpp — Trajectory generation for SO101 arm
//
// Provides:
//   - TrajectoryPoint    : time-stamped joint + EEF snapshot
//   - Trajectory         : ordered sequence of TrajectoryPoints
//   - TrajectoryGenerator: builds Trajectories from waypoints

#include "common.hpp"
#include "kinematics.hpp"
#include <vector>
#include <optional>

namespace tinyvla {

// ---------------------------------------------------------------------------
// TrajectoryPoint — one sample in a trajectory
// ---------------------------------------------------------------------------

struct TrajectoryPoint {
    double      time_s;         // time from trajectory start (seconds)
    JointState  joints;         // target joint state at this time
    Pose        eef_pose;       // EEF pose (for reference / logging)
    GripperCommand gripper_cmd; // gripper state at this point
};

// ---------------------------------------------------------------------------
// Trajectory — full sequence
// ---------------------------------------------------------------------------

class Trajectory {
public:
    Trajectory() = default;
    explicit Trajectory(std::vector<TrajectoryPoint> points);

    bool   empty()    const { return points_.empty(); }
    size_t size()     const { return points_.size(); }
    double duration() const;   // total time in seconds

    const TrajectoryPoint& operator[](size_t i) const { return points_[i]; }
    const TrajectoryPoint& front() const { return points_.front(); }
    const TrajectoryPoint& back()  const { return points_.back(); }

    // Sample trajectory at time t (cubic spline interpolation).
    // Clamps to [0, duration] outside the range.
    TrajectoryPoint sample(double t) const;

    // Returns all points
    const std::vector<TrajectoryPoint>& points() const { return points_; }

private:
    std::vector<TrajectoryPoint> points_;

    // Cubic Hermite spline coefficients for a single segment
    static double hermite(double t, double p0, double p1, double m0, double m1, double dt);
};

// ---------------------------------------------------------------------------
// TrajectoryGenerator — builds trajectories from waypoints
// ---------------------------------------------------------------------------

struct WaypointSpec {
    Pose           pose;            // desired EEF pose
    GripperCommand gripper_cmd;     // gripper action at this waypoint
    double         blend_radius_m;  // smoothing radius (0 = sharp corner)
};

struct TrajectoryConfig {
    double max_joint_vel_rad_s  = 1.0;   // rad/s per joint
    double max_joint_acc_rad_s2 = 2.0;   // rad/s² per joint
    double max_cart_vel_m_s     = 0.15;  // m/s end-effector
    double sample_dt_s          = 0.02;  // 50 Hz default
    int    ik_max_iter          = 300;
};

class TrajectoryGenerator {
public:
    explicit TrajectoryGenerator(
        const Kinematics&      kin,
        const TrajectoryConfig& cfg = {}
    );

    // Build a joint-space trajectory from a list of joint waypoints.
    // Uses trapezoidal velocity profile (ramps up, holds, ramps down).
    Trajectory joint_space(
        const std::vector<JointState>& waypoints,
        const std::vector<GripperCommand>& gripper_cmds = {}
    ) const;

    // Build a Cartesian-space trajectory from EEF waypoints.
    // Internally solves IK at each sample and falls back to joint-space.
    Trajectory cartesian_space(
        const JointState&             start_joints,
        const std::vector<WaypointSpec>& waypoints
    ) const;

    // Convenience: build a pick trajectory
    // Moves arm above object, descends, closes gripper, lifts.
    Trajectory pick(
        const JointState& start,
        const Pose&       object_pose,
        double            approach_height_m = 0.08,
        double            z_offset_mm       = 0.0
    ) const;

    // Convenience: build a place trajectory
    // Moves arm to above shelf, descends, opens gripper, retreats.
    Trajectory place(
        const JointState& start,
        const Pose&       shelf_pose,
        double            approach_height_m = 0.08
    ) const;

    // Combined pick-and-place: pick from object_pose, place at shelf_pose.
    Trajectory pick_and_place(
        const JointState& start,
        const Pose&       object_pose,
        const Pose&       shelf_pose,
        double            approach_height_m = 0.08,
        double            z_offset_mm       = 0.0
    ) const;

private:
    Kinematics      kin_;
    TrajectoryConfig cfg_;

    // Compute segment duration given joint displacement using trapezoidal profile
    double segment_duration(
        const JointState& from,
        const JointState& to
    ) const;

    // Interpolate joint states linearly (used for dense sampling)
    static JointState lerp_joints(
        const JointState& a,
        const JointState& b,
        double t   // [0,1]
    );
};

} // namespace tinyvla
