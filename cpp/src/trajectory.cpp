// trajectory.cpp — Trajectory generation implementation

#include "tinyvla/trajectory.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace tinyvla {

// ---------------------------------------------------------------------------
// Trajectory
// ---------------------------------------------------------------------------

Trajectory::Trajectory(std::vector<TrajectoryPoint> points)
    : points_(std::move(points)) {}

double Trajectory::duration() const {
    if (points_.empty()) return 0.0;
    return points_.back().time_s - points_.front().time_s;
}

// Cubic Hermite basis function
double Trajectory::hermite(double t, double p0, double p1,
                            double m0, double m1, double dt) {
    double u  = t / dt;
    double u2 = u * u;
    double u3 = u2 * u;
    double h00 = 2*u3 - 3*u2 + 1;
    double h10 = u3 - 2*u2 + u;
    double h01 = -2*u3 + 3*u2;
    double h11 = u3 - u2;
    return h00*p0 + h10*dt*m0 + h01*p1 + h11*dt*m1;
}

TrajectoryPoint Trajectory::sample(double t) const {
    if (points_.empty())
        throw std::runtime_error("Trajectory is empty");

    // Clamp
    if (t <= points_.front().time_s) return points_.front();
    if (t >= points_.back().time_s)  return points_.back();

    // Binary search for the segment
    size_t lo = 0, hi = points_.size() - 1;
    while (hi - lo > 1) {
        size_t mid = (lo + hi) / 2;
        if (points_[mid].time_s <= t) lo = mid;
        else                          hi = mid;
    }
    const auto& p0 = points_[lo];
    const auto& p1 = points_[hi];
    double dt = p1.time_s - p0.time_s;
    if (dt < 1e-9) return p0;
    double u = (t - p0.time_s) / dt;

    // Finite-difference tangents (Catmull-Rom)
    auto tangent = [&](size_t idx, int joint) -> double {
        if (idx == 0)
            return (points_[1].joints.position[joint] - points_[0].joints.position[joint])
                   / (points_[1].time_s - points_[0].time_s);
        if (idx == points_.size() - 1) {
            size_t n = points_.size();
            return (points_[n-1].joints.position[joint] - points_[n-2].joints.position[joint])
                   / (points_[n-1].time_s - points_[n-2].time_s);
        }
        return (points_[idx+1].joints.position[joint] - points_[idx-1].joints.position[joint])
               / (points_[idx+1].time_s - points_[idx-1].time_s);
    };

    TrajectoryPoint result;
    result.time_s    = t;
    result.gripper_cmd = (u < 0.5) ? p0.gripper_cmd : p1.gripper_cmd;

    for (int j = 0; j < NUM_JOINTS; ++j) {
        double m0 = tangent(lo, j);
        double m1 = tangent(hi, j);
        result.joints.position[j] = hermite(
            t - p0.time_s,
            p0.joints.position[j],
            p1.joints.position[j],
            m0, m1, dt
        );
        // Linear interpolation for velocity
        result.joints.velocity[j] = lerp(
            p0.joints.velocity[j],
            p1.joints.velocity[j],
            u
        );
    }
    return result;
}

// ---------------------------------------------------------------------------
// TrajectoryGenerator
// ---------------------------------------------------------------------------

TrajectoryGenerator::TrajectoryGenerator(
    const Kinematics&      kin,
    const TrajectoryConfig& cfg
) : kin_(kin), cfg_(cfg) {}

// ---- Segment duration (trapezoidal velocity profile) ----------------------

double TrajectoryGenerator::segment_duration(
    const JointState& from,
    const JointState& to
) const {
    double max_time = 0.0;
    for (int j = 0; j < NUM_JOINTS; ++j) {
        double dq  = std::abs(to.position[j] - from.position[j]);
        double v   = cfg_.max_joint_vel_rad_s;
        double a   = cfg_.max_joint_acc_rad_s2;
        // Trapezoid: time to ramp + hold + ramp
        double t_ramp = v / a;
        double d_ramp = 0.5 * a * t_ramp * t_ramp;
        double t_seg;
        if (dq <= 2.0 * d_ramp) {
            // Triangular profile (never reaches max velocity)
            t_seg = 2.0 * std::sqrt(dq / a);
        } else {
            t_seg = t_ramp * 2.0 + (dq - 2.0 * d_ramp) / v;
        }
        max_time = std::max(max_time, t_seg);
    }
    return std::max(max_time, cfg_.sample_dt_s);
}

// ---- Linear interpolation of joint states ---------------------------------

JointState TrajectoryGenerator::lerp_joints(
    const JointState& a,
    const JointState& b,
    double t
) {
    JointState q;
    for (int j = 0; j < NUM_JOINTS; ++j) {
        q.position[j] = tinyvla::lerp(a.position[j], b.position[j], t);
        q.velocity[j] = tinyvla::lerp(a.velocity[j], b.velocity[j], t);
    }
    return q;
}

// ---- Joint-space trajectory -----------------------------------------------

Trajectory TrajectoryGenerator::joint_space(
    const std::vector<JointState>&    waypoints,
    const std::vector<GripperCommand>& gripper_cmds
) const {
    if (waypoints.empty())
        return {};

    std::vector<TrajectoryPoint> pts;
    double t = 0.0;

    for (size_t seg = 0; seg + 1 < waypoints.size(); ++seg) {
        const JointState& q0 = waypoints[seg];
        const JointState& q1 = waypoints[seg + 1];
        double duration = segment_duration(q0, q1);
        GripperCommand gcmd = gripper_cmds.size() > seg
                            ? gripper_cmds[seg]
                            : GripperCommand::HOLD;

        int n_steps = std::max(2, (int)std::ceil(duration / cfg_.sample_dt_s));
        for (int i = 0; i < n_steps; ++i) {
            double u = (double)i / (n_steps - 1);
            double t_seg = t + u * duration;
            JointState q = lerp_joints(q0, q1, u);
            // Compute velocity as finite difference
            double du = 1.0 / (n_steps - 1);
            for (int j = 0; j < NUM_JOINTS; ++j)
                q.velocity[j] = (q1.position[j] - q0.position[j]) / duration;

            TrajectoryPoint tp;
            tp.time_s     = t_seg;
            tp.joints     = q;
            tp.eef_pose   = kin_.forward(q);
            tp.gripper_cmd = gcmd;
            pts.push_back(tp);
        }
        t += duration;
    }

    // Ensure the last point exists
    if (!pts.empty() && pts.back().joints.position != waypoints.back().position) {
        TrajectoryPoint tp;
        tp.time_s     = t;
        tp.joints     = waypoints.back();
        tp.eef_pose   = kin_.forward(waypoints.back());
        tp.gripper_cmd = gripper_cmds.empty()
                       ? GripperCommand::HOLD
                       : gripper_cmds.back();
        pts.push_back(tp);
    }

    return Trajectory(std::move(pts));
}

// ---- Cartesian-space trajectory -------------------------------------------

Trajectory TrajectoryGenerator::cartesian_space(
    const JointState&              start_joints,
    const std::vector<WaypointSpec>& waypoints
) const {
    if (waypoints.empty()) return {};

    std::vector<JointState>    jt_wps;
    std::vector<GripperCommand> g_wps;
    jt_wps.push_back(start_joints);

    JointState seed = start_joints;
    for (const auto& wp : waypoints) {
        auto maybe_q = kin_.inverse_analytical(wp.pose, seed);
        if (!maybe_q) maybe_q = kin_.inverse(wp.pose, seed);
        if (!maybe_q) {
            // Skip unreachable waypoints gracefully
            continue;
        }
        jt_wps.push_back(*maybe_q);
        g_wps.push_back(wp.gripper_cmd);
        seed = *maybe_q;
    }

    return joint_space(jt_wps, g_wps);
}

// ---- Pick trajectory -------------------------------------------------------

Trajectory TrajectoryGenerator::pick(
    const JointState& start,
    const Pose&       object_pose,
    double            approach_height_m,
    double            z_offset_mm
) const {
    // Build waypoints:
    // 1. Pre-grasp above the object
    // 2. Descend to grasp position
    // 3. Close gripper (hold)
    // 4. Lift back up

    double z_off = z_offset_mm / 1000.0;

    Pose above = object_pose;
    above.position.z += approach_height_m;

    Pose grasp = object_pose;
    grasp.position.z += z_off;

    Pose lift = grasp;
    lift.position.z += approach_height_m;

    return cartesian_space(start, {
        { above, GripperCommand::OPEN, 0.0 },
        { grasp, GripperCommand::OPEN, 0.0 },
        { grasp, GripperCommand::CLOSE, 0.0 },  // close at grasp point
        { lift,  GripperCommand::CLOSE, 0.0 },
    });
}

// ---- Place trajectory ------------------------------------------------------

Trajectory TrajectoryGenerator::place(
    const JointState& start,
    const Pose&       shelf_pose,
    double            approach_height_m
) const {
    Pose above = shelf_pose;
    above.position.z += approach_height_m;

    Pose place_pt = shelf_pose;

    Pose retreat = above;

    return cartesian_space(start, {
        { above,    GripperCommand::CLOSE, 0.0 },
        { place_pt, GripperCommand::CLOSE, 0.0 },
        { place_pt, GripperCommand::OPEN,  0.0 },  // open at shelf
        { retreat,  GripperCommand::OPEN,  0.0 },
    });
}

// ---- Pick-and-place ---------------------------------------------------------

Trajectory TrajectoryGenerator::pick_and_place(
    const JointState& start,
    const Pose&       object_pose,
    const Pose&       shelf_pose,
    double            approach_height_m,
    double            z_offset_mm
) const {
    double z_off = z_offset_mm / 1000.0;

    // Pre-grasp above object
    Pose above_obj = object_pose;
    above_obj.position.z += approach_height_m;

    // Grasp position
    Pose grasp = object_pose;
    grasp.position.z += z_off;

    // Lift position
    Pose lift = grasp;
    lift.position.z += approach_height_m;

    // Above shelf
    Pose above_shelf = shelf_pose;
    above_shelf.position.z += approach_height_m;

    // Place position
    Pose place_pt = shelf_pose;

    // Retreat
    Pose retreat = above_shelf;

    return cartesian_space(start, {
        { above_obj,   GripperCommand::OPEN,  0.0 },
        { grasp,       GripperCommand::OPEN,  0.0 },
        { grasp,       GripperCommand::CLOSE, 0.0 },  // grasp
        { lift,        GripperCommand::CLOSE, 0.0 },
        { above_shelf, GripperCommand::CLOSE, 0.0 },
        { place_pt,    GripperCommand::CLOSE, 0.0 },
        { place_pt,    GripperCommand::OPEN,  0.0 },  // release
        { retreat,     GripperCommand::OPEN,  0.0 },
    });
}

} // namespace tinyvla
