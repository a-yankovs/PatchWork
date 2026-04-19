#pragma once
// motion_planner.hpp — High-level pick-and-place motion planner
//
// Sits above trajectory.hpp: takes a symbolic skill command from the SLM
// ("pick_object", "place_in_shelf", "full_pick_and_place", etc.) and turns it
// into a ready-to-execute Trajectory.
//
// ReplayParams from robot_api.py are passed in as PlanParams here so the SLM's
// patches (z_offset_mm, speed_scale, approach_angle_deg, gripper_close_force)
// flow all the way to the motion layer.

#include "common.hpp"
#include "kinematics.hpp"
#include "trajectory.hpp"
#include <string>
#include <functional>
#include <optional>

namespace tinyvla {

// ---------------------------------------------------------------------------
// PlanParams — mirrors robot_api.ReplayParams (patched by the SLM)
// ---------------------------------------------------------------------------

struct PlanParams {
    double z_offset_mm        = 0.0;   // additional z shift at grasp (mm)
    double speed_scale        = 1.0;   // multiply all max velocities by this
    double approach_angle_deg = 0.0;   // extra tilt of the approach vector (deg)
    double gripper_close_force= 0.6;   // normalised force [0,1]
    double approach_height_m  = 0.08;  // default pre-grasp hover height
};

// ---------------------------------------------------------------------------
// PlanResult
// ---------------------------------------------------------------------------

struct PlanResult {
    Trajectory trajectory;
    Status     status    = Status::OK;
    std::string message;

    bool ok() const { return status == Status::OK && !trajectory.empty(); }
};

// ---------------------------------------------------------------------------
// SceneInfo — what the VLM gives us about the scene
// ---------------------------------------------------------------------------

struct SceneInfo {
    Pose object_pose;     // detected object pose in robot base frame
    Pose shelf_pose;      // target placement pose in robot base frame
    bool object_visible   = false;
    bool shelf_visible    = false;
};

// ---------------------------------------------------------------------------
// MotionPlanner
// ---------------------------------------------------------------------------

class MotionPlanner {
public:
    explicit MotionPlanner(
        const Kinematics&       kin,
        const TrajectoryConfig& traj_cfg = {}
    );

    // ---- Individual skill planners -----------------------------------------

    // Plan a pick motion (approach, grasp, lift).
    PlanResult plan_pick(
        const JointState& current,
        const SceneInfo&  scene,
        const PlanParams& params = {}
    ) const;

    // Plan a place motion (transit to shelf, descend, release, retract).
    PlanResult plan_place(
        const JointState& current,
        const SceneInfo&  scene,
        const PlanParams& params = {}
    ) const;

    // Plan a full pick-and-place in one trajectory.
    PlanResult plan_full_pick_and_place(
        const JointState& current,
        const SceneInfo&  scene,
        const PlanParams& params = {}
    ) const;

    // Plan moving arm to a named home/safe pose.
    PlanResult plan_home(
        const JointState& current
    ) const;

    // ---- Generic dispatcher ------------------------------------------------

    // Dispatches to one of the above based on skill_name string.
    // skill_name: "pick_object" | "place_in_shelf" | "full_pick_and_place" |
    //             "box_in_shelf" | "home"
    PlanResult plan(
        const std::string& skill_name,
        const JointState&  current,
        const SceneInfo&   scene,
        const PlanParams&  params = {}
    ) const;

    // ---- Getters -----------------------------------------------------------

    const Kinematics& kinematics() const { return kin_; }

private:
    Kinematics         kin_;
    TrajectoryGenerator gen_;

    // Named home joint configuration (arm folded safe)
    static JointState home_joints();

    // Apply approach_angle_deg to approach vector in a Pose
    static Pose apply_approach_angle(const Pose& base, double angle_deg);
};

} // namespace tinyvla
