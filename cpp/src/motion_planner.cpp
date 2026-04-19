// motion_planner.cpp — MotionPlanner implementation

#include "tinyvla/motion_planner.hpp"
#include <cmath>
#include <stdexcept>

namespace tinyvla {

// ---------------------------------------------------------------------------
// Home configuration — arm safely folded above base
// ---------------------------------------------------------------------------

JointState MotionPlanner::home_joints() {
    JointState q;
    q.position[0] = 0.0;           // base centred
    q.position[1] = -PI / 4.0;     // shoulder slightly up
    q.position[2] =  PI / 3.0;     // elbow bent
    q.position[3] = -PI / 6.0;     // wrist level
    q.position[4] = 0.0;
    q.position[5] = 0.0;
    return q;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

MotionPlanner::MotionPlanner(
    const Kinematics&       kin,
    const TrajectoryConfig& traj_cfg
) : kin_(kin), gen_(kin, traj_cfg) {}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

Pose MotionPlanner::apply_approach_angle(const Pose& base, double angle_deg) {
    if (std::abs(angle_deg) < 1e-3) return base;
    // Tilt the end-effector approach vector around the x-axis by angle_deg
    double angle_rad = angle_deg * DEG_TO_RAD;
    Rotation3 tilt = Rotation3::from_rpy(angle_rad, 0.0, 0.0);
    Pose tilted = base;
    tilted.rotation = base.rotation * tilt;
    return tilted;
}

// ---------------------------------------------------------------------------
// plan_pick
// ---------------------------------------------------------------------------

PlanResult MotionPlanner::plan_pick(
    const JointState& current,
    const SceneInfo&  scene,
    const PlanParams& params
) const {
    if (!scene.object_visible)
        return { {}, Status::IK_FAILED, "Object not visible in scene" };

    Pose grasp_pose = apply_approach_angle(scene.object_pose, params.approach_angle_deg);

    // Scale approach height with speed (faster moves need more height to decel)
    double h = params.approach_height_m * std::max(0.5, params.speed_scale);

    // Apply speed scaling to trajectory config
    TrajectoryConfig cfg;
    cfg.max_joint_vel_rad_s  = 1.0 * params.speed_scale;
    cfg.max_joint_acc_rad_s2 = 2.0 * params.speed_scale;
    cfg.max_cart_vel_m_s     = 0.15 * params.speed_scale;
    TrajectoryGenerator gen(kin_, cfg);

    auto traj = gen.pick(current, grasp_pose, h, params.z_offset_mm);
    if (traj.empty())
        return { {}, Status::IK_FAILED, "IK failed during pick planning" };

    return { std::move(traj), Status::OK, "pick planned OK" };
}

// ---------------------------------------------------------------------------
// plan_place
// ---------------------------------------------------------------------------

PlanResult MotionPlanner::plan_place(
    const JointState& current,
    const SceneInfo&  scene,
    const PlanParams& params
) const {
    if (!scene.shelf_visible)
        return { {}, Status::IK_FAILED, "Shelf not visible in scene" };

    TrajectoryConfig cfg;
    cfg.max_joint_vel_rad_s  = 1.0 * params.speed_scale;
    cfg.max_joint_acc_rad_s2 = 2.0 * params.speed_scale;
    cfg.max_cart_vel_m_s     = 0.15 * params.speed_scale;
    TrajectoryGenerator gen(kin_, cfg);

    auto traj = gen.place(current, scene.shelf_pose, params.approach_height_m);
    if (traj.empty())
        return { {}, Status::IK_FAILED, "IK failed during place planning" };

    return { std::move(traj), Status::OK, "place planned OK" };
}

// ---------------------------------------------------------------------------
// plan_full_pick_and_place
// ---------------------------------------------------------------------------

PlanResult MotionPlanner::plan_full_pick_and_place(
    const JointState& current,
    const SceneInfo&  scene,
    const PlanParams& params
) const {
    if (!scene.object_visible)
        return { {}, Status::IK_FAILED, "Object not visible — cannot plan pick" };
    if (!scene.shelf_visible)
        return { {}, Status::IK_FAILED, "Shelf not visible — cannot plan place" };

    Pose grasp_pose = apply_approach_angle(scene.object_pose, params.approach_angle_deg);

    TrajectoryConfig cfg;
    cfg.max_joint_vel_rad_s  = 1.0 * params.speed_scale;
    cfg.max_joint_acc_rad_s2 = 2.0 * params.speed_scale;
    cfg.max_cart_vel_m_s     = 0.15 * params.speed_scale;
    TrajectoryGenerator gen(kin_, cfg);

    auto traj = gen.pick_and_place(
        current,
        grasp_pose,
        scene.shelf_pose,
        params.approach_height_m,
        params.z_offset_mm
    );

    if (traj.empty())
        return { {}, Status::IK_FAILED, "IK failed during pick-and-place planning" };

    return { std::move(traj), Status::OK, "full pick-and-place planned OK" };
}

// ---------------------------------------------------------------------------
// plan_home
// ---------------------------------------------------------------------------

PlanResult MotionPlanner::plan_home(const JointState& current) const {
    auto traj = gen_.joint_space(
        { current, home_joints() },
        { GripperCommand::OPEN }
    );
    if (traj.empty())
        return { {}, Status::TRAJECTORY_EMPTY, "Home trajectory empty" };
    return { std::move(traj), Status::OK, "home planned OK" };
}

// ---------------------------------------------------------------------------
// Generic dispatcher
// ---------------------------------------------------------------------------

PlanResult MotionPlanner::plan(
    const std::string& skill_name,
    const JointState&  current,
    const SceneInfo&   scene,
    const PlanParams&  params
) const {
    if (skill_name == "pick_object") {
        return plan_pick(current, scene, params);
    }
    else if (skill_name == "place_in_shelf" || skill_name == "place_in_box") {
        return plan_place(current, scene, params);
    }
    else if (skill_name == "full_pick_and_place" || skill_name == "box_in_shelf") {
        return plan_full_pick_and_place(current, scene, params);
    }
    else if (skill_name == "home") {
        return plan_home(current);
    }
    else {
        return { {}, Status::UNKNOWN_ERROR,
                 "Unknown skill_name: " + skill_name };
    }
}

} // namespace tinyvla
