// bind_motion.cpp — pybind11 bindings for Trajectory + MotionPlanner

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "tinyvla/trajectory.hpp"
#include "tinyvla/motion_planner.hpp"

namespace py = pybind11;
using namespace tinyvla;

void bind_motion(py::module_& m) {

    // ---- WaypointSpec --------------------------------------------------------
    py::class_<WaypointSpec>(m, "WaypointSpec")
        .def(py::init<>())
        .def_readwrite("pose",           &WaypointSpec::pose)
        .def_readwrite("gripper_cmd",    &WaypointSpec::gripper_cmd)
        .def_readwrite("blend_radius_m", &WaypointSpec::blend_radius_m);

    // ---- TrajectoryConfig ---------------------------------------------------
    py::class_<TrajectoryConfig>(m, "TrajectoryConfig")
        .def(py::init<>())
        .def_readwrite("max_joint_vel_rad_s",   &TrajectoryConfig::max_joint_vel_rad_s)
        .def_readwrite("max_joint_acc_rad_s2",  &TrajectoryConfig::max_joint_acc_rad_s2)
        .def_readwrite("max_cart_vel_m_s",      &TrajectoryConfig::max_cart_vel_m_s)
        .def_readwrite("sample_dt_s",           &TrajectoryConfig::sample_dt_s);

    // ---- TrajectoryPoint ----------------------------------------------------
    py::class_<TrajectoryPoint>(m, "TrajectoryPoint")
        .def(py::init<>())
        .def_readwrite("time_s",      &TrajectoryPoint::time_s)
        .def_readwrite("joints",      &TrajectoryPoint::joints)
        .def_readwrite("eef_pose",    &TrajectoryPoint::eef_pose)
        .def_readwrite("gripper_cmd", &TrajectoryPoint::gripper_cmd);

    // ---- Trajectory ---------------------------------------------------------
    py::class_<Trajectory>(m, "Trajectory")
        .def(py::init<>())
        .def("empty",    &Trajectory::empty)
        .def("size",     &Trajectory::size)
        .def("duration", &Trajectory::duration)
        .def("sample",   &Trajectory::sample, py::arg("t"),
             "Interpolated sample at time t (cubic spline)")
        .def("front",    &Trajectory::front)
        .def("back",     &Trajectory::back)
        .def("points",   &Trajectory::points,
             py::return_value_policy::reference_internal,
             "All TrajectoryPoints as a list")
        .def("__len__",  &Trajectory::size)
        .def("__getitem__", [](const Trajectory& tr, size_t i) {
            if (i >= tr.size()) throw py::index_error();
            return tr[i];
        });

    // ---- TrajectoryGenerator ------------------------------------------------
    py::class_<TrajectoryGenerator>(m, "TrajectoryGenerator")
        .def(py::init<const Kinematics&, const TrajectoryConfig&>(),
             py::arg("kinematics"), py::arg("config") = TrajectoryConfig{})
        .def("joint_space", &TrajectoryGenerator::joint_space,
             py::arg("waypoints"), py::arg("gripper_cmds") = std::vector<GripperCommand>{})
        .def("cartesian_space", &TrajectoryGenerator::cartesian_space,
             py::arg("start_joints"), py::arg("waypoints"))
        .def("pick", &TrajectoryGenerator::pick,
             py::arg("start"), py::arg("object_pose"),
             py::arg("approach_height_m") = 0.08,
             py::arg("z_offset_mm") = 0.0)
        .def("place", &TrajectoryGenerator::place,
             py::arg("start"), py::arg("shelf_pose"),
             py::arg("approach_height_m") = 0.08)
        .def("pick_and_place", &TrajectoryGenerator::pick_and_place,
             py::arg("start"), py::arg("object_pose"), py::arg("shelf_pose"),
             py::arg("approach_height_m") = 0.08,
             py::arg("z_offset_mm") = 0.0);

    // ---- PlanParams ---------------------------------------------------------
    py::class_<PlanParams>(m, "PlanParams")
        .def(py::init<>())
        .def_readwrite("z_offset_mm",         &PlanParams::z_offset_mm)
        .def_readwrite("speed_scale",         &PlanParams::speed_scale)
        .def_readwrite("approach_angle_deg",  &PlanParams::approach_angle_deg)
        .def_readwrite("gripper_close_force", &PlanParams::gripper_close_force)
        .def_readwrite("approach_height_m",   &PlanParams::approach_height_m);

    // ---- SceneInfo ----------------------------------------------------------
    py::class_<SceneInfo>(m, "SceneInfo")
        .def(py::init<>())
        .def_readwrite("object_pose",    &SceneInfo::object_pose)
        .def_readwrite("shelf_pose",     &SceneInfo::shelf_pose)
        .def_readwrite("object_visible", &SceneInfo::object_visible)
        .def_readwrite("shelf_visible",  &SceneInfo::shelf_visible);

    // ---- PlanResult ---------------------------------------------------------
    py::class_<PlanResult>(m, "PlanResult")
        .def_readonly("trajectory", &PlanResult::trajectory)
        .def_readonly("status",     &PlanResult::status)
        .def_readonly("message",    &PlanResult::message)
        .def("ok", &PlanResult::ok);

    // ---- MotionPlanner ------------------------------------------------------
    py::class_<MotionPlanner>(m, "MotionPlanner")
        .def(py::init<const Kinematics&, const TrajectoryConfig&>(),
             py::arg("kinematics"), py::arg("traj_config") = TrajectoryConfig{})
        .def("plan_pick",             &MotionPlanner::plan_pick,
             py::arg("current"), py::arg("scene"), py::arg("params") = PlanParams{})
        .def("plan_place",            &MotionPlanner::plan_place,
             py::arg("current"), py::arg("scene"), py::arg("params") = PlanParams{})
        .def("plan_full_pick_and_place", &MotionPlanner::plan_full_pick_and_place,
             py::arg("current"), py::arg("scene"), py::arg("params") = PlanParams{})
        .def("plan_home",             &MotionPlanner::plan_home,
             py::arg("current"))
        .def("plan",                  &MotionPlanner::plan,
             py::arg("skill_name"), py::arg("current"),
             py::arg("scene"), py::arg("params") = PlanParams{},
             "Generic dispatcher: 'pick_object'|'place_in_shelf'|'full_pick_and_place'|'home'");
}
