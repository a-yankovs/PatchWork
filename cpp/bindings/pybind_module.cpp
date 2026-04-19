// pybind_module.cpp — Top-level pybind11 module definition
//
// Exposes the C++ motion stack to Python as `tinyvla_cpp`:
//
//   from tinyvla_cpp import (
//       Kinematics, MotionPlanner, TrajectoryGenerator,
//       RobotInterface, Controller,
//       JointState, Pose, Vec3, GripperCommand, Status, PlanParams, SceneInfo
//   )

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tinyvla/common.hpp"

namespace py = pybind11;
using namespace tinyvla;

// Forward declarations of bind_* functions
void bind_kinematics(py::module_& m);
void bind_motion(py::module_& m);
void bind_robot(py::module_& m);

PYBIND11_MODULE(tinyvla_cpp, m) {
    m.doc() = "tinyVLA C++ motion stack — kinematics, trajectory, planning, control";

    // ---- Enums (needed by all sub-modules) ----------------------------------

    py::enum_<Status>(m, "Status")
        .value("OK",                   Status::OK)
        .value("IK_FAILED",            Status::IK_FAILED)
        .value("JOINT_LIMIT_EXCEEDED", Status::JOINT_LIMIT_EXCEEDED)
        .value("TRAJECTORY_EMPTY",     Status::TRAJECTORY_EMPTY)
        .value("SERIAL_ERROR",         Status::SERIAL_ERROR)
        .value("TIMEOUT",              Status::TIMEOUT)
        .value("GRIPPER_FAULT",        Status::GRIPPER_FAULT)
        .value("COLLISION_DETECTED",   Status::COLLISION_DETECTED)
        .export_values();

    py::enum_<GripperCommand>(m, "GripperCommand")
        .value("OPEN",  GripperCommand::OPEN)
        .value("CLOSE", GripperCommand::CLOSE)
        .value("HOLD",  GripperCommand::HOLD)
        .export_values();

    // ---- Vec3 ---------------------------------------------------------------
    py::class_<Vec3>(m, "Vec3")
        .def(py::init<>())
        .def(py::init<double, double, double>(), py::arg("x"), py::arg("y"), py::arg("z"))
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z)
        .def("norm",  &Vec3::norm)
        .def("unit",  &Vec3::unit)
        .def("dot",   &Vec3::dot)
        .def("cross", &Vec3::cross)
        .def("__add__", &Vec3::operator+)
        .def("__sub__", &Vec3::operator-)
        .def("__mul__", &Vec3::operator*)
        .def("__repr__", [](const Vec3& v) {
            return "Vec3(" + std::to_string(v.x) + ", " +
                   std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
        });

    // ---- Rotation3 ----------------------------------------------------------
    py::class_<Rotation3>(m, "Rotation3")
        .def(py::init<>())
        .def_static("identity",  &Rotation3::identity)
        .def_static("from_rpy",  &Rotation3::from_rpy,
             py::arg("roll"), py::arg("pitch"), py::arg("yaw"))
        .def("transpose", &Rotation3::transpose)
        .def("to_rpy", [](const Rotation3& R) {
            double r, p, y;
            R.to_rpy(r, p, y);
            return py::make_tuple(r, p, y);
        }, "Returns (roll, pitch, yaw) in radians");

    // ---- Pose ---------------------------------------------------------------
    py::class_<Pose>(m, "Pose")
        .def(py::init<>())
        .def_readwrite("position", &Pose::position)
        .def_readwrite("rotation", &Pose::rotation)
        .def_static("identity",   &Pose::identity)
        .def_static("from_pos_rpy", &Pose::from_pos_rpy,
             py::arg("x"), py::arg("y"), py::arg("z"),
             py::arg("roll"), py::arg("pitch"), py::arg("yaw"))
        .def("__repr__", [](const Pose& p) {
            return "Pose(pos=(" + std::to_string(p.position.x) + ", " +
                   std::to_string(p.position.y) + ", " +
                   std::to_string(p.position.z) + "))";
        });

    // ---- JointState ---------------------------------------------------------
    py::class_<JointState>(m, "JointState")
        .def(py::init<>())
        .def_readwrite("position",    &JointState::position)
        .def_readwrite("velocity",    &JointState::velocity)
        .def_readwrite("effort",      &JointState::effort)
        .def_readwrite("timestamp_s", &JointState::timestamp_s)
        .def("within_limits",         &JointState::within_limits)
        .def("__repr__", [](const JointState& q) {
            std::string s = "JointState([";
            for (int i = 0; i < NUM_JOINTS; ++i) {
                s += std::to_string(q.position[i]);
                if (i < NUM_JOINTS-1) s += ", ";
            }
            return s + "])";
        });

    // ---- GripperState -------------------------------------------------------
    py::class_<GripperState>(m, "GripperState")
        .def(py::init<>())
        .def_readwrite("position",    &GripperState::position)
        .def_readwrite("force",       &GripperState::force)
        .def_readwrite("is_grasping", &GripperState::is_grasping);

    // ---- RobotState ---------------------------------------------------------
    py::class_<RobotState>(m, "RobotState")
        .def(py::init<>())
        .def_readwrite("joints",       &RobotState::joints)
        .def_readwrite("gripper",      &RobotState::gripper)
        .def_readwrite("eef_pose",     &RobotState::eef_pose)
        .def_readwrite("estop_active", &RobotState::estop_active);

    // ---- Constants ----------------------------------------------------------
    m.attr("NUM_JOINTS")  = NUM_JOINTS;
    m.attr("DEG_TO_RAD")  = DEG_TO_RAD;
    m.attr("RAD_TO_DEG")  = RAD_TO_DEG;

    // ---- Sub-module bindings ------------------------------------------------
    bind_kinematics(m);
    bind_motion(m);
    bind_robot(m);
}
