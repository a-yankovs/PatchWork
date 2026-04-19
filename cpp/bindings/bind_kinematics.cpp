// bind_kinematics.cpp — pybind11 bindings for Kinematics

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "tinyvla/kinematics.hpp"

namespace py = pybind11;
using namespace tinyvla;

void bind_kinematics(py::module_& m) {

    // ---- Transform4 -------------------------------------------------------
    py::class_<Transform4>(m, "Transform4")
        .def(py::init<>())
        .def_static("identity", &Transform4::identity)
        .def_static("from_dh", &Transform4::from_dh,
            py::arg("a"), py::arg("alpha"), py::arg("d"), py::arg("theta"))
        .def("translation", &Transform4::translation)
        .def("to_pose", &Transform4::to_pose)
        .def("__mul__", &Transform4::operator*)
        .def("__repr__", [](const Transform4& t) {
            auto p = t.translation();
            return "<Transform4 pos=(" + std::to_string(p.x) + ", " +
                   std::to_string(p.y) + ", " + std::to_string(p.z) + ")>";
        });

    // ---- DHParam ----------------------------------------------------------
    py::class_<DHParam>(m, "DHParam")
        .def(py::init<>())
        .def(py::init<double, double, double, double>(),
             py::arg("a"), py::arg("alpha"), py::arg("d"), py::arg("theta_offset"))
        .def_readwrite("a",            &DHParam::a)
        .def_readwrite("alpha",        &DHParam::alpha)
        .def_readwrite("d",            &DHParam::d)
        .def_readwrite("theta_offset", &DHParam::theta_offset);

    // ---- Kinematics -------------------------------------------------------
    py::class_<Kinematics>(m, "Kinematics")
        .def(py::init<>(), "Construct with default SO101 DH parameters")
        .def(py::init<const std::array<DHParam, NUM_JOINTS>&>(),
             py::arg("dh"), "Construct with custom DH table")

        // Forward kinematics
        .def("forward", &Kinematics::forward,
             py::arg("joints"),
             "Compute end-effector Pose from JointState (FK)")

        .def("forward_to", &Kinematics::forward_to,
             py::arg("joints"), py::arg("joint_index"),
             "FK up to joint_index (0-indexed)")

        // Inverse kinematics
        .def("inverse", &Kinematics::inverse,
             py::arg("target"), py::arg("seed"),
             py::arg("max_iter") = 200,
             py::arg("tol_pos")  = 1e-4,
             py::arg("tol_rot")  = 1e-3,
             "Numerical IK — returns JointState or None")

        .def("inverse_analytical", &Kinematics::inverse_analytical,
             py::arg("target"), py::arg("seed"),
             "Analytical IK (geometric, falls back to numerical)")

        // Jacobian (returns flat list, column-major, 6×6)
        .def("jacobian", [](const Kinematics& k, const JointState& q) {
            auto J = k.jacobian(q);
            return py::array_t<double>({ 6, NUM_JOINTS }, J.data());
        }, py::arg("joints"), "Geometric Jacobian (6×6 numpy array)")

        // Static helpers
        .def_static("clamp_to_limits", &Kinematics::clamp_to_limits)
        .def_static("within_limits",   &Kinematics::within_limits);
}
