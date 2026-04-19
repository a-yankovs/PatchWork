// bind_robot.cpp — pybind11 bindings for RobotInterface + Controller

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "tinyvla/robot_interface.hpp"
#include "tinyvla/controller.hpp"

namespace py = pybind11;
using namespace tinyvla;

void bind_robot(py::module_& m) {

    // ---- PIDGains -----------------------------------------------------------
    py::class_<PIDGains>(m, "PIDGains")
        .def(py::init<>())
        .def(py::init<double, double, double, double>(),
             py::arg("kp"), py::arg("ki"), py::arg("kd"), py::arg("i_clamp") = 1.0)
        .def_readwrite("kp",      &PIDGains::kp)
        .def_readwrite("ki",      &PIDGains::ki)
        .def_readwrite("kd",      &PIDGains::kd)
        .def_readwrite("i_clamp", &PIDGains::i_clamp);

    // ---- ControllerConfig ---------------------------------------------------
    py::class_<ControllerConfig>(m, "ControllerConfig")
        .def(py::init<>())
        .def_readwrite("loop_rate_hz",     &ControllerConfig::loop_rate_hz)
        .def_readwrite("position_tol_rad", &ControllerConfig::position_tol_rad)
        .def_readwrite("max_effort_nm",    &ControllerConfig::max_effort_nm);

    // ---- ControllerMode -----------------------------------------------------
    py::enum_<ControllerMode>(m, "ControllerMode")
        .value("POSITION", ControllerMode::POSITION)
        .value("TORQUE",   ControllerMode::TORQUE)
        .export_values();

    // ---- JointCommand -------------------------------------------------------
    py::class_<JointCommand>(m, "JointCommand")
        .def(py::init<>())
        .def_readwrite("position",  &JointCommand::position)
        .def_readwrite("velocity",  &JointCommand::velocity)
        .def_readwrite("effort",    &JointCommand::effort)
        .def_readwrite("gripper",   &JointCommand::gripper)
        .def_readwrite("mode",      &JointCommand::mode);

    // ---- Controller ---------------------------------------------------------
    py::class_<Controller>(m, "Controller")
        .def(py::init<const ControllerConfig&, ControllerMode>(),
             py::arg("config") = ControllerConfig{},
             py::arg("mode")   = ControllerMode::POSITION)
        .def("compute_command", &Controller::compute_command,
             py::arg("desired"), py::arg("actual"), py::arg("dt_s"),
             "Compute one control step (does not send to hardware)")
        .def("execute",
             [](Controller& ctrl,
                const Trajectory& traj,
                py::function cmd_cb,
                py::function state_cb,
                py::function gripper_cb,
                double timeout_s) {
                 return ctrl.execute(
                     traj,
                     [cmd_cb](const JointCommand& c) { py::gil_scoped_acquire g; cmd_cb(c); },
                     [state_cb]() -> JointState { py::gil_scoped_acquire g; return state_cb().cast<JointState>(); },
                     [gripper_cb](GripperCommand gc, double f) { py::gil_scoped_acquire g; gripper_cb(gc, f); },
                     timeout_s
                 );
             },
             py::arg("trajectory"),
             py::arg("cmd_callback"),
             py::arg("state_callback"),
             py::arg("gripper_callback"),
             py::arg("timeout_s") = 30.0,
             "Execute trajectory (blocking). Callbacks run at loop_rate_hz.")
        .def("request_stop", &Controller::request_stop)
        .def("estop",        &Controller::estop)
        .def("is_running",   &Controller::is_running)
        .def("set_gains",    &Controller::set_gains, py::arg("joint"), py::arg("gains"))
        .def("set_mode",     &Controller::set_mode,  py::arg("mode"));

    // ---- RobotInterfaceConfig -----------------------------------------------
    py::class_<RobotInterfaceConfig>(m, "RobotInterfaceConfig")
        .def(py::init<>())
        .def_readwrite("port",           &RobotInterfaceConfig::port)
        .def_readwrite("baud_rate",      &RobotInterfaceConfig::baud_rate)
        .def_readwrite("read_timeout_s", &RobotInterfaceConfig::read_timeout_s)
        .def_readwrite("mock_mode",      &RobotInterfaceConfig::mock_mode);

    // ---- RobotInterface -----------------------------------------------------
    py::class_<RobotInterface>(m, "RobotInterface")
        .def(py::init<const RobotInterfaceConfig&>(),
             py::arg("config") = RobotInterfaceConfig{})
        .def("open",       &RobotInterface::open,
             "Open serial port and enable motor torque")
        .def("close",      &RobotInterface::close,
             "Disable torque and close port")
        .def("is_open",    &RobotInterface::is_open)
        .def("read_joint_state",  &RobotInterface::read_joint_state,
             "Read current joint positions from hardware — returns JointState or None")
        .def("read_robot_state",  &RobotInterface::read_robot_state,
             "Read full robot state (joints + EEF pose) — returns RobotState or None")
        .def("send_command", &RobotInterface::send_command, py::arg("command"))
        .def("set_gripper",  &RobotInterface::set_gripper,
             py::arg("command"), py::arg("force_norm") = 0.6, py::arg("timeout_s") = 3.0)
        .def("emergency_stop", &RobotInterface::emergency_stop)
        .def("ping_motor",   &RobotInterface::ping_motor, py::arg("motor_id"))
        .def("start_state_streaming",
             [](RobotInterface& ri, py::function cb, double rate_hz) {
                 ri.start_state_streaming(
                     [cb](const RobotState& s) { py::gil_scoped_acquire g; cb(s); },
                     rate_hz
                 );
             },
             py::arg("callback"), py::arg("rate_hz") = 100.0)
        .def("stop_state_streaming", &RobotInterface::stop_state_streaming);
}
