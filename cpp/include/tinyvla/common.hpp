#pragma once
// common.hpp — Shared data types for the tinyVLA motion stack
// Used by: kinematics, trajectory, motion_planner, controller, robot_interface

#include <array>
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>
#include <chrono>

namespace tinyvla {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr int   NUM_JOINTS   = 6;
static constexpr double PI          = 3.14159265358979323846;
static constexpr double DEG_TO_RAD  = PI / 180.0;
static constexpr double RAD_TO_DEG  = 180.0 / PI;

// SO101 joint limits (radians)
static constexpr std::array<double, NUM_JOINTS> JOINT_MIN = {
    -PI,          // J1 base rotation
    -PI / 2.0,    // J2 shoulder
    -PI * 0.75,   // J3 elbow
    -PI,          // J4 wrist pitch
    -PI / 2.0,    // J5 wrist roll
    -PI           // J6 tool
};
static constexpr std::array<double, NUM_JOINTS> JOINT_MAX = {
     PI,          // J1
     PI * 0.75,   // J2
     PI * 0.75,   // J3
     PI,          // J4
     PI / 2.0,    // J5
     PI           // J6
};

// ---------------------------------------------------------------------------
// Error codes
// ---------------------------------------------------------------------------

enum class Status {
    OK = 0,
    IK_FAILED,              // inverse kinematics found no solution
    JOINT_LIMIT_EXCEEDED,   // solution outside joint range
    TRAJECTORY_EMPTY,       // no waypoints in trajectory
    SERIAL_ERROR,           // hardware communication failure
    TIMEOUT,                // operation did not complete in time
    GRIPPER_FAULT,          // gripper did not reach target state
    COLLISION_DETECTED,     // planned path too close to obstacle
    UNKNOWN_ERROR
};

inline std::string status_string(Status s) {
    switch (s) {
        case Status::OK:                   return "OK";
        case Status::IK_FAILED:            return "IK_FAILED";
        case Status::JOINT_LIMIT_EXCEEDED: return "JOINT_LIMIT_EXCEEDED";
        case Status::TRAJECTORY_EMPTY:     return "TRAJECTORY_EMPTY";
        case Status::SERIAL_ERROR:         return "SERIAL_ERROR";
        case Status::TIMEOUT:              return "TIMEOUT";
        case Status::GRIPPER_FAULT:        return "GRIPPER_FAULT";
        case Status::COLLISION_DETECTED:   return "COLLISION_DETECTED";
        default:                           return "UNKNOWN_ERROR";
    }
}

// ---------------------------------------------------------------------------
// Vec3 — 3D vector / point
// ---------------------------------------------------------------------------

struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;

    Vec3() = default;
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(double s)      const { return {x*s,   y*s,   z*s};   }
    Vec3 operator/(double s)      const { return {x/s,   y/s,   z/s};   }

    Vec3& operator+=(const Vec3& o) { x+=o.x; y+=o.y; z+=o.z; return *this; }

    double norm()  const { return std::sqrt(x*x + y*y + z*z); }
    Vec3   unit()  const { double n = norm(); return (n > 1e-12) ? *this/n : Vec3{}; }
    double dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    Vec3   cross(const Vec3& o) const {
        return { y*o.z - z*o.y,
                 z*o.x - x*o.z,
                 x*o.y - y*o.x };
    }
};

// ---------------------------------------------------------------------------
// Rotation3 — 3×3 rotation matrix (row-major)
// ---------------------------------------------------------------------------

struct Rotation3 {
    std::array<std::array<double,3>,3> m = {{{1,0,0},{0,1,0},{0,0,1}}};

    static Rotation3 identity() { return {}; }

    static Rotation3 from_rpy(double roll, double pitch, double yaw) {
        double cr = std::cos(roll),  sr = std::sin(roll);
        double cp = std::cos(pitch), sp = std::sin(pitch);
        double cy = std::cos(yaw),   sy = std::sin(yaw);
        Rotation3 R;
        R.m[0] = { cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr };
        R.m[1] = { sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr };
        R.m[2] = { -sp,    cp*sr,             cp*cr            };
        return R;
    }

    // Rotate a vector
    Vec3 operator*(const Vec3& v) const {
        return {
            m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z,
            m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z,
            m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z
        };
    }

    // Matrix multiply
    Rotation3 operator*(const Rotation3& o) const {
        Rotation3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                r.m[i][j] = 0;
                for (int k = 0; k < 3; ++k)
                    r.m[i][j] += m[i][k] * o.m[k][j];
            }
        return r;
    }

    Rotation3 transpose() const {
        Rotation3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.m[i][j] = m[j][i];
        return r;
    }

    // Extract RPY (roll-pitch-yaw) from rotation matrix
    void to_rpy(double& roll, double& pitch, double& yaw) const {
        pitch = std::atan2(-m[2][0], std::sqrt(m[0][0]*m[0][0] + m[1][0]*m[1][0]));
        if (std::abs(std::cos(pitch)) > 1e-6) {
            roll = std::atan2(m[2][1], m[2][2]);
            yaw  = std::atan2(m[1][0], m[0][0]);
        } else {
            roll = std::atan2(-m[1][2], m[1][1]);
            yaw  = 0.0;
        }
    }
};

// ---------------------------------------------------------------------------
// Pose — position + orientation (end-effector or waypoint)
// ---------------------------------------------------------------------------

struct Pose {
    Vec3     position;     // metres
    Rotation3 rotation;   // SO3

    static Pose identity() {
        return { Vec3{}, Rotation3::identity() };
    }

    // Convenience: build from position and RPY angles (radians)
    static Pose from_pos_rpy(double x, double y, double z,
                             double roll, double pitch, double yaw) {
        return { Vec3{x, y, z}, Rotation3::from_rpy(roll, pitch, yaw) };
    }

    // Transform another pose into this frame
    Pose operator*(const Pose& child) const {
        return { position + rotation * child.position,
                 rotation * child.rotation };
    }
};

// ---------------------------------------------------------------------------
// JointState — positions, velocities, efforts for all joints
// ---------------------------------------------------------------------------

struct JointState {
    std::array<double, NUM_JOINTS> position = {};   // radians
    std::array<double, NUM_JOINTS> velocity = {};   // rad/s
    std::array<double, NUM_JOINTS> effort   = {};   // Nm (torque)
    double timestamp_s = 0.0;                        // seconds since epoch

    bool within_limits() const {
        for (int i = 0; i < NUM_JOINTS; ++i) {
            if (position[i] < JOINT_MIN[i] || position[i] > JOINT_MAX[i])
                return false;
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// GripperState
// ---------------------------------------------------------------------------

enum class GripperCommand { OPEN, CLOSE, HOLD };

struct GripperState {
    double position     = 0.0;   // 0.0 = fully closed, 1.0 = fully open
    double force        = 0.0;   // normalised [0,1]
    bool   is_grasping  = false;
    GripperCommand last_cmd = GripperCommand::OPEN;
};

// ---------------------------------------------------------------------------
// RobotState — full snapshot
// ---------------------------------------------------------------------------

struct RobotState {
    JointState   joints;
    GripperState gripper;
    Pose         eef_pose;         // end-effector pose in world frame
    bool         estop_active = false;
};

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

inline double clamp(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline double lerp(double a, double b, double t) {
    return a + t * (b - a);
}

// Wrap angle to [-π, π]
inline double wrap_angle(double a) {
    while (a >  PI) a -= 2.0 * PI;
    while (a < -PI) a += 2.0 * PI;
    return a;
}

} // namespace tinyvla
