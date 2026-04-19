#pragma once
// kinematics.hpp — Forward and Inverse Kinematics for SO101 6-DOF arm
//
// DH convention (modified Denavit-Hartenberg):
//   a_i      : link length (metres)
//   alpha_i  : link twist  (radians)
//   d_i      : link offset (metres)
//   theta_i  : joint angle (radians) — variable for revolute joints
//
// SO101 approximate parameters (desktop robot, ~400 mm reach):
//   J1  base  yaw   : a=0,      alpha=π/2,  d=0.065
//   J2  shoulder    : a=0.100,  alpha=0,    d=0
//   J3  elbow       : a=0.096,  alpha=0,    d=0
//   J4  wrist pitch : a=0.0625, alpha=π/2,  d=0
//   J5  wrist roll  : a=0,      alpha=-π/2, d=0.0625
//   J6  tool/grip   : a=0,      alpha=0,    d=0.040

#include "common.hpp"
#include <optional>
#include <functional>

namespace tinyvla {

// ---------------------------------------------------------------------------
// 4×4 homogeneous transform (column-major storage for clarity)
// ---------------------------------------------------------------------------

struct Transform4 {
    // Stored as 4×4 row-major
    std::array<std::array<double,4>,4> m = {{
        {1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1}
    }};

    static Transform4 identity() { return {}; }

    // Build from modified DH parameters
    static Transform4 from_dh(double a, double alpha, double d, double theta);

    Transform4 operator*(const Transform4& o) const;

    // Extract position
    Vec3 translation() const { return { m[0][3], m[1][3], m[2][3] }; }

    // Extract 3×3 rotation
    Rotation3 rotation() const {
        Rotation3 R;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                R.m[i][j] = m[i][j];
        return R;
    }

    Pose to_pose() const { return { translation(), rotation() }; }
};

// ---------------------------------------------------------------------------
// DH parameters for one joint
// ---------------------------------------------------------------------------

struct DHParam {
    double a;      // link length (m)
    double alpha;  // link twist  (rad)
    double d;      // link offset (m)
    double theta_offset;  // constant offset added to the variable angle
};

// ---------------------------------------------------------------------------
// Kinematics — forward + inverse kinematics
// ---------------------------------------------------------------------------

class Kinematics {
public:
    // Build with default SO101 DH table
    Kinematics();

    // Build with custom DH table (useful for calibrated models)
    explicit Kinematics(const std::array<DHParam, NUM_JOINTS>& dh);

    // ---- Forward kinematics ------------------------------------------------

    // Full forward kinematics: returns end-effector pose in base frame
    Pose forward(const JointState& joints) const;

    // Partial FK: returns pose of joint i's frame (0-indexed)
    Pose forward_to(const JointState& joints, int joint_index) const;

    // Returns all intermediate frame transforms (useful for visualisation)
    std::array<Transform4, NUM_JOINTS+1> all_transforms(const JointState& joints) const;

    // ---- Inverse kinematics ------------------------------------------------

    // Numerical IK (Jacobian pseudo-inverse iteration).
    // seed       : initial joint guess (use current state for speed)
    // max_iter   : Newton steps (default 200)
    // tol_pos    : position error tolerance in metres (default 1e-4)
    // tol_rot    : orientation error tolerance in radians (default 1e-3)
    // Returns joint solution or std::nullopt on failure.
    std::optional<JointState> inverse(
        const Pose&       target,
        const JointState& seed,
        int               max_iter = 200,
        double            tol_pos  = 1e-4,
        double            tol_rot  = 1e-3
    ) const;

    // Analytical IK using SO101 geometry (faster, closed-form, partial).
    // Falls back to numerical if geometric solution doesn't converge.
    std::optional<JointState> inverse_analytical(
        const Pose&       target,
        const JointState& seed
    ) const;

    // ---- Jacobian ----------------------------------------------------------

    // Geometric Jacobian (6×6): maps joint velocities to EEF twist
    // Returns column-major 6×6 matrix as flat array [col0, col1, ...]
    std::array<double, 6*NUM_JOINTS> jacobian(const JointState& joints) const;

    // ---- Helpers -----------------------------------------------------------

    // Clamp joints to within limits
    static JointState clamp_to_limits(const JointState& joints);

    // Check joint limits
    static bool within_limits(const JointState& joints);

private:
    std::array<DHParam, NUM_JOINTS> dh_;

    // Build single-joint transform
    Transform4 joint_transform(int i, double theta) const;

    // Compute 6D pose error [dx, dy, dz, drx, dry, drz]
    std::array<double,6> pose_error(const Pose& current, const Pose& target) const;

    // Apply one Newton step; returns updated joints
    JointState newton_step(
        const JointState& q,
        const Pose&       target,
        double            damping
    ) const;
};

} // namespace tinyvla
