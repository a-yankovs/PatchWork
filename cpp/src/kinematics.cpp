// kinematics.cpp — FK / IK for SO101 6-DOF arm

#include "tinyvla/kinematics.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace tinyvla {

// ---------------------------------------------------------------------------
// Transform4
// ---------------------------------------------------------------------------

Transform4 Transform4::from_dh(double a, double alpha, double d, double theta) {
    // Modified DH convention
    double ca = std::cos(alpha), sa = std::sin(alpha);
    double ct = std::cos(theta), st = std::sin(theta);
    Transform4 T;
    T.m[0] = {  ct,     -st,      0,    a      };
    T.m[1] = {  st*ca,   ct*ca,  -sa,  -sa*d   };
    T.m[2] = {  st*sa,   ct*sa,   ca,   ca*d   };
    T.m[3] = {  0,       0,       0,    1      };
    return T;
}

Transform4 Transform4::operator*(const Transform4& o) const {
    Transform4 r;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            r.m[i][j] = 0;
            for (int k = 0; k < 4; ++k)
                r.m[i][j] += m[i][k] * o.m[k][j];
        }
    return r;
}

// ---------------------------------------------------------------------------
// Default SO101 DH table
// Approximate values for the SO101 desktop arm (metres, radians)
// ---------------------------------------------------------------------------

static const std::array<DHParam, NUM_JOINTS> SO101_DH = {{
    // a       alpha        d        theta_offset
    { 0.000,   PI/2.0,   0.065,      0.0 },   // J1: base
    { 0.100,   0.000,    0.000,     -PI/2.0 }, // J2: shoulder
    { 0.096,   0.000,    0.000,      0.0 },    // J3: elbow
    { 0.0625,  PI/2.0,   0.000,      0.0 },    // J4: wrist pitch
    { 0.000,  -PI/2.0,   0.0625,     0.0 },    // J5: wrist roll
    { 0.000,   0.000,    0.040,      0.0 },    // J6: tool
}};

// ---------------------------------------------------------------------------
// Kinematics
// ---------------------------------------------------------------------------

Kinematics::Kinematics() : dh_(SO101_DH) {}

Kinematics::Kinematics(const std::array<DHParam, NUM_JOINTS>& dh) : dh_(dh) {}

Transform4 Kinematics::joint_transform(int i, double theta) const {
    const auto& p = dh_[i];
    return Transform4::from_dh(p.a, p.alpha, p.d, theta + p.theta_offset);
}

// ---- Forward kinematics ----------------------------------------------------

std::array<Transform4, NUM_JOINTS+1> Kinematics::all_transforms(
    const JointState& joints) const
{
    std::array<Transform4, NUM_JOINTS+1> T;
    T[0] = Transform4::identity();
    for (int i = 0; i < NUM_JOINTS; ++i)
        T[i+1] = T[i] * joint_transform(i, joints.position[i]);
    return T;
}

Pose Kinematics::forward(const JointState& joints) const {
    auto T = Transform4::identity();
    for (int i = 0; i < NUM_JOINTS; ++i)
        T = T * joint_transform(i, joints.position[i]);
    return T.to_pose();
}

Pose Kinematics::forward_to(const JointState& joints, int joint_index) const {
    auto T = Transform4::identity();
    int n = std::min(joint_index + 1, NUM_JOINTS);
    for (int i = 0; i < n; ++i)
        T = T * joint_transform(i, joints.position[i]);
    return T.to_pose();
}

// ---- Jacobian (geometric) --------------------------------------------------

std::array<double, 6*NUM_JOINTS> Kinematics::jacobian(const JointState& joints) const {
    std::array<double, 6*NUM_JOINTS> J = {};
    auto transforms = all_transforms(joints);
    Vec3 p_ee = transforms[NUM_JOINTS].translation();

    for (int i = 0; i < NUM_JOINTS; ++i) {
        // z-axis of joint i in world frame
        const auto& Ti = transforms[i];
        Vec3 z_i = { Ti.m[0][2], Ti.m[1][2], Ti.m[2][2] };
        Vec3 p_i = Ti.translation();
        Vec3 dp  = p_ee - p_i;

        // Linear velocity part: z_i × (p_ee - p_i)
        Vec3 Jv = z_i.cross(dp);

        // Angular velocity part: z_i
        // Column i = [Jv; z_i]
        J[i*6 + 0] = Jv.x;
        J[i*6 + 1] = Jv.y;
        J[i*6 + 2] = Jv.z;
        J[i*6 + 3] = z_i.x;
        J[i*6 + 4] = z_i.y;
        J[i*6 + 5] = z_i.z;
    }
    return J;
}

// ---- Pose error (6D) -------------------------------------------------------

std::array<double,6> Kinematics::pose_error(const Pose& curr, const Pose& tgt) const {
    // Position error
    Vec3 dp = tgt.position - curr.position;

    // Rotation error via R_err = R_curr^T * R_tgt, then skew-symmetric part
    Rotation3 R_err = curr.rotation.transpose() * tgt.rotation;
    // Extract rotation vector from R_err (axis-angle, small angle approx ok here)
    double rx = R_err.m[2][1] - R_err.m[1][2];
    double ry = R_err.m[0][2] - R_err.m[2][0];
    double rz = R_err.m[1][0] - R_err.m[0][1];
    // Rotate back to world frame
    Vec3 rot_world = curr.rotation * Vec3{rx, ry, rz} * 0.5;

    return { dp.x, dp.y, dp.z, rot_world.x, rot_world.y, rot_world.z };
}

// ---- Newton step (damped least-squares) ------------------------------------

JointState Kinematics::newton_step(
    const JointState& q,
    const Pose&       target,
    double            damping
) const {
    auto J_flat = jacobian(q);
    auto curr   = forward(q);
    auto err    = pose_error(curr, target);

    // J^T (J J^T + λ²I)^{-1} err — for 6×6 this is a full matrix solve.
    // For speed we use the transpose form: dq = J^T (J J^T + λ²I)^{-1} e
    // Implemented as: dq = J^T * (J J^T + λ²I)^{-1} * e
    // We compute A = J J^T + λ²I (6×6) and solve A * x = e, then dq = J^T x

    // Build J (6 rows × 6 cols, stored row-major)
    // J_flat is col-major: col i = J_flat[i*6 .. i*6+5]
    std::array<std::array<double,6>,6> Jmat = {};
    for (int row = 0; row < 6; ++row)
        for (int col = 0; col < NUM_JOINTS; ++col)
            Jmat[row][col] = J_flat[col*6 + row];

    // A = J J^T + λ²I
    std::array<std::array<double,6>,6> A = {};
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            double s = 0;
            for (int k = 0; k < NUM_JOINTS; ++k)
                s += Jmat[i][k] * Jmat[j][k];
            A[i][j] = s;
        }
        A[i][i] += damping * damping;
    }

    // Solve A * x = err via Gauss elimination with partial pivoting
    std::array<double,6> b = { err[0],err[1],err[2],err[3],err[4],err[5] };
    for (int col = 0; col < 6; ++col) {
        // pivot
        int pivot = col;
        for (int row = col+1; row < 6; ++row)
            if (std::abs(A[row][col]) > std::abs(A[pivot][col])) pivot = row;
        std::swap(A[col], A[pivot]);
        std::swap(b[col], b[pivot]);
        if (std::abs(A[col][col]) < 1e-12) continue;
        for (int row = col+1; row < 6; ++row) {
            double f = A[row][col] / A[col][col];
            for (int c = col; c < 6; ++c) A[row][c] -= f * A[col][c];
            b[row] -= f * b[col];
        }
    }
    std::array<double,6> x = {};
    for (int i = 5; i >= 0; --i) {
        x[i] = b[i];
        for (int j = i+1; j < 6; ++j) x[i] -= A[i][j] * x[j];
        if (std::abs(A[i][i]) > 1e-12) x[i] /= A[i][i];
    }

    // dq = J^T * x
    JointState q_new = q;
    for (int j = 0; j < NUM_JOINTS; ++j) {
        double dq = 0;
        for (int i = 0; i < 6; ++i)
            dq += Jmat[i][j] * x[i];
        q_new.position[j] = wrap_angle(q.position[j] + dq);
    }
    return clamp_to_limits(q_new);
}

// ---- Numerical IK ----------------------------------------------------------

std::optional<JointState> Kinematics::inverse(
    const Pose&       target,
    const JointState& seed,
    int               max_iter,
    double            tol_pos,
    double            tol_rot
) const {
    JointState q = clamp_to_limits(seed);
    double damping = 0.05;

    for (int iter = 0; iter < max_iter; ++iter) {
        auto curr = forward(q);
        auto err  = pose_error(curr, target);

        double pos_err = std::sqrt(err[0]*err[0] + err[1]*err[1] + err[2]*err[2]);
        double rot_err = std::sqrt(err[3]*err[3] + err[4]*err[4] + err[5]*err[5]);

        if (pos_err < tol_pos && rot_err < tol_rot)
            return q;

        // Reduce damping as we converge
        damping = 0.05 * std::max(pos_err / 0.01, 0.1);
        damping = clamp(damping, 1e-4, 0.5);

        q = newton_step(q, target, damping);
    }
    return std::nullopt;
}

// ---- Analytical IK (geometric, SO101 approximation) -----------------------
// Uses a decoupled approach:
//   - J1 from target x,y (base rotation)
//   - J2, J3, J4 from the projected planar arm
//   - J5, J6 from wrist orientation

std::optional<JointState> Kinematics::inverse_analytical(
    const Pose&       target,
    const JointState& seed
) const {
    const Vec3& p = target.position;

    // J1 — base rotation
    double theta1 = std::atan2(p.y, p.x);

    // Wrist centre position (subtract tool + J6 offset along z_ee)
    Vec3 z_ee = { target.rotation.m[0][2],
                  target.rotation.m[1][2],
                  target.rotation.m[2][2] };
    double d6 = dh_[5].d;
    Vec3 wc = p - z_ee * d6;

    // Project wrist centre into J1 plane
    double r  = std::sqrt(wc.x*wc.x + wc.y*wc.y) - dh_[0].a;  // radial reach
    double s  = wc.z - dh_[0].d;                                  // vertical height

    double a2 = dh_[1].a;
    double a3 = dh_[2].a;
    double d4 = dh_[3].d;

    // Effective a3 includes wrist offset
    double a3_eff = std::sqrt(a3*a3 + d4*d4);

    double D = (r*r + s*s - a2*a2 - a3_eff*a3_eff) / (2.0 * a2 * a3_eff);
    if (std::abs(D) > 1.0) {
        // Out of reach — fall back to numerical
        return inverse(target, seed);
    }

    double theta3_offset = std::atan2(d4, a3);

    // Elbow-up solution
    double theta3 = std::atan2(std::sqrt(1.0 - D*D), D) - theta3_offset;

    double k1 = a2 + a3_eff * std::cos(theta3 + theta3_offset);
    double k2 = a3_eff * std::sin(theta3 + theta3_offset);
    double theta2 = std::atan2(s, r) - std::atan2(k2, k1);

    // Wrist angles (J4, J5, J6) — from desired orientation
    // R_0_3 = R1 * R2 * R3
    JointState q_partial{};
    q_partial.position[0] = theta1;
    q_partial.position[1] = theta2;
    q_partial.position[2] = theta3;
    Rotation3 R03 = forward_to(q_partial, 2).rotation;
    Rotation3 R36 = R03.transpose() * target.rotation;

    // ZYZ decomposition of R36 for wrist
    double theta4 = std::atan2(R36.m[1][2], R36.m[0][2]);
    double theta5 = std::atan2(
        std::sqrt(R36.m[0][2]*R36.m[0][2] + R36.m[1][2]*R36.m[1][2]),
        R36.m[2][2]);
    double theta6 = std::atan2(R36.m[2][1], -R36.m[2][0]);

    JointState q{};
    q.position[0] = wrap_angle(theta1 - dh_[0].theta_offset);
    q.position[1] = wrap_angle(theta2 - dh_[1].theta_offset);
    q.position[2] = wrap_angle(theta3 - dh_[2].theta_offset);
    q.position[3] = wrap_angle(theta4 - dh_[3].theta_offset);
    q.position[4] = wrap_angle(theta5 - dh_[4].theta_offset);
    q.position[5] = wrap_angle(theta6 - dh_[5].theta_offset);

    if (!within_limits(q))
        return inverse(target, seed);  // retry numerically

    return q;
}

// ---- Helpers ---------------------------------------------------------------

JointState Kinematics::clamp_to_limits(const JointState& joints) {
    JointState q = joints;
    for (int i = 0; i < NUM_JOINTS; ++i)
        q.position[i] = clamp(q.position[i], JOINT_MIN[i], JOINT_MAX[i]);
    return q;
}

bool Kinematics::within_limits(const JointState& joints) {
    return joints.within_limits();
}

} // namespace tinyvla
