#pragma once
// robot_interface.hpp — Serial hardware interface to the SO101 arm
//
// Communicates with the SO101 follower arm over a USB-serial port using
// the Dynamixel Protocol 2.0 packet format (same as used by LeRobot).
//
// Key responsibilities:
//   - Open / close the serial port
//   - Bulk read current joint positions, velocities, loads
//   - Bulk write target positions and gripper commands
//   - Emergency-stop (torque disable)
//
// This layer is intentionally thin. The MotionPlanner and Controller sit above
// it and call send_command() at the control loop rate.

#include "common.hpp"
#include "controller.hpp"   // for JointCommand
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>
#include <optional>

namespace tinyvla {

// ---------------------------------------------------------------------------
// Dynamixel Protocol 2.0 motor IDs for SO101
// ---------------------------------------------------------------------------

// Default IDs assigned by LeRobot / SO101 firmware
static constexpr std::array<uint8_t, NUM_JOINTS> MOTOR_IDS = {1, 2, 3, 4, 5, 6};
static constexpr uint8_t GRIPPER_MOTOR_ID = 7;   // seventh servo for gripper

// ---------------------------------------------------------------------------
// RobotInterfaceConfig
// ---------------------------------------------------------------------------

struct RobotInterfaceConfig {
    std::string port          = "/dev/ttyACM1";  // default SO101 follower port
    int         baud_rate     = 1000000;          // 1 Mbaud (Dynamixel standard)
    double      read_timeout_s = 0.01;            // per-packet timeout
    bool        mock_mode     = false;            // simulate hardware (no serial)
};

// ---------------------------------------------------------------------------
// RobotInterface
// ---------------------------------------------------------------------------

class RobotInterface {
public:
    using StateCallback   = std::function<void(const RobotState&)>;

    explicit RobotInterface(const RobotInterfaceConfig& cfg = {});
    ~RobotInterface();

    // Non-copyable
    RobotInterface(const RobotInterface&)            = delete;
    RobotInterface& operator=(const RobotInterface&) = delete;

    // ---- Connection ---------------------------------------------------------

    Status open();   // open serial port + enable torque on all joints
    void   close();  // disable torque + close port
    bool   is_open() const { return open_.load(); }

    // ---- State reading -------------------------------------------------------

    // Read current joint state from hardware (blocking, ~1 ms).
    std::optional<JointState> read_joint_state();

    // Read full robot state (joints + gripper + EEF pose from FK).
    std::optional<RobotState> read_robot_state();

    // ---- Command writing ----------------------------------------------------

    // Write position / velocity / effort command to servos.
    Status send_command(const JointCommand& cmd);

    // Move gripper: OPEN or CLOSE (blocks until done or timeout).
    Status set_gripper(GripperCommand cmd, double force_norm = 0.6,
                       double timeout_s = 3.0);

    // Emergency stop: disable all motor torque immediately.
    Status emergency_stop();

    // ---- Streaming state updates --------------------------------------------

    // Start a background thread that reads joint state at rate_hz
    // and calls cb whenever a new state is available.
    void start_state_streaming(StateCallback cb, double rate_hz = 100.0);
    void stop_state_streaming();

    // ---- Low-level helpers (for diagnostics) --------------------------------

    // Ping a specific motor ID; returns true if it responds.
    bool ping_motor(uint8_t motor_id);

    // Read/write a single Dynamixel control-table register (2-byte value).
    Status write_register(uint8_t motor_id, uint16_t address, uint16_t value);
    std::optional<uint16_t> read_register(uint8_t motor_id, uint16_t address);

    // Get config
    const RobotInterfaceConfig& config() const { return cfg_; }

private:
    RobotInterfaceConfig cfg_;
    int                  fd_   = -1;      // file descriptor (POSIX serial)
    std::atomic<bool>    open_ { false };

    // Background streaming thread
    std::thread          stream_thread_;
    std::atomic<bool>    streaming_ { false };
    StateCallback        state_cb_;
    std::mutex           state_mutex_;
    RobotState           last_state_;

    // Mock state (used in mock_mode)
    JointState           mock_joints_ {};

    // ---- Serial helpers -------------------------------------------------------
    Status serial_open();
    void   serial_close();
    Status serial_write(const std::vector<uint8_t>& data);
    int    serial_read(std::vector<uint8_t>& buf, size_t n, int timeout_ms);

    // ---- Dynamixel Protocol 2.0 --------------------------------------------
    std::vector<uint8_t> build_sync_write_position(
        const std::array<double, NUM_JOINTS>& positions
    );
    std::vector<uint8_t> build_sync_read_request(uint16_t addr, uint16_t len);
    bool parse_sync_read_response(
        const std::vector<uint8_t>& data,
        std::array<double, NUM_JOINTS>& positions
    );

    // Compute Dynamixel Protocol 2.0 CRC
    static uint16_t dxl_crc(const std::vector<uint8_t>& data, size_t len);

    // Convert rad ↔ Dynamixel position value (4096 ticks per revolution)
    static uint32_t rad_to_dxl(double rad)  { return (uint32_t)((rad + PI) / (2*PI) * 4095); }
    static double   dxl_to_rad(uint32_t dxl){ return (double)dxl / 4095.0 * 2*PI - PI; }

    void stream_loop(double rate_hz);
};

} // namespace tinyvla
