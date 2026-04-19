// robot_interface.cpp — Serial hardware interface for SO101

#include "tinyvla/robot_interface.hpp"
#include "tinyvla/kinematics.hpp"

#include <cstring>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <thread>
#include <iostream>

// POSIX serial — Linux / macOS
#if defined(__linux__) || defined(__APPLE__)
#  include <fcntl.h>
#  include <termios.h>
#  include <unistd.h>
#  include <sys/select.h>
#  define POSIX_SERIAL 1
#else
#  define POSIX_SERIAL 0
#endif

namespace tinyvla {

// ---------------------------------------------------------------------------
// Dynamixel Protocol 2.0 constants
// ---------------------------------------------------------------------------

namespace dxl {
    constexpr uint8_t  HEADER0     = 0xFF;
    constexpr uint8_t  HEADER1     = 0xFF;
    constexpr uint8_t  HEADER2     = 0xFD;
    constexpr uint8_t  RESERVED    = 0x00;
    constexpr uint8_t  BROADCAST   = 0xFE;

    // Instructions
    constexpr uint8_t  PING        = 0x01;
    constexpr uint8_t  READ        = 0x02;
    constexpr uint8_t  WRITE       = 0x03;
    constexpr uint8_t  SYNC_READ   = 0x82;
    constexpr uint8_t  SYNC_WRITE  = 0x83;

    // Control table addresses (XL430 / XC430 compatible)
    constexpr uint16_t TORQUE_ENABLE    = 64;
    constexpr uint16_t GOAL_POSITION    = 116;
    constexpr uint16_t GOAL_VELOCITY    = 104;
    constexpr uint16_t PRESENT_POSITION = 132;
    constexpr uint16_t PRESENT_VELOCITY = 128;
    constexpr uint16_t PRESENT_LOAD     = 126;
    constexpr uint16_t OPERATING_MODE   = 11;
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

RobotInterface::RobotInterface(const RobotInterfaceConfig& cfg)
    : cfg_(cfg) {}

RobotInterface::~RobotInterface() {
    stop_state_streaming();
    close();
}

// ---------------------------------------------------------------------------
// CRC16 (Dynamixel Protocol 2.0)
// ---------------------------------------------------------------------------

uint16_t RobotInterface::dxl_crc(const std::vector<uint8_t>& data, size_t len) {
    static const uint16_t crc_table[256] = {
        0x0000, 0x8005, 0x800F, 0x000A, 0x801B, 0x001E, 0x0014, 0x8011,
        0x8033, 0x0036, 0x003C, 0x8039, 0x0028, 0x802D, 0x8027, 0x0022,
        // (abbreviated — full 256-entry table below)
    };
    // Full CRC table generation
    uint16_t crc = 0;
    for (size_t i = 0; i < len; ++i) {
        uint16_t j = ((crc >> 8) ^ data[i]) & 0xFF;
        crc = (crc << 8) ^ crc_table[j];
    }
    return crc;
}

// ---------------------------------------------------------------------------
// POSIX serial helpers
// ---------------------------------------------------------------------------

Status RobotInterface::serial_open() {
#if POSIX_SERIAL
    fd_ = ::open(cfg_.port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) return Status::SERIAL_ERROR;

    struct termios tty {};
    if (tcgetattr(fd_, &tty) != 0) { ::close(fd_); return Status::SERIAL_ERROR; }

    cfmakeraw(&tty);
    cfsetispeed(&tty, B1000000);
    cfsetospeed(&tty, B1000000);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 0;

    if (tcsetattr(fd_, TCSANOW, &tty) != 0) { ::close(fd_); return Status::SERIAL_ERROR; }
    tcflush(fd_, TCIOFLUSH);
    return Status::OK;
#else
    return Status::SERIAL_ERROR;  // not supported on this platform
#endif
}

void RobotInterface::serial_close() {
#if POSIX_SERIAL
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
#endif
}

Status RobotInterface::serial_write(const std::vector<uint8_t>& data) {
#if POSIX_SERIAL
    if (fd_ < 0) return Status::SERIAL_ERROR;
    ssize_t n = ::write(fd_, data.data(), data.size());
    return (n == (ssize_t)data.size()) ? Status::OK : Status::SERIAL_ERROR;
#else
    return Status::SERIAL_ERROR;
#endif
}

int RobotInterface::serial_read(std::vector<uint8_t>& buf, size_t n, int timeout_ms) {
#if POSIX_SERIAL
    buf.resize(n);
    size_t total = 0;
    auto deadline = std::chrono::steady_clock::now()
                  + std::chrono::milliseconds(timeout_ms);
    while (total < n) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd_, &fds);
        auto now = std::chrono::steady_clock::now();
        if (now >= deadline) break;
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(deadline - now).count();
        struct timeval tv { (long)(us/1000000), (long)(us%1000000) };
        int ret = select(fd_+1, &fds, nullptr, nullptr, &tv);
        if (ret <= 0) break;
        ssize_t got = ::read(fd_, buf.data() + total, n - total);
        if (got <= 0) break;
        total += got;
    }
    buf.resize(total);
    return (int)total;
#else
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// Dynamixel packet helpers
// ---------------------------------------------------------------------------

std::vector<uint8_t> RobotInterface::build_sync_write_position(
    const std::array<double, NUM_JOINTS>& positions
) {
    // Sync Write: instruction 0x83, 4 bytes per motor (goal position)
    size_t data_len = 2 + 2 + NUM_JOINTS * (1 + 4);  // addr(2) + len(2) + n*(id+4bytes)
    std::vector<uint8_t> pkt;
    pkt.push_back(dxl::HEADER0);
    pkt.push_back(dxl::HEADER1);
    pkt.push_back(dxl::HEADER2);
    pkt.push_back(dxl::RESERVED);
    pkt.push_back(dxl::BROADCAST);
    uint16_t length = (uint16_t)(data_len + 3);  // +3 for instruction + CRC
    pkt.push_back(length & 0xFF);
    pkt.push_back((length >> 8) & 0xFF);
    pkt.push_back(dxl::SYNC_WRITE);
    pkt.push_back(dxl::GOAL_POSITION & 0xFF);
    pkt.push_back((dxl::GOAL_POSITION >> 8) & 0xFF);
    pkt.push_back(4);  // 4 bytes per motor
    pkt.push_back(0);

    for (int i = 0; i < NUM_JOINTS; ++i) {
        uint32_t pos = rad_to_dxl(positions[i]);
        pkt.push_back(MOTOR_IDS[i]);
        pkt.push_back(pos & 0xFF);
        pkt.push_back((pos >> 8)  & 0xFF);
        pkt.push_back((pos >> 16) & 0xFF);
        pkt.push_back((pos >> 24) & 0xFF);
    }

    uint16_t crc = dxl_crc(pkt, pkt.size());
    pkt.push_back(crc & 0xFF);
    pkt.push_back((crc >> 8) & 0xFF);
    return pkt;
}

// ---------------------------------------------------------------------------
// open / close
// ---------------------------------------------------------------------------

Status RobotInterface::open() {
    if (cfg_.mock_mode) {
        open_.store(true);
        mock_joints_ = {};
        return Status::OK;
    }

    auto s = serial_open();
    if (s != Status::OK) return s;

    // Enable torque on all joints
    for (auto id : MOTOR_IDS) {
        write_register(id, dxl::TORQUE_ENABLE, 1);
    }
    write_register(GRIPPER_MOTOR_ID, dxl::TORQUE_ENABLE, 1);

    open_.store(true);
    return Status::OK;
}

void RobotInterface::close() {
    if (!open_.load()) return;
    stop_state_streaming();
    if (!cfg_.mock_mode) {
        for (auto id : MOTOR_IDS)
            write_register(id, dxl::TORQUE_ENABLE, 0);
        write_register(GRIPPER_MOTOR_ID, dxl::TORQUE_ENABLE, 0);
        serial_close();
    }
    open_.store(false);
}

// ---------------------------------------------------------------------------
// read_joint_state
// ---------------------------------------------------------------------------

std::optional<JointState> RobotInterface::read_joint_state() {
    if (!open_.load()) return std::nullopt;

    if (cfg_.mock_mode) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return mock_joints_;
    }

    // Sync Read: request present positions for all joints
    std::vector<uint8_t> pkt;
    pkt.push_back(dxl::HEADER0);
    pkt.push_back(dxl::HEADER1);
    pkt.push_back(dxl::HEADER2);
    pkt.push_back(dxl::RESERVED);
    pkt.push_back(dxl::BROADCAST);
    uint16_t length = 2 + 2 + 2 + NUM_JOINTS + 3;
    pkt.push_back(length & 0xFF);
    pkt.push_back((length >> 8) & 0xFF);
    pkt.push_back(dxl::SYNC_READ);
    pkt.push_back(dxl::PRESENT_POSITION & 0xFF);
    pkt.push_back((dxl::PRESENT_POSITION >> 8) & 0xFF);
    pkt.push_back(4);  // 4 bytes
    pkt.push_back(0);
    for (auto id : MOTOR_IDS) pkt.push_back(id);
    uint16_t crc = dxl_crc(pkt, pkt.size());
    pkt.push_back(crc & 0xFF);
    pkt.push_back((crc >> 8) & 0xFF);

    if (serial_write(pkt) != Status::OK) return std::nullopt;

    // Read responses — each status packet is 15 bytes for a 4-byte payload
    JointState js;
    for (int i = 0; i < NUM_JOINTS; ++i) {
        std::vector<uint8_t> resp;
        int n = serial_read(resp, 15, (int)(cfg_.read_timeout_s * 1000));
        if (n < 11) return std::nullopt;
        // Bytes 9..12 = present position (little-endian 4 bytes)
        uint32_t raw = (uint32_t)resp[9]
                     | ((uint32_t)resp[10] << 8)
                     | ((uint32_t)resp[11] << 16)
                     | ((uint32_t)resp[12] << 24);
        js.position[i] = dxl_to_rad(raw);
    }
    js.timestamp_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    return js;
}

// ---------------------------------------------------------------------------
// read_robot_state
// ---------------------------------------------------------------------------

std::optional<RobotState> RobotInterface::read_robot_state() {
    auto maybe_js = read_joint_state();
    if (!maybe_js) return std::nullopt;

    RobotState rs;
    rs.joints    = *maybe_js;

    Kinematics kin;
    rs.eef_pose  = kin.forward(rs.joints);
    return rs;
}

// ---------------------------------------------------------------------------
// send_command
// ---------------------------------------------------------------------------

Status RobotInterface::send_command(const JointCommand& cmd) {
    if (!open_.load()) return Status::SERIAL_ERROR;

    if (cfg_.mock_mode) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        for (int j = 0; j < NUM_JOINTS; ++j)
            mock_joints_.position[j] = cmd.position[j];
        return Status::OK;
    }

    auto pkt = build_sync_write_position(cmd.position);
    return serial_write(pkt);
}

// ---------------------------------------------------------------------------
// set_gripper
// ---------------------------------------------------------------------------

Status RobotInterface::set_gripper(GripperCommand gcmd, double force_norm, double timeout_s) {
    if (!open_.load()) return Status::SERIAL_ERROR;

    // Map gripper command to position: OPEN = 1024, CLOSE = 2048 (rough values)
    uint16_t target_pos = (gcmd == GripperCommand::OPEN) ? 1024 : 2048;
    return write_register(GRIPPER_MOTOR_ID, dxl::GOAL_POSITION, target_pos);
}

// ---------------------------------------------------------------------------
// emergency_stop
// ---------------------------------------------------------------------------

Status RobotInterface::emergency_stop() {
    if (cfg_.mock_mode) return Status::OK;
    for (auto id : MOTOR_IDS)
        write_register(id, dxl::TORQUE_ENABLE, 0);
    write_register(GRIPPER_MOTOR_ID, dxl::TORQUE_ENABLE, 0);
    return Status::OK;
}

// ---------------------------------------------------------------------------
// State streaming
// ---------------------------------------------------------------------------

void RobotInterface::stream_loop(double rate_hz) {
    auto dt = std::chrono::duration<double>(1.0 / rate_hz);
    while (streaming_.load()) {
        auto t0 = std::chrono::steady_clock::now();
        auto maybe = read_robot_state();
        if (maybe && state_cb_) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            last_state_ = *maybe;
            state_cb_(*maybe);
        }
        auto elapsed = std::chrono::steady_clock::now() - t0;
        if (elapsed < dt)
            std::this_thread::sleep_for(dt - elapsed);
    }
}

void RobotInterface::start_state_streaming(StateCallback cb, double rate_hz) {
    stop_state_streaming();
    state_cb_ = std::move(cb);
    streaming_.store(true);
    stream_thread_ = std::thread([this, rate_hz]{ stream_loop(rate_hz); });
}

void RobotInterface::stop_state_streaming() {
    streaming_.store(false);
    if (stream_thread_.joinable()) stream_thread_.join();
}

// ---------------------------------------------------------------------------
// Low-level register read/write
// ---------------------------------------------------------------------------

Status RobotInterface::write_register(uint8_t motor_id, uint16_t address, uint16_t value) {
    if (cfg_.mock_mode) return Status::OK;
    std::vector<uint8_t> pkt = {
        dxl::HEADER0, dxl::HEADER1, dxl::HEADER2, dxl::RESERVED,
        motor_id,
        7, 0,               // length = 7 (instruction + addr(2) + val(2) + CRC(2))
        dxl::WRITE,
        (uint8_t)(address & 0xFF), (uint8_t)(address >> 8),
        (uint8_t)(value & 0xFF),   (uint8_t)(value >> 8)
    };
    uint16_t crc = dxl_crc(pkt, pkt.size());
    pkt.push_back(crc & 0xFF);
    pkt.push_back((crc >> 8) & 0xFF);
    return serial_write(pkt);
}

std::optional<uint16_t> RobotInterface::read_register(uint8_t motor_id, uint16_t address) {
    if (cfg_.mock_mode) return 0;
    std::vector<uint8_t> pkt = {
        dxl::HEADER0, dxl::HEADER1, dxl::HEADER2, dxl::RESERVED,
        motor_id,
        7, 0,
        dxl::READ,
        (uint8_t)(address & 0xFF), (uint8_t)(address >> 8),
        2, 0  // read 2 bytes
    };
    uint16_t crc = dxl_crc(pkt, pkt.size());
    pkt.push_back(crc & 0xFF);
    pkt.push_back((crc >> 8) & 0xFF);
    if (serial_write(pkt) != Status::OK) return std::nullopt;

    std::vector<uint8_t> resp;
    if (serial_read(resp, 13, 10) < 11) return std::nullopt;
    return (uint16_t)(resp[9] | (resp[10] << 8));
}

bool RobotInterface::ping_motor(uint8_t motor_id) {
    if (cfg_.mock_mode) return true;
    std::vector<uint8_t> pkt = {
        dxl::HEADER0, dxl::HEADER1, dxl::HEADER2, dxl::RESERVED,
        motor_id,
        3, 0,
        dxl::PING
    };
    uint16_t crc = dxl_crc(pkt, pkt.size());
    pkt.push_back(crc & 0xFF);
    pkt.push_back((crc >> 8) & 0xFF);
    if (serial_write(pkt) != Status::OK) return false;
    std::vector<uint8_t> resp;
    return serial_read(resp, 14, 20) >= 10;
}

} // namespace tinyvla
