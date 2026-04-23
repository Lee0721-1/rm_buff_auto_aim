#pragma once

#include "buff_interfaces/msg/buff_aiming_data.hpp"
#include "gary_msgs/msg/auto_aim.hpp"
#include "gary_msgs/msg/auto_aim_debug.hpp"
#include <geometry_msgs/msg/point_stamped.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <memory>

// 云台类
class Gimbal
{
public:
    Gimbal() : pitch_(0), yaw_(0) {}
    Gimbal(sensor_msgs::msg::JointState::SharedPtr joint_state)
        : pitch_(joint_state->position[1]), yaw_(joint_state->position[0]) {}
    Gimbal(double pitch, double yaw) : pitch_(pitch), yaw_(yaw) {}
    bool update(sensor_msgs::msg::JointState::SharedPtr joint_state);

    double get_pitch() const { return pitch_; }
    double get_yaw() const { return yaw_; }
    void set_pitch(double pitch) { pitch_ = pitch; }
    void set_yaw(double yaw) { yaw_ = yaw; }

private:
    double pitch_;
    double yaw_;
};

// 弹道解算结果结构体
struct BallisticResult {
    double pitch;   // 枪口绝对仰角 (rad)
    double t_fly;   // 子弹飞行时间 (s)
};

// 能量机关数据类
class Buff
{
public:
    Buff() = default;
    Buff(const buff_interfaces::msg::BuffAimingData::SharedPtr msg);
    bool update(const buff_interfaces::msg::BuffAimingData::SharedPtr msg);

    bool is_tracking() const { return is_tracking_; }
    bool is_bigbuff()    const { return is_bigbuff_; }
    int8_t spin_direction() const { return spin_direction_; }

    double r_x_3d() const { return r_x_3d_; }
    double r_y_3d() const { return r_y_3d_; }
    double r_z_3d() const { return r_z_3d_; }
    double target_x_3d() const { return target_x_3d_; }
    double target_y_3d() const { return target_y_3d_; }
    double target_z_3d() const { return target_z_3d_; }
    double sin_a() const { return sin_a_; }
    double sin_omega() const { return sin_omega_; }
    double sin_phi() const { return sin_phi_; }
    double sin_b() const { return sin_b_; }
    double fit_start_time_sec() const { return fit_start_time_sec_; }

private:
    bool is_tracking_ = false;          // 是否拟合成功
    bool is_bigbuff_ = false;           // true=大符, false=小符
    int8_t spin_direction_ = 0;         // -1=逆时针, 0=未知, 1=顺时针
    double r_x_3d_ = 0.0;
    double r_y_3d_ = 0.0;
    double r_z_3d_ = 0.0;
    double target_x_3d_ = 0.0;
    double target_y_3d_ = 0.0;
    double target_z_3d_ = 0.0;
    double sin_a_ = 0.0;
    double sin_omega_ = 0.0;
    double sin_phi_ = 0.0;
    double sin_b_ = 0.0;
    double fit_start_time_sec_ = 0.0;
};

// 能量机关解算器
class BuffSolver
{
public:
    BuffSolver(rclcpp::Clock::SharedPtr clock);

    void set_params(
        double bullet_speed,
        double shoot_delay,
        double physical_radius,
        double gravity,
        double air_k,
        double converge_thresh,
        double sim_dt,
        int max_iteration,
        double stop_error);

    bool update(
        const buff_interfaces::msg::BuffAimingData::SharedPtr msg,
        const sensor_msgs::msg::JointState::SharedPtr current_gimbal_msg);

    // 获取结果的接口
    double get_plan_yaw()   const { return plan_yaw_; }
    double get_plan_pitch() const { return plan_pitch_; }
    double get_distance()   const { return distance_; }
    gary_msgs::msg::AutoAimDebug get_debug_msg() const;

private:
    BallisticResult solveBallistic(double distance_xy, double distance_z);
    geometry_msgs::msg::Point predictTargetPosition(const Buff& buff, double delta_t) const;

    // 参数
    double bullet_speed_ = 24.0;
    double shoot_delay_ = 0.05;
    double physical_radius_ = 0.75;
    double gravity_ = 9.81;
    double air_k_ = 0.001;
    double converge_thresh_ = 0.002;
    double sim_dt_ = 0.005;
    int max_iteration_ = 5;
    double stop_error_ = 0.001;

    // 中间状态
    Buff buff_;
    Gimbal current_gimbal_;
    double plan_yaw_ = 0.0;
    double plan_pitch_ = 0.0;
    double distance_ = 0.0;
    BallisticResult final_result_;
    rclcpp::Clock::SharedPtr clock_;
};