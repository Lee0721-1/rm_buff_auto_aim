#include "buff_solver.hpp"
#include <cmath>
#include <rclcpp/logging.hpp>

// --------------- Gimbal ---------------
bool Gimbal::update(sensor_msgs::msg::JointState::SharedPtr joint_state)
{
    if (joint_state->position.size() < 2)
    {
        RCLCPP_WARN(rclcpp::get_logger("BuffSolver"), "JointState message does not contain enough position data.");
        return false;
    }
    pitch_ = joint_state->position[1];
    yaw_ = joint_state->position[0];
    return true;
}

// --------------- Buff ---------------
Buff::Buff(const buff_interfaces::msg::BuffAimingData::SharedPtr msg)
    : is_tracking_(msg->is_tracking),
      is_bigbuff_(msg->is_bigbuff),
      spin_direction_(msg->spin_direction),
      r_x_3d_(msg->r_x_3d),
      r_y_3d_(msg->r_y_3d),
      r_z_3d_(msg->r_z_3d),
      target_x_3d_(msg->target_x_3d),
      target_y_3d_(msg->target_y_3d),
      target_z_3d_(msg->target_z_3d),
      sin_a_(msg->sin_a),
      sin_omega_(msg->sin_omega),
      sin_phi_(msg->sin_phi),
      sin_b_(msg->sin_b),
      fit_start_time_sec_(msg->fit_start_time_sec)
{
}

bool Buff::update(const buff_interfaces::msg::BuffAimingData::SharedPtr msg)
{
    is_tracking_  = msg->is_tracking;
    is_bigbuff_   = msg->is_bigbuff;
    spin_direction_ = msg->spin_direction;
    r_x_3d_       = msg->r_x_3d;
    r_y_3d_       = msg->r_y_3d;
    r_z_3d_       = msg->r_z_3d;
    target_x_3d_  = msg->target_x_3d;
    target_y_3d_  = msg->target_y_3d;
    target_z_3d_  = msg->target_z_3d;
    sin_a_        = msg->sin_a;
    sin_omega_    = msg->sin_omega;
    sin_phi_      = msg->sin_phi;
    sin_b_        = msg->sin_b;
    fit_start_time_sec_ = msg->fit_start_time_sec;
    return true;
}

// --------------- BuffSolver ---------------
BuffSolver::BuffSolver(rclcpp::Clock::SharedPtr clock)
    : clock_(std::move(clock))
{
}

void BuffSolver::set_params(
    double bullet_speed,
    double shoot_delay,
    double physical_radius,
    double gravity,
    double air_k,
    double converge_thresh,
    double sim_dt,
    int max_iteration,
    double stop_error)
{
    bullet_speed_     = bullet_speed;
    shoot_delay_      = shoot_delay;
    physical_radius_  = physical_radius;
    gravity_          = gravity;
    air_k_            = air_k;
    converge_thresh_  = converge_thresh;
    sim_dt_           = sim_dt;
    max_iteration_    = max_iteration;
    stop_error_       = stop_error;
}

geometry_msgs::msg::Point BuffSolver::predictTargetPosition(const Buff& buff, double delta_t) const
{
    // 计算当前目标在 Y-Z 平面内的角度（从 Z 轴正方向起算，逆时针为正）
    double theta0 = std::atan2(buff.target_z_3d() - buff.r_z_3d(),
                               buff.target_y_3d() - buff.r_y_3d());
    double delta_theta = 0.0;
    const double eps = 1e-6;

    if (buff.is_bigbuff()) {
        // 大符：角速度 = A*sin(ωt + φ) + B
        double A = buff.sin_a();
        double omega = buff.sin_omega();
        double phi = buff.sin_phi();
        double B = buff.sin_b();

        if (std::abs(omega) > eps && A > eps) {
            double t0 = buff.fit_start_time_sec();
            double t_pred = t0 + delta_t;
            delta_theta = B * delta_t
                - (A / omega) * (std::cos(omega * t_pred + phi) - std::cos(omega * t0 + phi));
        } else {
            delta_theta = B * delta_t;
        }
    } else {
        // 小符：匀速 pi/3 rad/s
        delta_theta = M_PI / 3.0 * delta_t;
    }

    double theta_pred = theta0 + delta_theta;

    geometry_msgs::msg::Point p;
    p.x = buff.r_x_3d();                       // 深度不变
    p.y = buff.r_y_3d() + physical_radius_ * std::cos(theta_pred);
    p.z = buff.r_z_3d() + physical_radius_ * std::sin(theta_pred);
    return p;
}

BallisticResult BuffSolver::solveBallistic(double distance_xy, double distance_z)
{
    const double v0 = bullet_speed_;
    const double g  = gravity_;
    const double k  = air_k_;
    const double dt = sim_dt_;

    double v2 = v0 * v0;
    double v4 = v2 * v2;
    double r = std::sqrt(v4 - g * (g * distance_xy * distance_xy + 2 * distance_z * v2));
    double pitch = std::isnan(r) ? std::atan2(distance_z, distance_xy)
                                 : std::atan2(v2 - r, g * distance_xy);

    double y_aim = distance_z;
    double t_fly = 0.0;

    for (int iter = 0; iter < max_iteration_; ++iter) {
        double x = 0.0, y = 0.0, t = 0.0;
        double vx = v0 * std::cos(pitch);
        double vy = v0 * std::sin(pitch);

        while (x < distance_xy) {
            double v = std::sqrt(vx*vx + vy*vy);
            double ax = -k * v * vx;
            double ay = -g - k * v * vy;
            vx += ax * dt;
            vy += ay * dt;
            x  += vx * dt;
            y  += vy * dt;
            t  += dt;
            if (t > 2.0) break;
        }

        double error = distance_z - y;
        t_fly = t;

        if (std::abs(error) < stop_error_)
            break;

        y_aim += error;
        pitch = std::atan2(y_aim, distance_xy);
    }

    return {-pitch, t_fly};
}

bool BuffSolver::update(
    const buff_interfaces::msg::BuffAimingData::SharedPtr msg,
    const sensor_msgs::msg::JointState::SharedPtr current_gimbal_msg)
{
    // 更新内部数据
    if (!buff_.update(msg) || !current_gimbal_.update(current_gimbal_msg))
        return false;

    // 消息延迟
    double msg_latency = (clock_->now() - msg->header.stamp).seconds();
    if (msg_latency < 0.0) msg_latency = 0.0;

    // 初始飞行时间估计
    double t_fly = std::max(std::hypot(buff_.r_x_3d(), buff_.r_y_3d()) / bullet_speed_, 0.05);
    bool converged = false;

    for (int iter = 0; iter < max_iteration_; ++iter) {
        double predict_delay = msg_latency + t_fly + shoot_delay_;
        geometry_msgs::msg::Point pred = predictTargetPosition(buff_, predict_delay);

        double dist_xy = std::hypot(pred.x, pred.y);
        double dist_z  = pred.z;

        BallisticResult res = solveBallistic(dist_xy, dist_z);

        if (std::abs(t_fly - res.t_fly) < converge_thresh_) {
            final_result_ = res;
            plan_yaw_   = std::atan2(pred.y, pred.x);
            plan_pitch_ = res.pitch;
            distance_   = std::hypot(dist_xy, dist_z);
            converged = true;
            break;
        }

        t_fly = res.t_fly;
        // 即使不收敛也用最后一次结果
        final_result_ = res;
        plan_yaw_   = std::atan2(pred.y, pred.x);
        plan_pitch_ = res.pitch;
        distance_   = std::hypot(dist_xy, dist_z);
    }

    if (!converged) {
        RCLCPP_WARN_THROTTLE(rclcpp::get_logger("BuffSolver"), *clock_, 1000,
                             "Ballistic iteration did not converge, using last result.");
    }

    return true;
}

gary_msgs::msg::AutoAimDebug BuffSolver::get_debug_msg() const
{
    gary_msgs::msg::AutoAimDebug debug;
    debug.plan_yaw   = plan_yaw_;
    debug.plan_pitch = plan_pitch_;
    debug.yaw_diff   = -(plan_yaw_ - current_gimbal_.get_yaw());
   
    return debug;
}








