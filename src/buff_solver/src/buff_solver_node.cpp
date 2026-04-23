#include <cmath>
#include <memory>
#include <string>

#include "buff_solver.hpp"

class BuffSolverNode : public rclcpp::Node
{
public:
    explicit BuffSolverNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : Node("buff_solver", options), buff_solver_(get_clock())
    {
        // ---------- 参数声明 ----------
        this->declare_parameter<std::string>("odom_frame", "odom");
        this->declare_parameter<std::string>("gimbal_frame", "gimbal_link");
        this->declare_parameter<double>("bullet_speed", 24.0);
        this->declare_parameter<double>("shoot_delay", 0.05);
        this->declare_parameter<double>("physical_radius", 0.75);
        this->declare_parameter<double>("gravity", 9.81);
        this->declare_parameter<double>("air_k", 0.001);
        this->declare_parameter<double>("converge_thresh", 0.002);
        this->declare_parameter<double>("sim_dt", 0.005);
        this->declare_parameter<bool>("debug", true);
        this->declare_parameter<int>("max_iteration", 5);
        this->declare_parameter<double>("stop_error", 0.001);

        // 读取参数
        odom_frame_        = this->get_parameter("odom_frame").as_string();
        gimbal_frame_      = this->get_parameter("gimbal_frame").as_string();
        double bullet_speed      = this->get_parameter("bullet_speed").as_double();
        double shoot_delay       = this->get_parameter("shoot_delay").as_double();
        double physical_radius   = this->get_parameter("physical_radius").as_double();
        double gravity           = this->get_parameter("gravity").as_double();
        double air_k             = this->get_parameter("air_k").as_double();
        double converge_thresh   = this->get_parameter("converge_thresh").as_double();
        double sim_dt            = this->get_parameter("sim_dt").as_double();
        debug_                   = this->get_parameter("debug").as_bool();
        int max_iteration        = this->get_parameter("max_iteration").as_int();
        double stop_error        = this->get_parameter("stop_error").as_double();

        // 注入参数给解算器
        buff_solver_.set_params(bullet_speed, shoot_delay, physical_radius,
                                gravity, air_k, converge_thresh, sim_dt,
                                max_iteration, stop_error);

        // 订阅云台状态
        joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::JointState::SharedPtr joint_msg)
            {
                if (!joint_msg) {
                    RCLCPP_ERROR(this->get_logger(), "Received null joint state message");
                    return;
                }
                current_gimbal_msg_ = joint_msg;
            }
        );

        // 订阅能量机关数据
        aiming_sub_ = this->create_subscription<buff_interfaces::msg::BuffAimingData>(
            "/buff/aiming_data", rclcpp::SensorDataQoS(),
            [this](const buff_interfaces::msg::BuffAimingData::SharedPtr msg)
            {
                if (!msg) {
                    RCLCPP_ERROR(this->get_logger(), "Received null buff aiming data");
                    return;
                }

                if (!current_gimbal_msg_) {
                    RCLCPP_WARN(this->get_logger(), "Current gimbal state not received yet.");
                    return;
                }

                auto autoaim_msg = std::make_unique<gary_msgs::msg::AutoAIM>();
                
                // 未跟踪或拟合失败
                if (!msg->is_tracking) {
                    publishZeroCommand(std::move(autoaim_msg));
                    if (debug_) {
                        gary_msgs::msg::AutoAimDebug debug_msg;
                        debug_pub_->publish(debug_msg);
                    }
                    return;
                }

                // 更新解算器
                if (!buff_solver_.update(msg, current_gimbal_msg_)) {
                    RCLCPP_WARN(this->get_logger(), "Failed to update BuffSolver");
                    return;
                }

                // 获取解算结果
                double plan_yaw   = buff_solver_.get_plan_yaw();
                double plan_pitch = buff_solver_.get_plan_pitch();
                double distance   = buff_solver_.get_distance();

                double pitch_diff = -(plan_pitch - current_gimbal_msg_->position[1]);
                double yaw_diff   = plan_yaw - current_gimbal_msg_->position[0];

                // 填充并发布
                autoaim_msg->yaw   = yaw_diff;
                autoaim_msg->pitch = pitch_diff;
                autoaim_msg->target_distance = distance;
                autoaim_msg->header.stamp = this->now();
                autoaim_msg->header.frame_id = gimbal_frame_;
                autoaim_pub_->publish(std::move(autoaim_msg));

                // 调试
                if (debug_) {
                    auto debug_msg = buff_solver_.get_debug_msg();
                    debug_msg.yaw_diff = yaw_diff;
                    debug_pub_->publish(debug_msg);
                }
            }
        );

        // 发布器
        autoaim_pub_ = this->create_publisher<gary_msgs::msg::AutoAIM>(
            "/autoaim/target", rclcpp::SensorDataQoS());

        if (debug_) {
            debug_pub_ = this->create_publisher<gary_msgs::msg::AutoAimDebug>(
                "/autoaim/debug", rclcpp::SensorDataQoS());
        }
    }

private:
    void publishZeroCommand(std::unique_ptr<gary_msgs::msg::AutoAIM> msg)
    {
        msg->yaw = 0.0;
        msg->pitch = 0.0;
        msg->target_distance = 0.0;
        msg->header.stamp = this->now();
        msg->header.frame_id = gimbal_frame_;
        autoaim_pub_->publish(std::move(msg));
    }

    BuffSolver buff_solver_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    rclcpp::Subscription<buff_interfaces::msg::BuffAimingData>::SharedPtr aiming_sub_;
    rclcpp::Publisher<gary_msgs::msg::AutoAIM>::SharedPtr autoaim_pub_;
    rclcpp::Publisher<gary_msgs::msg::AutoAimDebug>::SharedPtr debug_pub_;

    sensor_msgs::msg::JointState::SharedPtr current_gimbal_msg_;
    std::string odom_frame_;
    std::string gimbal_frame_;
    bool debug_ = true;
};

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(BuffSolverNode)