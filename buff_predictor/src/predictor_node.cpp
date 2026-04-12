#include "predictor.hpp"
#include "coordinate_solver.hpp"
#include "config.hpp"
#include <memory>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include "buff_interfaces/msg/buff_target.hpp"
#include "buff_interfaces/msg/buff_aiming_data.hpp"
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <tf2/utils.h>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>

class BuffPredictorNode : public rclcpp::Node
{
public:
    BuffPredictorNode() : Node("buff_predictor_node")
    {
        // 声明参数（可从 YAML 文件加载）
        this->declare_parameter<double>("physics.arm_length", 1.5);
        this->declare_parameter<double>("physics.r_radius", 0.1);
        this->declare_parameter<int>("filter.avg_window", 5);
        this->declare_parameter<int>("filter.median_window", 5);
        this->declare_parameter<double>("predictor.fit_start_time", 1.5);
        this->declare_parameter<double>("predictor.min_omega", 1.884);
        this->declare_parameter<int>("predictor.lm_maxfev", 12000);
        this->declare_parameter<double>("predictor.lm_xtol", 1.0e-7);
        this->declare_parameter<bool>("debug_mode", false);
        this->declare_parameter<std::string>("frames.world", "odom");
        this->declare_parameter<std::string>("frames.camera", "camera_optical_frame");
        this->declare_parameter<double>("camera.fx", 1303.675283386667);
        this->declare_parameter<double>("camera.fy", 1303.675283386667);
        this->declare_parameter<double>("camera.cx", 720.0);
        this->declare_parameter<double>("camera.cy", 540.0);

        // 读取参数
        physical_arm_length_ = this->get_parameter("physics.arm_length").as_double();
        debug_mode_ = this->get_parameter("debug_mode").as_bool();
        world_frame_ = this->get_parameter("frames.world").as_string();
        camera_frame_ = this->get_parameter("frames.camera").as_string();

        // 初始化滤波器
        radius_filter_ = std::make_unique<MovAvg>(15);

        // 初始化预测器（可传入滤波窗口参数）
        predictor_ = std::make_unique<Big_Buff_Predictor>();
        // 注：Big_Buff_Predictor 构造函数内部硬编码了窗口大小，可扩展接口支持参数化

        // 角度观测器
        angle_observer_ = std::make_unique<angleObserver>(clockMode::unknown);

        // 相机参数
        double fx = this->get_parameter("camera.fx").as_double();
        double fy = this->get_parameter("camera.fy").as_double();
        double cx = this->get_parameter("camera.cx").as_double();
        double cy = this->get_parameter("camera.cy").as_double();
        CameraParams cam_params(fx, fy, cx, cy);
        predictor_3d_ = std::make_unique<Predictor3D>(cam_params, physical_arm_length_);
        CoordinateSolver::CameraIntrinsics coord_params = {fx, fy, cx, cy};
        coord_solver_ = std::make_unique<CoordinateSolver>(coord_params);

        // 订阅与发布
        target_sub_ = this->create_subscription<buff_interfaces::msg::BuffTarget>(
            "/buff_target", rclcpp::SensorDataQoS(),
            [this](const buff_interfaces::msg::BuffTarget::SharedPtr msg) { targetCallback(msg); });

        aiming_pub_ = this->create_publisher<buff_interfaces::msg::BuffAimingData>(
            "/buff/aiming_data", rclcpp::SensorDataQoS());

        // TF
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        if (debug_mode_) {
            tracker_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>("/tracker/debug_point", 10);
        }

        RCLCPP_INFO(this->get_logger(), "BuffPredictorNode started. World frame: %s, Camera frame: %s",
                    world_frame_.c_str(), camera_frame_.c_str());
    }

private:
    bool transformPointToWorld(
        const geometry_msgs::msg::PointStamped& input,
        const builtin_interfaces::msg::Time& stamp,
        geometry_msgs::msg::PointStamped& output)
    {
        if (input.header.frame_id == world_frame_) {
            output = input;
            return true;
        }
        try {
            auto tf_stamped = tf_buffer_->lookupTransform(
                world_frame_, input.header.frame_id, stamp, tf2::durationFromSec(0.05));
            tf2::doTransform(input, output, tf_stamped);
            return true;
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "TF error (world=%s, source=%s): %s",
                world_frame_.c_str(), input.header.frame_id.c_str(), ex.what());
            return false;
        }
    }

    void targetCallback(const buff_interfaces::msg::BuffTarget::SharedPtr msg)
    {
        // 计算向量与半径
        float vector_x = msg->target_center_x - msg->r_center_x;
        float vector_y = msg->target_center_y - msg->r_center_y;
        double pixel_arm_length = msg->radius;
        if (pixel_arm_length < 1.0) pixel_arm_length = 1.0;

        // 角度连续化
        float continues_angle = angle_observer_->update(vector_x, vector_y, pixel_arm_length);
        frame_count_++;

        // 使用完整的绝对时间戳（秒）
        rclcpp::Time stamp(msg->header.stamp);
        double stamp_sec = stamp.seconds();

        // 记录 CSV（debug）
        if (debug_mode_) {
            static std::ofstream csv_file("log/angle_time.csv", std::ios::app);
            static bool wrote_header = false;
            if (csv_file.is_open()) {
                if (!wrote_header) {
                    csv_file << "stamp_sec,continues_angle\n";
                    wrote_header = true;
                }
                csv_file << std::fixed << std::setprecision(6) << stamp_sec << "," << continues_angle << "\n";
            }
        }

        // 更新预测器（获取正弦参数）
        auto result = predictor_->update(continues_angle, stamp_sec);
        bool fit_success = result.first;

        // 3D 解算
        float depth = predictor_3d_->compute_depth(pixel_arm_length);

        // R 标在相机坐标系下的 3D 点
        auto r_cam_pt = coord_solver_->pixelToCamera(msg->r_center_x, msg->r_center_y, depth);
        geometry_msgs::msg::PointStamped r_cam;
        r_cam.header.stamp = msg->header.stamp;
        r_cam.header.frame_id = camera_frame_;
        r_cam.point.x = r_cam_pt.x;
        r_cam.point.y = r_cam_pt.y;
        r_cam.point.z = r_cam_pt.z;

        // 目标点在相机坐标系
        auto target_cam_pt = coord_solver_->pixelToCamera(msg->target_center_x, msg->target_center_y, depth);
        geometry_msgs::msg::PointStamped target_cam;
        target_cam.header.stamp = msg->header.stamp;
        target_cam.header.frame_id = camera_frame_;
        target_cam.point.x = target_cam_pt.x;
        target_cam.point.y = target_cam_pt.y;
        target_cam.point.z = target_cam_pt.z;

        // TF 转换到世界坐标系
        geometry_msgs::msg::PointStamped r_world, target_world;
        if (!transformPointToWorld(r_cam, msg->header.stamp, r_world) ||
            !transformPointToWorld(target_cam, msg->header.stamp, target_world)) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "TF failed, skipping frame");
            return;
        }

        if (debug_mode_) {
            tracker_pub_->publish(r_world);
            tracker_pub_->publish(target_world);
        }

        // 组装 BuffAimingData 消息
        auto aiming_msg = buff_interfaces::msg::BuffAimingData();
        aiming_msg.header.stamp = msg->header.stamp;          // 保持原始时间戳
        aiming_msg.header.frame_id = world_frame_;
        aiming_msg.frame_number = frame_count_;
        aiming_msg.is_tracking = fit_success;
        aiming_msg.r_center_x_3d = r_world.point.x;
        aiming_msg.r_center_y_3d = r_world.point.y;
        aiming_msg.r_center_z_3d = r_world.point.z;
        aiming_msg.pixel_r_center_x = msg->r_center_x;
        aiming_msg.pixel_r_center_y = msg->r_center_y;
        aiming_msg.pixel_radius = pixel_arm_length;
        aiming_msg.target_x_3d = target_world.point.x;
        aiming_msg.target_y_3d = target_world.point.y;
        aiming_msg.target_z_3d = target_world.point.z;

        if (fit_success) {
            const auto sin_para = predictor_->get_sin_para();
            if (sin_para.size() >= 4) {
                aiming_msg.sin_a = sin_para[0];
                aiming_msg.sin_omega = sin_para[1];
                aiming_msg.sin_phi = sin_para[2];
                aiming_msg.sin_b = sin_para[3];
            } else {
                aiming_msg.is_tracking = false;
            }
        } else {
            aiming_msg.is_tracking = false;
        }

        aiming_pub_->publish(aiming_msg);
    }

    // 成员变量
    rclcpp::Subscription<buff_interfaces::msg::BuffTarget>::SharedPtr target_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr tracker_pub_;
    rclcpp::Publisher<buff_interfaces::msg::BuffAimingData>::SharedPtr aiming_pub_;

    std::unique_ptr<Big_Buff_Predictor> predictor_;
    std::unique_ptr<angleObserver> angle_observer_;
    std::unique_ptr<Predictor3D> predictor_3d_;
    std::unique_ptr<CoordinateSolver> coord_solver_;
    std::unique_ptr<MovAvg> radius_filter_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    double physical_arm_length_;
    bool debug_mode_;
    std::string world_frame_;
    std::string camera_frame_;
    int frame_count_ = 0;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BuffPredictorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
// 注册为组件
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(BuffPredictorNode)