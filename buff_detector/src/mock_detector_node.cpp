/**
 * @file mock_detector_node.cpp
 * @brief 模拟能量机关检测节点，用于无 ONNX Runtime 环境下的功能测试。
 *        发布模拟的 BuffTarget 消息，供 predictor 和 solver 节点使用。
 */

#include <rclcpp/rclcpp.hpp>
#include "buff_interfaces/msg/buff_target.hpp"
#include <cmath>

class MockDetectorNode : public rclcpp::Node
{
public:
    MockDetectorNode() : Node("mock_detector_node")
    {
        publisher_ = this->create_publisher<buff_interfaces::msg::BuffTarget>("/buff_target", 10);
        
        // 20Hz 发布
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),
            std::bind(&MockDetectorNode::publishMockData, this));
            
        RCLCPP_INFO(this->get_logger(), "Mock Detector started, publishing simulated BuffTarget at 20Hz.");
    }

private:
    void publishMockData()
    {
        auto msg = buff_interfaces::msg::BuffTarget();
        msg.header.stamp = this->now();
        msg.header.frame_id = "camera_optical_frame";
        msg.is_tracking = true;
        
        // 模拟一个半径为120像素的匀速旋转目标
        msg.radius = 120.0f;
        msg.r_center_x = 640.0f;
        msg.r_center_y = 480.0f;
        
        // 动态角度：每秒旋转 2π/5 弧度（约72°/s）
        static double angle = 0.0;
        angle += 0.1;  // 每帧增加0.1弧度 ≈ 5.7°
        if (angle > 2.0 * M_PI) angle -= 2.0 * M_PI;
        
        msg.target_center_x = 640.0f + msg.radius * std::cos(angle);
        msg.target_center_y = 480.0f + msg.radius * std::sin(angle);
        
        // 【修正点】直接传递对象 msg，而不是 *msg
        publisher_->publish(msg);
    }

    rclcpp::Publisher<buff_interfaces::msg::BuffTarget>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MockDetectorNode>());
    rclcpp::shutdown();
    return 0;
}