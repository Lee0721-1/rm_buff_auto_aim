#include "buff_detector/config.hpp"
#include "buff_detector/detector.hpp"
#include "buff_interfaces/msg/buff_target.hpp"
#include <opencv2/core/mat.hpp>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>


class BuffDetectorNode : public rclcpp::Node
{
public:
    BuffDetectorNode() : Node("buff_detector_node")
    {
        // 声明参数
        bool debug = this->declare_parameter<bool>("detector.debug", true);
        double conf_thresh = this->declare_parameter<double>("detector.conf_thresh", 0.5);
        double iou_thresh = this->declare_parameter<double>("detector.iou_thresh", 0.5);
        double inside_shade_rate = this->declare_parameter<double>("detector.inside_shade_rate", 0.7);
        double outside_shade_rate = this->declare_parameter<double>("detector.outside_shade_rate", 1.39);
        int dilate_kernel = this->declare_parameter<int>("detector.dilate_kernel", 7);
        int max_lost_frame = this->declare_parameter<int>("detector.max_lost_frame", 5);
        std::string model_path = this->declare_parameter<std::string>("detector.model_path", "model/Fan.onnx");

        // HSV 参数：从 YAML 读取数组
        std::vector<int64_t> lower_hsv_vec = this->declare_parameter<std::vector<int64_t>>("detector.hsv_lower", {0, 40, 220});
        std::vector<int64_t> upper_hsv_vec = this->declare_parameter<std::vector<int64_t>>("detector.hsv_upper", {70, 255, 255});
        cv::Scalar lower_hsv(lower_hsv_vec[0], lower_hsv_vec[1], lower_hsv_vec[2]);
        cv::Scalar upper_hsv(upper_hsv_vec[0], upper_hsv_vec[1], upper_hsv_vec[2]);

        detector_ = std::make_unique<BuffDetector>(
        debug, conf_thresh, iou_thresh, inside_shade_rate, outside_shade_rate,
        dilate_kernel, max_lost_frame, model_path, lower_hsv, upper_hsv);

        // 订阅图像
        img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_raw",
            rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
            [this](const sensor_msgs::msg::Image::SharedPtr msg) { imageCallback(msg); });

        // 发布目标信息
        target_pub_ = this->create_publisher<buff_interfaces::msg::BuffTarget>("/buff_target", 10);

        if (debug) {
            debug_image_pub_ = image_transport::create_publisher(this, "/buff_detector/debug_image");
        }
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // 将ROS图像消息转换为OpenCV格式
        cv::Mat frame;
        try {
            frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        if (frame.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty frame.");
            return;
        }

        try {
            processFrame(frame, msg->header.stamp);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "processFrame threw exception: %s", e.what());
        } catch (...) {
            RCLCPP_ERROR(this->get_logger(), "processFrame threw unknown exception");
        }
    }

    void processFrame(cv::Mat frame, const rclcpp::Time& frame_stamp)
    {
        // 处理第一帧
        if (!is_tracking_)
        {
            auto frame_clone = frame.clone(); // 克隆帧以避免修改原始帧
            if (detector_->init(frame_clone))
            {
                is_tracking_ = true;
                RCLCPP_INFO(this->get_logger(), "Detector initialized.");
            }
        }

        // 更新检测
        if (is_tracking_)
        {
            if (!detector_->update(frame))
            {
                is_tracking_ = false;
                RCLCPP_WARN(this->get_logger(), "Detector lost target, re-initialization required.");
            }
        }

        cv::Mat debug_frame;
        if constexpr (IS_DEBUG) {
            // 获取调试图像
            debug_frame = detector_->get_debug_frame();
        }

        // 获取中心点坐标
        cv::Point2f fan_center = detector_->get_first_target_blade_bbox().get_center_2f();
        cv::Point2f r_center = detector_->get_current_R_box().get_center_2f();

        // 发布目标信息
        auto target_msg = buff_interfaces::msg::BuffTarget();
        target_msg.header.stamp = frame_stamp;
        target_msg.radius = detector_->get_radius(); // 获取能量机关半径
        target_msg.target_center_x = fan_center.x;
        target_msg.target_center_y = fan_center.y;
        target_msg.r_center_x = r_center.x;
        target_msg.r_center_y = r_center.y;
        target_msg.is_tracking = is_tracking_;
        target_pub_->publish(target_msg);

        // 显示调试信息
        if constexpr (IS_DEBUG)
        {
            // debug_frame 为空时回退显示原始帧，保证 imshow 始终有效输入
            const cv::Mat& display_frame = debug_frame.empty() ? frame : debug_frame;
            try
            {
                cv::imshow("visualization", display_frame);
            }
            catch (const cv::Exception& e)
            {
                RCLCPP_WARN(this->get_logger(), "imshow failed %s", e.what());
            }
            // 移到 try 外：保证无论 imshow 是否成功都会调用 waitKey
            // 否则 imshow 抛异常时 waitKey 被跳过，导致 OpenCV 事件循环不运行，窗口不显示
            cv::waitKey(1);
            
            // 发布调试图像到 ROS Topic
            if (!debug_frame.empty())
            {
                auto debug_img_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", debug_frame).toImageMsg();
                debug_img_msg->header.stamp = frame_stamp;
                debug_img_msg->header.frame_id = "camera";
                debug_image_pub_.publish(debug_img_msg);
            }
        }
    }

    // ROS2 通信
    rclcpp::Publisher<buff_interfaces::msg::BuffTarget>::SharedPtr target_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;

    // 检测器
    std::unique_ptr<BuffDetector> detector_;
    image_transport::Publisher debug_image_pub_;

    // 状态
    bool is_tracking_ = false;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BuffDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
