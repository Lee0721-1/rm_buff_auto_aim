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

        std::vector<int64_t> lower_hsv_vec = this->declare_parameter<std::vector<int64_t>>("detector.hsv_lower", {0, 40, 220});
        std::vector<int64_t> upper_hsv_vec = this->declare_parameter<std::vector<int64_t>>("detector.hsv_upper", {70, 255, 255});
        cv::Scalar lower_hsv(lower_hsv_vec[0], lower_hsv_vec[1], lower_hsv_vec[2]);
        cv::Scalar upper_hsv(upper_hsv_vec[0], upper_hsv_vec[1], upper_hsv_vec[2]);

        detector_ = std::make_unique<BuffDetector>(
            debug, conf_thresh, iou_thresh, inside_shade_rate, outside_shade_rate,
            dilate_kernel, max_lost_frame, model_path, lower_hsv, upper_hsv);

        img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
            std::bind(&BuffDetectorNode::imageCallback, this, std::placeholders::_1));

        target_pub_ = this->create_publisher<buff_interfaces::msg::BuffTarget>("/buff_target", 10);

        if (debug)
            debug_image_pub_ = image_transport::create_publisher(this, "/buff_detector/debug_image");
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
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
        if (!is_tracking_)
        {
            auto frame_clone = frame.clone();
            if (detector_->init(frame_clone))
            {
                is_tracking_ = true;
                RCLCPP_INFO(this->get_logger(), "Detector initialized.");
            }
        }

        if (is_tracking_)
        {
            if (!detector_->update(frame))
            {
                is_tracking_ = false;
                RCLCPP_WARN(this->get_logger(), "Detector lost target, re-initialization required.");
            }
        }

        // 发布目标信息
        auto target_msg = buff_interfaces::msg::BuffTarget();
        target_msg.header.stamp = frame_stamp;
        target_msg.radius = detector_->get_radius();
        cv::Point2f fan_center = detector_->get_first_target_blade_bbox().get_center_2f();
        cv::Point2f r_center = detector_->get_current_R_box().get_center_2f();
        target_msg.target_center_x = fan_center.x;
        target_msg.target_center_y = fan_center.y;
        target_msg.r_center_x = r_center.x;
        target_msg.r_center_y = r_center.y;
        target_msg.is_tracking = is_tracking_;
        target_pub_->publish(target_msg);

        // 调试显示
        cv::Mat debug_frame = detector_->get_debug_frame();
        if (debug_image_pub_.getTopic() != "")
        {
            if (!debug_frame.empty())
            {
                auto debug_img_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", debug_frame).toImageMsg();
                debug_img_msg->header.stamp = frame_stamp;
                debug_img_msg->header.frame_id = "camera";
                debug_image_pub_.publish(debug_img_msg);
            }
            const cv::Mat& display_frame = debug_frame.empty() ? frame : debug_frame;
            try {
                cv::imshow("visualization", display_frame);
                cv::waitKey(1);
            } catch (const cv::Exception& e) {
                RCLCPP_WARN(this->get_logger(), "imshow failed %s", e.what());
            }
        }
    }

    rclcpp::Publisher<buff_interfaces::msg::BuffTarget>::SharedPtr target_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
    std::unique_ptr<BuffDetector> detector_;
    image_transport::Publisher debug_image_pub_;
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

// 注册为组件
#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(BuffDetectorNode)

