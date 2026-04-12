#include "buff_detector/detector.hpp"
#include "buff_detector/yolo_inference.hpp"

#include <iostream>
#include <unordered_map>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>

// 构造函数：初始化所有参数成员
BuffDetector::BuffDetector(
    bool debug,
    double conf_thresh,
    double iou_thresh,
    double inside_shade_rate,
    double outside_shade_rate,
    int dilate_kernel,
    int max_lost_frame,
    const std::string& model_path,
    const cv::Scalar& lower_hsv,
    const cv::Scalar& upper_hsv)
    : debug_(debug),
      conf_thresh_(conf_thresh),
      iou_thresh_(iou_thresh),
      inside_shade_rate_(inside_shade_rate),
      outside_shade_rate_(outside_shade_rate),
      dilate_kernel_(dilate_kernel),
      max_lost_frame_(max_lost_frame),
      model_path_(model_path),
      lower_hsv_(lower_hsv),
      upper_hsv_(upper_hsv)
{
}

RotationRectangle::RotationRectangle(const cv::Mat& points, const cv::Point2f& R_Box_center)
{
    // 存储点和对应距离的pair
    std::vector<std::pair<cv::Point2f, float>> points_with_distances = {
        {points.at<cv::Point2f>(0), 0},
        {points.at<cv::Point2f>(1), 0},
        {points.at<cv::Point2f>(2), 0},
        {points.at<cv::Point2f>(3), 0}};

    cv::Point2f p1, p2, p3, p4;

    // 计算每个点到R标的距离
    for (auto& point : points_with_distances)
    {
        point.second = euclidean_distance(point.first, R_Box_center);
    }

    // 按到R标的距离从大到小排序
    std::sort(
        points_with_distances.begin(), points_with_distances.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    // p1 p2 是距离R标最远的两个点
    p1 = points_with_distances[0].first;
    p2 = points_with_distances[1].first;

    // p3为距离p1最远的，剩下的就是p4
    if (euclidean_distance(points_with_distances[2].first, p1) > euclidean_distance(points_with_distances[3].first, p1))
    {
        p3 = points_with_distances[2].first;
        p4 = points_with_distances[3].first;
    }
    else
    {
        p3 = points_with_distances[3].first;
        p4 = points_with_distances[2].first;
    }

    points_ = {p1, p2, p3, p4};
}

cv::Point2f RotationRectangle::get_line_center(const cv::Point2f& p1, const cv::Point2f& p2)
{
    return cv::Point2f((p1.x + p2.x) / 2.0f, (p1.y + p2.y) / 2.0f);
}

cv::Point2f RotationRectangle::get_center_2f() const
{
    float x = (points_[0].x + points_[1].x + points_[2].x + points_[3].x) / 4.0f;
    float y = (points_[0].y + points_[1].y + points_[2].y + points_[3].y) / 4.0f;
    return cv::Point2f(x, y);
}

cv::Point2i RotationRectangle::get_center_2i() const
{
    int x = static_cast<int>((points_[0].x + points_[1].x + points_[2].x + points_[3].x) / 4.0f);
    int y = static_cast<int>((points_[0].y + points_[1].y + points_[2].y + points_[3].y) / 4.0f);
    return cv::Point2i(x, y);
}

float RotationRectangle::get_width() const
{
    return euclidean_distance(points_[0], points_[1]);
}

float RotationRectangle::get_height() const
{
    return euclidean_distance(points_[1], points_[2]);
}

float RotationRectangle::get_area() const
{
    return get_width() * get_height();
}

float RotationRectangle::get_dist_top(const cv::Point2f& point) const
{
    //  定义p1p2为顶边，即距离R标最远的为顶边
    cv::Point2f top = get_line_center(points_[0], points_[1]);
    return euclidean_distance(top, point);
}

float RotationRectangle::get_dist_bottom(const cv::Point2f& point) const
{
    //  定义p3p4为底边，即距离R标最近的为底边
    cv::Point2f bottom = get_line_center(points_[2], points_[3]);
    return euclidean_distance(bottom, point);
}

void RotationRectangle::move_by(float x, float y)
{
    cv::Point2f changed(x, y);
    for (auto& point : points_)
        point += changed;
}

// BBox 实现
BBox::BBox(const std::vector<BBox>& boxes)
{
    // 合并多个检测框为一个大框
    float x_min = std::numeric_limits<float>::max();
    float y_min = std::numeric_limits<float>::max();
    float x_max = std::numeric_limits<float>::lowest();
    float y_max = std::numeric_limits<float>::lowest();

    for (const auto& box : boxes)
    {
        x_min = std::min(x_min, box.get_point_min().x);
        y_min = std::min(y_min, box.get_point_min().y);
        x_max = std::max(x_max, box.get_point_max().x);
        y_max = std::max(y_max, box.get_point_max().y);
    }
    point_min_ = cv::Point2f(x_min, y_min);
    point_max_ = cv::Point2f(x_max, y_max);
}

float BBox::operator&(const BBox& other) const
{
    // 计算交集区域
    float xmax = std::min(point_max_.x, other.point_max_.x);
    float ymax = std::min(point_max_.y, other.point_max_.y);
    float xmin = std::max(point_min_.x, other.point_min_.x);
    float ymin = std::max(point_min_.y, other.point_min_.y);
    BBox cross_box(xmin, ymin, xmax, ymax);
    if (cross_box.get_width() <= 0 || cross_box.get_height() <= 0)
        return 0.0;
    return cross_box.get_area();
}

float BBox::operator|(const BBox& other) const
{
    float cross_area = *this & other;
    float union_area = this->get_area() + other.get_area() - cross_area;
    return union_area;
}

float BBox::operator^(const BBox& other) const
{
    float cross_area = *this & other;
    float union_area = *this | other;
    return cross_area / union_area;
}

BBox BBox::boundof(const BBox& other) const
{
    float xmin = std::min(point_min_.x, other.point_min_.x);
    float ymin = std::min(point_min_.y, other.point_min_.y);
    float xmax = std::max(point_max_.x, other.point_max_.x);
    float ymax = std::max(point_max_.y, other.point_max_.y);
    return BBox(xmin, ymin, xmax, ymax);
}

cv::Point2f BBox::get_center_2f() const
{
    return cv::Point2f((point_min_.x + point_max_.x) / 2.0f, (point_min_.y + point_max_.y) / 2.0f);
}

cv::Point2i BBox::get_center_2i() const
{
    return cv::Point2i(static_cast<int>((point_min_.x + point_max_.x) / 2.0f),
                       static_cast<int>((point_min_.y + point_max_.y) / 2.0f));
}

cv::Point2f BBox::get_point_min() const { return point_min_; }
cv::Point2f BBox::get_point_max() const { return point_max_; }

float BBox::get_area() const { return get_width() * get_height(); }
float BBox::get_width() const { return point_max_.x - point_min_.x; }
float BBox::get_height() const { return point_max_.y - point_min_.y; }

BBox BBox::copy_bbox_by_center(cv::Point2f new_center) const
{
    float half_width = get_width() / 2.0f;
    float half_height = get_height() / 2.0f;
    return BBox(new_center.x - half_width, new_center.y - half_height,
                new_center.x + half_width, new_center.y + half_height);
}

void BBox::move_by(float x, float y)
{
    cv::Point2f changed(x, y);
    point_max_ += changed;
    point_min_ += changed;
}

// YOLO 检测初始化
bool BuffDetector::detect_by_yolo(cv::Mat& frame)
{
    static YOLO_V8 yolo_detector;
    static bool loaded = false;
    if (!loaded)
    {
        // yolo参数
        DL_INIT_PARAM yolo_param;
        yolo_detector.classes = {"fan_blade", "R"};
        yolo_param.rectConfidenceThreshold = static_cast<float>(conf_thresh_);
        yolo_param.iouThreshold = static_cast<float>(iou_thresh_);
        yolo_param.modelPath = model_path_;
        yolo_param.imgSize = {480, 640};
        yolo_param.modelType = YOLO_DETECT_V8;
        yolo_param.cudaEnable = false;
        yolo_detector.CreateSession(yolo_param);
        loaded = true;
    }

    // 进行yolo检测
    std::vector<DL_RESULT> output_list;
    yolo_detector.RunSession(frame, output_list);

    // 检测结果和信度列表
    std::vector<std::pair<BBox, float>> R_detections;
    std::vector<std::pair<BBox, float>> fan_detections;

    for (const auto& output : output_list)
    {
        // 边界检查，确保 ROI 在图像范围内
        int x = std::max(0, output.box.x);
        int y = std::max(0, output.box.y);
        int w = std::min(output.box.width, frame.cols - x);
        int h = std::min(output.box.height, frame.rows - y);
        if (w <= 0 || h <= 0) continue;

        BBox box(static_cast<float>(x), static_cast<float>(y),
                 static_cast<float>(x + w), static_cast<float>(y + h));

        if (output.classId == 1)
            fan_detections.push_back({box, output.confidence});
        else if (output.classId == 0)
            R_detections.push_back({box, output.confidence});
    }

    if (!fan_detections.empty() && !R_detections.empty())
    {
        auto best_fan = std::max_element(fan_detections.begin(), fan_detections.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        target_blades_.clear();
        target_blades_.push_back(FanBlade{BBox(best_fan->first), FanBladeState::target, -1});

        auto best_R = std::max_element(R_detections.begin(), R_detections.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        current_R_box_ = BBox(best_R->first);
        return true;
    }
    else
    {
        if (debug_)
            RCLCPP_WARN(rclcpp::get_logger("BuffDetector"),
                        "YOLO init failed (got %zu fans, %zu Rs)",
                        fan_detections.size(), R_detections.size());
        return false;
    }
}

bool BuffDetector::preprocess_image(cv::Mat& frame)
{
    // 转换为HSV颜色空间
    cv::cvtColor(frame, frame, cv::COLOR_BGR2HSV);
    cv::inRange(frame, lower_hsv_, upper_hsv_, frame);
    cv::Mat kernel(dilate_kernel_, dilate_kernel_, CV_8U, cv::Scalar(1));
    cv::dilate(frame, frame, kernel);

    if (debug_)
        cv::imshow("preprocessed_image", frame);
    return true;
}

bool BuffDetector::update_fan_blades(cv::Mat& image)
{
    // 创建遮罩遮挡无关部分
    const cv::Mat mask = cv::Mat::ones(image.size(), CV_8UC1);
    cv::circle(mask, current_R_box_.get_center_2i(),
               static_cast<int>(buff_radius_ * outside_shade_rate_), cv::Scalar(0), -1);
    image.setTo(0, mask);
    cv::circle(image, current_R_box_.get_center_2i(),
               static_cast<int>(buff_radius_ * inside_shade_rate_), cv::Scalar(0), -1);

    if (debug_)
        cv::imshow("FAN_BLADE_ROI", image);

    // 轮廓检测
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<BBox> detected_bounding_list;
    std::vector<FanBlade> lighted_fan_blade_list;

    // 遍历所有轮廓，筛选出可能的扇叶
    for (const auto& contour : contours)
    {
        // 将轮廓拟合为最小外接旋转矩形
        cv::Mat rect_pts;
        cv::boxPoints(cv::minAreaRect(contour), rect_pts);
        RotationRectangle rotation_rect(rect_pts, current_R_box_.get_center_2f());

        // 筛选条件：底部到R标距离、顶部到R标距离、面积要求
        if (0.4f * buff_radius_ < rotation_rect.get_dist_bottom(current_R_box_.get_center_2f()) &&
            rotation_rect.get_dist_top(current_R_box_.get_center_2f()) < 1.5f * buff_radius_ &&
            rotation_rect.get_area() > 2.0f * current_R_box_.get_area())
        {
            // 将备选扇叶轮廓转换为BBox检测框
            cv::Rect rect = cv::boundingRect(contour);
            if (debug_)
                cv::rectangle(debug_frame_, rect, cv::Scalar(128, 0, 128), 2);
            detected_bounding_list.push_back(BBox(
                static_cast<float>(rect.x), static_cast<float>(rect.y),
                static_cast<float>(rect.x + rect.width), static_cast<float>(rect.y + rect.height)));
        }
    }

    // 初始化分支
    if (blade_list_[0].box.get_area() == 0 && detected_bounding_list.size() == 2)
    {
        // big buff第一帧初始化：检测到2个扇叶
        buff_type_ = BuffType::big_buff;
        lighted_fan_blade_list.push_back(FanBlade{detected_bounding_list[0], FanBladeState::target, 0});
        lighted_fan_blade_list.push_back(FanBlade{detected_bounding_list[1], FanBladeState::target, 1});
        RCLCPP_INFO(rclcpp::get_logger("BuffDetector"), "big buff initialized");
    }
    else if (blade_list_[0].box.get_area() == 0 && detected_bounding_list.size() == 1)
    {
        // small buff第一帧初始化：检测到1个扇叶
        buff_type_ = BuffType::small_buff;
        lighted_fan_blade_list.push_back(FanBlade{detected_bounding_list[0], FanBladeState::target, 0});
        RCLCPP_INFO(rclcpp::get_logger("BuffDetector"), "small buff initialized");
    }
    else if (blade_list_[0].box.get_area() == 0)
    {
        // 初始化失败：检测到的扇叶数量异常
        RCLCPP_WARN(rclcpp::get_logger("BuffDetector"), "Failed to initialize fan blades");
        return false;
    }

    // 匹配跟踪逻辑（保持原有）
    auto last_fan_blade_list = blade_list_;
    cv::Point2f offset = current_R_box_.get_center_2f() - last_R_box_.get_center_2f();

    // 遍历上一帧扇叶列表，匹配当前帧检测框
    for (auto& last_fan_blade : last_fan_blade_list)
    {
        // 将上一帧扇叶框移动到当前R标位置
        last_fan_blade.box.move_by(offset.x, offset.y);
        std::vector<BBox> matched_boxes;
        for (const auto& detected_bounding : detected_bounding_list)
        {
            if (IoU(last_fan_blade.box, detected_bounding) > 0)
                matched_boxes.push_back(detected_bounding);
        }

        if (matched_boxes.size() == 1)
            lighted_fan_blade_list.push_back(FanBlade{matched_boxes[0], FanBladeState::none, last_fan_blade.id});
        else if (matched_boxes.size() >= 2)
            lighted_fan_blade_list.push_back(FanBlade{BBox(matched_boxes), FanBladeState::none, last_fan_blade.id});
    }

    // 状态更新（保持原有，略作参数替换）
    if (lighted_fan_blade_list.empty())
    {
        if (debug_) RCLCPP_WARN(rclcpp::get_logger("BuffDetector"), "No fan blade is lighted.");
        return false;
    }
    else if (buff_type_ == BuffType::big_buff)
    {
        if (lighted_fan_blade_list.size() == 1 || lighted_fan_blade_list.size() == 2)
        {
            for (size_t i = 0; i < 5; i++)
            {
                blade_list_[i].state = FanBladeState::unlighted;
                blade_list_[i].id = static_cast<int>(i);
            }
            for (size_t i = 0; i < lighted_fan_blade_list.size(); ++i)
            {
                int id = lighted_fan_blade_list[i].id;
                blade_list_[id] = lighted_fan_blade_list[i];
                blade_list_[id].state = FanBladeState::target;
            }
        }
        else if (lighted_fan_blade_list.size() >= 3)
        {
            if (debug_) RCLCPP_DEBUG(rclcpp::get_logger("BuffDetector"),
                                     "Detected %zu lighted blades (>=3), keeping previous state",
                                     lighted_fan_blade_list.size());
            for (auto& lighted_fan_blade : lighted_fan_blade_list)
            {
                int id = lighted_fan_blade.id;
                if (id >= 0 && id < 5)
                {
                    lighted_fan_blade.state = blade_list_[id].state;
                    blade_list_[id].box = lighted_fan_blade.box;
                }
            }
        }
        else
        {
            if (debug_) RCLCPP_WARN(rclcpp::get_logger("BuffDetector"), "Unexpected blade count: %zu", lighted_fan_blade_list.size());
            return false;
        }
    }
    else if (buff_type_ == BuffType::small_buff)
    {
        if (static_cast<int>(lighted_fan_blade_list.size()) > lighted_blade_num_)
        {
            for (auto& fan_blade : lighted_fan_blade_list)
            {
                int id = fan_blade.id;
                if (id >= 0 && id < 5)
                {
                    if (blade_list_[id].state == FanBladeState::target)
                    {
                        fan_blade.state = FanBladeState::shotted;
                        blade_list_[id] = fan_blade;
                    }
                    else if (blade_list_[id].state == FanBladeState::unlighted)
                    {
                        fan_blade.state = FanBladeState::target;
                        blade_list_[id] = fan_blade;
                    }
                }
            }
        }
        else if (static_cast<int>(lighted_fan_blade_list.size()) == lighted_blade_num_)
        {
            for (const auto& fan_blade : lighted_fan_blade_list)
            {
                int id = fan_blade.id;
                if (id >= 0 && id < 5)
                    blade_list_[id].box = fan_blade.box;
            }
        }
        else
        {
            if (debug_) RCLCPP_WARN(rclcpp::get_logger("BuffDetector"), "Unexpected blade count: %zu", lighted_fan_blade_list.size());
            return false;
        }
    }

    // 同步 target 列表
    target_blades_.clear();
    for (const auto& fan_blade : blade_list_)
        if (fan_blade.state == FanBladeState::target)
            target_blades_.push_back(fan_blade);

    if (debug_)
    {
        for (const auto& fan_blade : target_blades_)
            cv::rectangle(debug_frame_, fan_blade.box.get_point_min(), fan_blade.box.get_point_max(), cv::Scalar(0, 255, 0), 3);
    }

    lighted_blade_num_ = static_cast<int>(lighted_fan_blade_list.size());

    if (target_blades_.empty())
    {
        RCLCPP_WARN(rclcpp::get_logger("BuffDetector"), "No target fan blade tracked.");
        return false;
    }

    const FanBlade& ref_target = target_blades_.front();
    for (auto& fan_blade : blade_list_)
    {
        float angle = 2.0f * M_PI / 5.0f * (fan_blade.id - ref_target.id);
        fan_blade.box = ref_target.box.copy_bbox_by_center(
            rotate_point(ref_target.box.get_center_2f(), current_R_box_.get_center_2f(), angle));
    }
    return true;
}

bool BuffDetector::update_R_box(cv::Mat& image, bool is_init)
{
    auto last_R_box = current_R_box_;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<BBox> candidate_boxes;

    for (const auto& contour : contours)
    {
        cv::Rect rect = cv::boundingRect(contour);
        BBox rtn_box(static_cast<float>(rect.x), static_cast<float>(rect.y),
                     static_cast<float>(rect.x + rect.width), static_cast<float>(rect.y + rect.height));
        float distance_to_last_center = euclidean_distance(rtn_box.get_center_2f(), last_R_box.get_center_2f());

        if (is_init)
        {
            if (distance_to_last_center < 0.1f * buff_radius_)
                candidate_boxes.push_back(rtn_box);
        }
        else
        {
            if (rtn_box.get_width() > last_R_box.get_width() * 0.8f &&
                rtn_box.get_width() < last_R_box.get_width() * 1.2f &&
                rtn_box.get_height() > last_R_box.get_height() * 0.8f &&
                rtn_box.get_height() < last_R_box.get_height() * 1.2f &&
                distance_to_last_center < 2.0f * buff_radius_)
            {
                candidate_boxes.push_back(rtn_box);
            }
        }
    }

    auto boxes_sorted_by_iou = compare_by_IoU(last_R_box, candidate_boxes, IoU_Type::CIoU);
    if (boxes_sorted_by_iou.empty())
    {
        RCLCPP_WARN(rclcpp::get_logger("BuffDetector"), "No valid R box detected");
        return false;
    }

    last_R_box_ = last_R_box;
    current_R_box_ = boxes_sorted_by_iou[0].first;
    return true;
}

bool BuffDetector::init(cv::Mat& frame)
{
    if (debug_) debug_frame_ = frame.clone();

    if (!detect_by_yolo(frame)) return false;

    blade_list_.clear();
    for (size_t i = 0; i < 5; i++)
        blade_list_.push_back(FanBlade{BBox(), FanBladeState::unlighted, static_cast<int>(i)});

    last_R_box_ = current_R_box_;
    lighted_blade_num_ = 0;

    if (target_blades_.empty())
    {
        RCLCPP_ERROR(rclcpp::get_logger("BuffDetector"), "No target fan blade after YOLO init.");
        return false;
    }
    buff_radius_ = euclidean_distance(current_R_box_.get_center_2f(), target_blades_.front().box.get_center_2f());

    try
    {
        preprocess_image(frame);
    }
    catch (const cv::Exception& e)
    {
        RCLCPP_ERROR(rclcpp::get_logger("BuffDetector"), "OpenCV exception: %s", e.what());
        return false;
    }
    return update_R_box(frame, true);
}

bool BuffDetector::update(cv::Mat& frame)
{
    if (debug_) debug_frame_ = frame.clone();

    try
    {
        preprocess_image(frame);
    }
    catch (const cv::Exception& e)
    {
        RCLCPP_ERROR(rclcpp::get_logger("BuffDetector"), "OpenCV exception: %s", e.what());
        return false;
    }

    bool r_box_ok = update_R_box(frame);
    bool fan_blades_ok = update_fan_blades(frame);

    if (r_box_ok && fan_blades_ok)
        lost_frame_count_ = 0;
    else
    {
        lost_frame_count_++;
        if (debug_ && lost_frame_count_ <= 3)
            RCLCPP_DEBUG(rclcpp::get_logger("BuffDetector"),
                         "Tracking failed (R_box: %s, Fan_blades: %s, lost_count: %d/%d)",
                         r_box_ok ? "OK" : "FAIL", fan_blades_ok ? "OK" : "FAIL",
                         lost_frame_count_, max_lost_frame_);
    }

    if (debug_)
    {
        cv::rectangle(debug_frame_, current_R_box_.get_point_min(), current_R_box_.get_point_max(), cv::Scalar(255, 255, 0), 2);
        for (const auto& fan_blade : blade_list_)
        {
            if (fan_blade.state == FanBladeState::unlighted)
                cv::rectangle(debug_frame_, fan_blade.box.get_point_min(), fan_blade.box.get_point_max(), cv::Scalar(0, 0, 255), 2);
            cv::putText(debug_frame_,
                        "id=" + std::to_string(fan_blade.id) + "|" +
                        (fan_blade.state == FanBladeState::target ? "target" :
                         (fan_blade.state == FanBladeState::shotted ? "shotted" : "unlighted")),
                        fan_blade.box.get_center_2i(), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }
    }

    if (lost_frame_count_ >= max_lost_frame_)
    {
        if (debug_) RCLCPP_WARN(rclcpp::get_logger("BuffDetector"),
                                "Lost target for %d consecutive frames, re-initialization required",
                                lost_frame_count_);
        return false;
    }
    return true;
}

// 全局辅助函数
float euclidean_distance(const cv::Point2f& p1, const cv::Point2f& p2)
{
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

float IoU(const BBox& box1, const BBox& box2) { return box1 ^ box2; }

float GIoU(const BBox& box1, const BBox& box2)
{
    float bound_area = box1.boundof(box2).get_area();
    float union_area = box1 | box2;
    return IoU(box1, box2) - (bound_area - union_area) / bound_area;
}

float DIoU(const BBox& box1, const BBox& box2)
{
    float center_distance = euclidean_distance(box1.get_center_2f(), box2.get_center_2f());
    float bound_diagonal = euclidean_distance(box1.boundof(box2).get_point_min(), box1.boundof(box2).get_point_max());
    return IoU(box1, box2) - (center_distance * center_distance) / (bound_diagonal * bound_diagonal);
}

float CIoU(const BBox& box1, const BBox& box2)
{
    constexpr float kEps = 1e-6f;
    const float w1 = box1.get_width(), h1 = box1.get_height();
    const float w2 = box2.get_width(), h2 = box2.get_height();
    const float t = std::atan((w1 + kEps) / (h1 + kEps)) - std::atan((w2 + kEps) / (h2 + kEps));
    const float v = (4.0f / (M_PI * M_PI)) * t * t;
    const float iou = IoU(box1, box2);
    const float alpha = v / (std::max(1.0f - iou + v, kEps));
    return DIoU(box1, box2) - alpha * v;
}

std::vector<std::pair<BBox, float>> compare_by_IoU(
    const BBox& last_box, const std::vector<BBox>& boxes, const IoU_Type& iou_type)
{
    std::vector<std::pair<BBox, float>> targets;
    if (boxes.empty()) return targets;

    for (size_t i = 0; i < boxes.size(); i++)
    {
        float score = 0.0f;
        switch (iou_type)
        {
            case IoU_Type::IoU:  score = IoU(last_box, boxes[i]); break;
            case IoU_Type::GIoU: score = GIoU(last_box, boxes[i]); break;
            case IoU_Type::DIoU: score = DIoU(last_box, boxes[i]); break;
            case IoU_Type::CIoU: score = CIoU(last_box, boxes[i]); break;
        }
        targets.push_back({boxes[i], score});
    }

    std::sort(targets.begin(), targets.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    return targets;
}

cv::Point2f rotate_point(const cv::Point2f& point, const cv::Point2f& center, float angle_rad)
{
    const float cos = std::cos(angle_rad), sin = std::sin(angle_rad);
    const float x = point.x - center.x, y = point.y - center.y;
    return {x * cos - y * sin + center.x, x * sin + y * cos + center.y};
}