#pragma once

#include <opencv2/opencv.hpp>



float euclidean_distance(const cv::Point2f& p1, const cv::Point2f& p2);
cv::Point2f rotate_point(const cv::Point2f& point, const cv::Point2f& center, float angle_rad);

enum class IoU_Type { IoU, GIoU, DIoU, CIoU };

enum class FanBladeState { target, unlighted, shotted, none };

enum class BuffType { small_buff, big_buff };

class RotationRectangle
{
private:
    std::vector<cv::Point2f> points_; // 四个顶点

public:
    RotationRectangle(const cv::Mat& points, const cv::Point2f& R_Box_center);          // 构造函数
    static cv::Point2f get_line_center(const cv::Point2f& p1, const cv::Point2f& p2);  // 计算线段中心点
    cv::Point2f get_center_2f() const;                                                  // 计算旋转矩形中心点
    cv::Point2i get_center_2i() const;                   // 计算旋转矩形中心点（整数类型）
    float get_width() const;                             // 计算旋转矩形宽度
    float get_height() const;                            // 计算旋转矩形高度
    float get_area() const;                              // 计算旋转矩形面积
    float get_dist_top(const cv::Point2f& point) const;    // 获取顶边中点到一点的距离
    float get_dist_bottom(const cv::Point2f& point) const; // 获取底边中点到一点的距离
    const std::vector<cv::Point2f>& get_points() const { return points_; } // 获取四个顶点
    void move_by(float x, float y);                                        // 根据x，y值进行移动
};

class BBox
{
private:
    cv::Point2f point_min_; // 左上角顶点
    cv::Point2f point_max_; // 右下角顶点

public:
    BBox(float x_min, float y_min, float x_max, float y_max) : point_min_(x_min, y_min), point_max_(x_max, y_max) {}
    BBox(const std::vector<BBox>& boxes);                   // 用多个BBox构造一个BBox（包围盒）
    BBox() : point_min_(0, 0), point_max_(0, 0) {}          // 默认构造函数定点坐标都设为0
    float operator&(const BBox& other) const;               // 计算两个BBox的重合部分面积(交集)
    float operator|(const BBox& other) const;               // 计算两个BBox的总面积(并集)
    float operator^(const BBox& other) const;               // 计算IoU(重合部分面积与总面积的比值)
    BBox boundof(const BBox& other) const;                  // 计算box和other的最小外接矩形
    cv::Point2f get_center_2f() const;                      // 计算BBox中心点
    cv::Point2i get_center_2i() const;                      // 计算BBox中心点（整数类型）
    cv::Point2f get_point_min() const;                      // 获取左上角顶点
    cv::Point2f get_point_max() const;                      // 获取右下角顶点
    float get_area() const;                                 // 计算BBox面积
    float get_width() const;                                // 计算BBox宽度
    float get_height() const;                               // 计算BBox高度
    BBox copy_bbox_by_center(cv::Point2f new_center) const; // 通过中心点复制BBox
    void move_by(float x, float y);                         // 根据x，y值进行移动
};

float IoU(const BBox& box1, const BBox& box2);
float GIoU(const BBox& box1, const BBox& box2);
float DIoU(const BBox& box1, const BBox& box2);
float CIoU(const BBox& box1, const BBox& box2);
std::vector<std::pair<BBox, float>> compare_by_IoU(
    const BBox& last_box, const std::vector<BBox>& boxes,
    const IoU_Type& iou_type = IoU_Type::IoU);

struct FanBlade {
    BBox box;
    FanBladeState state = FanBladeState::none;
    int id = -1;
};

class BuffDetector {
public:
    // 构造函数接收所有可调参数
    BuffDetector(
        bool debug,
        double conf_thresh,
        double iou_thresh,
        double inside_shade_rate,
        double outside_shade_rate,
        int dilate_kernel,
        int max_lost_frame,
        const std::string& model_path,
        const cv::Scalar& lower_hsv,
        const cv::Scalar& upper_hsv
    );

    bool init(cv::Mat& frame);
    bool update(cv::Mat& frame);
    BBox get_first_target_blade_bbox() const;
    const std::vector<FanBlade>& get_target_fan_blades() const;
    BBox get_current_R_box() const;
    float get_radius() const;
    const cv::Mat& get_debug_frame() const;

private:
    // 参数成员
    bool debug_;
    double conf_thresh_;
    double iou_thresh_;
    double inside_shade_rate_;
    double outside_shade_rate_;
    int dilate_kernel_;
    int max_lost_frame_;
    std::string model_path_;
    cv::Scalar lower_hsv_;
    cv::Scalar upper_hsv_;

    // 状态成员
    BBox current_R_box_;
    BBox last_R_box_;
    std::vector<FanBlade> target_blades_;
    std::vector<FanBlade> blade_list_;
    float buff_radius_ = 0.0f;
    int lighted_blade_num_ = 0;
    int lost_frame_count_ = 0;
    cv::Mat debug_frame_;
    BuffType buff_type_ = BuffType::small_buff;

    // 内部方法
    bool detect_by_yolo(cv::Mat& frame);
    bool preprocess_image(cv::Mat& frame);
    bool update_R_box(cv::Mat& image, bool is_init = false);
    bool update_fan_blades(cv::Mat& image);
};