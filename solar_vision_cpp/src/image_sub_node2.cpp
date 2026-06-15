#ifndef VISION_DETECT_NODE_HPP_
#define VISION_DETECT_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <cmath>

// 自定义消息头文件（请确保 CMakeLists.txt 和 package.xml 已正确配置依赖）
#include "vision_detect_msgs/msg/vision_localization.hpp"
#include "vision_detect_msgs/msg/heading_error.hpp"
#include "vision_detect_msgs/msg/edge_observation.hpp"
#include "vision_detect_msgs/msg/vision_status.hpp"

// 边缘信息结构体
struct EdgeInfo {
    bool visible = false;
    std::string type = "";
    std::string side = "";
    std::string safety_level = "EDGE_CLEAR";
    double distance_cm = -1.0;
};

// 视觉流水线输出结果结构体
struct ProcessResult {
    cv::Mat output_img;
    cv::Mat closed_edges;
    std::optional<double> heading_error;
    int cross_event = 0;
    EdgeInfo edge_info;
};

// ==========================================
// 纯算法类：光伏板网格线与边缘检测器
// ==========================================
class PVLineDetector {
public:
    PVLineDetector();
    ~PVLineDetector() = default;

    ProcessResult process_frame(const cv::Mat& img);

    // 物理参数
    double camera_height_cm = 15.0;  // 相机距离光伏板面的高度 (cm)
    double camera_pitch_deg = 10.0;  // 相机前倾俯仰角 (度)
    
    // 相机内参 (经验值占位)
    double fx = 400.0;
    double fy = 400.0;
    double cx = 320.0;
    double cy = 240.0;

    // 常量定义
    static constexpr const char* EDGE_TYPE_BLOCK = "block_edge"; // 大板悬崖
    static constexpr const char* EDGE_TYPE_CELL = "cell_edge";   // 小板缝隙

    static constexpr const char* SAFETY_STOP = "EDGE_STOP";
    static constexpr const char* SAFETY_WARN = "EDGE_CAUTION";
    static constexpr const char* SAFETY_SAFE = "EDGE_CLEAR";

private:
    cv::Ptr<cv::CLAHE> clahe_;
    int tripwire_y_;
    int trigger_zone_;
    bool line_is_crossing_;

    double pixel_to_distance_cm(double pixel_val, char axis);
    int detect_line_crossing(const cv::Mat& blurred_gray, const std::vector<cv::Vec4i>& lines, cv::Mat& output_img);
    std::optional<double> extract_heading_error(const std::vector<cv::Vec4i>& lines);
    EdgeInfo detect_edge_state(const cv::Mat& blurred_gray, const std::vector<cv::Vec4i>& lines, cv::Mat& output_img);

    // 内部数据结构，用于直线聚类
    struct RawLine {
        double y_center;
        double k;
    };
    struct MergedLine {
        double y_center;
        double sum_k;
        int count;
    };
};

// ==========================================
// ROS 2 节点类：视觉感知节点
// ==========================================
class VisionDetectNode : public rclcpp::Node {
public:
    VisionDetectNode();
    ~VisionDetectNode() = default;

private:
    // 回调函数
    void video_loop_callback();
    void publish_localization();
    void cam_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    // void image_callback(const sensor_msgs::msg::Image::SharedPtr msg); // 预留给真实相机的回调

    // 参数变量
    double cam_offset_x_;
    double cam_offset_y_;
    double cam_yaw_offset_;
    int vision_timeout_;
    bool enable_debug_;

    // 状态机变量
    std::string current_state_;
    int block_id_;
    int cell_row_;
    int cell_col_;
    int inner_row_;
    int inner_col_;

    std::string travel_axis_;
    int travel_sign_;

    // 核心对象
    PVLineDetector line_detector_;
    cv::VideoCapture cap_;
    std::string video_path_;

    // ROS 2 通信接口
    rclcpp::Publisher<vision_detect_msgs::msg::VisionLocalization>::SharedPtr pub_localization_;
    rclcpp::Publisher<vision_detect_msgs::msg::HeadingError>::SharedPtr pub_heading_error_;
    rclcpp::Publisher<vision_detect_msgs::msg::EdgeObservation>::SharedPtr pub_edge_obs_;
    rclcpp::Publisher<vision_detect_msgs::msg::VisionStatus>::SharedPtr pub_status_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_debug_;

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_cam_info_;
    // rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_image_;

    rclcpp::TimerBase::SharedPtr timer_video_;
    rclcpp::TimerBase::SharedPtr timer_publish_loc_;
};

#endif  // VISION_DETECT_NODE_HPP_

// ==========================================
// PVLineDetector 实现
// ==========================================
PVLineDetector::PVLineDetector() 
    : tripwire_y_(240), trigger_zone_(5), line_is_crossing_(false) {
    // 初始化 CLAHE，对抗光伏板局部反光
    clahe_ = cv::createCLAHE(2.0, cv::Size(8, 8));
}

ProcessResult PVLineDetector::process_frame(const cv::Mat& img) {
    ProcessResult result;
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(640, 480));
    result.output_img = resized_img.clone(); // 严格使用 clone() 深拷贝

    // ==========================================
    // 第一阶段：图像预处理 (降噪与边缘强化)
    // ==========================================
    cv::Mat gray, enhanced_gray, blurred;
    cv::cvtColor(resized_img, gray, cv::COLOR_BGR2GRAY);
    clahe_->apply(gray, enhanced_gray);

    // 双边滤波降噪
    cv::bilateralFilter(enhanced_gray, blurred, 9, 75, 75);

    // 动态自适应 Canny 边缘检测
    // 使用 cv::mean 或将图像转为 1D 求 median (此处使用均值近似 median 以追求极速，或手动求中位数)
    cv::Scalar mean_scalar = cv::mean(blurred);
    double v = mean_scalar[0];
    double sigma = 0.33;
    int lower = std::clamp(static_cast<int>((1.0 - sigma) * v), 0, 255);
    int upper = std::clamp(static_cast<int>((1.0 + sigma) * v), 0, 255);
    
    cv::Mat edges;
    cv::Canny(blurred, edges, lower, upper);

    // 形态学闭运算
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    cv::morphologyEx(edges, result.closed_edges, cv::MORPH_CLOSE, kernel);

    // ==========================================
    // 第二阶段：几何特征提取 (寻找工业级直线)
    // ==========================================
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(result.closed_edges, lines, 1, CV_PI / 180.0, 80, 100, 40);

    // ==========================================
    // 第三阶段：状态机逻辑运算
    // ==========================================
    if (!lines.empty()) {
        result.heading_error = extract_heading_error(lines);
        result.cross_event = detect_line_crossing(blurred, lines, result.output_img);

        if (result.heading_error.has_value()) {
            double deg = result.heading_error.value() * 180.0 / CV_PI;
            char text[50];
            snprintf(text, sizeof(text), "Yaw Err: %.2f deg", deg);
            cv::putText(result.output_img, text, cv::Point(20, 40), 
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        }
    }

    result.edge_info = detect_edge_state(blurred, lines, result.output_img);

    return result;
}

double PVLineDetector::pixel_to_distance_cm(double pixel_val, char axis) {
    double pitch_rad = camera_pitch_deg * CV_PI / 180.0;
    
    if (axis == 'y') {
        double alpha = std::atan((cy - pixel_val) / fy);
        double distance = camera_height_cm * std::tan(pitch_rad + alpha);
        return std::round(distance * 100.0) / 100.0;
    } else if (axis == 'x') {
        double center_dist = camera_height_cm * std::tan(pitch_rad);
        double beta = std::atan(std::abs(pixel_val - cx) / fx);
        double slant_dist = std::sqrt(camera_height_cm * camera_height_cm + center_dist * center_dist);
        double lateral_distance = slant_dist * std::tan(beta);
        return std::round(lateral_distance * 100.0) / 100.0;
    }
    return -1.0;
}

int PVLineDetector::detect_line_crossing(const cv::Mat& blurred_gray, const std::vector<cv::Vec4i>& lines, cv::Mat& output_img) {
    int height = output_img.rows;
    int width = output_img.cols;
    int img_cx = width / 2;
    tripwire_y_ = height / 2;
    trigger_zone_ = 5;

    cv::line(output_img, cv::Point(0, tripwire_y_), cv::Point(width, tripwire_y_), cv::Scalar(255, 0, 0), 2);

    if (lines.empty()) return 0;

    std::vector<RawLine> raw_lines;
    for (const auto& line : lines) {
        double x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
        double dx = std::abs(x2 - x1);
        double dy = std::abs(y2 - y1);
        
        if (dx > dy && x1 != x2) {
            double k = (y2 - y1) / (x2 - x1);
            double b = y1 - k * x1;
            double y_center = k * img_cx + b;
            raw_lines.push_back({y_center, k});
        }
    }

    if (raw_lines.empty()) return 0;

    // 聚类：按 y_center 排序
    std::sort(raw_lines.begin(), raw_lines.end(), [](const RawLine& a, const RawLine& b) {
        return a.y_center < b.y_center;
    });

    std::vector<MergedLine> merged_lines;
    for (const auto& rl : raw_lines) {
        if (merged_lines.empty() || std::abs(rl.y_center - merged_lines.back().y_center) > 40.0) {
            merged_lines.push_back({rl.y_center, rl.k, 1});
        } else {
            auto& back = merged_lines.back();
            back.y_center = (back.y_center * back.count + rl.y_center) / (back.count + 1);
            back.sum_k += rl.k;
            back.count += 1;
        }
    }

    // ==========================================
    // 核心升级：倾斜自适应 CT 扫描 (手动双重循环代替 Numpy)
    // ==========================================
    std::vector<double> valid_y_centers;
    int x_start = std::max(0, img_cx - 50);
    int x_end = std::min(width, img_cx + 50);
    int sample_width = x_end - x_start;

    if (sample_width <= 0) return 0;

    for (const auto& m : merged_lines) {
        double y_center = m.y_center;
        double k = m.sum_k / m.count;
        int y_int = static_cast<int>(y_center);
        
        double local_max = -1.0;
        double local_min = 300.0; // 灰度最大255
        std::vector<double> slice_roi(41, 0.0);

        for (int offset = -20; offset <= 20; ++offset) {
            double sum_val = 0.0;
            for (int x = x_start; x < x_end; ++x) {
                // 严谨的类型转换和边界钳制 (std::clamp)
                int y = std::clamp(static_cast<int>(k * (x - img_cx) + y_center + offset), 0, height - 1);
                sum_val += blurred_gray.at<uchar>(y, x);
            }
            double mean_val = sum_val / sample_width;
            slice_roi[offset + 20] = mean_val;
            
            if (mean_val > local_max) local_max = mean_val;
            if (mean_val < local_min) local_min = mean_val;
        }

        double contrast = local_max - local_min;
        int left_y = static_cast<int>(k * (0 - img_cx) + y_center);
        int right_y = static_cast<int>(k * (width - img_cx) + y_center);

        // 安检门 1：对比度过滤
        if (contrast < 40.0) {
            cv::line(output_img, cv::Point(0, left_y), cv::Point(width, right_y), cv::Scalar(0, 0, 255), 1);
            cv::putText(output_img, "WEAK (" + std::to_string(static_cast<int>(contrast)) + ")", 
                        cv::Point(img_cx + 60, y_int), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            continue;
        }

        // 安检门 2：厚度过滤
        double threshold = local_min + contrast * 0.5;
        int thickness = 0;
        for (double val : slice_roi) {
            if (val > threshold) thickness++;
        }

        if (thickness >= 15) {
            cv::line(output_img, cv::Point(0, left_y), cv::Point(width, right_y), cv::Scalar(255, 0, 255), 2);
            cv::putText(output_img, "THICK (" + std::to_string(thickness) + ")", 
                        cv::Point(img_cx + 60, y_int), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 2);
            continue;
        }

        // 完美格子
        valid_y_centers.push_back(y_center);
        cv::line(output_img, cv::Point(0, left_y), cv::Point(width, right_y), cv::Scalar(255, 255, 0), 2);
    }

    int cross_event = 0;
    double min_dist = 9999.0;
    double closest_y_center = -1.0;

    for (double y_c : valid_y_centers) {
        double dist = std::abs(y_c - tripwire_y_);
        if (dist < min_dist) {
            min_dist = dist;
            closest_y_center = y_c;
        }
    }

    if (closest_y_center != -1.0 && min_dist <= trigger_zone_) {
        if (!line_is_crossing_) {
            cross_event = 1;
            line_is_crossing_ = true;
            cv::circle(output_img, cv::Point(img_cx, static_cast<int>(closest_y_center)), 20, cv::Scalar(0, 255, 0), -1);
        }
    } else {
        if (closest_y_center == -1.0 || min_dist > 20.0) {
            line_is_crossing_ = false;
        }
    }

    return cross_event;
}

std::optional<double> PVLineDetector::extract_heading_error(const std::vector<cv::Vec4i>& lines) {
    if (lines.empty()) return std::nullopt;

    struct AngleInfo { double angle; double length; };
    std::vector<AngleInfo> valid_angles;

    for (const auto& line : lines) {
        double dx = line[2] - line[0];
        double dy = line[3] - line[1];

        if (dy > 0) { dx = -dx; dy = -dy; }
        if (dy == 0) continue;

        double angle_rad = std::atan2(dx, -dy);
        
        if (std::abs(angle_rad) < (CV_PI / 4.0)) {
            double length = std::sqrt(dx * dx + dy * dy);
            valid_angles.push_back({angle_rad, length});
        }
    }

    if (valid_angles.empty()) return std::nullopt;

    double total_weight = 0.0;
    double weighted_angle_sum = 0.0;
    for (const auto& va : valid_angles) {
        total_weight += va.length;
        weighted_angle_sum += va.angle * va.length;
    }

    return weighted_angle_sum / total_weight;
}

EdgeInfo PVLineDetector::detect_edge_state(const cv::Mat& blurred_gray, const std::vector<cv::Vec4i>& lines, cv::Mat& output_img) {
    EdgeInfo info;
    int height = blurred_gray.rows;
    int width = blurred_gray.cols;
    cx = width / 2.0;
    cy = height / 2.0;

    int lookahead_y = static_cast<int>(height * 0.5);
    int left_margin = static_cast<int>(width * 0.15);
    int right_margin = static_cast<int>(width * 0.85);

    cv::line(output_img, cv::Point(0, lookahead_y), cv::Point(width, lookahead_y), cv::Scalar(0, 0, 150), 2);

    int lines_in_front = 0, lines_on_right = 0, lines_on_left = 0;
    int min_y_front = height, max_x_right = 0, min_x_left = width;

    for (const auto& line : lines) {
        int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
        if (std::abs(x2 - x1) > std::abs(y2 - y1)) {
            min_y_front = std::min({min_y_front, y1, y2});
            max_x_right = std::max({max_x_right, x1, x2});
            min_x_left = std::min({min_x_left, x1, x2});

            if (std::min(y1, y2) < lookahead_y) lines_in_front++;
            if (std::min(x1, x2) < left_margin) lines_on_left++;
            if (std::max(x1, x2) > right_margin) lines_on_right++;
        }
    }

    // 1. 悬崖判定与动态物理测距
    if (lines_in_front == 0) {
        info.distance_cm = (min_y_front == height) ? 0.0 : pixel_to_distance_cm(min_y_front, 'y');
        info.visible = true; info.type = EDGE_TYPE_BLOCK; info.side = "front"; info.safety_level = SAFETY_STOP;
        cv::putText(output_img, "FRONT CLIFF! Dist: " + std::to_string(info.distance_cm) + "cm", 
                    cv::Point(width / 2 - 150, lookahead_y - 20), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        return info;
    }
    if (lines_on_right == 0) {
        int edge_x = (max_x_right > 0) ? max_x_right : right_margin;
        info.distance_cm = pixel_to_distance_cm(edge_x, 'x');
        info.visible = true; info.type = EDGE_TYPE_BLOCK; info.side = "right"; info.safety_level = SAFETY_STOP;
        return info;
    }
    if (lines_on_left == 0) {
        int edge_x = (min_x_left < width) ? min_x_left : left_margin;
        info.distance_cm = pixel_to_distance_cm(edge_x, 'x');
        info.visible = true; info.type = EDGE_TYPE_BLOCK; info.side = "left"; info.safety_level = SAFETY_STOP;
        return info;
    }

    // 2. 小板缝隙测距 (使用 cv::reduce 替代 np.mean(axis=0))
    cv::Mat col_means;
    cv::reduce(blurred_gray, col_means, 0, cv::REDUCE_AVG, CV_32F);
    
    double max_mean_val;
    cv::Point max_loc;
    cv::minMaxLoc(col_means, nullptr, &max_mean_val, nullptr, &max_loc);

    if (max_mean_val > 180.0) {
        info.distance_cm = pixel_to_distance_cm(max_loc.x, 'x');
        info.visible = true; info.type = EDGE_TYPE_CELL; info.side = ""; info.safety_level = SAFETY_WARN;
        cv::line(output_img, cv::Point(max_loc.x, 0), cv::Point(max_loc.x, height), cv::Scalar(255, 0, 255), 3);
        return info;
    }

    return info; // 默认 CLEAR
}


// ==========================================
// VisionDetectNode 实现
// ==========================================
VisionDetectNode::VisionDetectNode() 
    : Node("vision_detect_node"), current_state_("NORMAL"), block_id_(0), 
      cell_row_(-1), cell_col_(-1), inner_row_(-1), inner_col_(-1), 
      travel_axis_("block_u"), travel_sign_(1) 
{
    // 声明参数
    this->declare_parameter<double>("camera_to_base_x_cm", 0.0);
    this->declare_parameter<double>("camera_to_base_y_cm", 0.0);
    this->declare_parameter<double>("camera_yaw_offset_rad", 0.0);
    this->declare_parameter<int>("vision_timeout_ms", 1000);
    this->declare_parameter<std::string>("camera_topic", "/camera/image_raw");
    this->declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");
    this->declare_parameter<bool>("publish_debug_image", false);

    cam_offset_x_ = this->get_parameter("camera_to_base_x_cm").as_double();
    cam_offset_y_ = this->get_parameter("camera_to_base_y_cm").as_double();
    cam_yaw_offset_ = this->get_parameter("camera_yaw_offset_rad").as_double();
    vision_timeout_ = this->get_parameter("vision_timeout_ms").as_int();
    enable_debug_ = this->get_parameter("publish_debug_image").as_bool();

    // 视频测试路径
    video_path_ = "/home/cat/ros2_ws/testvideo/1.mp4";
    cap_.open(video_path_);
    if (!cap_.isOpened()) {
        RCLCPP_ERROR(this->get_logger(), "严重错误：无法打开视频文件 %s！", video_path_.c_str());
    } else {
        RCLCPP_INFO(this->get_logger(), "成功加载视频文件：%s", video_path_.c_str());
    }

    // 初始化 Publisher
    pub_localization_ = this->create_publisher<vision_detect_msgs::msg::VisionLocalization>("/vision/localization", 10);
    pub_heading_error_ = this->create_publisher<vision_detect_msgs::msg::HeadingError>("/vision/heading_error", 10);
    pub_edge_obs_ = this->create_publisher<vision_detect_msgs::msg::EdgeObservation>("/vision/edge_observation", 10);
    pub_status_ = this->create_publisher<vision_detect_msgs::msg::VisionStatus>("/vision/status", 10);
    pub_debug_ = this->create_publisher<sensor_msgs::msg::Image>("/vision/debug_image", 10);

    // 初始化 Subscriber (示例：内参回调)
    std::string info_topic = this->get_parameter("camera_info_topic").as_string();
    sub_cam_info_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        info_topic, 10, std::bind(&VisionDetectNode::cam_info_callback, this, std::placeholders::_1));

    // 定时器：30fps 视频流模拟 & 10Hz 定位发布
    timer_video_ = this->create_wall_timer(
        std::chrono::milliseconds(33), std::bind(&VisionDetectNode::video_loop_callback, this));
    timer_publish_loc_ = this->create_wall_timer(
        std::chrono::milliseconds(100), std::bind(&VisionDetectNode::publish_localization, this));

    RCLCPP_INFO(this->get_logger(), "视觉感知节点 (C++版) 已启动，视频模式测试中...");
}

void VisionDetectNode::cam_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    line_detector_.fx = msg->k[0];
    line_detector_.cx = msg->k[2];
    line_detector_.fy = msg->k[4];
    line_detector_.cy = msg->k[5];
}

void VisionDetectNode::video_loop_callback() {
    if (!cap_.isOpened()) return;

    cv::Mat cv_image;
    cap_ >> cv_image;

    if (cv_image.empty()) {
        RCLCPP_INFO(this->get_logger(), "--- 视频播放完毕，自动重新循环 ---");
        cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
        return;
    }

    try {
        ProcessResult res = line_detector_.process_frame(cv_image);

        // 1. 组装 EdgeObservation
        vision_detect_msgs::msg::EdgeObservation msg_edge;
        msg_edge.edge_visible = res.edge_info.visible;
        msg_edge.edge_side = res.edge_info.visible ? res.edge_info.side : "";
        msg_edge.d_edge_cm = (res.edge_info.distance_cm >= 0) ? (res.edge_info.distance_cm + cam_offset_x_) : -1.0;
        msg_edge.edge_type = res.edge_info.type;
        msg_edge.safety_level = res.edge_info.safety_level;
        pub_edge_obs_->publish(msg_edge);

        // 2. 状态机逻辑
        if (current_state_ != "FAULT") {
            if (res.cross_event == 1) {
                RCLCPP_INFO(this->get_logger(), ">>> 触发过线事件！(方向: %d)", travel_sign_);
                if (travel_axis_ == "block_u") inner_col_ += travel_sign_;
                else inner_row_ += travel_sign_;
            }

            if (res.edge_info.visible && res.edge_info.type == PVLineDetector::EDGE_TYPE_CELL) {
                RCLCPP_WARN(this->get_logger(), ">>> 跨越小板边界！");
                if (travel_axis_ == "block_u") {
                    cell_col_ += travel_sign_;
                    inner_col_ = (travel_sign_ == 1) ? 0 : 99;
                } else {
                    cell_row_ += travel_sign_;
                    inner_row_ = (travel_sign_ == 1) ? 0 : 99;
                }
            }
        }

        // 3. 组装 HeadingError
        vision_detect_msgs::msg::HeadingError heading_msg;
        heading_msg.header.stamp = this->get_clock()->now();
        heading_msg.header.frame_id = "body";
        
        if (res.heading_error.has_value()) {
            heading_msg.valid = true;
            heading_msg.heading_error_rad = res.heading_error.value() * travel_sign_;
        } else {
            heading_msg.valid = false;
            heading_msg.heading_error_rad = 0.0;
        }
        pub_heading_error_->publish(heading_msg);

        // 显示结果图像
        cv::imshow("Detected Lines & Yaw", res.output_img);
        cv::waitKey(1);

    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "视频帧处理失败: %s", e.what());
    }
}

void VisionDetectNode::publish_localization() {
    vision_detect_msgs::msg::VisionLocalization msg;
    msg.header.stamp = this->get_clock()->now();
    msg.header.frame_id = "pv_map";
    msg.localization_state = current_state_;
    msg.block_id = block_id_;
    msg.cell_row = cell_row_;
    msg.cell_col = cell_col_;
    msg.inner_row = inner_row_;
    msg.inner_col = inner_col_;
    pub_localization_->publish(msg);

    vision_detect_msgs::msg::VisionStatus status_msg;
    status_msg.state = current_state_;
    status_msg.camera_ok = cap_.isOpened();
    pub_status_->publish(status_msg);
}

// ==========================================
// 节点主函数入口
// ==========================================
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VisionDetectNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    cv::destroyAllWindows();
    return 0;
}