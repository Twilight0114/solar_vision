import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import traceback
#from rover_fcu_bridge.msg import RoverStatus
from vision_detect_msgs.msg import VisionLocalization, HeadingError, EdgeObservation, VisionStatus


class PVLineDetector:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.tripwire_y = 240
        self.trigger_zone = 5 
        self.line_is_crossing = False
        
        # ==========================================
        # 机物理参数与状态枚举定义
        # ==========================================
        # 1. 物理安装参数 (你需要根据小车实际情况测量修改)
        self.camera_height_cm = 15.0  # 相机距离光伏板面的高度 (cm)
        self.camera_pitch_deg = 10.0  # 相机前倾俯仰角 (度)
        
        # 2. 相机内参 (可以通过 ROS2 camera_calibration 包标定获得，这里用经验值占位)
        self.fx = 400.0  # X轴焦距 (像素)
        self.fy = 400.0  # Y轴焦距 (像素)
        self.cx = 320.0  # 图像中心 X
        self.cy = 240.0  # 图像中心 Y
        
        # 3. 严格映射 
        self.EDGE_TYPES = {
            "BLOCK": "block_edge", # 大板悬崖
            "CELL": "cell_edge"    # 小板缝隙
        }
        self.SAFETY_LEVELS = {
            "STOP": "EDGE_STOP",
            "WARN": "EDGE_CAUTION",
            "SAFE": "EDGE_CLEAR"
        }

    def process_frame(self, img):
            """
            接收从视频流或 ROS 传来的实时图像矩阵，统筹整条 OpenCV 视觉流水线
            """
            img = cv2.resize(img, (640, 480))
            output_img = img.copy()
            # ==========================================
            # 第一阶段：图像预处理 (降噪与边缘强化)
            # ==========================================
            # 1. 灰度化与光照归一化 (CLAHE)，对抗光伏板局部反光
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            enhanced_gray = self.clahe.apply(gray)

            # 2. 双边滤波降噪 (极大地平滑了反光区域，同时保留了清晰的物理边缘)
            blurred = cv2.bilateralFilter(enhanced_gray, 9, 75, 75)

            # 3. 动态自适应 Canny 边缘检测
            v = np.median(blurred)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            edges = cv2.Canny(blurred, lower, upper)

            # 4. 形态学闭运算 (连接因为光照不均而断裂的细小线条)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # ==========================================
            # 第二阶段：几何特征提取 (寻找工业级直线)
            # ==========================================
            # 5. 概率霍夫直线变换 (使用极其严格的参数过滤碎线和噪点)
            # threshold=100: 特征极强才算线
            # minLineLength=150: 忽略短小的焊点干扰线
            # maxLineGap=40: 允许线条在穿过白点时自动闭合
            lines = cv2.HoughLinesP(closed_edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=40)

            heading_error = None
            cross_event = 0
            
            # ==========================================
            # 第三阶段：状态机逻辑运算 (偏角、计数、边缘)
            # ==========================================
            if lines is not None:
                # 提取偏转角度 (Yaw)
                heading_error = self.extract_heading_error(lines)
                
                # 检测是否跨过横线 (触发计数)
                # 【注意传入了 blurred 矩阵】
                cross_event = self.detect_line_crossing(blurred, lines, output_img)
                
                # 如果算出了角度，把角度用绿字打印在画面左上角
                if heading_error is not None:
                    deg = math.degrees(heading_error)
                    cv2.putText(output_img, f"Yaw Err: {deg:.2f} deg", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 检测大板悬崖与小板缝隙 (传入滤波后的灰度图用于亮度投影统计)
            edge_info = self.detect_edge_state(blurred, lines, output_img)
            
            # ==========================================
            # 第四阶段：向外层节点抛出数据
            # ==========================================
            return output_img, closed_edges, heading_error, cross_event, edge_info
        
    def pixel_to_distance_cm(self, pixel_val, axis='y'):
        """
        利用针孔模型将像素坐标转换为物理距离
        """
        pitch_rad = math.radians(self.camera_pitch_deg)
        
        if axis == 'y':
            # 计算前方距离 (基于 Y 坐标)
            # 画面越靠上 (pixel_y越小)，距离越远
            alpha = math.atan((self.cy - pixel_val) / self.fy)
            distance = self.camera_height_cm * math.tan(pitch_rad + alpha)
            return round(distance, 2)
            
        elif axis == 'x':
            # 计算侧向距离 (基于 X 坐标)
            # 这里简化为：先算出画面中心的投影距离，再按比例推算横向物理宽度
            center_dist = self.camera_height_cm * math.tan(pitch_rad)
            # 光线从光心到侧边的横向夹角
            beta = math.atan(abs(pixel_val - self.cx) / self.fx)
            # 斜边距离（光心到地面投影点）
            slant_dist = math.sqrt(self.camera_height_cm**2 + center_dist**2)
            lateral_distance = slant_dist * math.tan(beta)
            return round(lateral_distance, 2)
            
        return -1.0

    
    def detect_line_crossing(self, blurred_gray, lines, output_img):
        height, width = output_img.shape[:2]
        cx = int(width / 2) # 画面的中心 X
        self.tripwire_y = int(height / 2)
        self.trigger_zone = 5 
        
        cv2.line(output_img, (0, self.tripwire_y), (width, self.tripwire_y), (255, 0, 0), 2)
        
        if lines is None or len(lines) == 0:
            return 0
            
        cross_event = 0
        raw_lines = []
        
        # 1. 提取所有横向为主的线，并计算它们在画面正中央的 Y 坐标和斜率 K
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            
            if dx > dy: 
                if x1 == x2: continue
                k = (y2 - y1) / (x2 - x1) # 计算斜率
                b = y1 - k * x1
                y_center = k * cx + b     # 计算该直线延长线与画面中轴线的交点
                raw_lines.append((y_center, k))

        # 2. 聚类：把同一根线上的碎段合并
        raw_lines.sort(key=lambda item: item[0])
        merged_lines = []
        for y_c, k in raw_lines:
            if not merged_lines or abs(y_c - merged_lines[-1][0]) > 40:
                merged_lines.append([y_c, k, 1]) # [当前平均 y_center, sum_k, count]
            else:
                count = merged_lines[-1][2] + 1
                merged_lines[-1][0] = (merged_lines[-1][0] * merged_lines[-1][2] + y_c) / count
                merged_lines[-1][1] += k
                merged_lines[-1][2] = count

        # ==========================================
        # 3. 核心升级：倾斜自适应 CT 扫描 (Rotated Scan)
        # ==========================================
        valid_y_centers = []
        
        # 锁定 X 采样范围 (中心 ±50)
        sample_xs = np.arange(max(0, cx - 50), min(width, cx + 50))
        
        for m in merged_lines:
            y_center = m[0]
            k = m[1] / m[2] # 提取平均斜率
            y_int = int(y_center)
            
            slice_roi = []
            # 在垂直方向上以 offset 为偏移量进行平移扫描 (上下 20 像素)
            for offset in range(-20, 21):
                # 让采样线和光伏板格子拥有完全相同的倾斜角度
                sample_ys = np.clip(np.int32(k * (sample_xs - cx) + y_center + offset), 0, height - 1)
                mean_val = np.mean(blurred_gray[sample_ys, sample_xs])
                slice_roi.append(mean_val)
                
            slice_roi = np.array(slice_roi)
            local_max = np.max(slice_roi)
            local_min = np.min(slice_roi)
            contrast = local_max - local_min
            
            # 计算这条线的左右端点，用于在画面上画出完美的倾斜线
            left_y = int(k * (0 - cx) + y_center)
            right_y = int(k * (width - cx) + y_center)
            
            # 安检门 1：对比度过滤
            if contrast < 40: 
                cv2.line(output_img, (0, left_y), (width, right_y), (0, 0, 255), 1)
                cv2.putText(output_img, f"WEAK ({int(contrast)})", (cx + 60, y_int), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                continue 
                
            # 安检门 2：厚度过滤 (收严至 15，强力杀灭小板边缘)
            threshold = local_min + contrast * 0.5
            thickness = np.sum(slice_roi > threshold)
            
            if thickness >= 15: 
                cv2.line(output_img, (0, left_y), (width, right_y), (255, 0, 255), 2)
                cv2.putText(output_img, f"THICK ({thickness})", (cx + 60, y_int), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                continue 
                
            # 完美格子
            valid_y_centers.append(y_center)
            cv2.line(output_img, (0, left_y), (width, right_y), (255, 255, 0), 2)

        # 4. 寻找距离绊线最近的有效格子并触发状态机
        min_dist = 9999
        closest_y_center = -1
        for y_c in valid_y_centers:
            dist = abs(y_c - self.tripwire_y)
            if dist < min_dist:
                min_dist = dist
                closest_y_center = y_c

        if closest_y_center != -1 and min_dist <= self.trigger_zone:
            if not self.line_is_crossing:
                cross_event = 1
                self.line_is_crossing = True
                cv2.circle(output_img, (cx, int(closest_y_center)), 20, (0, 255, 0), -1)
        else:
            if closest_y_center == -1 or min_dist > 20:
                self.line_is_crossing = False
            
        return cross_event
    

    def extract_heading_error(self, lines):
        """
        从提取出的线段中，计算机器人的偏转角度（弧度）
        返回值: error_rad (如果没找到合适的线则返回 None)
        """
        if lines is None or len(lines) == 0:
            return None

        valid_angles = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 1. 统一向量方向：让线段始终从下往上指 (因为图像坐标系 y 轴向下)
            dx = x2 - x1
            dy = y2 - y1
            if dy > 0:
                dx = -dx
                dy = -dy
                
            # 2. 计算相对垂直中轴线的夹角 (弧度)
            # 垂直(|)为 0; 往右倾斜(/)为正; 往左倾斜(\)为负
            # 避免除以 0 的情况
            if dy == 0:
                continue 
            angle_rad = math.atan2(dx, -dy)
            
            # 3. 过滤掉横向的线 (只取夹角在 -45度 到 +45度 之间的竖线)
            if abs(angle_rad) < (math.pi / 4):
                length = math.sqrt(dx**2 + dy**2)
                valid_angles.append((angle_rad, length))
                
        # 4. 如果没有找到竖线，说明可能全是横线或者全是噪点
        if not valid_angles:
            return None
            
        # 5. 加权平均：越长的线段话语权越大
        total_weight = sum(length for _, length in valid_angles)
        weighted_angle = sum(angle * length for angle, length in valid_angles) / total_weight
        
        return weighted_angle
    

    def detect_edge_state(self, blurred_gray, lines, output_img):
        edge_visible = False
        edge_type = ""
        safety_level = self.SAFETY_LEVELS["SAFE"]
        d_edge_cm = -1.0 

        height, width = blurred_gray.shape
        self.cx = width / 2
        self.cy = height / 2

        lookahead_y = int(height * 0.5)
        left_margin = int(width * 0.15)
        right_margin = int(width * 0.85)

        cv2.line(output_img, (0, lookahead_y), (width, lookahead_y), (0, 0, 150), 2) 

        lines_in_front = 0
        lines_on_right = 0
        lines_on_left = 0
        
        # 【核心新增】：记录物理边界在画面中的极值坐标
        min_y_front = height
        max_x_right = 0
        min_x_left = width

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)

                if dx > dy: # 仅处理横向网格
                    # 动态捕捉最边缘的网格线
                    min_y_front = min(min_y_front, y1, y2)
                    max_x_right = max(max_x_right, x1, x2)
                    min_x_left = min(min_x_left, x1, x2)
                    
                    if min(y1, y2) < lookahead_y:
                        lines_in_front += 1
                    if min(x1, x2) < left_margin:
                        lines_on_left += 1
                    if max(x1, x2) > right_margin:
                        lines_on_right += 1

       # ==========================================
        # 1. 悬崖判定与【动态物理测距】
        # ==========================================
        if lines_in_front == 0:
            if min_y_front == height:
                d_edge_cm = 0.0
            else:
                d_edge_cm = self.pixel_to_distance_cm(min_y_front, axis='y')
            cv2.putText(output_img, f"FRONT CLIFF! Dist: {d_edge_cm}cm", (int(width/2)-150, lookahead_y - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # 【修改点】：返回 BLOCK 类型，以及方向 "front"
            return True, self.EDGE_TYPES["BLOCK"], "front", self.SAFETY_LEVELS["STOP"], d_edge_cm

        if lines_on_right == 0:
            edge_x = max_x_right if max_x_right > 0 else right_margin
            d_edge_cm = self.pixel_to_distance_cm(edge_x, axis='x')
            cv2.putText(output_img, f"RIGHT CLIFF ({d_edge_cm}cm)", (right_margin - 220, int(height/2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # 【修改点】：返回 BLOCK 类型，以及方向 "right"
            return True, self.EDGE_TYPES["BLOCK"], "right", self.SAFETY_LEVELS["STOP"], d_edge_cm
            
        if lines_on_left == 0:
            edge_x = min_x_left if min_x_left < width else left_margin
            d_edge_cm = self.pixel_to_distance_cm(edge_x, axis='x')
            cv2.putText(output_img, f"LEFT CLIFF ({d_edge_cm}cm)", (left_margin + 20, int(height/2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # 【修改点】：返回 BLOCK 类型，以及方向 "left"
            return True, self.EDGE_TYPES["BLOCK"], "left", self.SAFETY_LEVELS["STOP"], d_edge_cm

        # ==========================================
        # 2. 小板缝隙测距
        # ==========================================
        col_means = np.mean(blurred_gray, axis=0)
        max_mean_val = np.max(col_means)
        max_mean_x = int(np.argmax(col_means))

        if max_mean_val > 180:  
            d_edge_cm = self.pixel_to_distance_cm(max_mean_x, axis='x')
            cv2.line(output_img, (max_mean_x, 0), (max_mean_x, height), (255, 0, 255), 3)
            cv2.putText(output_img, f"CELL EDGE ({d_edge_cm}cm)", (max_mean_x + 10, int(height/2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            # 【修改点】：返回 CELL 类型，方向留空 ""
            return True, self.EDGE_TYPES["CELL"], "", self.SAFETY_LEVELS["WARN"], d_edge_cm

        return edge_visible, edge_type, "", safety_level, d_edge_cm



class VisionDetectNode(Node):
    def __init__(self):
        super().__init__('vision_detect_node')
        # 声明 API 要求的节点参数
        self.declare_parameter('camera_to_base_x_cm', 0.0)
        self.declare_parameter('camera_to_base_y_cm', 0.0)
        self.declare_parameter('camera_yaw_offset_rad', 0.0)
        self.declare_parameter('vision_timeout_ms', 1000)

        # 可以在这里读取并在后续计算物理距离时作为偏移量补偿（目前先挂载，防止 launch 报错）
        self.cam_offset_x = self.get_parameter('camera_to_base_x_cm').value
        self.cam_offset_y = self.get_parameter('camera_to_base_y_cm').value
        self.cam_yaw_offset = self.get_parameter('camera_yaw_offset_rad').value
        self.vision_timeout = self.get_parameter('vision_timeout_ms').value

        self.bridge = CvBridge()
        self.line_detector = PVLineDetector()

        # 核心状态机变量
        self.current_state = "NORMAL" #测试默认NORMAL，实际初始为 FAULT，等待 planner 唤醒
        self.block_id = 0
        self.cell_row = -1
        self.cell_col = -1
        self.inner_row = -1
        self.inner_col = -1

        #测试用的视频路径，实际部署时改为摄像头订阅
        self.video_path = '/home/cat/ros2_ws/testvideo/1.mp4' 
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f"严重错误：无法打开视频文件 {self.video_path}！")
        else:
            self.get_logger().info(f"成功加载视频文件：{self.video_path}") 

        # --- 运行上下文 ---
        self.travel_axis = "block_u"
        self.travel_sign = 1
        #  声明并获取参数
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('publish_debug_image', False) # 默认关闭调试图节省算力
        
        cam_topic = self.get_parameter('camera_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        self.enable_debug = self.get_parameter('publish_debug_image').value

        # # ================== 1. 订阅 Topic ==================
        # self.sub_image = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        # self.sub_cam_info = self.create_subscription(CameraInfo, '/camera/camera_info', self.cam_info_callback, 10)
        # self.sub_init = self.create_subscription(VisionInit, '/mission_planner/vision_init', self.init_callback, 10)
        # self.sub_ctx = self.create_subscription(TravelContext, '/mission_planner/travel_context', self.context_callback, 10)
        
        # 订阅 FCU 状态
        # self.sub_fcu = self.create_subscription(RoverStatus, '/fcu/status', self.fcu_callback, 10)
        # self.current_fcu_speed = 0.0
        
        # # ================== 2. 发布 Topic ==================
        self.pub_localization = self.create_publisher(VisionLocalization, '/vision/localization', 10)
        self.pub_heading_error = self.create_publisher(HeadingError, '/vision/heading_error', 10)
        self.pub_edge_obs = self.create_publisher(EdgeObservation, '/vision/edge_observation', 10)
        self.pub_status = self.create_publisher(VisionStatus, '/vision/status', 10)
        self.pub_debug = self.create_publisher(Image, '/vision/debug_image', 10)
        
        # # 周期性发布状态和定位
        # self.create_timer(0.1, self.timer_publish_loc_and_status)
        
        # 创建一个定时器，模拟 30fps 的视频流 (约 0.033 秒触发一次)
        self.timer_video = self.create_timer(0.033, self.video_loop_callback)
        self.timer_publish_loc = self.create_timer(0.1, self.publish_localization)
        self.get_logger().info('视觉感知节点已启动，视频模式测试中...')

    def fcu_callback(self, msg):
        # 记录当前车体真实速度 (m/s)
        self.current_fcu_speed = msg.linear_velocity_mps

# ---------------- 业务回调处理 ----------------
    def init_callback(self, msg):
        self.block_id = msg.block_id
        self.cell_row = msg.cell_row
        self.cell_col = msg.cell_col
        self.inner_row = msg.inner_row
        self.inner_col = msg.inner_col
        self.current_state = "NORMAL"
        self.get_logger().info(f"收到初始化: block={self.block_id}, inner=[{self.inner_row}, {self.inner_col}]")

    def context_callback(self, msg):
        self.travel_axis = msg.travel_axis
        self.travel_sign = msg.travel_sign

    def cam_info_callback(self, msg):
        # 动态更新内参用于距离计算
        self.line_detector.fx = msg.k[0]
        self.line_detector.cx = msg.k[2]
        self.line_detector.fy = msg.k[4]
        self.line_detector.cy = msg.k[5]
        
            
    def video_loop_callback(self):
            """
            定时器触发，每次从视频文件中抠出一帧进行处理
            """
            if not self.cap.isOpened():
                return
                
            # 1. 读取一帧画面
            ret, cv_image = self.cap.read()
            
            # 2. 如果视频播完了，自动重置到第一帧，实现“无限洗脑循环”调参！
            if not ret:
                self.get_logger().info("--- 视频播放完毕，自动重新循环 ---")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return

            try:
                # 1. 喂给大脑，注意这里还是接住 5 个返回值
                result_img, edges_img, heading_error_rad, cross_event, edge_info = self.line_detector.process_frame(cv_image)
                
                # ==========================================
                # 2. 解包边缘信息 (现在里面有 5 个元素了！)
                # ==========================================
                edge_visible, edge_type, edge_side_val, safety_level, d_edge_cm = edge_info
                
                msg_edge = EdgeObservation()
                msg_edge.edge_visible = edge_visible
                msg_edge.edge_side = edge_side_val if edge_visible else "" # 动态赋予方向
                # 将“相机距离”转换为“车体距离”
                if d_edge_cm >= 0:
                    msg_edge.d_edge_cm = float(d_edge_cm + self.cam_offset_x)
                else:
                    msg_edge.d_edge_cm = -1.0  
                msg_edge.edge_type = edge_type
                msg_edge.safety_level = safety_level
                self.pub_edge_obs.publish(msg_edge)
                
                # ==========================================
                # 3. 跨越边界与格子计数状态机 (整合上下文)
                # ==========================================
                if self.current_state != "FAULT":
                    # A. 跨越内部格子
                    if cross_event == 1:
                        self.get_logger().info(f'>>> 触发过线事件！(方向: {self.travel_sign})')
                        if self.travel_axis == "block_u":
                            self.inner_col += self.travel_sign
                        else:
                            self.inner_row += self.travel_sign

                    # B. 跨越小板缝隙 (Cell Edge)
                    if edge_visible and edge_type == self.line_detector.EDGE_TYPES["CELL"]:
                        self.get_logger().warn('>>> 跨越小板边界！')
                        if self.travel_axis == "block_u":
                            self.cell_col += self.travel_sign
                            self.inner_col = 0 if self.travel_sign == 1 else 99 # 跨板后重置内部格子
                        else:
                            self.cell_row += self.travel_sign
                            self.inner_row = 0 if self.travel_sign == 1 else 99
                
                # ==========================================
                # 4. 发布 HeadingError 消息 (结合行车方向翻转符号)
                # ==========================================
                heading_msg = HeadingError()
                heading_msg.header.stamp = self.get_clock().now().to_msg()
                heading_msg.header.frame_id = "body"
                if heading_error_rad is not None:
                    heading_msg.valid = True
                    # 【核心修正】：如果倒着走，航向误差必须乘上 travel_sign 翻转过来！
                    heading_msg.heading_error_rad = float(heading_error_rad * self.travel_sign)
                else:
                    heading_msg.valid = False
                    heading_msg.heading_error_rad = 0.0
                self.pub_heading_error.publish(heading_msg)
                
                # 显示结果
                cv2.imshow("Detected Lines & Yaw", result_img)
                cv2.waitKey(1)
                        
            except Exception as e:
                self.get_logger().error(f'视频帧处理失败:\n{traceback.format_exc()}')

    # def image_callback(self, msg):
    #     if self.current_state == "FAULT":
    #         return # 没被初始化，不浪费算力
            
    #     cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    #     result_img, edges_img, heading_error, cross_event, edge_info = self.line_detector.process_frame(cv_image)
        
    #     # ==========================================
    #     # 核心逻辑：结合上下文进行状态机运转
    #     # ==========================================
    #     # 1. 处理过内部格子事件
    #     if cross_event == 1:
    #         if self.travel_axis == "block_u":
    #             self.inner_col += self.travel_sign
    #         else:
    #             self.inner_row += self.travel_sign
                
    #     # 2. 处理跨越小板边界 (Cell Edge) 事件
    #     edge_visible, edge_type, safety_level, d_edge_cm = edge_info
    #     if edge_visible and edge_type == self.line_detector.EDGE_TYPES["GAP"]:
    #         # 当跨过缝隙时，根据方向更新 cell_row / cell_col
    #         # 并重置 inner_row / inner_col 
    #         pass # (具体逻辑需要对照 grid_localization.md 的细节)

    #     # 3. 发布 EdgeObservation 和 HeadingError (代码略，和你之前写的一样)
    #     result_img, edges_img, heading_error_rad, cross_event, edge_info = self.line_detector.process_frame(cv_image)
                
    #             # ==========================================
    #             # 2. 在这里组装并发布 EdgeObservation 消息！
    #             # ==========================================
    #             # 解包刚收到的边缘信息（注意现在是 4 个参数了，多了物理距离 d_edge_cm）
    #     edge_visible, edge_type, safety_level, d_edge_cm = edge_info
                
    #     msg_edge = EdgeObservation()
    #     msg_edge.edge_visible = edge_visible
    #     msg_edge.edge_side = "front"
    #     if d_edge_cm >= 0:
            # msg_edge.d_edge_cm = float(d_edge_cm + offset_x)
        # else:
            # msg_edge.d_edge_cm = -1.0
    #     msg_edge.edge_type = edge_type
    #     msg_edge.safety_level = safety_level
                
    #     # 用节点类 (VisionDetectNode) 的发布者发布消息！
    #     self.pub_edge_obs.publish(msg_edge)
        
    #     # ==========================================
    #     # 3. 组装并发布 HeadingError 消息
    #     # ==========================================
    #     heading_msg = HeadingError()
    #     heading_msg.header.stamp = self.get_clock().now().to_msg()
    #     heading_msg.header.frame_id = "body"
    #     if heading_error_rad is not None:
    #         heading_msg.valid = True
    #         heading_msg.heading_error_rad = float(heading_error_rad)
    #     else:
    #         heading_msg.valid = False
    #         heading_msg.heading_error_rad = 0.0
    #     self.pub_heading_error.publish(heading_msg)
                
    #     # 4. 发布 Debug 图像 (只在参数允许时发布)
        # if self.enable_debug:
        #     debug_msg = self.bridge.cv2_to_imgmsg(result_img, encoding="bgr8")
        #     self.pub_debug.publish(debug_msg)

    def publish_localization(self):
        # ==========================================
        # 1. 发布 Localization 消息
        # ==========================================
        msg = VisionLocalization()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "pv_map"
        msg.localization_state = self.current_state
        msg.block_id = self.block_id
        msg.cell_row = self.cell_row
        msg.cell_col = self.cell_col
        msg.inner_row = self.inner_row
        msg.inner_col = self.inner_col
        self.pub_localization.publish(msg)

        # ==========================================
        # 2. 发布 VisionStatus 消息
        # ==========================================
        status_msg = VisionStatus()
        status_msg.state = self.current_state
        
        # 动态判定相机是否正常：如果在测视频，看 cap 是否打开；如果在真实小车上，看是否持续收到图像
        if hasattr(self, 'cap'):
            status_msg.camera_ok = self.cap.isOpened()
        else:
            status_msg.camera_ok = True # 如果改为真实相机订阅，这里可以加上更严谨的超时检测
            
        self.pub_status.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = VisionDetectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()