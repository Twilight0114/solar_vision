import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from vision_detect_msgs.msg import VisionLocalization, HeadingError, EdgeObservation

# ==============================================================
# 将你的算法类稍微改造一下，使其适应实时视频流输入
# ==============================================================
class PVLineDetector:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.tripwire_y = 240
        self.trigger_zone = 5 
        self.line_is_crossing = False
        
        # ==========================================
        # 新增：相机物理参数与状态枚举定义
        # ==========================================
        # 1. 物理安装参数 (你需要根据小车实际情况测量修改)
        self.camera_height_cm = 15.0  # 相机距离光伏板面的高度 (cm)
        self.camera_pitch_deg = 10.0  # 相机前倾俯仰角 (度)
        
        # 2. 相机内参 (可以通过 ROS2 camera_calibration 包标定获得，这里用经验值占位)
        self.fx = 400.0  # X轴焦距 (像素)
        self.fy = 400.0  # Y轴焦距 (像素)
        self.cx = 320.0  # 图像中心 X
        self.cy = 240.0  # 图像中心 Y
        
        # 3. 严格映射 API.md 的枚举字典
        self.EDGE_TYPES = {
            "FRONT": "block_edge_front",
            "LEFT": "block_edge_left",
            "RIGHT": "block_edge_right",
            "GAP": "cell_edge"
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
        self.tripwire_y = int(height / 2)
        self.trigger_zone = 5 
        
        cv2.line(output_img, (0, self.tripwire_y), (width, self.tripwire_y), (255, 0, 0), 2)
        
        if lines is None or len(lines) == 0:
            return 0
            
        cross_event = 0
        horizontal_y_list = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > abs(y2 - y1):
                horizontal_y_list.append((y1 + y2) / 2)

        horizontal_y_list.sort()
        merged_y_list = []
        for y in horizontal_y_list:
            if not merged_y_list or abs(y - merged_y_list[-1]) > 40:
                merged_y_list.append(y)
            else:
                merged_y_list[-1] = (merged_y_list[-1] + y) / 2

        # 对比度 + 厚度双重安检，过滤掉那些无关的弱纹理和粗壮白边，只留下真正的格子线

        valid_y_list = []
        for y in merged_y_list:
            y_int = int(y)
            y_start = max(0, y_int - 20)
            y_end = min(height, y_int + 20)
            
            # 【修复 1】扩大扫描宽度：从中间切取 100 个像素宽，求平均，彻底抹平噪点
            mid_x = int(width / 2)
            slice_roi = np.mean(blurred_gray[y_start:y_end, max(0, mid_x-50):min(width, mid_x+50)], axis=1)
            
            local_max = np.max(slice_roi)
            local_min = np.min(slice_roi)
            contrast = local_max - local_min
            
            # ----------------------------------------
            # 安检门 1：对比度过滤（专杀图一的无关弱纹理）
            # 如果对比度太低，说明它根本不是带有高亮焊点的粗格子
            # ----------------------------------------
            if contrast < 40: 
                # 用细红线画出，并打印它那可怜的对比度
                cv2.line(output_img, (0, y_int), (width, y_int), (0, 0, 255), 1)
                cv2.putText(output_img, f"WEAK ({int(contrast)})", (10, y_int-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                continue # 【关键修复：拒绝放行，直接检查下一根】
                
            # ----------------------------------------
            # 安检门 2：厚度过滤（专杀图二的粗壮白边）
            # ----------------------------------------
            threshold = local_min + contrast * 0.5
            thickness = np.sum(slice_roi > threshold)
            
            if thickness >= 15:
                # 用粉色画出，并打印厚度
                cv2.line(output_img, (0, y_int), (width, y_int), (255, 0, 255), 2)
                cv2.putText(output_img, f"THICK ({thickness})", (10, y_int-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                continue # 【关键修复：拒绝放行，直接检查下一根】
                
            # ----------------------------------------
            # 恭喜，通过双重安检，这是完美的格子！
            # ----------------------------------------
            valid_y_list.append(y)
            cv2.line(output_img, (0, y_int), (width, y_int), (255, 255, 0), 2)

        # 寻找最近的有效格子并触发状态机
        min_dist = 9999
        closest_horizontal_y = -1
        for y in valid_y_list:
            dist = abs(y - self.tripwire_y)
            if dist < min_dist:
                min_dist = dist
                closest_horizontal_y = y

        if closest_horizontal_y != -1 and min_dist <= self.trigger_zone:
            if not self.line_is_crossing:
                cross_event = 1
                self.line_is_crossing = True
                cv2.circle(output_img, (int(width/2), int(closest_horizontal_y)), 20, (0, 255, 0), -1)
        else:
            if closest_horizontal_y == -1 or min_dist > 20:
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
            # 此时的物理边缘，就是画面里那根最靠上的线 (min_y_front)！
            if min_y_front == height:
                d_edge_cm = 0.0 # 画面里没线了，说明已经开出去了
            else:
                d_edge_cm = self.pixel_to_distance_cm(min_y_front, axis='y')
                
            cv2.putText(output_img, f"FRONT CLIFF! Dist: {d_edge_cm}cm", (int(width/2)-150, lookahead_y - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return True, self.EDGE_TYPES["FRONT"], self.SAFETY_LEVELS["STOP"], d_edge_cm

        if lines_on_right == 0:
            # 追踪最右侧那根线的 X 坐标
            edge_x = max_x_right if max_x_right > 0 else right_margin
            d_edge_cm = self.pixel_to_distance_cm(edge_x, axis='x')
            cv2.putText(output_img, f"RIGHT CLIFF ({d_edge_cm}cm)", (right_margin - 220, int(height/2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return True, self.EDGE_TYPES["RIGHT"], self.SAFETY_LEVELS["STOP"], d_edge_cm
            
        if lines_on_left == 0:
            # 追踪最左侧那根线的 X 坐标
            edge_x = min_x_left if min_x_left < width else left_margin
            d_edge_cm = self.pixel_to_distance_cm(edge_x, axis='x')
            cv2.putText(output_img, f"LEFT CLIFF ({d_edge_cm}cm)", (left_margin + 20, int(height/2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return True, self.EDGE_TYPES["LEFT"], self.SAFETY_LEVELS["STOP"], d_edge_cm

        # ==========================================
        # 2. 小板缝隙测距（横移时动态改变）
        # ==========================================
        col_means = np.mean(blurred_gray, axis=0)
        max_mean_val = np.max(col_means)
        max_mean_x = int(np.argmax(col_means))

        if max_mean_val > 180:  
            # 小车的摄像头往左往右偏时，max_mean_x 会跟着变，距离也会变！
            d_edge_cm = self.pixel_to_distance_cm(max_mean_x, axis='x')
            cv2.line(output_img, (max_mean_x, 0), (max_mean_x, height), (255, 0, 255), 3)
            cv2.putText(output_img, f"CELL EDGE ({d_edge_cm}cm)", (max_mean_x + 10, int(height/2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            return True, self.EDGE_TYPES["GAP"], self.SAFETY_LEVELS["WARN"], d_edge_cm

        return edge_visible, edge_type, safety_level, d_edge_cm



class VisionDetectNode(Node):
    def __init__(self):
        super().__init__('vision_detect_node')
        
        self.bridge = CvBridge()
        
        # 实例化你的核心算法类！
        self.line_detector = PVLineDetector()

        # 核心状态机变量
        self.current_state = "NORMAL"
        self.block_id = 0
        self.cell_row = -1
        self.cell_col = -1
        self.inner_row = -1
        self.inner_col = -1

        # 订阅相机图像
        #self.sub_image = self.create_subscription(
            #Image, '/camera/image_raw', self.image_callback, 10)
        self.video_path = '/home/cat/ros2_ws/testvideo/1.mp4' 
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            self.get_logger().error(f"严重错误：无法打开视频文件 {self.video_path}！")
        else:
            self.get_logger().info(f"成功加载视频文件：{self.video_path}")
            
        # 创建一个定时器，模拟 30fps 的视频流 (约 0.033 秒触发一次)
        self.timer_video = self.create_timer(0.033, self.video_loop_callback)

        # 发布者
        self.pub_localization = self.create_publisher(VisionLocalization, '/vision/localization', 10)
        self.pub_heading_error = self.create_publisher(HeadingError, '/vision/heading_error', 10)
        self.pub_edge_obs = self.create_publisher(EdgeObservation, '/vision/edge_observation', 10)
    
        self.timer_publish_loc = self.create_timer(0.1, self.publish_localization)
        self.get_logger().info('视觉感知节点已启动，正在应用 OpenCV 传统视觉算法...')

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
                # 1. 把图像喂给“大脑”，接住返回的 5 个数据
                result_img, edges_img, heading_error_rad, cross_event, edge_info = self.line_detector.process_frame(cv_image)
                
                # ==========================================
                # 2. 在这里组装并发布 EdgeObservation 消息！
                # ==========================================
                # 解包刚收到的边缘信息（注意现在是 4 个参数了，多了物理距离 d_edge_cm）
                edge_visible, edge_type, safety_level, d_edge_cm = edge_info
                
                msg_edge = EdgeObservation()
                msg_edge.edge_visible = edge_visible
                msg_edge.edge_side = "front"
                msg_edge.d_edge_cm = float(d_edge_cm)  # 赋予物理厘米数
                msg_edge.edge_type = edge_type
                msg_edge.safety_level = safety_level
                
                # 用节点类 (VisionDetectNode) 的发布者发布消息！
                self.pub_edge_obs.publish(msg_edge)
                
                # ==========================================
                # 3. 组装并发布 HeadingError 消息
                # ==========================================
                heading_msg = HeadingError()
                heading_msg.header.stamp = self.get_clock().now().to_msg()
                heading_msg.header.frame_id = "body"
                if heading_error_rad is not None:
                    heading_msg.valid = True
                    heading_msg.heading_error_rad = float(heading_error_rad)
                else:
                    heading_msg.valid = False
                    heading_msg.heading_error_rad = 0.0
                self.pub_heading_error.publish(heading_msg)
                
                # ==========================================
                # 4. 打印过线事件，维护状态机
                # ==========================================
                if self.current_state != "FAULT" and cross_event == 1:
                    self.get_logger().info('>>> 触发过线事件！(内部格子计数 +1)')
                    self.inner_col += 1
                
                # 显示结果
                cv2.imshow("Detected Lines & Yaw", result_img)
                # cv2.imshow("Binary Edges", edges_img) # 看情况开这个窗口
                cv2.waitKey(1)
                        
            except Exception as e:
                self.get_logger().error(f'视频帧处理失败: {e}')

    # def image_callback(self, msg):
    #     try:
    #         # 1. 获取实时图像
    #         cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
    #         # 2. 将图像送入你的流水线进行处理
    #         result_img, edges_img, heading_error_rad, cross_event, edge_info = self.line_detector.process_frame(cv_image)
    #         # 解包边缘信息
    #         edge_visible, edge_type, safety_level = edge_info
            
    #         # ==========================================
    #         # 组装并发布 EdgeObservation 消息
    #         # ==========================================
    #         msg_edge = EdgeObservation()
    #         msg_edge.edge_visible = edge_visible
    #         msg_edge.edge_side = "front" # 简化的摄像头默认在前方
    #         msg_edge.d_edge_cm = -1.0    # C++阶段可以通过相机内参算真实距离，目前用-1代替
    #         msg_edge.edge_type = edge_type
    #         msg_edge.safety_level = safety_level

    #         self.pub_edge_obs.publish(msg_edge)
    #         # ==========================================
    #         # 结合 travel_context 更新内部格子状态！
    #         # ==========================================
    #         # 假设机器人已经初始化，且当前沿着 block_u 轴以 +1 方向清扫
    #         if self.current_state != "FAULT" and cross_event == 1:
    #             self.get_logger().info('>>> 触发过线事件！')
                
    #             # 这里就是严格执行 grid_localization.md 中的逻辑
    #             # 假定 travel_axis 为 'block_u', travel_sign 为 +1
    #             self.inner_col += 1
                
    #             #TODO: 跨过这块小板后的越界处理（留给小板边缘检测来做）
            
    #         # 3. 实时显示结果 (不要用 plt.show，用 cv2.imshow)
    #         # 开两个窗口，一个看最终的线，一个看底层的二值化边缘，这对调参非常有帮助！
    #         cv2.imshow("Detected Lines", result_img)
    #         cv2.imshow("Binary Edges", edges_img)
    #         cv2.waitKey(1)
            
    #     except Exception as e:
    #         self.get_logger().error(f'图像处理失败: {e}')

    def publish_localization(self):
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