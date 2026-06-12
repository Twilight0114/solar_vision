import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('my_camera_publisher')
        
        # 1. 创建一个发布者，往 '/image_raw' 话题发数据
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        
        # 2. 设置一个定时器，每 0.033 秒（约 30 帧）触发一次读取
        timer_period = 0.033 
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # 3. 初始化 OpenCV 摄像头 (0代表 /dev/video0)
        self.cap = cv2.VideoCapture(0)
        # 强制设置分辨率，避免玄学问题
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.bridge = CvBridge()
        self.get_logger().info('自定义相机节点已启动，正在发布 /image_raw ...')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            try:
                # 4. 把 OpenCV 的图像转换成 ROS 2 的 Image 消息
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                # 5. 发布出去！
                self.publisher_.publish(msg)
            except Exception as e:
                self.get_logger().error(f'发布失败: {e}')
        else:
            self.get_logger().warning('未能从摄像头读取到画面！')

def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CameraPublisher()
    
    try:
        rclpy.spin(camera_publisher)
    except KeyboardInterrupt:
        pass
    
    # 善后清理
    camera_publisher.cap.release()
    camera_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()