import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2

class MultiObjectThermalTracker(Node):
    def __init__(self):
        super().__init__('multi_object_thermal_tracker')
        self.publisher_ = self.create_publisher(Image, '/parrot1/camera/thermal/image_color', 10)
        self.subscription = self.create_subscription(Image, '/parrot1/camera/thermal/image', self.thermal_callback, 10)
        self.get_logger().info('Multi-Object Tracker Active. Ready for flight...')

    def thermal_callback(self, msg):
        # 1. Convert ROS Image to OpenCV Matrix
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
        
        # 2. Apply Auto-Gain Contrast & Color for RViz
        clipped_img = np.clip(img, 90, 115) 
        normalized_img = cv2.normalize(clipped_img, None, 0, 255, cv2.NORM_MINMAX)
        color_img = cv2.applyColorMap(normalized_img, cv2.COLORMAP_INFERNO)
        
        # 3. Object Detection (Find anything hotter than the ambient grass)
        # We threshold at 98 to filter out the background and isolate the warm objects
        _, mask = cv2.threshold(img, 98, 255, cv2.THRESH_BINARY)
        
        # Find blobs of heat
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for cnt in contours:
            # Ignore tiny noise artifacts
            if cv2.contourArea(cnt) > 50:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Extract the thermal data strictly inside this bounding box
                roi = img[y:y+h, x:x+w]
                max_intensity = np.max(roi)
                max_k = max_intensity * 3.0
                
                # Format the temperature text
                temp_str = "> 765 K" if max_intensity >= 255 else f"{max_k} K"
                
                # Draw the Bounding Box (Green) and text (White)
                cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(color_img, temp_str, (x, y - 5), font, 0.5, (255, 255, 255), 2)
        
        # 4. Publish to RViz
        out_msg = Image()
        out_msg.header = msg.header
        out_msg.height, out_msg.width = msg.height, msg.width
        out_msg.encoding = 'bgr8'
        out_msg.step = msg.width * 3
        out_msg.data = color_img.tobytes()
        self.publisher_.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MultiObjectThermalTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()