#v3 Navigation buttons, live feed thermal sensor, RGB Camera, Lidar scan

from flask import Flask, render_template_string, jsonify, Response
import os
import threading
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
import cv2
import numpy as np

app = Flask(__name__)

# -------------------------------------------------------------
# 1. ROS 2 Node (Teleop, Cameras, & LiDAR)
# -------------------------------------------------------------
class WebTeleopNode(Node):
    def __init__(self):
        super().__init__('web_teleop_node')
        
        self.cmd_pub = self.create_publisher(Twist, '/parrot1/cmd_vel', 10)
        self.current_twist = Twist()
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        self.rgb_frame = None
        self.thermal_frame = None
        self.lidar_frame = None
        self.x, self.y, self.z = 0.0, 0.0, 0.0
        
        # Subscriptions
        self.create_subscription(Image, '/parrot1/camera/image', self.rgb_cb, 10)
        self.create_subscription(Image, '/parrot1/camera/thermal/image', self.thermal_cb, 10)
        self.create_subscription(Odometry, '/parrot1/odometry', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/parrot1/scan', self.lidar_cb, 10)

        self.get_logger().info('Live Dashboard Node Started (New UI Layout).')

    def set_command(self, lx=0.0, ly=0.0, lz=0.0, az=0.0):
        self.current_twist.linear.x = float(lx)
        self.current_twist.linear.y = float(ly)
        self.current_twist.linear.z = float(lz)
        self.current_twist.angular.z = float(az)

    def timer_callback(self):
        self.cmd_pub.publish(self.current_twist)
        
    def rgb_cb(self, msg):
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', img)
            self.rgb_frame = buffer.tobytes()
        except Exception:
            pass
            
    def thermal_cb(self, msg):
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
            clipped = np.clip(img, 90, 115)
            normalized = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX)
            color_img = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
            _, buffer = cv2.imencode('.jpg', color_img)
            self.thermal_frame = buffer.tobytes()
        except Exception:
            pass

    def lidar_cb(self, msg):
        try:
            img_size = 400
            radar_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            center = (img_size // 2, img_size // 2)
            cv2.circle(radar_img, center, 4, (0, 0, 255), -1)
            
            scale = 15.0 
            cv2.circle(radar_img, center, int(5 * scale), (50, 50, 50), 1)
            cv2.circle(radar_img, center, int(10 * scale), (50, 50, 50), 1)

            angle = msg.angle_min
            for r in msg.ranges:
                if msg.range_min < r < msg.range_max and not np.isinf(r):
                    x_ros = r * np.cos(angle)
                    y_ros = r * np.sin(angle)
                    img_x = int(center[0] - y_ros * scale)
                    img_y = int(center[1] - x_ros * scale)
                    if 0 <= img_x < img_size and 0 <= img_y < img_size:
                        cv2.circle(radar_img, (img_x, img_y), 2, (0, 255, 0), -1)
                angle += msg.angle_increment

            _, buffer = cv2.imencode('.jpg', radar_img)
            self.lidar_frame = buffer.tobytes()
        except Exception:
            pass
            
    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z

# Initialize ROS 2
rclpy.init(args=None)
ros_node = WebTeleopNode()
threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True).start()

# -------------------------------------------------------------
# 2. Web UI Template (Updated Layout)
# -------------------------------------------------------------
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <title>Drone Rescue Web UI</title>
    <style>
      body { font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; text-align: center; margin: 0; padding: 20px;}
      .panel { background: #2a2a2a; padding: 20px; border-radius: 12px; box-shadow: 0 6px 12px rgba(0,0,0,0.4); }
      
      /* Top Section: Video Feeds Horizontal layout */
      .top-row { display: flex; justify-content: space-between; gap: 20px; margin-bottom: 20px; width: 100%; max-width: 1600px; margin-left: auto; margin-right: auto; }
      .video-box { flex: 1; display: flex; flex-direction: column; align-items: center; }
      .video-feed { width: 100%; max-width: 500px; border: 3px solid #444; border-radius: 8px; background: #000; }
      .feed-title { margin: 0 0 10px 0; font-size: 1.2rem; color: #00ccff; }
      
      /* Bottom Section: Controls to the left */
      .bottom-row { display: flex; justify-content: flex-start; max-width: 1600px; margin-left: auto; margin-right: auto; }
      .controls-wrapper { display: inline-block; text-align: center; }
      
      /* Telemetry display */
      .telemetry-box { background: #002233; border: 2px solid #0088cc; padding: 15px; border-radius: 8px; font-size: 1.2rem; font-family: monospace; margin-bottom: 20px; color: #00ffcc; max-width: 1600px; margin-left: auto; margin-right: auto;}
      .coord { font-weight: bold; font-size: 1.5rem; color: #fff; margin: 0 10px; }
      
      /* D-pad buttons */
      .d-pad { display: grid; grid-template-columns: repeat(3, 70px); gap: 10px; justify-content: center; margin-bottom: 15px;}
      .btn { width: 70px; height: 70px; background: #444; color: white; border: none; border-radius: 12px; font-size: 26px; cursor: pointer; user-select: none; transition: background 0.1s; }
      .btn:active { background: #00aa00; }
      .btn-wide { width: 230px; font-size: 16px; padding: 12px; cursor: pointer; border: none; border-radius: 8px; color: white; margin-top: 5px; transition: background 0.1s;}
      .btn-auto { background: #006699; font-weight: bold; }
      .btn-auto:active { background: #0099ff; }
      .stop-btn { background: #aa0000; }
      .stop-btn:active { background: #ff3333; }
      
      /* Responsive fallback for smaller screens */
      @media (max-width: 1000px) {
          .top-row { flex-direction: column; }
          .bottom-row { justify-content: center; }
      }
    </style>
  </head>
  <body>
    <h1>🚁 Drone Intelligence & Live Control</h1>
    
    <!-- Telemetry HUD -->
    <div class="telemetry-box">
        📍 LIVE POSITION &rarr; 
        X: <span class="coord" id="pos-x">0.00</span>m | 
        Y: <span class="coord" id="pos-y">0.00</span>m | 
        ALT: <span class="coord" id="pos-z">0.00</span>m
    </div>

    <!-- TOP ROW: 3 Horizontal Live Feeds -->
    <div class="top-row">
        <div class="panel video-box">
            <h3 class="feed-title">🔴 Thermal Acquisition</h3>
            <img class="video-feed" src="/video_feed/thermal" alt="Waiting for Thermal Feed...">
        </div>
        <div class="panel video-box">
            <h3 class="feed-title">🟢 LiDAR Radar Mapping</h3>
            <img class="video-feed" src="/video_feed/lidar" alt="Waiting for LiDAR Feed...">
        </div>
        <div class="panel video-box">
            <h3 class="feed-title">🔵 True-Color RGB Visual</h3>
            <img class="video-feed" src="/video_feed/rgb" alt="Waiting for RGB Feed...">
        </div>
    </div>

    <!-- BOTTOM ROW: Controls fixed to Bottom-Left -->
    <div class="bottom-row">
        <div class="panel controls-wrapper">
            <h2 style="margin-top:0;">Flight Systems</h2>
            <div class="d-pad">
                <button class="btn" onmousedown="sendCommand('forward_left')" onmouseup="sendCommand('stop')">↖</button>
                <button class="btn" onmousedown="sendCommand('forward')" onmouseup="sendCommand('stop')">↑</button>
                <button class="btn" onmousedown="sendCommand('forward_right')" onmouseup="sendCommand('stop')">↗</button>
                
                <button class="btn" onmousedown="sendCommand('left')" onmouseup="sendCommand('stop')">←</button>
                <button class="btn stop-btn" onclick="sendCommand('stop')">⏹</button>
                <button class="btn" onmousedown="sendCommand('right')" onmouseup="sendCommand('stop')">→</button>
                
                <button class="btn" onmousedown="sendCommand('back_left')" onmouseup="sendCommand('stop')">↙</button>
                <button class="btn" onmousedown="sendCommand('backward')" onmouseup="sendCommand('stop')">↓</button>
                <button class="btn" onmousedown="sendCommand('back_right')" onmouseup="sendCommand('stop')">↘</button>
            </div>
            
            <div class="d-pad" style="grid-template-columns: repeat(4, 70px);">
                <button class="btn" onmousedown="sendCommand('up')" onmouseup="sendCommand('stop')">⇡</button>
                <button class="btn" onmousedown="sendCommand('down')" onmouseup="sendCommand('stop')">⇣</button>
                <button class="btn" onmousedown="sendCommand('turn_left')" onmouseup="sendCommand('stop')">↺</button>
                <button class="btn" onmousedown="sendCommand('turn_right')" onmouseup="sendCommand('stop')">↻</button>
            </div>
            
            <div style="display: flex; gap: 10px; justify-content: center; margin-top: 15px; flex-wrap: wrap;">
                <button class="btn-wide btn-auto" onclick="sendCommand('auto_ccw')">🔄 Auto Patrol (CCW)</button>
                <button class="btn-wide btn-auto" onclick="sendCommand('auto_cw')">🔃 Auto Patrol (CW)</button>
            </div>
        </div>
    </div>

    <script>
        function sendCommand(cmd) { fetch('/cmd/' + cmd); }

        setInterval(function() {
            fetch('/telemetry')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('pos-x').innerText = data.x.toFixed(2);
                    document.getElementById('pos-y').innerText = data.y.toFixed(2);
                    document.getElementById('pos-z').innerText = data.z.toFixed(2);
                });
        }, 100);
    </script>
  </body>
</html>
"""

# -------------------------------------------------------------
# 3. Flask Routes (MJPEG Streaming & APIs)
# -------------------------------------------------------------
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

def generate_frames(camera_type):
    while True:
        frame = None
        if camera_type == 'rgb' and ros_node.rgb_frame is not None:
            frame = ros_node.rgb_frame
        elif camera_type == 'thermal' and ros_node.thermal_frame is not None:
            frame = ros_node.thermal_frame
        elif camera_type == 'lidar' and ros_node.lidar_frame is not None:
            frame = ros_node.lidar_frame
            
        if frame is None:
            time.sleep(0.1)
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)

@app.route('/video_feed/<camera_type>')
def video_feed(camera_type):
    return Response(generate_frames(camera_type), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/telemetry')
def telemetry():
    return jsonify({"x": ros_node.x, "y": ros_node.y, "z": ros_node.z})

@app.route('/cmd/<direction>')
def command(direction):
    speed = 0.5
    lx, ly, lz, az = 0.0, 0.0, 0.0, 0.0
    
    if direction == 'forward': lx = speed
    elif direction == 'backward': lx = -speed
    elif direction == 'left': ly = speed
    elif direction == 'right': ly = -speed
    elif direction == 'forward_left': lx = speed; ly = speed
    elif direction == 'forward_right': lx = speed; ly = -speed
    elif direction == 'back_left': lx = -speed; ly = speed
    elif direction == 'back_right': lx = -speed; ly = -speed
    elif direction == 'up': lz = speed
    elif direction == 'down': lz = -speed
    elif direction == 'turn_left': az = speed
    elif direction == 'turn_right': az = -speed
    elif direction == 'auto_ccw': lx = speed; az = 0.5
    elif direction == 'auto_cw': lx = speed; az = -0.5
    elif direction == 'stop': pass 

    ros_node.set_command(lx, ly, lz, az)
    return jsonify({"status": "success"})

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()