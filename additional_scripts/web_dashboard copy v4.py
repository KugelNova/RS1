#v4 Navigation buttons, live feed thermal sensor, RGB Camera, Lidar scan, add manual navigation control with co-ordinate entry (GPS autopilot), added speed control

from flask import Flask, render_template_string, jsonify, Response
import os
import threading
import time
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
import cv2
import numpy as np

app = Flask(__name__)

# -------------------------------------------------------------
# 1. ROS 2 Node (Teleop, Cameras, LiDAR & Autopilot)
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
        
        # Telemetry & State
        self.x, self.y, self.z, self.yaw = 0.0, 0.0, 0.0, 0.0
        
        # Autopilot Variables
        self.is_navigating = False
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.target_speed = 0.5  # New dynamic speed variable
        
        self.create_subscription(Image, '/parrot1/camera/image', self.rgb_cb, 10)
        self.create_subscription(Image, '/parrot1/camera/thermal/image', self.thermal_cb, 10)
        self.create_subscription(Odometry, '/parrot1/odometry', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/parrot1/scan', self.lidar_cb, 10)

    def set_command(self, lx=0.0, ly=0.0, lz=0.0, az=0.0):
        self.is_navigating = False 
        self.current_twist.linear.x = float(lx)
        self.current_twist.linear.y = float(ly)
        self.current_twist.linear.z = float(lz)
        self.current_twist.angular.z = float(az)
        
    def goto_target(self, x, y, z, speed):
        self.target_x = float(x)
        self.target_y = float(y)
        self.target_z = float(z)
        self.target_speed = float(speed)
        self.is_navigating = True

    def timer_callback(self):
        if self.is_navigating:
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            dz = self.target_z - self.z
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < 0.3 and abs(dz) < 0.2:
                self.is_navigating = False
                self.current_twist = Twist() 
            else:
                # Altitude Control (Capped by dynamic speed)
                lz = max(-self.target_speed, min(self.target_speed, dz * 0.5))
                
                # Yaw Control (Kept at 0.5 max so the drone doesn't spin violently)
                target_yaw = math.atan2(dy, dx)
                yaw_error = math.atan2(math.sin(target_yaw - self.yaw), math.cos(target_yaw - self.yaw))
                az = max(-0.5, min(0.5, yaw_error * 1.2))
                
                # Forward Speed Control (Capped by dynamic speed)
                if abs(yaw_error) < 0.5:
                    lx = max(-self.target_speed, min(self.target_speed, distance * 0.5))
                else:
                    lx = 0.0
                    
                self.current_twist.linear.x = float(lx)
                self.current_twist.linear.y = 0.0
                self.current_twist.linear.z = float(lz)
                self.current_twist.angular.z = float(az)

        self.cmd_pub.publish(self.current_twist)
        
    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def rgb_cb(self, msg):
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', img)
            self.rgb_frame = buffer.tobytes()
        except Exception: pass
            
    def thermal_cb(self, msg):
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
            clipped = np.clip(img, 90, 115)
            normalized = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX)
            color_img = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
            _, buffer = cv2.imencode('.jpg', color_img)
            self.thermal_frame = buffer.tobytes()
        except Exception: pass

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
        except Exception: pass

# Initialize ROS 2
rclpy.init(args=None)
ros_node = WebTeleopNode()
threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True).start()

# -------------------------------------------------------------
# 2. Web UI Template
# -------------------------------------------------------------
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <title>Drone Rescue Web UI</title>
    <style>
      body { font-family: Arial, sans-serif; background: #121212; color: #fff; margin: 0; display: flex; height: 100vh; overflow: hidden; }
      
      .sidebar { width: 350px; background: #1e1e1e; padding: 20px; border-right: 3px solid #333; display: flex; flex-direction: column; align-items: center; overflow-y: auto; box-shadow: 4px 0 10px rgba(0,0,0,0.5); z-index: 10; }
      .sidebar h2 { margin-top: 0; color: #00ccff; border-bottom: 1px solid #444; padding-bottom: 10px; width: 100%; text-align: center; }
      
      .d-pad { display: grid; grid-template-columns: repeat(3, 70px); gap: 10px; justify-content: center; margin-bottom: 25px;}
      .btn { width: 70px; height: 70px; background: #3a3a3a; color: white; border: 2px solid #555; border-radius: 12px; font-size: 26px; cursor: pointer; user-select: none; transition: background 0.1s, border-color 0.1s; }
      .btn:active { background: #00aa00; border-color: #00ff00; }
      .btn-wide { width: 230px; font-size: 16px; padding: 12px; cursor: pointer; border: none; border-radius: 8px; color: white; margin-bottom: 10px; transition: background 0.1s;}
      .btn-auto { background: #005588; font-weight: bold; border: 2px solid #0088cc; }
      .btn-auto:active { background: #0099ff; }
      .stop-btn { background: #880000; border-color: #ff3333; }
      .stop-btn:active { background: #ff0000; }
      
      /* Updated Autopilot Panel */
      .autopilot-panel { background: #2a2a2a; padding: 15px; border-radius: 12px; border: 1px solid #444; width: 90%; display: flex; flex-direction: column; align-items: center; margin-top: 10px; }
      .wp-input { width: 20%; padding: 8px; background: #111; color: #fff; border: 1px solid #555; border-radius: 4px; text-align: center; }
      
      .main-content { flex: 1; padding: 20px; display: flex; flex-direction: column; align-items: center; overflow-y: auto; background: #121212; }
      
      .telemetry-box { background: #002233; border: 2px solid #0088cc; padding: 15px 30px; border-radius: 8px; font-size: 1.2rem; font-family: monospace; margin-bottom: 25px; color: #00ffcc; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.4); }
      .coord { font-weight: bold; font-size: 1.5rem; color: #fff; margin: 0 15px; }
      
      .video-container { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: center; gap: 20px; margin-bottom: 30px; width: 100%; }
      .video-box { background: #1e1e1e; padding: 15px; border-radius: 12px; border: 1px solid #333; display: flex; flex-direction: column; align-items: center; box-shadow: 0 4px 8px rgba(0,0,0,0.4); }
      .video-feed { width: 320px; height: 240px; border: 2px solid #444; border-radius: 8px; background: #000; object-fit: cover; }
      .feed-title { margin: 0 0 10px 0; font-size: 1.1rem; color: #ccc; }
      
      .map-container { display: flex; flex-direction: column; align-items: center; background: #1e1e1e; padding: 15px; border-radius: 12px; border: 1px solid #333; box-shadow: 0 4px 8px rgba(0,0,0,0.4); }
      canvas { border: 2px solid #444; border-radius: 8px; background: #0a0a0a; cursor: crosshair;}
    </style>
  </head>
  <body>
    
    <div class="sidebar">
        <h2>🕹️ Flight Systems</h2>
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
        
        <div style="margin-top: 10px;">
            <button class="btn-wide btn-auto" onclick="sendCommand('auto_ccw')">🔄 Auto Patrol (CCW)</button>
            <button class="btn-wide btn-auto" onclick="sendCommand('auto_cw')">🔃 Auto Patrol (CW)</button>
        </div>
        
        <!-- Updated Autopilot Panel with Speed Input -->
        <div class="autopilot-panel">
            <h3 style="margin-top: 0; color: #00ffcc; font-size: 1rem;">📍 GPS Autopilot</h3>
            <div style="display: flex; gap: 5px; justify-content: center; margin-bottom: 15px; width: 100%;">
                <input type="number" id="wp-x" class="wp-input" placeholder="X(m)" step="0.5">
                <input type="number" id="wp-y" class="wp-input" placeholder="Y(m)" step="0.5">
                <input type="number" id="wp-z" class="wp-input" placeholder="Z(m)" value="1.8" step="0.5">
                <input type="number" id="wp-speed" class="wp-input" placeholder="V(m/s)" value="0.5" step="0.5" title="Speed">
            </div>
            <button class="btn-wide" style="background: #008800; font-weight: bold; width: 100%;" onclick="sendWaypoint()">🚀 Launch to Coordinates</button>
            <button class="btn-wide stop-btn" style="width: 100%; margin-bottom: 0;" onclick="sendCommand('stop')">🛑 Cancel Navigation</button>
        </div>
    </div>

    <div class="main-content">
        <h1 style="margin-top:0;">🚁 Drone Intelligence Command</h1>
        <div class="telemetry-box">
            📍 LIVE POSITION &rarr; 
            X: <span class="coord" id="pos-x">0.00</span>m | 
            Y: <span class="coord" id="pos-y">0.00</span>m | 
            ALT: <span class="coord" id="pos-z">0.00</span>m
        </div>
        <div class="video-container">
            <div class="video-box">
                <h3 class="feed-title">🔴 Thermal Acquisition</h3>
                <img class="video-feed" src="/video_feed/thermal" alt="Waiting for Thermal...">
            </div>
            <div class="video-box">
                <h3 class="feed-title">🟢 LiDAR Radar</h3>
                <img class="video-feed" src="/video_feed/lidar" alt="Waiting for LiDAR...">
            </div>
            <div class="video-box">
                <h3 class="feed-title">🔵 True-Color RGB</h3>
                <img class="video-feed" src="/video_feed/rgb" alt="Waiting for RGB...">
            </div>
        </div>
        
        <div class="map-container">
            <h3 class="feed-title">🗺️ Global Positioning & Target Map</h3>
            <canvas id="droneMap" width="600" height="400"></canvas>
        </div>
    </div>

    <script>
        function sendCommand(cmd) { fetch('/cmd/' + cmd); }
        
        // Updated JS with string fallback for empty boxes
        function sendWaypoint() {
            let x = document.getElementById('wp-x').value || "0.0";
            let y = document.getElementById('wp-y').value || "0.0";
            let z = document.getElementById('wp-z').value || "1.8";
            let speed = document.getElementById('wp-speed').value || "0.5";
            fetch(`/goto/${x}/${y}/${z}/${speed}`);
        }

        const canvas = document.getElementById('droneMap');
        const ctx = canvas.getContext('2d');
        const pixelsPerMeter = 12; 
        let flightPath = [];

        function drawMap(data) {
            ctx.fillStyle = '#0a0a0a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            const cx = canvas.width / 2;
            const cy = canvas.height / 2;

            ctx.strokeStyle = '#222';
            ctx.lineWidth = 1;
            ctx.beginPath();
            for(let i = 0; i < canvas.width; i += (pixelsPerMeter * 5)) {
                ctx.moveTo(i, 0); ctx.lineTo(i, canvas.height);
                ctx.moveTo(0, i); ctx.lineTo(canvas.width, i);
            }
            ctx.stroke();

            flightPath.push({x: data.x, y: data.y});
            if(flightPath.length > 300) flightPath.shift();

            ctx.strokeStyle = '#00ffcc';
            ctx.lineWidth = 2;
            ctx.beginPath();
            for(let i = 0; i < flightPath.length; i++) {
                let px = cx - (flightPath[i].y * pixelsPerMeter);
                let py = cy - (flightPath[i].x * pixelsPerMeter);
                if(i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }
            ctx.stroke();

            if (data.navigating) {
                let tx_px = cx - (data.ty * pixelsPerMeter);
                let ty_px = cy - (data.tx * pixelsPerMeter);
                
                ctx.strokeStyle = '#ffff00';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(tx_px - 8, ty_px - 8); ctx.lineTo(tx_px + 8, ty_px + 8);
                ctx.moveTo(tx_px + 8, ty_px - 8); ctx.lineTo(tx_px - 8, ty_px + 8);
                ctx.stroke();
                
                let d_px = cx - (data.y * pixelsPerMeter);
                let d_py = cy - (data.x * pixelsPerMeter);
                ctx.strokeStyle = 'rgba(255, 255, 0, 0.4)';
                ctx.setLineDash([5, 5]);
                ctx.beginPath();
                ctx.moveTo(d_px, d_py);
                ctx.lineTo(tx_px, ty_px);
                ctx.stroke();
                ctx.setLineDash([]); 
            }

            let d_px = cx - (data.y * pixelsPerMeter);
            let d_py = cy - (data.x * pixelsPerMeter);
            ctx.fillStyle = '#ff3333';
            ctx.beginPath();
            ctx.arc(d_px, d_py, 6, 0, 2*Math.PI);
            ctx.fill();
            
            ctx.strokeStyle = 'rgba(255, 51, 51, 0.5)';
            ctx.beginPath();
            ctx.arc(d_px, d_py, 12, 0, 2*Math.PI);
            ctx.stroke();
        }

        setInterval(function() {
            fetch('/telemetry')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('pos-x').innerText = data.x.toFixed(2);
                    document.getElementById('pos-y').innerText = data.y.toFixed(2);
                    document.getElementById('pos-z').innerText = data.z.toFixed(2);
                    drawMap(data);
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
    return jsonify({
        "x": ros_node.x, "y": ros_node.y, "z": ros_node.z,
        "navigating": ros_node.is_navigating,
        "tx": ros_node.target_x, "ty": ros_node.target_y
    })

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

# Updated Route to string to prevent decimal crash, handling 4 variables
@app.route('/goto/<string:x>/<string:y>/<string:z>/<string:speed>')
def goto_target(x, y, z, speed):
    ros_node.goto_target(float(x), float(y), float(z), float(speed))
    return jsonify({"status": "navigating"})

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()