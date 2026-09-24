#v2 web app with navigational buttons

from flask import Flask, render_template_string, send_from_directory, jsonify
import os
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

app = Flask(__name__)
SNAP_DIR = "/tmp/rescue_snaps"

# -------------------------------------------------------------
# 1. ROS 2 Web Teleop Node (Continuous Publishing)
# -------------------------------------------------------------
class WebTeleopNode(Node):
    def __init__(self):
        super().__init__('web_teleop_node')
        self.cmd_pub = self.create_publisher(Twist, '/parrot1/cmd_vel', 10)
        self.current_twist = Twist()
        
        # Publish the command continuously at 10Hz to prevent Gazebo timeouts
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('Web Teleop Node Started with 10Hz continuous publish.')

    def set_command(self, lx=0.0, ly=0.0, lz=0.0, az=0.0):
        self.current_twist.linear.x = float(lx)
        self.current_twist.linear.y = float(ly)
        self.current_twist.linear.z = float(lz)
        self.current_twist.angular.z = float(az)

    def timer_callback(self):
        self.cmd_pub.publish(self.current_twist)

# Initialize ROS 2 and start the thread
rclpy.init(args=None)
ros_node = WebTeleopNode()

def ros_spin_thread():
    rclpy.spin(ros_node)

threading.Thread(target=ros_spin_thread, daemon=True).start()

# -------------------------------------------------------------
# 2. Web UI Template (HTML + CSS + JS)
# -------------------------------------------------------------
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <title>Drone Rescue Web UI</title>
    <style>
      body { font-family: Arial, sans-serif; background: #222; color: #fff; text-align: center; }
      .container { display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin-top: 20px; }
      .card { background: #333; padding: 15px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); }
      img { max-width: 300px; height: auto; border: 2px solid #555; border-radius: 4px; }
      
      /* Controller Styles */
      .control-panel { background: #333; padding: 20px; border-radius: 10px; display: inline-block; margin-bottom: 20px;}
      .d-pad { display: grid; grid-template-columns: repeat(3, 60px); gap: 10px; justify-content: center; margin-bottom: 15px;}
      .btn { width: 60px; height: 60px; background: #444; color: white; border: 2px solid #555; border-radius: 8px; font-size: 24px; cursor: pointer; user-select: none; }
      .btn:active { background: #00aa00; border-color: #00ff00; }
      .btn-wide { width: 200px; font-size: 16px; padding: 10px; cursor: pointer; border: 2px solid #555; border-radius: 8px; color: white;}
      .btn-auto { background: #005588; font-weight: bold; }
      .btn-auto:active { background: #0088cc; }
      .stop-btn { background: #880000; }
      .stop-btn:active { background: #ff0000; }
    </style>
  </head>
  <body>
    <h1>🚁 Drone Intelligence & Control UI</h1>
    
    <!-- Control Panel -->
    <div class="control-panel">
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
        
        <div class="d-pad" style="grid-template-columns: repeat(4, 60px);">
            <button class="btn" onmousedown="sendCommand('up')" onmouseup="sendCommand('stop')">⇡</button>
            <button class="btn" onmousedown="sendCommand('down')" onmouseup="sendCommand('stop')">⇣</button>
            <button class="btn" onmousedown="sendCommand('turn_left')" onmouseup="sendCommand('stop')">↺</button>
            <button class="btn" onmousedown="sendCommand('turn_right')" onmouseup="sendCommand('stop')">↻</button>
        </div>
        
        <!-- Auto Patrol Buttons (Using onclick instead of mouseup/down to lock them in) -->
        <div style="display: flex; gap: 10px; justify-content: center; margin-top: 15px;">
            <button class="btn-wide btn-auto" onclick="sendCommand('auto_ccw')">🔄 Auto Patrol (CCW)</button>
            <button class="btn-wide btn-auto" onclick="sendCommand('auto_cw')">🔃 Auto Patrol (CW)</button>
        </div>
    </div>

    <div>
        <button onclick="location.reload()" style="padding: 10px; background: #0088cc; color: white; border: none; border-radius: 5px; cursor: pointer; margin-bottom: 10px;">🔄 Refresh Snapshots</button>
    </div>

    <!-- Snapshots -->
    <div class="container">
      {% for img in images %}
        <div class="card">
          <h3>{{ img }}</h3>
          <img src="/images/{{ img }}" alt="{{ img }}">
        </div>
      {% endfor %}
    </div>

    <script>
        // Sends HTTP request to Flask backend
        function sendCommand(cmd) {
            fetch('/cmd/' + cmd);
        }
    </script>
  </body>
</html>
"""

# -------------------------------------------------------------
# 3. Flask Routes
# -------------------------------------------------------------
@app.route('/')
def index():
    if not os.path.exists(SNAP_DIR):
        images = []
    else:
        images = sorted([f for f in os.listdir(SNAP_DIR) if f.endswith('.png')])
    return render_template_string(HTML_TEMPLATE, images=images)

@app.route('/images/<path:filename>')
def get_image(filename):
    return send_from_directory(SNAP_DIR, filename)

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
    
    # Auto Patrol logic (Uses set_command so the timer keeps looping it)
    elif direction == 'auto_ccw': lx = speed; az = 0.5
    elif direction == 'auto_cw': lx = speed; az = -0.5
    
    elif direction == 'stop': pass # All zeros

    ros_node.set_command(lx, ly, lz, az)
    return jsonify({"status": "success", "command": direction})

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        # Clean up ROS 2 node on exit
        ros_node.destroy_node()
        rclpy.shutdown()