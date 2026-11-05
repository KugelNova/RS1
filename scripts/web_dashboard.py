#v5 Navigation buttons, live feed thermal sensor, RGB Camera, Lidar scan, add manual navigation control with co-ordinate entry (GPS autopilot), added speed control, drone orientation  visualizer added, added controls to angular orientation and normalise its orientation

from flask import Flask, render_template_string, jsonify, Response, request
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
import xml.etree.ElementTree as ET
from rescue_drone_node import RescueDroneNode

app = Flask(__name__)


# Initialize ROS 2
rclpy.init(args=None)
ros_node = RescueDroneNode()
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
      
      .sidebar { width: 370px; background: #1e1e1e; padding: 20px; border-right: 3px solid #333; display: flex; flex-direction: column; overflow-y: auto; z-index: 10; flex-shrink: 0;}
      
      /* New Status Panel */
      .status-panel { background: #002233; border: 2px solid #0088cc; border-radius: 8px; padding: 15px; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.4); }
      .status-panel h3 { margin: 0 0 10px 0; color: #00ccff; text-align: center; text-transform: uppercase; font-size: 1rem; border-bottom: 1px solid #005588; padding-bottom: 5px;}
      .status-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 1.1rem; }
      .status-val { color: #00ffcc; font-weight: bold; font-family: monospace; font-size: 1.2rem;}
      .status-alert { color: #ff3333; font-weight: bold; font-family: monospace; font-size: 1.2rem;}
      .rescue-alert { color: #ffa870; font-size: 0.85rem; margin-bottom: 8px; white-space: pre-line; }
      
      /* Accordion Menus */
      .accordion { background-color: #2a2a2a; color: #fff; cursor: pointer; padding: 15px; width: 100%; border: 1px solid #444; border-radius: 6px; text-align: left; outline: none; font-size: 16px; font-weight: bold; transition: 0.3s; margin-bottom: 5px; }
      .active, .accordion:hover { background-color: #005588; border-color: #00ccff; }
      .accordion:after { content: '+'; font-size: 16px; float: right; margin-left: 5px; }
      .active:after { content: '-'; }

      .panel { padding: 15px 0; display: none; background-color: transparent; overflow: hidden; }
      
      .d-pad { display: grid; grid-template-columns: repeat(3, 70px); gap: 10px; justify-content: center; margin-bottom: 15px;}
      .btn { width: 70px; height: 70px; background: #3a3a3a; color: white; border: 2px solid #555; border-radius: 12px; font-size: 26px; cursor: pointer; user-select: none; transition: background 0.1s, border-color 0.1s; }
      .btn:active { background: #00aa00; border-color: #00ff00; }
      
      .btn-wide { width: 100%; font-size: 16px; padding: 12px; cursor: pointer; border: none; border-radius: 8px; color: white; margin-bottom: 10px; font-weight: bold;}
      .stop-btn { background: #880000; border-color: #ff3333; border: 2px solid; }
      
      .hazard-box { background: #2a1a1a; border: 1px solid #662222; padding: 10px; border-radius: 8px; margin-bottom: 15px; }
      .hazard-box h4 { margin: 0 0 10px 0; color: #ff9933; text-align: center; }
      .param-inputs { display: flex; gap: 5px; margin-bottom: 10px; }
      .param-inputs input { width: 100%; padding: 8px; background: #111; color: #fff; border: 1px solid #555; border-radius: 4px; text-align: center; }
      
      .main-content { position: relative; flex: 1; padding: 15px; display: flex; flex-direction: column; background: #121212; min-width: 0; }
      .attitude-widget { position: absolute; bottom: 15px; right: 15px; background: rgba(30, 30, 30, 0.85); padding: 10px; border-radius: 8px; border: 1px solid #444; z-index: 100; display: flex; flex-direction: column; align-items: center; box-shadow: 0 4px 10px rgba(0,0,0,0.6); backdrop-filter: blur(4px); }
      .attitude-widget h3 { font-size: 0.8rem; margin: 0 0 5px 0; color: #ccc; text-transform: uppercase; }
      .attitude-widget canvas { background: #0a0a0a; border: 1px solid #555; border-radius: 4px; }
      .dashboard-grid { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; gap: 15px; flex: 1; min-height: 0; }
      .grid-box { background: #1e1e1e; padding: 10px; border-radius: 12px; border: 1px solid #333; display: flex; flex-direction: column; align-items: center; box-shadow: 0 4px 8px rgba(0,0,0,0.4); overflow: hidden; }
      .grid-box h3 { margin: 0 0 8px 0; font-size: 1.1rem; color: #ccc; flex-shrink: 0; }
      .grid-content { width: 100%; flex: 1; min-height: 0; display: flex; justify-content: center; align-items: center; background: #000; border-radius: 8px; border: 2px solid #444; overflow: hidden; }
      .grid-content img, .grid-content canvas { width: 100%; height: 100%; object-fit: contain; }
    </style>
  </head>
  <body>
    
    <!-- LEFT SIDEBAR -->
    <div class="sidebar">
        
        <!-- Top Status Area -->
        <div class="status-panel">
            <h3>Radar & Telemetry</h3>
            <div class="status-row"><span>Confirmed Humans:</span> <span class="status-val" id="det-humans">0</span></div>
            <div class="rescue-alert" id="latest-rescue-alert" role="status" hidden></div>
            <div class="status-row"><span>Hazard Warning:</span> <span class="status-val" id="det-hazards">NONE</span></div>
            <div class="status-row"><span>Position X:</span> <span class="status-val" id="pos-x">0.00</span></div>
            <div class="status-row"><span>Position Y:</span> <span class="status-val" id="pos-y">0.00</span></div>
            <div class="status-row"><span>Altitude Z:</span> <span class="status-val" id="pos-z">0.00</span></div>
        </div>

        <!-- Flight Controls Accordion -->
        <button class="accordion" onclick="togglePanel('nav-panel', this)">Flight & Navigation</button>
        <div class="panel" id="nav-panel">
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
                <button class="btn" style="font-size: 16px;" onmousedown="sendCommand('roll_left')" onmouseup="sendCommand('stop')">Roll<br>Left</button>
                <button class="btn" style="font-size: 16px;" onmousedown="sendCommand('pitch_down')" onmouseup="sendCommand('stop')">Nose<br>Down</button>
                <button class="btn" style="font-size: 16px;" onmousedown="sendCommand('pitch_up')" onmouseup="sendCommand('stop')">Nose<br>Up</button>
                <button class="btn" style="font-size: 16px;" onmousedown="sendCommand('roll_right')" onmouseup="sendCommand('stop')">Roll<br>Right</button>
            </div>
            
            <div class="d-pad" style="grid-template-columns: repeat(4, 70px);">
                <button class="btn" onmousedown="sendCommand('up')" onmouseup="sendCommand('stop')">⇡</button>
                <button class="btn" onmousedown="sendCommand('down')" onmouseup="sendCommand('stop')">⇣</button>
                <button class="btn" onmousedown="sendCommand('turn_left')" onmouseup="sendCommand('stop')">↺</button>
                <button class="btn" onmousedown="sendCommand('turn_right')" onmouseup="sendCommand('stop')">↻</button>
            </div>
            
            <button class="btn-wide" style="background: #8800aa;" onclick="sendCommand('normalize')">⚖️ Auto-Level</button>
            <div style="display:flex; gap:10px;">
                <button class="btn-wide" style="background: #005588;" onclick="sendCommand('auto_ccw')">🔄 Patrol (CCW)</button>
                <button class="btn-wide" style="background: #005588;" onclick="sendCommand('auto_cw')">🔃 Patrol (CW)</button>
            </div>
        </div>

        <!-- Hazard Management Accordion -->
        <button class="accordion" onclick="togglePanel('haz-panel', this)">Hazard Management</button>
        <div class="panel" id="haz-panel">
            
            <!-- Fire Config -->
            <div class="hazard-box">
                <h4>🔥 Fire Simulation</h4>
                <div class="param-inputs">
                    <input type="number" id="fire-x" placeholder="X" value="5.0" step="0.5" title="X Coordinate">
                    <input type="number" id="fire-y" placeholder="Y" value="0.0" step="0.5" title="Y Coordinate">
                    <input type="number" id="fire-z" placeholder="Z" value="0.0" step="0.5" title="Z Coordinate">
                    <input type="number" id="fire-s" placeholder="Size" value="1.0" step="0.5" title="Scale factor">
                </div>
                <div style="display:flex; gap:10px;">
                    <button class="btn-wide" style="background: #cc5500;" onclick="spawnDynamicHazard('fire')">Spawn</button>
                    <button class="btn-wide stop-btn" onclick="clearHazard('runtime_fire_gui')">Clear</button>
                </div>
            </div>

            <!-- Flood Config -->
            <div class="hazard-box" style="border-color: #224466;">
                <h4 style="color: #66b3ff;">🌊 Flood Simulation</h4>
                <div class="param-inputs">
                    <input type="number" id="flood-x" placeholder="X" value="0.0" step="0.5">
                    <input type="number" id="flood-y" placeholder="Y" value="0.0" step="0.5">
                    <input type="number" id="flood-z" placeholder="Z" value="0.4" step="0.5">
                    <input type="number" id="flood-s" placeholder="Size" value="1.0" step="0.5">
                </div>
                <div style="display:flex; gap:10px;">
                    <button class="btn-wide" style="background: #0055aa;" onclick="spawnDynamicHazard('flood')">Spawn</button>
                    <button class="btn-wide stop-btn" onclick="clearHazard('runtime_flood_gui')">Clear</button>
                </div>
            </div>
            
        </div>
    </div>

    <!-- RIGHT MAIN (GRID) -->
    <div class="main-content">
        <div class="attitude-widget">
            <h3>📐 Orientation</h3>
            <canvas id="attitudeMap" width="160" height="160"></canvas>
        </div>
        <div class="dashboard-grid">
            <div class="grid-box">
                <h3>🔴 Thermal</h3>
                <div class="grid-content"><img src="/video_feed/thermal"></div>
            </div>
            <div class="grid-box">
                <h3>🔵 True-Color RGB</h3>
                <div class="grid-content"><img src="/video_feed/rgb"></div>
            </div>
            <div class="grid-box">
                <h3>🟢 LiDAR Radar</h3>
                <div class="grid-content"><img src="/video_feed/lidar"></div>
            </div>
            <div class="grid-box">
                <h3>🗺️ Global Positioning Map</h3>
                <div class="grid-content"><canvas id="droneMap" width="800" height="600"></canvas></div>
            </div>
        </div>
    </div>

    <script>
        // Accordion Logic
        function togglePanel(panelId, btn) {
            btn.classList.toggle("active");
            var panel = document.getElementById(panelId);
            if (panel.style.display === "block") {
                panel.style.display = "none";
            } else {
                panel.style.display = "block";
            }
        }
        

        // Commands
        function sendCommand(cmd) { fetch('/cmd/' + cmd); }
        
        function spawnDynamicHazard(type) {
            let x = document.getElementById(`${type}-x`).value || "0.0";
            let y = document.getElementById(`${type}-y`).value || "0.0";
            let z = document.getElementById(`${type}-z`).value || "0.0";
            let s = document.getElementById(`${type}-s`).value || "1.0";
            fetch(`/hazard/spawn/${type}?x=${x}&y=${y}&z=${z}&scale=${s}`)
                .then(r => r.json()).then(console.log);
        }
        
        function clearHazard(target) {
            fetch(`/hazard/remove/${target}`).then(r => r.json()).then(console.log);
        }

        // Telemetry Mapping
        const mapCanvas = document.getElementById('droneMap');
        const mapCtx = mapCanvas.getContext('2d');
        const pixelsPerMeter = 16; 
        let flightPath = [];

        function drawMap(data) {
            mapCtx.fillStyle = '#0a0a0a';
            mapCtx.fillRect(0, 0, mapCanvas.width, mapCanvas.height);
            const cx = mapCanvas.width / 2;
            const cy = mapCanvas.height / 2;

            // Draw Grid
            mapCtx.strokeStyle = '#222';
            mapCtx.lineWidth = 1;
            mapCtx.beginPath();
            for(let i = 0; i < mapCanvas.width; i += (pixelsPerMeter * 5)) {
                mapCtx.moveTo(i, 0); mapCtx.lineTo(i, mapCanvas.height);
                mapCtx.moveTo(0, i); mapCtx.lineTo(mapCanvas.width, i);
            }
            mapCtx.stroke();

            let d_px = cx - (data.y * pixelsPerMeter);
            let d_py = cy - (data.x * pixelsPerMeter);
            
            mapCtx.save();
            mapCtx.translate(d_px, d_py);
            mapCtx.rotate(-data.yaw);
            mapCtx.fillStyle = '#ff3333';
            mapCtx.beginPath();
            mapCtx.moveTo(14, 0); mapCtx.lineTo(-8, -8); mapCtx.lineTo(-8, 8);
            mapCtx.fill();
            mapCtx.restore();
        }

        const attCanvas = document.getElementById('attitudeMap');
        const attCtx = attCanvas.getContext('2d');

        function rotate3D(pt, roll, pitch, yaw) {
            let y1 = pt.y * Math.cos(roll) - pt.z * Math.sin(roll);
            let z1 = pt.y * Math.sin(roll) + pt.z * Math.cos(roll);
            let x2 = pt.x * Math.cos(pitch) + z1 * Math.sin(pitch);
            let z2 = -pt.x * Math.sin(pitch) + z1 * Math.cos(pitch);
            let x3 = x2 * Math.cos(yaw) - y1 * Math.sin(yaw);
            let y3 = x2 * Math.sin(yaw) + y1 * Math.cos(yaw);
            return {x: x3, y: y3, z: z2};
        }

        function drawAttitude(roll, pitch, yaw) {
            attCtx.fillStyle = '#0a0a0a';
            attCtx.fillRect(0, 0, attCanvas.width, attCanvas.height);
            const cx = attCanvas.width / 2, cy = attCanvas.height / 2;
            let xr = rotate3D({x: 40, y: 0, z: 0}, roll, pitch, yaw);
            let yr = rotate3D({x: 0, y: 40, z: 0}, roll, pitch, yaw);
            let zr = rotate3D({x: 0, y: 0, z: 40}, roll, pitch, yaw);

            function project(pt) { return { x: cx - pt.y + pt.x * 0.4, y: cy - pt.z - pt.x * 0.4 }; }
            let pX = project(xr), pY = project(yr), pZ = project(zr), pOrigin = project({x:0,y:0,z:0});

            attCtx.lineWidth = 3;
            attCtx.strokeStyle = '#ff3333'; attCtx.beginPath(); attCtx.moveTo(pOrigin.x, pOrigin.y); attCtx.lineTo(pX.x, pX.y); attCtx.stroke();
            attCtx.strokeStyle = '#33ff33'; attCtx.beginPath(); attCtx.moveTo(pOrigin.x, pOrigin.y); attCtx.lineTo(pY.x, pY.y); attCtx.stroke();
            attCtx.strokeStyle = '#3333ff'; attCtx.beginPath(); attCtx.moveTo(pOrigin.x, pOrigin.y); attCtx.lineTo(pZ.x, pZ.y); attCtx.stroke();

            // Add text overlay
            attCtx.fillStyle = '#fff';
            attCtx.font = "10px monospace";
            attCtx.fillText(`R:${(roll * 180/Math.PI).toFixed(0)}° P:${(pitch * 180/Math.PI).toFixed(0)}°`, 5, 15);
            attCtx.fillText(`Y:${(yaw * 180/Math.PI).toFixed(0)}°`, 5, 28);
        }

        setInterval(function() {
            fetch('/telemetry')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('pos-x').innerText = data.x.toFixed(2);
                    document.getElementById('pos-y').innerText = data.y.toFixed(2);
                    document.getElementById('pos-z').innerText = data.z.toFixed(2);
                    
                    document.getElementById('det-humans').innerText = data.human_count;
                    const alert = data.rescue_alert;
                    if (alert) {
                        const alertEl = document.getElementById('latest-rescue-alert');
                        let message = 'Confirmed human #' + alert.id + ' at X ' +
                            alert.x_m.toFixed(2) + ', Y ' + alert.y_m.toFixed(2) +
                            ', Z ' + alert.z_m.toFixed(2) + ' m (' + alert.frame_id + ')';
                        if (alert.waypoint_frame_id) {
                            message += '\\nGround waypoint at X ' + alert.waypoint_x_m.toFixed(2) +
                                ', Y ' + alert.waypoint_y_m.toFixed(2) +
                                ', Z ' + alert.waypoint_z_m.toFixed(2) +
                                ' m (' + alert.waypoint_frame_id + ')';
                        }
                        if (alert.location_source === 'rgbd_lidar') {
                            message += '\\nLocation refined with LiDAR';
                        }
                        if (alertEl.textContent !== message) {
                            alertEl.hidden = false;
                            alertEl.textContent = message;
                        }
                    }
                    drawAttitude(data.roll, data.pitch, data.yaw);
                    
                    const hazEl = document.getElementById('det-hazards');
                    hazEl.innerText = data.hazard;
                    if(data.hazard !== "NONE") {
                        hazEl.className = "status-alert";
                    } else {
                        hazEl.className = "status-val";
                    }
                    
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
    hazard = "NONE"
    fire_r = getattr(ros_node, 'fire_rgb', False)
    fire_t = getattr(ros_node, 'fire_thermal', False)
    
    if fire_r and fire_t:
        hazard = "FIRE CONFIRMED"
    elif fire_r and not fire_t:
        hazard = "FIRE DETECTED ?"
    elif getattr(ros_node, 'flood_detected', False):
        hazard = f"FLOOD Z:{ros_node.z:.1f}"
        
    return jsonify({
        "x": ros_node.x, "y": ros_node.y, "z": ros_node.z,
        "roll": ros_node.roll, "pitch": ros_node.pitch, "yaw": ros_node.yaw,
        "state": ros_node.flight_state,
        "navigating": ros_node.is_navigating,
        "human_count": ros_node.human_count,
        "rescue_alert": ros_node.latest_rescue_alert,
        "hazard": hazard
    })

@app.route('/cmd/<direction>')
def command(direction):
    speed = 1.0 
    lx, ly, lz, ax, ay, az = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
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
    
    elif direction == 'pitch_down': ay = speed
    elif direction == 'pitch_up': ay = -speed
    elif direction == 'roll_left': ax = -speed
    elif direction == 'roll_right': ax = speed
    
    elif direction == 'auto_ccw': lx = speed; az = 0.5
    elif direction == 'auto_cw': lx = speed; az = -0.5
    
    elif direction == 'stop' or direction == 'normalize': pass 

    ros_node.set_command(lx, ly, lz, ax, ay, az)
    return jsonify({"status": "success"})

@app.route('/goto/<string:x>/<string:y>/<string:z>/<string:speed>')
def goto_target(x, y, z, speed):
    ros_node.goto_target(float(x), float(y), float(z), float(speed))
    return jsonify({"status": "navigating"})

import os
import subprocess

@app.route('/hazard/spawn/<hazard_type>')
def spawn_hazard(hazard_type):
    try:
        x = request.args.get('x', default='0.0')
        y = request.args.get('y', default='0.0')
        z = request.args.get('z', default='0.0')
        scale = float(request.args.get('scale', default='1.0'))
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        worlds_dir = os.path.abspath(os.path.join(current_dir, '..', 'worlds'))
        
        # --- FIXED LOGIC ---
        # We now explicitly inject the target GUI name directly into the temporary XML structure
        # to guarantee Gazebo registers the entity with this name, bypassing ros_gz_sim rename bugs.
        def create_scaled_sdf(base_sdf_path, scale_factor, entity_name):
            tree = ET.parse(base_sdf_path)
            root = tree.getroot()
            
            for model in root.findall('.//model'):
                model.set('name', entity_name)
            
            for tag in ['.//size', './/scale']:
                for elem in root.findall(tag):
                    dims = [float(d) * scale_factor for d in elem.text.split()]
                    elem.text = ' '.join([f"{d:.4f}" for d in dims])
                    
            temp_path = f"/tmp/scaled_{os.path.basename(base_sdf_path)}"
            tree.write(temp_path)
            return temp_path

        if hazard_type == 'fire':
            fire_sdf = os.path.join(worlds_dir, 'custom_fire.sdf')
            scaled_sdf = create_scaled_sdf(fire_sdf, scale, 'runtime_fire_gui')
            cmd = f"ros2 run ros_gz_sim create -world rescue_forest -file {scaled_sdf} -name runtime_fire_gui -x {x} -y {y} -z {z}"
            subprocess.Popen(cmd, shell=True)
            
        elif hazard_type == 'flood':
            flood_sdf = os.path.join(worlds_dir, 'flood_water.sdf')
            scaled_sdf = create_scaled_sdf(flood_sdf, scale, 'runtime_flood_gui')
            cmd = f"ros2 run ros_gz_sim create -world rescue_forest -file {scaled_sdf} -name runtime_flood_gui -x {x} -y {y} -z {z}"
            subprocess.Popen(cmd, shell=True)
            
        return jsonify({"status": "success", "action": f"spawned {hazard_type} at {x},{y},{z} (Scale: {scale})"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    
@app.route('/hazard/remove/<hazard_name>')
def remove_hazard(hazard_name):
    try:
        # --- FIXED LOGIC ---
        # 1. Switched req/rep message types in cmd_gz from `ignition.msgs` to `gz.msgs` for Harmonic+ compatibility
        # 2. Used proper escaped double-quotes (\") in the --req payload to satisfy Protobuf text format requirements
        cmd_gz = f"gz service -s /world/rescue_forest/remove --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean --timeout 2000 --req 'name: \"{hazard_name}\", type: MODEL'"
        cmd_ign = f"ign service -s /world/rescue_forest/remove --reqtype ignition.msgs.Entity --reptype ignition.msgs.Boolean --timeout 2000 --req 'name: \"{hazard_name}\", type: MODEL'"
        
        import os
        os.system(f"{cmd_gz} || {cmd_ign}")
        return jsonify({"status": "success", "action": f"removed {hazard_name}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
        
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()