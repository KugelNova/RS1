from flask import Flask, render_template, jsonify, Response
import threading
import time
import rclpy
from rescue_drone_node import WebTeleopNode

app = Flask(__name__)

rclpy.init(args=None)
ros_node = WebTeleopNode()
threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True).start()

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

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed/<camera_type>')
def video_feed(camera_type):
    return Response(generate_frames(camera_type), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/telemetry')
def telemetry():
    return jsonify({
        "x": ros_node.x, "y": ros_node.y, "z": ros_node.z,
        "roll": ros_node.roll, "pitch": ros_node.pitch, "yaw": ros_node.yaw,
        "state": ros_node.flight_state,
        "navigating": ros_node.is_navigating,
        "tx": ros_node.target_x, "ty": ros_node.target_y,
        "human_detected": ros_node.human_detected,
        "detect_x": ros_node.detect_x,
        "detect_y": ros_node.detect_y
    })

@app.route('/cmd/<direction>')
def command(direction):
    speed = 0.5
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

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()