#!/usr/bin/env python3
"""RGB object detector with thermal-world compatible coordinates.

Run as a separate ROS 2 node; the localizer remains responsible for thermal
candidate segmentation, geometric localisation, and quality gating.

Requires: ultralytics, cv2, numpy, rclpy, tf2_ros. First use of a YOLO
weights name downloads weights; supply a local --ros-args -p model:=... path
for offline operation.
"""
import json
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, TransformException

from target_geometry import pixel_to_ground

ANIMAL_LABELS = frozenset(('bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                           'elephant', 'bear', 'zebra', 'giraffe'))


def decode_rgb(msg):
    """Return uint8 HxWx3 BGR array, respecting padding/step."""
    encoding = msg.encoding.lower()
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(encoding)
    if channels is None:
        raise ValueError('Unsupported RGB encoding: ' + msg.encoding)
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if len(raw) < msg.step * msg.height:
        raise ValueError('Incomplete RGB image buffer')
    padded = raw[:msg.step * msg.height].reshape(msg.height, msg.step)
    pix = padded[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
    if encoding == 'bgr8':
        return np.ascontiguousarray(pix)
    if encoding == 'rgb8':
        return np.ascontiguousarray(pix[:, :, ::-1])
    if encoding == 'bgra8':
        return np.ascontiguousarray(pix[:, :, :3])
    return np.ascontiguousarray(pix[:, :, [2, 1, 0]])


class RgbSemanticDetector(Node):
    def __init__(self):
        super().__init__('rgb_semantic_detector')
        self.declare_parameter('rgb_topic', '/parrot1/camera/image')
        self.declare_parameter('camera_frame', 'parrot1_camera_link')
        self.declare_parameter('fixed_frame', 'parrot1_odom')
        self.declare_parameter('model', 'yolo26n.pt')
        self.declare_parameter('confidence', 0.35)
        self.declare_parameter('image_size', 640)
        self.declare_parameter('inference_hz', 1.0)
        self.declare_parameter('horizontal_fov_rad', 2.0944)
        self.declare_parameter('ground_z_m', 0.0)
        self.declare_parameter('max_target_distance_m', 30.0)
        self.declare_parameter('cpu_only', True)

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError('Install ultralytics for the Python interpreter used by ROS 2. '
                               'See README_HUMAN_CLASSIFIER.md') from exc
        self.model = YOLO(self.get_parameter('model').value)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.rgb = None
        self.last_processed = -1.0
        self.last_warn = 0.0
        self.pub = self.create_publisher(String, '/parrot1/rescue/rgb_objects', 10)
        self.create_subscription(Image, self.get_parameter('rgb_topic').value,
                                 self.on_rgb, qos_profile_sensor_data)
        self.create_timer(0.1, self.process)
        self.get_logger().info('RGB classifier ready; human=person, animal=COCO animal classes. '
                               'Unrecognised wildlife stays UNKNOWN, not HUMAN.')

    def on_rgb(self, msg):
        self.rgb = msg

    def warn(self, message):
        if time.monotonic()-self.last_warn > 4.0:
            self.get_logger().warning(message)
            self.last_warn = time.monotonic()

    def process(self):
        msg = self.rgb
        if msg is None:
            return
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # Use simulation time (not wall clock) for frame interval and fusion.
        if stamp_sec <= self.last_processed or stamp_sec - self.last_processed < 1.0 / max(0.1, float(self.get_parameter('inference_hz').value)):
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                self.get_parameter('fixed_frame').value,
                self.get_parameter('camera_frame').value,
                Time.from_msg(msg.header.stamp))
        except TransformException as exc:
            self.warn(f'RGB pose TF unavailable: {exc}')
            return
        self.last_processed = stamp_sec
        try:
            frame = decode_rgb(msg)
            results = self.model.predict(
                source=frame, conf=float(self.get_parameter('confidence').value),
                imgsz=int(self.get_parameter('image_size').value),
                device='cpu' if self.get_parameter('cpu_only').value else None,
                verbose=False)
        except Exception as exc:
            self.warn(f'YOLO inference failed: {exc}')
            return

        origin = tf.transform.translation
        rot = tf.transform.rotation
        xyz0 = (origin.x, origin.y, origin.z)
        quat = (rot.x, rot.y, rot.z, rot.w)
        objects = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                label = str(result.names[cls_id]).lower()
                kind = 'human' if label == 'person' else 'animal' if label in ANIMAL_LABELS else None
                if kind is None:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                foot_u, foot_v = (x1 + x2) / 2.0, y2
                xyz = pixel_to_ground(
                    foot_u, foot_v, msg.width, msg.height,
                    float(self.get_parameter('horizontal_fov_rad').value),
                    xyz0, quat,
                    float(self.get_parameter('ground_z_m').value),
                    float(self.get_parameter('max_target_distance_m').value))
                if xyz is None or not all(math.isfinite(v) for v in xyz):
                    continue
                objects.append({
                    'label': label, 'classification': kind,
                    'confidence': round(float(box.conf[0].item()), 4),
                    'xyz_m': [round(float(v), 3) for v in xyz],
                    'bbox_xyxy': [round(v, 1) for v in (x1, y1, x2, y2)],
                    'foot_pixel': [round(foot_u, 1), round(foot_v, 1)]})
        payload = {
            'sim_time_sec': stamp_sec,
            'frame_id': self.get_parameter('fixed_frame').value,
            'camera_frame': self.get_parameter('camera_frame').value,
            'objects': objects,
        }
        self.pub.publish(String(data=json.dumps(payload)))
        self.get_logger().info('RGB classified %d human(s), %d animal(s)' % (
            sum(x['classification'] == 'human' for x in objects),
            sum(x['classification'] == 'animal' for x in objects)))


def main(args=None):
    rclpy.init(args=args)
    node = RgbSemanticDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
