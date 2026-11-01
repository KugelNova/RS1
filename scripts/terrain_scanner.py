#!/usr/bin/env python3

import csv
import math
import os

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_msgs.msg import TFMessage


class TerrainProfiler(Node):

    def __init__(self):
        super().__init__('terrain_scanner')

        self.create_subscription(
            TFMessage,
            '/world/simple_trees/dynamic_pose/info',
            self.pose_callback,
            10
        )

        self.create_subscription(
            PointCloud2,
            '/parrot1/camera/depth/points',
            self.cloud_callback,
            10
        )

        self.drone_position = None
        self.drone_rotation = None

        self.grid_size = 0.5
        self.max_sensor_range = 25.0
        self.point_step = 20

        # Each grid cell stores [lowest_z, number_of_points].
        self.terrain_grid = {}

        self.output_path = os.path.expanduser(
            '~/41068_ws/terrain_profile.csv'
        )

        self.save_timer = self.create_timer(5.0, self.save_profile)

        # Camera pose relative to the drone base.
        self.camera_translation = np.array([0.25, 0.0, 0.25])

        pitch = math.radians(45.0)

        self.camera_rotation = np.array([
            [math.cos(pitch), 0.0, math.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(pitch), 0.0, math.cos(pitch)]
        ])

        self.get_logger().info('Terrain profiler started')

    def quaternion_to_matrix(self, x, y, z, w):
        return np.array([
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w)
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w)
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y)
            ]
        ])

    def pose_callback(self, msg):
        for transform in msg.transforms:
            if transform.child_frame_id == 'parrot1':
                translation = transform.transform.translation
                rotation = transform.transform.rotation

                self.drone_position = np.array([
                    translation.x,
                    translation.y,
                    translation.z
                ])

                self.drone_rotation = self.quaternion_to_matrix(
                    rotation.x,
                    rotation.y,
                    rotation.z,
                    rotation.w
                )
                break

    def cloud_callback(self, msg):
        if self.drone_position is None:
            return

        points = point_cloud2.read_points(
            msg,
            field_names=('x', 'y', 'z'),
            skip_nans=True
        )

        accepted = 0

        for index, point in enumerate(points):
            if index % self.point_step != 0:
                continue

            camera_point = np.array([
                float(point[0]),
                float(point[1]),
                float(point[2])
            ])

            distance = np.linalg.norm(camera_point)

            if distance < 0.3 or distance > self.max_sensor_range:
                continue

            base_point = (
                self.camera_rotation @ camera_point
                + self.camera_translation
            )

            world_point = (
                self.drone_rotation @ base_point
                + self.drone_position
            )

            world_x, world_y, world_z = world_point

            # Reject unlikely measurements above the drone.
            if world_z > self.drone_position[2] + 0.5:
                continue

            cell_x = math.floor(world_x / self.grid_size)
            cell_y = math.floor(world_y / self.grid_size)
            cell = (cell_x, cell_y)

            if cell not in self.terrain_grid:
                self.terrain_grid[cell] = [world_z, 1]
            else:
                # Lowest observed point is treated as ground.
                self.terrain_grid[cell][0] = min(
                    self.terrain_grid[cell][0],
                    world_z
                )
                self.terrain_grid[cell][1] += 1

            accepted += 1

        self.get_logger().info(
            f'Accepted {accepted} points; '
            f'mapped {len(self.terrain_grid)} cells'
        )

    def save_profile(self):
        if not self.terrain_grid:
            return

        with open(self.output_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow([
                'x',
                'y',
                'terrain_z',
                'sample_count'
            ])

            for (cell_x, cell_y), data in sorted(
                self.terrain_grid.items()
            ):
                world_x = (cell_x + 0.5) * self.grid_size
                world_y = (cell_y + 0.5) * self.grid_size
                terrain_z = data[0]
                sample_count = data[1]

                writer.writerow([
                    world_x,
                    world_y,
                    terrain_z,
                    sample_count
                ])

        self.get_logger().info(
            f'Saved terrain profile to {self.output_path}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = TerrainProfiler()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_profile()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
