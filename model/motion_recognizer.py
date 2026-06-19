from collections import deque
import numpy as np
import math

class MotionRecognizer:

    def __init__(self):
        pass

    def trajectory_length(self, points):

        if len(points) < 2:
            return 0

        length = 0.0

        for i in range(1, len(points)):
            p0 = np.array(points[i - 1])
            p1 = np.array(points[i])

            length += np.linalg.norm(p1 - p0)

        return length

    def is_stopped(self, points):

        if len(points) < 10:
            return False

        pts = np.array(points, dtype=np.float32)

        center = np.mean(pts, axis=0)

        spread = np.mean(
            np.linalg.norm(pts - center, axis=1)
        )

        return spread < 12

    def is_circle(self, points):

        if len(points) < 25:
            return False

        pts = np.array(points, dtype=np.float32)

        center = np.mean(pts, axis=0)

        radii = np.linalg.norm(
            pts - center,
            axis=1
        )

        mean_r = np.mean(radii)

        if mean_r < 20:
            return False

        circularity = np.std(radii) / mean_r

        start_end_dist = np.linalg.norm(
            pts[0] - pts[-1]
        )

        circumference = 2 * math.pi * mean_r

        path_len = self.trajectory_length(points)

        return (
            circularity < 0.35
            and start_end_dist < mean_r
            and path_len > 0.7 * circumference
        )

    def classify(self, point_history):

        points = [
            p for p in point_history
            if p[0] != 0 and p[1] != 0
        ]

        if len(points) < 10:
            return "TRACKING"

        if self.is_stopped(points):
            return "STOPPED"

        if self.is_circle(points):
            return "CIRCLE"

        displacement = np.linalg.norm(
            np.array(points[-1]) -
            np.array(points[0])
        )

        if displacement > 40:
            return "MOVING"

        return "TRACKING"