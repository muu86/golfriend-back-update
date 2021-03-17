import numpy as np
from anal_poses.utils import p3_angle


# 1번 자세
class FollowThrough:
    def __init__(self, kp):
        self.kp = kp
        self.feedback = dict()

    def run(self):
        return self.feedback