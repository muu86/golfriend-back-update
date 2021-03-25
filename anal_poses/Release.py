import numpy as np
from anal_poses.utils import p3_angle


# 1번 자세
class Release:
    def __init__(self, kp, face_on=True):
        self.kp = kp
        self.face_on = face_on
        self.feedback = dict()

    def run(self):
        return self.feedback