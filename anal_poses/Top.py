import numpy as np
from anal_poses.utils import p3_angle


class Top:
    def __init__(self, kp):
        self.kp = kp
        self.feedback = dict()

    # 왼 팔의 구부러짐 체크
    def bending_left_arm(self):
        lshoulder = self.kp[3][5]
        lelbow = self.kp[3][6]
        lwrist = self.kp[3][7]

        angle = p3_angle(lshoulder, lelbow, lwrist)

        if 120 <= angle:
            self.feedback["bending_arms"] = {
                0: 2,
                1: angle,
                2: "팔 구부러짐이 없습니다."
            }
        elif 100 <= angle:
            self.feedback["bending_arms"] = {
                0: 1,
                1: angle,
                2: "손을 몸에서 멀리 밀면 클럽이 더 먼 거리를 이동하게 됩니다. 샷의 일관성 또한 향상됩니다. "
            }
        else:
            self.feedback["bending_arms"] = {
                0: 0,
                1: angle,
                2: "손을 몸에서 멀리 밀면 클럽이 더 먼 거리를 이동하게 됩니다. 샷의 일관성 또한 향상됩니다."
            }

    def reverse_pivot(self):
        lshoulder = self.kp[3][5]
        lfoot = self.kp[3][14]
        rfoot = self.kp[3][11]

        angle = p3_angle(lshoulder, lfoot, rfoot)

        if 50 <= angle <= 85:
            self.feedback["reverse_pivot"] = {
                0: 2,
                1: angle,
                2: "체중 이동이 정상적입니다.",
            }
        elif 85 < angle <= 100:
            self.feedback["reverse_pivot"] = {
                0: 1,
                1: angle,
                2: "백스윙 자세에서 체중이 앞발로 이동하고 있습니다. 역피봇은 볼의 윗부분을 맟출 확률을 높이고 thin shots를 유도합니다.",
            }
        else:
            self.feedback["reverse_pivot"] = {
                0: 0,
                1: angle,
                2: "백스윙 자세에서 체중이 앞발로 이동하고 있습니다. 역피봇은 볼의 윗부분을 맟출 확률을 높이고 thin shots를 유도합니다.",
            }

    def left_knee_moving(self):
        lhip = self.kp[3][12]
        lknee = self.kp[3][13]
        lfoot = self.kp[3][14]

        angle = p3_angle(lhip, lknee, lfoot)

        if 165 <= angle:
            self.feedback["left_knee_moving"] = {
                0: 2,
                1: angle,
                2: "good"
            }
        elif 150 <= angle:
            self.feedback["left_knee_moving"] = {
                0: 1,
                1: angle,
                2: "체중 이동 중 왼 다리가 과다하게 이동해서는 안 됩니다. 왼 무릎이 공을 바라본다고 생각하세요."
            }
        else:
            self.feedback["left_knee_moving"] = {
                0: 0,
                1: angle,
                2: "체중 이동 중 왼 다리가 과다하게 이동해서는 안 됩니다. 왼 무릎이 공을 바라본다고 생각하세요."
            }

    def run(self):
        self.bending_left_arm()
        self.reverse_pivot()
        self.left_knee_moving()

        return self.feedback