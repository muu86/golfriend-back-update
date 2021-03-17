import numpy as np
from anal_poses.utils import p3_angle
from anal_poses.utils import p2_diff


# 2번 자세
class BackSwing:
    def __init__(self, kp):
        self.kp = kp
        self.feedback = dict()

    # 스웨이 체크
    # 골반과 몸통이 회전하는 것이 아니라 오른쪽으로 밀리면서 체중 이동 하는 것
    def sway(self):
        lhip_address = self.kp[0][9]
        lhip_backswing = self.kp[2][9]

        diff = np.array(lhip_address) - np.array(lhip_backswing)

        # 640 * 640 이미지 기준
        # -50~ 50 사이
        if -10 <= diff[0] <= 10:
            self.feedback["sway"] = {
                0: 2,
                1: diff[0],
                2: "상체가 축을 중심으로 회전하고 있습니다.",
            }
        # elif -20 <= diff[0] <= 20:
        #     self.feedback["sway"] = {
        #         0: 4,
        #         1: diff[0]
        #     }
        elif -30 <= diff[0] <= 30:
            self.feedback["sway"] = {
                0: 1,
                1: diff[0],
                2: "골반과 몸통이 축을 중심으로 회전하지 않고, 좌우로 밀리고 있습니다. 스윙의 축이 변하여 정확한 임팩트가 어렵고 거리 손실을 보게 됩니다."
            }
        else:
            self.feedback["sway"] = {
                0: 0,
                1: diff[0],
                2: "골반과 몸통이 축을 중심으로 회전하지 않고, 좌우로 밀리고 있습니다. 스윙의 축이 변하여 정확한 임팩트가 어렵고 거리 손실을 보게 됩니다."
            }

    # 헤드 포지션 체크
    # 어드레스 시 코의 위치와 백스윙 시 코의 위치 체크
    # 좌우의 움직임보다 위 아래로의 움직임이 중요
    # y 축의 변화를 체크한다
    def head_position(self):
        nose_address = self.kp[0][0]
        nose_backswing = self.kp[2][0]
        diff = np.array(nose_address) - np.array(nose_backswing)

        if -20 <= diff[1] <= 20:
            self.feedback['head_position'] = {
                0: 2,
                1: diff[1],
                2: "척추 각도가 안정적입니다."
            }
        elif -40 <= diff[1] <= 40:
            self.feedback['head_position'] = {
                0: 1,
                1: diff[1],
                2: "척추 각도가 무너져 머리가 상하로 움직이고 있습니다. 어드레스 시 만들었던 척추 각도가 임팩트까지 유지되어야 합니다. 척추 각도가 무너지면 스윙 궤도 또한 불안정해져 일관된 샷을 칠 수 없습니다."
            }
        else:
            self.feedback['head_position'] = {
                0: 0,
                1: diff[1],
                2: "척추 각도가 무너져 머리가 상하로 움직이고 있습니다. 어드레스 시 만들었던 척추 각도가 임팩트까지 유지되어야 합니다. 척추 각도가 무너지면 스윙 궤도 또한 불안정해져 일관된 샷을 칠 수 없습니다."
            }

    def reverse_pivot(self):
        lshoulder = self.kp[2][5]
        lfoot = self.kp[2][14]
        rfoot = self.kp[2][11]

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

    def foot_fliyng(self):
        lf_address = self.kp[0][21]
        lf_backswing = self.kp[2][21]
        diff = p2_diff(lf_address, lf_backswing)

        if -25 <= diff[1] <= 25:
            self.feedback["foot_flying"] = {
                0: 2,
                1: diff[1],
                2: ""
            }
        elif diff[1] < -25:
            self.feedback["foot_flying"] = {
                0: 0,
                1: diff[1],
                2: "백스윙 시 왼발을 들면 더 큰 회전력을 얻을 수 있지만 스윙의 일관성이 떨어질 수 있습니다. 권장되는 방법은 아닙니다."
            }

    # 왼 팔의 구부러짐 체크
    def bending_left_arm(self):
        lshoulder = self.kp[2][2]
        lelbow = self.kp[2][3]
        lwrist = self.kp[2][4]

        angle = p3_angle(lshoulder, lelbow, lwrist)

        if 140 <= angle:
            self.feedback["bending_arms"] = {
                0: 2,
                1: angle,
                2: "팔 구부러짐이 없습니다."
            }
        elif 125 <= angle:
            self.feedback["bending_arms"] = {
                0: 1,
                1: angle,
                2: "손을 몸에서 멀리 밀면 클럽이 더 먼 거리를 이동하게 됩니다. 샷의 일관성 또한 향상됩니다."
            }
        else:
            self.feedback["bending_arms"] = {
                0: 0,
                1: angle,
                2: "손을 몸에서 멀리 밀면 클럽이 더 먼 거리를 이동하게 됩니다. 샷의 일관성 또한 향상됩니다."
            }

    def left_knee_moving(self):
        lhip = self.kp[2][12]
        lknee = self.kp[2][13]
        lfoot = self.kp[2][14]

        angle = p3_angle(lhip, lknee, lfoot)

        if 170 <= angle:
            self.feedback["left_knee_moving"] = {
                0: 2,
                1: angle,
                2: "good"
            }
        elif 160 <= angle:
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

    # 모든 함수를 실행시킴
    def run(self):
        self.sway()
        self.head_position()
        self.reverse_pivot()
        self.foot_fliyng()
        self.bending_left_arm()
        self.left_knee_moving()

        return self.feedback