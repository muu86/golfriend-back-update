import numpy as np
from anal_poses.utils import p3_angle


# 1번 자세
class TakeAway:
    def __init__(self, kp):
        self.kp = kp
        self.feedback = dict()

    # 테이크어웨이 시 클럽을 멀리 보낼 수록 큰 스윙 궤도를 만들 수 있다.
    # 이른 힌지는 클럽 헤드의 이동 거리를 감소시켜 비거리 감소를 가져온다.
    # 클럽 : 25
    # 손목: 4
    # 어깨 : 2
    def early_cocking(self):
        club = self.kp[1][25]
        # 모델이 클럽 헤드를 제대로 감지하지 못 한 경우 패스
        if club[0] == 0:
            self.feedback["early_cocking"] = {
                0: 0,
                1: 0,
                2: "클럽 헤드를 감지하지 못 했습니다."
            }

        wrist = self.kp[1][4]
        shoulder = self.kp[1][2]

        angle = p3_angle(club, wrist, shoulder)

        if 100 <= angle:
            self.feedback["early_cocking"] = {
                0: 2,
                1: angle,
                2: ""
            }
        elif 90 <= angle:
            self.feedback["early_cocking"] = {
                0: 1,
                1: angle,
                2: "코킹을 늦춰보세요. 이른 코킹은 비거리 손실을 가져옵니다."
            }
        else:
            self.feedback["early_cocking"] = {
                0: 0,
                1: angle,
                2: "코킹을 늦춰보세요. 이른 코킹은 비거리 손실을 가져옵니다."
            }

    # 왼 팔의 구부러짐 체크
    def bending_left_arm(self):
        lshoulder = self.kp[1][2]
        lelbow = self.kp[1][3]
        lwrist = self.kp[1][4]

        angle = p3_angle(lshoulder, lelbow, lwrist)

        if 155 <= angle:
            self.feedback["bending_arms"] = {
                0: 2,
                1: angle,
                2: "팔 구부러짐이 없습니다."
            }
        elif 140 <= angle:
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

    def run(self):
        self.early_cocking()
        self.bending_left_arm()

        return self.feedback
