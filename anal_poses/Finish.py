import numpy as np
from anal_poses.utils import p3_angle
from anal_poses.utils import p2_diff
from anal_poses.utils import add_korean_keyword
from anal_poses.utils import key_to_str

# 1번 자세
class Finish:
    def __init__(self, kp, face_on=True):
        self.kp = kp
        self.face_on = face_on
        self.feedback = dict()
        self.height = self.kp[0][1][1] - self.kp[0][11][1]

    def weight_moving(self):
        lhip = self.kp[7][8]
        lknee = self.kp[7][13]
        lfoot = self.kp[7][14]

        angle = p3_angle(lhip, lknee, lfoot)
        if 160 <= angle:
            self.feedback["weight_moving"] = {
                0: 2,
                1: angle,
                2: "왼발로 완전히 체중이동"
            }
        elif 150 <= angle:
            self.feedback["weight_moving"] = {
                0: 2,
                1: angle,
                2: "피니쉬 자세에서 체중은 왼발로 완전히 이동해야 합니다."
            }
        else:
            self.feedback["weight_moving"] = {
                0: 2,
                1: angle,
                2: "피니쉬 자세에서 체중은 왼발로 완전히 이동해야 합니다."
            }

    def run(self):
        if self.face_on:
            self.weight_moving()
        # # 결과 인덱스 3번에 한국어 간단 설명 추가
        # add_korean_keyword(self.feedback, KOREAN_KEYWORD)
        #
        # # 모든 키를 스트링으로 바꾼 결과 리턴
        # return key_to_str(self.feedback)
        return self.feedback


KOREAN_KEYWORD = {
    "weight_moving": "체중 이동 체크",
    "target": "엉덩이가 목표물과 직각 유지",
    "side_target": "왼발과 허리가 일직선 유지",
}