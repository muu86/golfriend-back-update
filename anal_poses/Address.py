import numpy as np
from anal_poses.utils import p3_angle
from anal_poses.utils import add_korean_keyword
from anal_poses.utils import key_to_str


# 1번 자세
class Address:
    def __init__(self, kp, face_on=True):
        self.kp = kp
        self.face_on = face_on
        self.feedback = dict()
        self.height = self.kp[0][1][1] - self.kp[0][11][1]

    # 도훈 만듬 (스탠스 거리)
    def stance(self):
        shoulders = self.kp[0][5][0] - self.kp[0][2][0]
        stance_diff = self.kp[0][14][0] - self.kp[0][11][0]

        diff = (shoulders - stance_diff) / self.height
        # 640 * 640 이미지 기준
        # -50~ 50 사이
        if 0.0 <= diff <= 0.3:
            self.feedback["stance"] = {
                0: 2,
                1: diff,
                2: "스탠스가 안정적입니다.",
            }
        elif 0.3 <= diff:
            self.feedback["stance"] = {
                0: 1,
                1: diff,
                2: "스탠스가 넓습니다."
            }
        elif diff <= 0.0:
            self.feedback["stance"] = {
                0: 1,
                1: diff,
                2: "스탠스가 좁습니다."
            }

    # 도훈 만듬 (측면 무릎 각도)
    def knee_angle(self):
        upfoot = self.kp[0][9]
        middlefoot = self.kp[0][10]
        downfoot = self.kp[0][11]

        angle = p3_angle(upfoot, middlefoot, downfoot)

        if 154 <= angle <= 156:
            self.feedback["knee_angle"] = {
                0: 2,
                1: angle,
                2: "good"
            }
        elif 150 <= angle <= 161:
            self.feedback["knee_angle"] = {
                0: 1,
                1: angle,
                2: "어드레스 시 무릎은 적당히 굽혀주세요"
            }
        else:
            self.feedback["knee_angle"] = {
                0: 0,
                1: angle,
                2: "어드레스 시 무릎은 적당히 굽혀주세요"
            }

    # 도훈 만듬 (측면 허리 각도)
    def back_angle(self):
        oue = self.kp[0][9]
        chukan = self.kp[0][10]
        sita = self.kp[0][11]

        angle = p3_angle(oue, chukan, sita)

        if 148 <= angle <= 153:
            self.feedback["back_angle"] = {
                0: 2,
                1: angle,
                2: "척추 각도가 안정적입니다."
            }
        elif 144 <= angle <= 147:
            self.feedback["back_angle"] = {
                0: 1,
                1: angle,
                2: "양손이 편하게 움직일 수 있도록 골반과 허리에 적당한 각도를 유지해주세요"
            }
        else:
            self.feedback["back_angle"] = {
                0: 0,
                1: angle,
                2: "양손이 편하게 움직일 수 있도록 골반과 허리에 적당한 각도를 유지해주세요"
            }

    def run(self):
        if self.face_on:
            self.stance()
        else:
            self.knee_angle()
            self.back_angle()

        # 결과 인덱스 3번에 한국어 간단 설명 추가
        add_korean_keyword(self.feedback, KOREAN_KEYWORD)

        # 모든 키를 스트링으로 바꾼 결과 리턴
        return self.feedback


KOREAN_KEYWORD = {
    "stance" : "스탠스 넓이",
    "knee_angle" : "무릎 각도",
    "back_angle" : "등 각도"
}