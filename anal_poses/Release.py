import numpy as np
from anal_poses.utils import p3_angle
from anal_poses.utils import p2_diff
from anal_poses.utils import add_korean_keyword
from anal_poses.utils import key_to_str


# 1번 자세
class Release:
    def __init__(self, kp, face_on=True):
        self.kp = kp
        self.face_on = face_on
        self.feedback = dict()
        self.height = self.kp[0][1][1] - self.kp[0][11][1]

    def chicken_wing(self):
        lshoulder = self.kp[6][5]
        lelbow = self.kp[6][6]
        lwrist = self.kp[6][7]

        angle = p3_angle(lshoulder, lelbow, lwrist)

        if 160 <= angle:
            self.feedback["chicken_wing"] = {
                0: 2,
                1: angle,
                2: "치킨윙 체크"
            }
        elif 150 <= angle:
            self.feedback["chicken_wing"] = {
                0: 1,
                1: angle,
                2: "임팩트 후 팔이 곧게 펴지는 것이 좋습니다. 릴리즈 자세에서 클럽이 손을 앞으로 당기는 느낌이 들도록 연습해보세요."
            }
        else:
            self.feedback["chicken_wing"] = {
                0: 0,
                1: angle,
                2: "임팩트 후 팔이 곧게 펴지는 것이 좋습니다. 릴리즈 자세에서 클럽이 손을 앞으로 당기는 느낌이 들도록 연습해보세요."
            }

    # 측면 허리각도
    def back_angle(self):
        lshoulder = self.kp[6][1]
        lhip = self.kp[6][8]
        lwrist = self.kp[6][9]

        angle = p3_angle(lshoulder, lhip, lwrist)

        if 70 <= angle <= 85:
            self.feedback["back_angle"] = {
                0: 2,
                1: angle,
                2: "Good."
            }
        elif 65 <= angle <= 92:
            self.feedback["back_angle"] = {
                0: 1,
                1: angle,
                2: "So So"
            }
        else:
            self.feedback["back_angle"] = {
                0: 0,
                1: angle,
                2: "bad"
            }

    def run(self):
        if self.face_on:
            self.chicken_wing()

        # 결과 인덱스 3번에 한국어 간단 설명 추가
        # add_korean_keyword(self.feedback, KOREAN_KEYWORD)

        # 모든 키를 스트링으로 바꾼 결과 리턴
        # return key_to_str(self.feedback)
        return self.feedback


KOREAN_KEYWORD = {
    "chicken_wing": "치킨윙 체크",
    "back_angle": "허리 각도 유지"
}