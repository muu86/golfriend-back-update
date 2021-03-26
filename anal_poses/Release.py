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

    def run(self):
        # 결과 인덱스 3번에 한국어 간단 설명 추가
        # add_korean_keyword(self.feedback, KOREAN_KEYWORD)

        # 모든 키를 스트링으로 바꾼 결과 리턴
        # return key_to_str(self.feedback)
        return self.feedback
