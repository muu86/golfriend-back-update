import numpy as np
from anal_poses.utils import p3_angle
from anal_poses.utils import p2_diff
from anal_poses.utils import add_korean_keyword
from anal_poses.utils import key_to_str


# 1번 자세
class DownSwing:
    def __init__(self, kp, face_on=True):
        self.kp = kp
        self.face_on = face_on
        self.feedback = dict()
        self.height = self.kp[0][1][1] - self.kp[0][11][1]

    # 도훈 만듬
    def sway(self):
        lhip_address = self.kp[0][9]
        lhip_downswing = self.kp[4][9]

        diff = p2_diff(lhip_address, lhip_downswing) / self.height

        if 0.0 <= diff[0] <= 1.0:
            self.feedback["sway"] = {
                0: 2,
                1: diff[0],
                2: "스웨이 체크"
            }
        elif -0.1 <= diff[0] <= 0.2:
            self.feedback["sway"] = {
                0: 2,
                1: diff[0],
                2: "몸이 회전하지 않고 좌우로 이동하고 있습니다. 비거리 손실을 보게되고 스윙 궤도가 중심에서 벗어나 임팩트의 일관성이 떨어지게 됩니다."
            }
        else:
            self.feedback["sway"] = {
                0: 0,
                1: diff[0],
                2: "몸이 회전하지 않고 좌우로 이동하고 있습니다. 비거리 손실을 보게되고 스윙 궤도가 중심에서 벗어나 임팩트의 일관성이 떨어지게 됩니다."
            }

    # 도훈 만듬 왼 발로 체중이동이 돼야 리버스 피벗이 안됨
    def reverse_pivot(self):
        lshoulder = self.kp[4][5]
        lfoot = self.kp[4][14]
        diff = p2_diff(lshoulder, lfoot)[0] / self.height

        if 0.05 <= diff <= 0.15:
            self.feedback["reverse_pivot"] = {
                0: 2,
                1: diff,
                2: "리버스 피벗 체크"
            }
        elif 5 <= diff <= 16:
            self.feedback["reverse_pivot"] = {
                0: 1,
                1: diff,
                2: "오른발에서 왼발로 체중이동되는 것 아닌 왼발에서 오른발로 체중이 이동하고 있습니다.(리버스 피벗)"
            }
        else:
            self.feedback["reverse_pivot"] = {
                0: 0,
                1: diff,
                2: "오른발에서 왼발로 체중이동되는 것 아닌 왼발에서 오른발로 체중이 이동하고 있습니다.(리버스 피벗)"
            }

    # 도훈 만듬 다운 스윙시 손목 각도

    def wrist_angle_downswing(self):
        kkumchi = self.kp[4][3]
        lwrits = self.kp[4][4]
        clubhead = self.kp[4][25]

        angle = p3_angle(kkumchi, lwrits, clubhead)

        if clubhead[0] == 0:
            self.feedback["wrist_angle_downswing"] = {
                0: 0,
                1: angle,
                2: "클럽 헤드를 감지하지 못 했습니다."
            }

        if angle <= 90:
            self.feedback["wrist_angle_downswing"] = {
                0: 2,
                1: angle,
                2: "손목 코킹 체크"
            }
        elif angle <= 125:
            self.feedback["wrist_angle_downswing"] = {
                0: 1,
                1: angle,
                2: "백스윙 시 세팅된 손목의 코킹이 다운스윙 때도 유지되어야 합니다. 임팩트 후 릴리즈 자세에서 코킹을 풀도록 연습해보세요."
            }
        else:
            self.feedback["wrist_angle_downswing"] = {
                0: 0,
                1: angle,
                2: "백스윙 시 세팅된 손목의 코킹이 다운스윙 때도 유지되어야 합니다. 임팩트 후 릴리즈 자세에서 코킹을 풀도록 연습해보세요."
            }



    # 도훈 만듬 (측면) 클럽헤드의 위치 파악
    # def keep_hinge(self):
    #     lelbow = self.kp[4][3]
    #     lwrits = self.kp[4][4]
    #     club = self.kp[4][25]
    #
    #     if
    #
    #     angle = p3_angle(lelbow, lwrits, club)
    #
    #     if 140 <= angle <= 170:
    #         self.feedback["keep_hinge"] = {
    #             0: 2,
    #             1: angle,
    #             2: "Good."
    #         }
    #     elif 120 <= angle <= 180:
    #         self.feedback["keep_hinge"] = {
    #             0: 1,
    #             1: angle,
    #             2: "So So"
    #         }
    #     else:
    #         self.feedback["keep_hinge"] = {
    #             0: 0,
    #             1: angle,
    #             2: "bad"
    #         }

    def run(self):
        if self.face_on:
            self.sway()
            self.reverse_pivot()
            self.wrist_angle_downswing()

        # 결과 인덱스 3번에 한국어 간단 설명 추가
        add_korean_keyword(self.feedback, KOREAN_KEYWORD)

        # 모든 키를 스트링으로 바꾼 결과 리턴
        return self.feedback


KOREAN_KEYWORD = {
    "sway": "다운스윙시 스웨이",
    "reverse_pivot": "리버스 피벗",
    "wrist_angle_downswing": "다운 스윙 중에 손목 각도 유지",
    "keep_hinge": "클럽 헤드의 움직임"
}

