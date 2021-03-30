import numpy as np
from anal_poses.utils import p3_angle
from anal_poses.utils import p2_diff
from anal_poses.utils import add_korean_keyword
from anal_poses.utils import key_to_str


class Top:
    def __init__(self, kp, face_on=True):
        self.kp = kp
        self.face_on = face_on
        self.feedback = dict()
        self.height = self.kp[0][1][1] - self.kp[0][11][1]

    # 왼 팔의 구부러짐 체크
    def bending_left_arm(self):
        lshoulder = self.kp[3][5]
        lelbow = self.kp[3][6]
        lwrist = self.kp[3][7]

        angle = p3_angle(lshoulder, lelbow, lwrist)

        if 120 <= angle:
            self.feedback["bending_left_arm"] = {
                0: 2,
                1: angle,
                2: "팔 구부러짐이 없습니다."
            }
        elif 100 <= angle:
            self.feedback["bending_left_arm"] = {
                0: 1,
                1: angle,
                2: "손을 몸에서 멀리 밀면 클럽이 더 먼 거리를 이동하게 됩니다. 샷의 일관성 또한 향상됩니다. "
            }
        else:
            self.feedback["bending_left_arm"] = {
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
                2: "체중 이동 시 왼 다리 이동 체크"
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

    # 도훈 만듬
    def head_postion(self):
        nose_address = self.kp[0][0]
        nose_top = self.kp[3][0]

        diff = p2_diff(nose_address, nose_top) / self.height

        if -0.1 <= diff[1] <= 0.5 :
            self.feedback["head_position"] = {
                0: 2,
                1: diff[1],
                2: "등 각도 유지"
            }
        elif -0.2 <= diff[1] <= 1.0:
            self.feedback["head_position"] = {
                0: 1,
                1: diff[1],
                2: "어드레스 시 유지한 등과 골반의 각도를 유지하세요. 머리가 상하로 움직이고 있습니다."
            }
        else:
            self.feedback["head_position"] = {
                0: 0,
                1: diff[1],
                2: "어드레스 시 유지한 등과 골반의 각도를 유지하세요. 머리가 상하로 움직이고 있습니다."
            }

    # 도훈 만듬
    def back_face_target(self):
        lshoulder = self.kp[3][5]
        rshoulder = self.kp[3][2]
        lfoot = self.kp[3][14]

        angle = p3_angle(lshoulder, rshoulder, lfoot)

        if 70 <= angle <= 80:
            self.feedback["back_face_target"] = {
                0: 2,
                1: angle,
                2: "적당한 회전"
            }
        elif 65 <= angle <= 88:
            self.feedback["back_face_target"] = {
                0: 1,
                1: angle,
                2: "탑자세에서 어깨가 공을 바라보는 정도로 회전하는 것이 이상적입니다. 너무 느슨하거나 지나친 회전은 피해야 합니다."
            }
        else:
            self.feedback["back_face_target"] = {
                0: 0,
                1: angle,
                2: "탑자세에서 어깨가 공을 바라보는 정도로 회전하는 것이 이상적입니다. 너무 느슨하거나 지나친 회전은 피해야 합니다."
            }

    def parallel_shaft(self):
        rfoot = self.kp[3][11]
        lwrits = self.kp[3][4]
        club = self.kp[3][25]

        if club[0] == 0:
            self.feedback["parallel_shaft"] = {
                0: 2,
                1: 0,
                2: "클럽 헤드를 감지하지 못 했습니다."
            }

        angle = p3_angle(rfoot, lwrits, club)

        if 95 <= angle <= 115:
            self.feedback["parallel_shaft"] = {
                0: 2,
                1: angle,
                2: "클럽이 지면과 평행"
            }
        elif 90 <= angle <= 120:
            self.feedback["parallel_shaft"] = {
                0: 1,
                1: angle,
                2: "적절한 손목 힌지와 몸의 회전 시 클럽이 지면과 평행하게 되는 것이 자연스럽습니다. 너무 적거나 지나친 회전은 피해야합니다."
            }
        else:
            self.feedback["parallel_shaft"] = {
                0: 0,
                1: angle,
                2: "적절한 손목 힌지와 몸의 회전 시 클럽이 지면과 평행하게 되는 것이 자연스럽습니다. 너무 적거나 지나친 회전은 피해야합니다."
            }
    '''
    --------------------------------------------------------------------
    측면
    --------------------------------------------------------------------
    '''
    # 측면 왼팔의 구부러짐
    def left_wrist_flat(self):
        lshoulder = self.kp[3][4]
        head = self.kp[3][25]
        lwrist = self.kp[3][5]

        angle = p3_angle(lshoulder, head, lwrist)

        if 0 <= angle <= 18:
            self.feedback["left_wrist_flat"] = {
                0: 2,
                1: angle,
                2: "손목 힌지 체크"
            }
        elif 0 <= angle <= 60:
            self.feedback["left_wrist_flat"] = {
                0: 1,
                1: angle,
                2: "손목을 안쪽으로 말거나 바깥으로 접은 채 스윙 시 공에 사이드 스핀을 발생시킬 수 있어요. 적절한 손목 힌지가 공의 직진성을 높입니다."
            }
        else:
            self.feedback["left_wrist_flat"] = {
                0: 0,
                1: angle,
                2: "손목을 안쪽으로 말거나 바깥으로 접은 채 스윙 시 공에 사이드 스핀을 발생시킬 수 있어요. 적절한 손목 힌지가 공의 직진성을 높입니다."
            }

    def run(self):
        if self.face_on:
            self.bending_left_arm()
            self.reverse_pivot()
            self.left_knee_moving()
            self.head_postion()
            self.back_face_target()
            self.parallel_shaft()
        else:
            self.left_wrist_flat()

        # 결과 인덱스 3번에 한국어 간단 설명 추가
        add_korean_keyword(self.feedback, KOREAN_KEYWORD)

        # 모든 키를 스트링으로 바꾼 결과 리턴
        return self.feedback


KOREAN_KEYWORD = {
    "bending_left_arm": "왼 팔의 구부러짐",
    "reverse_pivot": "리버스 피벗",
    "left_knee_moving": "공을 향해 왼쪽 무릎 이동",
    "head_position": "헤드 포지션 유지",
    "back_face_target": "적당한 회전",
    "parallel_shaft": "클럽이 지면과 평행",
    '''
    ----------------------
    '''
    "left_wrist_flat": "손목 꺾임"
}