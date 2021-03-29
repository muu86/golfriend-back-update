import numpy as np
from anal_poses.utils import p3_angle
from anal_poses.utils import p2_diff
from anal_poses.utils import add_korean_keyword
from anal_poses.utils import key_to_str


# 1번 자세
class Impact:
    def __init__(self, kp, face_on=True):
        self.kp = kp
        self.face_on = face_on
        self.feedback = dict()
        self.height = self.kp[0][1][1] - self.kp[0][11][1]

    def reverse_pivot(self):
        lshoulder = self.kp[5][5]
        lfoot = self.kp[5][14]
        diff = p2_diff(lshoulder, lfoot)[0] / self.height

        if 0.05 <= diff <= 0.2:
            self.feedback["reverse_pivot"] = {
                0: 2,
                1: diff,
                2: "리버스 피벗 체크"
            }
        elif 0.0 <= diff <= 0.3:
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

    def keep_the_lag(self):
        lwrist = self.kp[5][5]
        club = self.kp[5][25]
        diff = p2_diff(lwrist, club)[0] / self.height

        if club[0] == 0:
            self.feedback["keep_the_lag"] = {
                0: 0,
                1: diff,
                2: "클럽 헤드를 감지하지 못 했습니다."
            }
            return

        if 0.0 <= diff <= 1.0:
            self.feedback["keep_the_lag"] = {
                0: 2,
                1: diff,
                2: "임팩트까지 래깅 유지"
            }
        elif -0.5 <= diff <= 0.0:
            self.feedback["keep_the_lag"] = {
                0: 1,
                1: diff,
                2: "어드레스 시 손목을 클럽보다 앞에 두고 클럽을 살짝 기울여 세팅하게 됩니다. 임팩트 시에도 손목 위치를 동일하게 하여 래깅을 유지해야 합니다."
            }
        else:
            self.feedback["keep_the_lag"] = {
                0: 0,
                1: diff,
                2: "어드레스 시 손목을 클럽보다 앞에 두고 클럽을 살짝 기울여 세팅하게 됩니다. 임팩트 시에도 손목 위치를 동일하게 하여 래깅을 유지해야 합니다."
            }

    def head_position(self):
        nose_address = self.kp[0][0]
        nose_backswing = self.kp[5][0]
        diff = p2_diff(nose_address, nose_backswing)[0] / self.height

        if -0.2 <= diff <= 0.2:
            self.feedback['head_position'] = {
                0: 2,
                1: diff,
                2: "척추 각도가 안정적입니다.",
            }
        elif -0.5 <= diff <= 0.5:
            self.feedback['head_position'] = {
                0: 1,
                1: diff,
                2: "척추 각도가 무너져 머리가 상하로 움직이고 있습니다. 어드레스 시 만들었던 척추 각도가 임팩트까지 유지되어야 합니다. 척추 각도가 무너지면 스윙 궤도 또한 불안정해져 일관된 샷을 칠 수 없습니다.",
            }
        else:
            self.feedback['head_position'] = {
                0: 0,
                1: diff,
                2: "척추 각도가 무너져 머리가 상하로 움직이고 있습니다. 어드레스 시 만들었던 척추 각도가 임팩트까지 유지되어야 합니다. 척추 각도가 무너지면 스윙 궤도 또한 불안정해져 일관된 샷을 칠 수 없습니다.",
            }

    def run(self):
        if self.face_on:
            self.reverse_pivot()
            self.keep_the_lag()
            self.head_position()

        # 결과 인덱스 3번에 한국어 간단 설명 추가
        add_korean_keyword(self.feedback, KOREAN_KEYWORD)

        # 모든 키를 스트링으로 바꾼 결과 리턴
        # return key_to_str(self.feedback)
        return self.feedback


KOREAN_KEYWORD = {
    "reverse_pivot": "리버스 피벗",
    "keep_the_lag": "래깅 유지",
    "head_position": "헤드 포지션",
    "back_angle": "백스윙때의 척추의 각도와 임팩트시 척추의 각도"
}