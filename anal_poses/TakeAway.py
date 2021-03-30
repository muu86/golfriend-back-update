import numpy as np
from anal_poses.utils import p3_angle
from anal_poses.utils import p2_diff
from anal_poses.utils import add_korean_keyword
from anal_poses.utils import key_to_str


# 1번 자세
class TakeAway:
    def __init__(self, kp, face_on=True):
        self.kp = kp
        self.face_on = face_on
        self.feedback = dict()
        self.height = self.kp[0][1][1] - self.kp[0][11][1]

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
            return

        wrist = self.kp[1][4]
        shoulder = self.kp[1][2]

        angle = p3_angle(club, wrist, shoulder)

        if 100 <= angle:
            self.feedback["early_cocking"] = {
                0: 2,
                1: angle,
                2: "이른 코킹 체크"
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
            self.feedback["bending_left_arm"] = {
                0: 2,
                1: angle,
                2: "팔 구부러짐이 없습니다."
            }
        elif 140 <= angle:
            self.feedback["bending_left_arm"] = {
                0: 1,
                1: angle,
                2: "손을 몸에서 멀리 밀면 클럽이 더 먼 거리를 이동하게 됩니다. 샷의 일관성 또한 향상됩니다."
            }
        else:
            self.feedback["bending_left_arm"] = {
                0: 0,
                1: angle,
                2: "손을 몸에서 멀리 밀면 클럽이 더 먼 거리를 이동하게 됩니다. 샷의 일관성 또한 향상됩니다."
            }

    # 도훈 만듬 어깨 손 삼각형 유지
    def keep_triangle(self):
        rshoulder = self.kp[1][2]
        relbow = self.kp[1][3]
        rwrist = self.kp[1][4]

        angle = p3_angle(rshoulder, relbow, rwrist)

        if 145 <= angle <= 180:
            self.feedback["keep_triangle"] = {
                0: 2,
                1: angle,
                2: "어깨, 손 삼각형 유지"
            }
        elif 140 <= angle <= 180:
            self.feedback["keep_triangle"] = {
                0: 1,
                1: angle,
                2: "어드레스 때 세팅한 어깨와 손의 삼각형을 유지하세요."
            }
        else:
            self.feedback["keep_triangle"] = {
                0: 0,
                1: angle,
                2: "어드레스 때 세팅한 어깨와 손의 삼각형을 유지하세요."
            }

    '''
    --------------------------------------------------------------------
    측면
    --------------------------------------------------------------------
    '''
    # 도훈 만듬 (측면 Toe_up 자세에서 손목위치)
    def wrist(self):
            lfoot = self.kp[1][11]
            club = self.kp[1][25]
            rfoot = self.kp[1][14]

            if club[0] == 0:
                self.feedback["wrist"] = {
                    0: 0,
                    1: 0,
                    2: "클럽 헤드를 감지하지 못 했습니다."
                }
                return

            angle = p3_angle(lfoot, club, rfoot)

            if 145 <= angle <= 180:
                self.feedback["wrist"] = {
                    0: 2,
                    1: angle,
                    2: "클럽 헤드 움직임이 좋아요."
                }
            elif 140 <= angle <= 180:
                self.feedback["wrist"] = {
                    0: 1,
                    1: angle,
                    2: "테이크 어웨이 시 클럽 헤드를 볼과 평행하게 움직이세요. 클럽이 몸 안쪽이나 바깥쪽을 가리키지 않도록 하세요."
                }
            else:
                self.feedback["wrist"] = {
                    0: 0,
                    1: angle,
                    2: "테이크 어웨이 시 클럽 헤드를 볼과 평행하게 움직이세요. 클럽이 몸 안쪽이나 바깥쪽을 가리키지 않도록 하세요."
                }

    def run(self):
        if self.face_on:
            self.early_cocking()
            self.bending_left_arm()
            self.keep_triangle()
        else:
            self.wrist()

        # 결과 인덱스 3번에 한국어 간단 설명 추가
        add_korean_keyword(self.feedback, KOREAN_KEYWORD)

        # 모든 키를 스트링으로 바꾼 결과 리턴
        return self.feedback


KOREAN_KEYWORD = {
    "early_cocking": "얼리 코킹",
    "bending_left_arm": "왼 팔의 구부러짐",
    "keep_triangle": "어깨와 팔꿈치의 삼각형 유지",
    '''
    --------------------------------------------
    '''
    "wrist": "목표 방향과 손목의 방향 유지"
}