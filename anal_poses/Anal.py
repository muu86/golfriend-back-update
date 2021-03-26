import numpy as np
# import address
import json
from anal_poses.Address import Address
from anal_poses.TakeAway import TakeAway
from anal_poses.BackSwing import BackSwing
from anal_poses.Top import Top
from anal_poses.DownSwing import DownSwing
from anal_poses.Impact import Impact
from anal_poses.Release import Release
from anal_poses.Finish import Finish


class Anal:
    result = dict()

    def __init__(self, kp, face_on=True):
        self.kp = kp
        self.address = Address(kp, face_on=True)
        self.takeaway = TakeAway(kp, face_on=True)
        self.backswing = BackSwing(kp, face_on=True)
        self.top = Top(kp, face_on=True)
        self.downswing = DownSwing(kp, face_on=True)
        self.impact = Impact(kp, face_on=True)
        self.release = Release(kp, face_on=True)
        self.finish = Finish(kp, face_on=True)

    def check_all(self):
        self.result["0"] = self.address.run()
        self.result["1"] = self.takeaway.run()
        self.result["2"] = self.backswing.run()
        self.result["3"] = self.top.run()
        self.result["4"] = self.downswing.run()
        self.result["5"] = self.impact.run()
        self.result["6"] = self.release.run()
        self.result["7"] = self.finish.run()
        return self.result

