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
from anal_poses.FollowThrough import FollowThrough


class Anal:
    result = dict()

    def __init__(self, kp):
        self.kp = kp
        self.address = Address(kp)
        self.takeaway = TakeAway(kp)
        self.backswing = BackSwing(kp)
        self.top = Top(kp)
        self.downswing = DownSwing(kp)
        self.impact = Impact(kp)
        self.release = Release(kp)
        self.followthrough = FollowThrough(kp)

    def check_all(self):
        self.result[0] = self.address.run()
        self.result[1] = self.takeaway.run()
        self.result[2] = self.backswing.run()
        self.result[3] = self.top.run()
        self.result[4] = self.downswing.run()
        self.result[5] = self.impact.run()
        self.result[6] = self.release.run()
        self.result[7] = self.followthrough.run()
        return self.result

