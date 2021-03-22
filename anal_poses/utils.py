import numpy as np
import json


def p3_angle(a, b, c):
    a = np.array(a)[[0, 1]]
    b = np.array(b)[[0, 1]]
    c = np.array(c)[[0, 1]]

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def p2_diff(a, b):
    a = np.array(a)[[0, 1]]
    b = np.array(b)[[0, 1]]
    return a - b


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
