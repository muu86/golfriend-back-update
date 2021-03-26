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


def add_korean_keyword(feedback, word_dict):
    for key in feedback.keys():
        feedback[key][3] = word_dict[key]


def key_to_str(feedback):
    new_dict = dict()
    for key, item in feedback.items():
        for key1, item1 in feedback[key].items():
            if not new_dict[key]:
                new_dict[key] = {}
            new_dict[key][str(key)] = item1
    return new_dict


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
