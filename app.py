import os
import sys
import cv2
import json
# import pickle
import numpy as np
from datetime import datetime
from flask import Flask, request, send_from_directory, abort, jsonify
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

from models.yolo.yolo import yolo
from anal_poses import Anal
from anal_poses.utils import MyEncoder

from pymongo import MongoClient
from bson.json_util import dumps
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity

# openpose 패스 설정
op_path = "C:/openpose/bin/python/openpose/Release"
sys.path.append(op_path)
os.environ['PATH'] = os.environ['PATH'] + ';' + 'C:/openpose/bin'

# openpose import
try:
    import pyopenpose as op
except ImportError as e:
    raise e

# golfdb import
try:
    from models.golfdb.test_video import SampleVideo, event_names
    from models.golfdb.dataloader import ToTensor, Normalize
    from models.golfdb.model import EventDetector
except ImportError as e:
    raise e

# pymongo 연결
conn = MongoClient('127.0.0.1')
db = conn.golfriend
col = db.data

# 플라스크 시작
app = Flask(__name__)

app.config['SECRET_KEY'] = 'mjmj'

jwt = JWTManager(app)

# 비디오 파일, 이미지 파일 저장할 디렉토리
VIDEO_SAVE_PATH = 'data/videos/'
IMAGE_SAVE_PATH = 'data/images/output_images/'


@app.route('/uploads', methods=['POST'])
@jwt_required()
def upload_file():
    # result를 디비에 저장하기 위해 jwt 로 본인확인
    current_user = get_jwt_identity()
    print(current_user, '님이 분석을 요청하였습니다')

    # 현재 시간
    # 몽고 디비에 data 업데이트 할 때 사용
    now = datetime.now()
    # 이미지를 저장할 이름을 현재 시간으로 설정 XXXX.XXXX
    image_save_name = datetime.timestamp(now)

    file = request.files['video']
    video_path = os.path.join(VIDEO_SAVE_PATH, file.filename)
    # print('현재경로는: ', os.getcwd())
    # print('비디오 파일 저장 경로는: ', video_path)
    file.save(video_path)
    print('비디오 저장')

    """
    -----------------------
    골프 db 에서 모델을 가져온다
    -----------------------
    """
    print('golfdb 시작')
    ds = SampleVideo(video_path, transform=transforms.Compose([ToTensor(),
                                    Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])]))

    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    model = EventDetector(pretrain=True,
                              width_mult=1.,
                              lstm_layers=1,
                              lstm_hidden=256,
                              bidirectional=True,
                              dropout=False)

    save_dict = torch.load('models/golfdb/weight/swingnet_1800.pth.tar')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    print("Loaded model weights")

    seq_length = 64

    print('Testing...')
    for sample in dl:
        images = sample['images']
        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
            logits = model(image_batch.cuda())
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1

    events = np.argmax(probs, axis=0)[:-1]
    print('Predicted event frames: {}'.format(events))

    """
    cv2 비디오 캡쳐 오픈
    """
    cap = cv2.VideoCapture(video_path)

    confidence = []
    for i, e in enumerate(events):
        confidence.append(probs[e, i])
    print('Condifence: {}'.format([np.round(c, 3) for c in confidence]))

    """
        openpose 객체 오픈
        및 파라미터 설정
        """
    params = dict()
    params["model_folder"] = "C:\\openpose\\models\\"
    params["number_people_max"] = 1
    params["net_resolution"] = "-1x240"

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    datum = op.Datum()

    # 각 프레임의 키포인트를 dict 로 모음
    key_data = dict()

    for i, e in enumerate(events):
        cap.set(cv2.CAP_PROP_POS_FRAMES, e)
        _, img = cap.read()
        # cv2.putText(img, '{:.3f}'.format(confidence[i]), (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255))

        # golfdb 가 뽑아낸 이미지를 op 객체에
        datum.cvInputData = img
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # yolo 로 뽑은 club head 데이터 추가
        club_head_points, confiedences = yolo(frame=img, size=416, score_threshold=0.3, nms_threshold=0.3)
        x = club_head_points[0]
        y = club_head_points[1]
        w = club_head_points[2]
        h = club_head_points[3]
        club_head_list = np.array([x, y, confiedences])

        _kp = datum.poseKeypoints[0]
        key_data[i] = np.append(_kp, [club_head_list], axis=0)

        # 오픈포즈 outputData에 헤드 좌표도 추가
        output_image = cv2.rectangle(datum.cvOutputData, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imwrite(f'{IMAGE_SAVE_PATH}{image_save_name}_{str(i)}.png', output_image)

    cap.release()

    # 뽑아낸 키포인트를 분석
    swing_anal = Anal.Anal(key_data)

    result = swing_anal.check_all()

    # result에 이미지가 저장되는 경로 넣어줌
    result["image_path"] = image_save_name

    # 디비 업데이트 시 현재 날짜 필드
    date_field = now.strftime('%Y-%m-%d')
    col.update_one(
        { "email": current_user},
        { "$push": {
                f"swingData.{date_field}": json.dumps(result)
            }
        }
    )
    print(f'{current_user} : {date_field} 스윙 분석 업데이트')

    return json.dumps(result, cls=MyEncoder)


@app.route('/images/<path:image_name>/<int:i>')
def get_images(image_name, i):
    print(image_name)
    print(i)
    try:
        return send_from_directory(
            "C:\\Users\\USER\\PycharmProjects\\golfriend\\data\\images\\output_images",
            filename=f"{image_name}_{i}.png",
            as_attachment=True)
    except FileNotFoundError:
        abort(404)


@app.route('/signup', methods=['POST'])
def sign_up():
    print('회원 가입 라우트')
    email = request.json.get('email')
    last_name = request.json.get('lastname')
    first_name = request.json.get('firstname')
    password = request.json.get('password')

    print(f'{email}, {last_name}, {first_name}, {password} 가입 신청')

    user = {
        'lastName': last_name,
        'firstName': first_name,
        'password': password,
        'email': email
    }

    col.insert(user)

    access_token = create_access_token(identity=email)

    return jsonify(access_token)


@app.route('/login', methods=['POST'])
def login():
    email = request.json.get('email')
    password = request.json.get('password')

    doc = col.find_one({
        'email': email,
        'password': password,
    })
    # for doc in docs:
    #     print(doc)
    if doc:
        access_token = create_access_token(identity=email)
        return jsonify(access_token)
    else:
        print('none')
        return 'bad'


@app.route('/latest-swing')
@jwt_required()
def latest_swing():
    current_user = get_jwt_identity()
    print(current_user)

    doc = col.find_one(
        {"email": current_user},
        {"swingData.2021-03-23": {"$slice": -1}}
    )
    return jsonify(current_user), 200


@app.route('/test/update', methods=['POST'])
@jwt_required()
def test_update():
    date = datetime.now().strftime('%Y-%m-%d')
    data = request.json.get('data')
    current_user = get_jwt_identity()
    print(current_user)
    # col.update_one(
    #     {'email': 'ghgh'},
    #     { "$push": {
    #         f"data.{date}": "gogogo"
    #     }}
    # )
    # print('done')

    doc = col.find_one(
        { "email": current_user},
        { "swingData.2021-03-23": { "$slice": -1} }
    )
    print(doc)
    print(type(doc))
    print(doc["swingData"]["2021-03-23"][0])
    print(len(doc["swingData"]["2021-03-23"]))
    print(type(doc["swingData"]["2021-03-23"][0]))
    print(dumps(doc))
    print(type(dumps(doc)))

    # doc = col.find_one(
    #     {
    #         "email": "bb",
    #     }
    # )
    #
    # a = doc["data"]
    # print(len(a))

    return 'update done'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)