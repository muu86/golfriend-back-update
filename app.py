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

from pymongo import MongoClient, ASCENDING, DESCENDING
from bson.json_util import dumps
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity

# openpose 패스 설정
op_path = "C:/openpose/bin/python/openpose/Release"
sys.path.append(op_path)
os.environ['PATH'] = os.environ['PATH'] + ';' + 'C:/openpose/bin'

# openpose 경로 집
# sys.path.append("C:\\openpose\\build\\python\\openpose\\Release")
# os.environ['PATH'] = os.environ['PATH'] + ';' + 'C:\\penpose\\build\\bin'

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
    video_path = os.path.join(VIDEO_SAVE_PATH, f"{str(image_save_name)}.mp4")
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

        # 오픈포즈 outputData에 클럽 헤드 좌표도 추가
        output_image = cv2.rectangle(datum.cvOutputData, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imwrite(f'{IMAGE_SAVE_PATH}{image_save_name}_{str(i)}.png', output_image)

    cap.release()

    # 뽑아낸 키포인트를 분석
    swing_anal = Anal.Anal(key_data, face_on=True)

    result = swing_anal.check_all()

    # result에 이미지가 저장되는 경로 넣어줌
    result["filePath"] = image_save_name

    # 결과 데이터에 업로드 날짜 추가
    upload_date = now.strftime('%Y-%m-%d')
    result["date"] = upload_date
    # col.update_one(
    #     { "email": current_user},
    #     { "$push": {
    #             f"swingData": json.dumps(result, ensure_ascii=False)
    #         }
    #     }
    # )
    col.update_one(
        {"email": current_user},
        {"$push": {"swingData": json.dumps(result, ensure_ascii=False)}}
    )
    # print(json.dumps(result))
    print(f'{current_user} : {upload_date}, 스윙 분석 업데이트')

    # 분석 횟 수가 일정 횟수 이상이면 뱃지 발급
    current_user_doc = col.find_one({ "email": current_user })
    counts = len(current_user_doc["swingData"])
    print('분석 횟수는: ', counts)
    if counts == 1:
        col.update_one(
            { 'email': current_user },
            { '$push': {
                'badges': 'analysis_1'
            }}
        )
    if counts == 10:
        col.update_one(
            { "email": current_user},
            { "$push": {
                "badges": "analysis_10"
            }}
        )
    if counts == 30:
        col.update_one(
            { "email": current_user},
            { "$push": {
                "badges": "analysis_50"
            }}
        )
    if counts == 50 and 'analysis_50' not in current_user_doc["badges"]:
        col.update_one(
            { "email": current_user},
            { "$push": {
                "badges": "analysis_50"
            }}
        )

    # return json.dumps(result, cls=MyEncoder)
    return result


# 클라이언트가 가지고 있는 토큰이 유효기간이 지났는지 체크
# 유효기간 지났다면 401 status 될 것
@app.route('/check-token')
@jwt_required()
def check_token():
    return 'good'


@app.route('/signup', methods=['POST'])
def sign_up():
    print('회원 가입 라우트')
    email = request.json.get('email')
    last_name = request.json.get('lastname')
    first_name = request.json.get('firstname')
    password = request.json.get('password')

    print(f'{email}, {last_name}, {first_name}, {password} 가입 신청')

    # 처음 가입 시 signUp 배지 발급
    user = {
        'lastName': last_name,
        'firstName': first_name,
        'password': password,
        'email': email,
        'badges': [
            'sign_up',
        ],
        'likes': [],
        'comments': [],
        'swingData': []
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


@app.route('/get-past-swing')
@jwt_required()
def get_past_swing():
    current_user = get_jwt_identity()
    print(current_user)

    swing_index = request.args.get('index')
    doc = col.find_one(
        {"email": current_user},
        {"swingData": {"$slice": -1 - int(swing_index)}}
    )
    
    print(swing_index, '번 데이터에 대한 요청')
    if 'swingData' in doc.keys():
        try:
            return doc['swingData'][-1 - int(swing_index)]
        except IndexError:
            return 'no more data'
    else:
        return json.dumps([]), 200


# 유저가 어떤 뱃지를 갖고 있는지 체크하는 라우트
@app.route('/get-user-info')
@jwt_required()
def get_user_info():
    current_user = get_jwt_identity()
    doc = col.find_one({ "email": current_user })
    print(type(doc))
    return {
        "userName": doc["firstName"],
        "badges": doc["badges"]
    }


# 이미지 요청 처리
@app.route('/get-image/<image_name>')
@jwt_required()
def get_image(image_name):
    current_user = get_jwt_identity()
    print('이미지 요청: ', current_user)

    # requested_image = request.args.get('name')
    print(f'{current_user}님이 이미지: {image_name}요청')
    return send_from_directory('data/images/output_images', f"{image_name}.png")
    # # 뱃지 이미지 요청 시
    # if request.args.get("type") == "badges":
    #     # 유저가 어떤 뱃지를 갖고 있는 지 검색
    #     badge = request.args.get("name")
    #     print(f'{current_user}님이 뱃지 이미지: {badge} 요청')
    #     return send_from_directory('data/images/badges', f"{badge}")


# 유저 뱃지
@app.route('/get-badge-image/<badge_name>')
@jwt_required()
def get_badge_image(badge_name):
    current_user = get_jwt_identity()
    # badge = request.args.get('name')
    return send_from_directory('data/images/badges', f"{badge_name}.png")


# 소셜에 비디오를 업로드
@app.route('/post-social-video', methods=['POST'])
@jwt_required()
def post_social_video():
    current_user = get_jwt_identity()
    video_name = request.json.get('video')
    print(video_name)
    now = datetime.now()
    upload_date = now.strftime('%Y-%m-%d-%H-%M')
    db.social.insert_one({
        "email": current_user,
        "date": upload_date,
        "videoName": video_name,
        "likes": [],
        "comments": []
    })
    col.update_one(
        { 'email': current_user },
        { '$push': {
            'socialUploads': str(video_name)
        }}
    )
    social_uploads_counts = len(col.find_one({ 'email': current_user })["socialUploads"])
    if social_uploads_counts == 1:
        col.update_one(
            { 'email': current_user },
            { '$push': {
                'badges': 'share_1'
            }}
        )
    if social_uploads_counts == 10:
        col.update_one(
            { 'email': current_user },
            { '$push': {
                'badges': 'share_10'
            }}
        )
    if social_uploads_counts == 30:
        col.update_one(
            { 'email': current_user },
            { '$push': {
                'badges': 'share_30'
            }}
        )
    if social_uploads_counts == 50:
        col.update_one(
            {'email': current_user},
            {'$push': {
                'badges': 'share_50'
            }}
        )
    return "success"


# 비디오 요청 처리
@app.route('/get-social')
# @jwt_required()
def get_social():
    index = int(request.args.get('index'))
    docs = db.social.find({}, { "_id": False }).sort("_id", DESCENDING)
    docs = list(docs)
    docs_indexed = docs[(5 * index) : (5 * (index + 1)) ]

    if not docs_indexed:
        return 'no more data'
    return { "socialData": docs_indexed }


@app.route('/get-social-video')
def get_social_video():
    video_name = request.args.get('name')
    return send_from_directory('data/videos', f"{video_name}.mp4")


@app.route('/get-social-video-likes')
@jwt_required()
def get_social_video_likes():
    current_user = get_jwt_identity()
    video_name = request.args.get('name')
    return {
        'likes': col.find_one({ 'email': current_user })['likes']
     }


@app.route('/update-like')
@jwt_required()
def update_like():
    current_user = get_jwt_identity()
    video_name = request.args.get('name')
    like_type = request.args.get('type')
    if like_type == "plus":
        # 현재 좋아요 숫자에서 + 1
        likes = db.social.find_one({'videoName': video_name})['likes']
        db.social.update_one(
            {'videoName': video_name},
            {'$push': {
                'likes': current_user
            }}
        )
        # 커런트 유저의 좋아요 목록 업데이트
        col.update_one(
            {'email': current_user},
            {'$push': {
                'likes': video_name
            }}
        )
    elif like_type == "minus":
        # 비디오의 좋아요 리스트에서 유저 제거
        video_likeList = db.social.find_one({'videoName': video_name})['likes']
        removed_user = video_likeList.pop(video_likeList.index(current_user))
        db.social.update_one(
            {'videoName': video_name},
            {'$set': {
                'likes': video_likeList
                }
            }
        )
        # 유저의 좋아요 목록에서 비디오 제거
        user_likeList = col.find_one({ "email": current_user })['likes']
        removed_video = user_likeList.pop(user_likeList.index(video_name))
        col.update_one(
            { 'email': current_user },
            { '$set': { 'likes': user_likeList }}
        )

        print(len(db.social.find_one({ 'videoName': video_name })['likes']))
    return { 'counts': len(db.social.find_one({ 'videoName': video_name })['likes'])}


@app.route('/post-comment', methods=['POST'])
@jwt_required()
def post_comment():
    current_user = get_jwt_identity()
    comment = request.json.get('comment')
    video_name = request.json.get('videoName')
    col.update_one(
        { 'email': current_user },
        { '$push': {
            'comments': {
                'videoName': video_name,
                'comment': comment
            }
        }}
    )
    db.social.update_one(
        { 'videoName': video_name },
        { '$push': {
            'comments': {
                'email': current_user,
                'comment': comment
            }
        }}
    )
    return "succes"


@app.route('/test/update')
def test_update():
    # current_user = get_jwt_identity()
    # print(current_user)

    doc = col.find_one(
        {"email": 'dd'},
        {"swingData": {"$slice": -30}}
    )

    return { 'swingData': doc['swingData']}


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)