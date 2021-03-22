

@app.route('/uploads', methods=['POST'])
# def upload_file():
#
#     file = request.files['video']
#     video_path = os.path.join(VIDEO_SAVE_PATH, file.filename)
#     print('현재경로는: ', os.getcwd())
#     print('비디오 파일 저장 경로는: ', video_path)
#     file.save(video_path)
#     print('비디오 저장')
#
    """
    -----------------------
    골프 db 에서 모델을 가져온다
    -----------------------
    """
#     print('golfdb 시작')
#     ds = SampleVideo(video_path, transform=transforms.Compose([ToTensor(),
#                                     Normalize([0.485, 0.456, 0.406],
#                                               [0.229, 0.224, 0.225])]))
#
#     dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
#
#     model = EventDetector(pretrain=True,
#                               width_mult=1.,
#                               lstm_layers=1,
#                               lstm_hidden=256,
#                               bidirectional=True,
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

        cv2.imwrite(f'{IMAGE_SAVE_PATH}{IMAGE_SAVE_NAME}_{str(i)}.png', output_image)

    cap.release()

    # 뽑아낸 키포인트를 분석
    swing_anal = Anal.Anal(key_data)

    result = swing_anal.check_all()

    # result에 이미지가 저장되는 경로 넣어줌
    result["image_path"] = IMAGE_SAVE_NAME

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
    print('hi')
    last_name = request.form['lastName']
    first_name = request.form['firstName']
    print(last_name, first_name)
    return 'success'