import cv2
import numpy as np


def yolo(frame, size, score_threshold, nms_threshold):

    classes = ['club-head']
    # with open('9k.names', 'r') as f:
    #     classes = [line.strip() for line in f.readlines()]
    # print(classes)

    # YOLO 네트워크 불러오기
    net = cv2.dnn.readNet(f"models/yolo/weight/yolo-obj_last.weights", "models/yolo/weight/yolo-obj.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # 클래스의 갯수만큼 랜덤 RGB 배열을 생성
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # 이미지의 높이, 너비, 채널 받아오기
    height, width, channels = frame.shape

    # 네트워크에 넣기 위한 전처리
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (size, size), (0, 0, 0), True, crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(blob)

    # 결과 받아오기
    outs = net.forward(output_layers)

    # 각각의 데이터를 저장할 빈 리스트
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.1:
                # 탐지된 객체의 너비, 높이 및 중앙 좌표값 찾기
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 객체의 사각형 테두리 중 좌상단 좌표값 찾기
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 후보 박스(x, y, width, height)와 confidence(상자가 물체일 확률) 출력
    print(f"boxes: {boxes}")
    print(f"confidences: {confidences}")

    # Non Maximum Suppression (겹쳐있는 박스 중 confidence 가 가장 높은 박스를 선택)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=score_threshold, nms_threshold=nms_threshold)

    # 후보 박스 중 선택된 박스의 인덱스 출력
    print(f"indexes: ", end='')
    for index in indexes:
        print(index, end=' ')

    print("indexes : ", indexes)
    print(type(indexes))

    print("\n\n============================== classes ==============================")

    # 한 개도 뽑아내지 못 했다면
    if len(boxes) == 0:
        return [0, 0, 20, 20], 0

    # threshold 를 넘지 못하면
    # indexes = ()  => out of range
    if indexes == ():
        return [0, 0, 20, 20], 0

    # 박스만 리턴하도록 수정했음
    return boxes[indexes[0][0]], confidences[indexes[0][0]]

    # for i in range(len(boxes)):
    #     if i in indexes:
    #         x, y, w, h = boxes[i]
    #         class_name = classes[class_ids[i]]
    #         label = f"{class_name} {confidences[i]:.2f}"
    #         color = colors[class_ids[i]]
    #
    #         # 사각형 테두리 그리기 및 텍스트 쓰기
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    #         cv2.rectangle(frame, (x - 1, y), (x + len(class_name) * 13 + 65, y - 25), color, -1)
    #         cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
    #
    #         # 탐지된 객체의 정보 출력
    #         print(f"[{class_name}({i})] conf: {confidences[i]} / x: {x} / y: {y} / width: {w} / height: {h}")
    #
    # return frame


if __name__ == '__main__':
    size_list = [320, 416, 608]
    path = 'data/images/1000_6.png'

    frame = cv2.imread(path)

    yolo_img = yolo(frame=frame, size=size_list[2], score_threshold=0.1, nms_threshold=0.1)

    # cv2.imshow("Output_Yolo", yolo_img)
    # cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()

