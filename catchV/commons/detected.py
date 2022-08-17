import cv2
import pickle
import time

from deepface import DeepFace
from deepface.commons import distance
from deepface.detectors import FaceDetector

model_name = 'Facenet512' # fix
detector_backend = 'mtcnn' # fix

def splitData(obj):
    """
    받아온 얼굴 사진을 인코딩 및 좌표 처리 해주는 함수
    :param:
        obj(list):3차원으로 된 리스트
    :return:
        encodings(list):사진의 얼굴 부분의 값을 추출해 2중 list [[emb1],[emb2], ... ]
        boxes(list): 사진의 얼굴 부분의 좌표 값 2중 list [[startX, startY, endX, endY], ... ]
    """
    encodings = []
    boxes = []
    for o in obj:
        try :
            encodings.append(DeepFace.represent(img_path=o[0], model_name=model_name, detector_backend=detector_backend)) # embeded 한 데이터를 저장한다.
            startX, StartY, endX, endY = o[1][0], o[1][1], o[1][0]+o[1][2], o[1][1]+o[1][3]    # 각 좌표를 저장
            boxes.append([startX, StartY, endX, endY])
        except:
            print("not detected")
    return encodings, boxes


def detectAndDisplay(image, id, data):
    """
    이미지에 그림을 그려주며 판단해 주는 함수.
    :parameter
        :param image: frame 사진 한장
        :param id: detect 할 아이디
        :param data: encoding data
    :return
        : boolean (미완성)
    """
    start_time = time.time()
    face_detector = FaceDetector.build_model(detector_backend)
    obj = FaceDetector.detect_faces(face_detector=face_detector,
                                    detector_backend=detector_backend, img=image)
    encodings, boxes = splitData(obj)
    threshold = distance.findThreshold(model_name, 'cosine')  # 정답 0.3
    dist_cnt = [0 for i in range(len(encodings))]

    print(dist_cnt)
    for i, (encoding, (startX, startY, endX, endY)) in enumerate(zip(encodings, boxes)):
        for da in data[id]:
            dist = distance.findCosineDistance(encoding, da)  # 거리값을 하나씩 계산 cosine 로 사용
            print(dist)
            if dist <= threshold:  # threshold보다 작으면 cnt
                dist_cnt[i] += 1

    try:
        max_cnt = max(dist_cnt)
    except:
        print("not max_cnt")
        max_cnt = 0

    for i, (encoding, (startX, startY, endX, endY)) in enumerate(zip(encodings, boxes)):
        name = "UnKnown"  # 정답 기준을 못넘었을 경우
        color = (0, 255, 0)  # 기본 green 색상

        if dist_cnt[i] == max_cnt and max_cnt != 0:
            cnt_avg = max_cnt / len(data[id])
            name = id + " " + str(round(cnt_avg, 3) * 100) + "%"
            color = (255, 0, 0)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)  # 얼굴 사각형 그려주기
        y = startY - 15 if startY - 15 > 15 else startY + 15  # 화면을 벗어날수 있어 예외처리
        cv2.putText(image, name, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, color, 2)  # 글자 입력 처리

    end_time = time.time()
    process_time = end_time - start_time
    print("=== A frame took {:.3f} seconds".format(process_time))
    cv2.imshow("image", image)  # cv2.imshow("image", image)
