import time

import cv2
import numpy as np
from deepface import DeepFace
from deepface.commons import distance

model_name = 'Facenet512'  # fix
detector_backend = 'skip'  # fix mtcnn -> yolo(skip)


def splitData(obj):
    """
    :param obj:3차원으로 된 리스트
    받아온 얼굴 사진을 인코딩 및 좌표 처리 해주는 함수
    :return:
        encodings(list):사진의 얼굴 부분의 값을 추출해 2중 list [[emb1],[emb2], ... ] <br/>
        boxes(list): 사진의 얼굴 부분의 좌표 값 2중 list [[startX, startY, endX, endY], ... ]
    """
    encodings = []
    boxes = []
    for o in obj:
        try:
            encodings.append(DeepFace.represent(img_path=o[0], model_name=model_name,
                                                detector_backend=detector_backend))  # embeded 한 데이터를 저장한다.
            startX, StartY, endX, endY = o[1][0], o[1][1], o[1][0] + o[1][2], o[1][1] + o[1][3]  # 각 좌표를 저장
            boxes.append([startX, StartY, endX, endY])
        except:
            print("not detected")
    return encodings, boxes


# 벡터와 행렬 cosine distance
def findDistance(target, compare):
    u = target
    v = compare
    print(f'u={u.shape}, v={v.shape}')

    u_dot_v = np.sum(u * v, axis=1)

    # find the norm of u and each row of v
    mod_u = np.sqrt(np.sum(u * u))
    mod_v = np.sqrt(np.sum(v * v, axis=1))

    # just apply the definition
    final = 1 - u_dot_v / (mod_u * mod_v)

    return final


def detectAndDisplay_yolo_df(image, df, pro_df, model):
    """
    :param image: frame 사진 한장
    :param df: dataframe 된 고객 사진
    :param pro_df: pro data in DataFrame

    이미지에 그림을 그려주며 판단해 주는 함수.

    :return:match_check[(name, bool), ...]
    """
    # --------time check-------- #
    start_time = time.time()
    face_detect_tic = time.time()
    # --------time check-------- #


    names = []

    # ---------YOLOv5 적용 코드 -------------
    bboxes, _ = model.predict(image) # [[[432, 134, 545, 280], [72, 221, 171, 358], [209, 225, 314, 357],...]]
    bboxes = bboxes[0]
    #--------------------------------------
    face_detect_toc = time.time()

    threshold = distance.findThreshold(model_name, 'cosine')  # 정답 0.3

    face_detect_toc = time.time()

    match_check = []

    if len(bboxes) > 0:

        for i in range(len(bboxes)):
            best_distance, best_distance_pro = 1, 1

            x, y, x_h, y_h = bboxes[i]
            imcrop = image[y:y_h + 1, x:x_h + 1, :]
            check = (None, False)
            frame_encoding = DeepFace.represent(img_path=imcrop, model_name=model_name,
                                                detector_backend=detector_backend, enforce_detection=False)

            col_name = f'distance_from_{i}'

            df[col_name] = findDistance(np.array(frame_encoding), np.array(df['embedding'].values.tolist()))
            df = df.sort_values(by=[col_name])
            finalist = df.iloc[0]
            final_name = finalist['candidate']
            best_distance = finalist[col_name]

            pro_df[col_name] = findDistance(np.array(frame_encoding), np.array(pro_df['embedding2'].values.tolist()))
            pro_df = pro_df.sort_values(by=[col_name])
            finalist_pro = pro_df.iloc[0]
            final_name_pro = finalist_pro['candidate2']
            best_distance_pro = finalist_pro[col_name]

            color = (0, 255, 0)
            name = None
            if best_distance < best_distance_pro:
                if best_distance <= threshold:
                    color = (255, 0, 0)
                    name = final_name
                    check = (name, False)
            else:
                if best_distance_pro <= threshold:
                    color = (255, 0, 0)
                    name = final_name_pro
                    check = (name, True)

            match_check.append(check)
            cv2.rectangle(image, (x, y), (x_h, y_h), color, 2)
            cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # --------time check-------- #
    end_time = time.time()
    process_time = end_time - start_time
    detection_time = face_detect_toc - face_detect_tic
    print("=== A frame took {:.3f} seconds".format(process_time))
    print(f'Detection took {detection_time} seconds')
    print(f'Comparison takes {process_time - detection_time} seconds')
    # --------time check-------- #
    image = cv2.resize(src=image, dsize=(800, 600))
    cv2.namedWindow("frame", flags=cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", width=800, height=600)
    cv2.imshow("frame", image)

    return match_check
