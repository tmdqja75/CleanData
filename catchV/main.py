from commons import functions, detected
from commons.facedetect import get_boxes_points, FaceDetector
from glob import glob
from deepface import DeepFace
from tqdm import tqdm

import os
import pickle
import cv2
import platform
import numpy as np
import pandas as pd


def embeded_file(datapath):
    """
    이미지를 embed 해주고 삭제해 주는 함수.
    :param
        datapath(string):기본 경로
    :return
        rb(dictionary):{id:[[emb_data1], [emb_data2], ...], id2:[[emb_data1], [emb_data2], ...], ...}
    """
    img_path_list = []
    img_path_list.extend(glob(os.path.join(datapath, '*/*.*')))

    model_name = 'Facenet512'
    detector_backend = 'skip'

    emb_dic = {}
    print(img_path_list)
    for img_path in tqdm(img_path_list):
        if platform.system() == 'Windows':
            split_key = '\\'
        else:
            split_key ='/'
            
        img_path_split = img_path.split(split_key)
        d_name = img_path_split[-2] # 디렉토리 이름
        
        # ----Face Detection with YOLO V2-----
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (1920, 1080))
        frame_copy = np.array(frame)
        
        
        yolo_model = FaceDetector()
        yolo_boxes = yolo_model.detect(frame, 0.7)
        
        print(f'{len(yolo_boxes)} faces detected')
        
        yb = get_boxes_points(yolo_boxes, frame.shape)

        if len(yb) >= 2:
            continue

        lx, ly, rx, ry = yb[0]
        img_crop = frame_copy[ly:ry+1, lx:rx+1, :]
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
        # -----------------------------------
        if img_crop:
            embedding = DeepFace.represent(img_path=img_crop, model_name=model_name, detector_backend=detector_backend, enforce_detection=False) # img_path 변경
        else: # 얼굴 인식 못할시 원본, mtcnn
            embedding = DeepFace.represent(img_path=img_path, model_name=model_name, detector_backend='mtcnn', enforce_detection=False) # img_path 변경
        # os.remove(img_path)
        if d_name not in emb_dic.keys():
            emb_dic[d_name] = [embedding]
        else:
            emb_dic[d_name].append(embedding)
    # functions.rm_dir(img_path_list)

    rb = functions.make_pickle(emb_dic, datapath, pro=False)

    return rb


def proActorEmbeded(datapath):
    img_path_list = []
    img_path_list.extend(glob(os.path.join(datapath, 'AvList/*/*.*')))

    model_name = 'Facenet512'
    detector_backend = 'skip'

    emb_dic = {}
    print(img_path_list)
    for img_path in tqdm(img_path_list):
        if platform.system() == 'Windows':
            split_key = '\\'
        else:
            split_key = '/'

        img_path_split = img_path.split(split_key)
        d_name = img_path_split[-2]  # 디렉토리 이름

        # ----Face Detection with YOLO V2-----
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (1920, 1080))
        frame_copy = np.array(frame)

        yolo_model = FaceDetector()
        yolo_boxes = yolo_model.detect(frame, 0.7)

        yb = get_boxes_points(yolo_boxes, frame.shape)

        if len(yb) >= 2:
            continue
        try :
            lx, ly, rx, ry = yb[0]
            img_crop = frame_copy[ly:ry + 1, lx:rx + 1, :]
            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
            # -----------------------------------
            embedding = DeepFace.represent(img_path=img_crop, model_name=model_name, detector_backend=detector_backend,
                                               enforce_detection=False)  # img_path 변경
            b = True if embedding else False
            print(embedding[0], f"{b} yolo deteted")
        except:  # 얼굴 인식 못할시 원본, mtcnn
            embedding = DeepFace.represent(img_path=img_path, model_name=model_name, detector_backend='mtcnn',
                                           enforce_detection=False)  # img_path 변경
            b = True if embedding else False
            print(embedding[0], f"{b} mtcnn deteted")
        # os.remove(img_path)
        if d_name not in emb_dic.keys():
            emb_dic[d_name] = [embedding]
        else:
            emb_dic[d_name].append(embedding)
    # functions.rm_dir(img_path_list)

    rb = functions.make_pickle(emb_dic, datapath, pro=True)

    return rb


# def display(file_name, encoding_file, id):
#     """
#     :parameter
#         :param file_name: VideoCapture 실행 시킬 경로.
#         :param encoding_file: encoding 된 데이터 들
#     :return:
#     """
#     cap = cv2.VideoCapture(file_name)
#     writer = None
#     data = pickle.loads(open(encoding_file, "rb").read())
#
#     if not cap.isOpened:
#         print('-- (!) Check your cap or video root (!) --')
#         exit(0)
#     while True:
#         ret, frame = cap.read()
#         if frame is None:
#             print('-- (!) Check your cap or video root (!) --')
#             cap.release()
#             writer.release()
#
#         detected.detectAndDisplay(frame, id, data)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             cap.release()
#             break
#
#     cv2.destroyAllWindows()
    
def display_df(file_name, encoding_df,):
    """
    :parameter
        :param file_name: VideoCapture 실행 시킬 경로.
        :param encoding_file: encoding 된 데이터 들
    :return:
    """
    cap = cv2.VideoCapture(file_name)
    
    writer = None

    if not cap.isOpened:
        print('-- (!) Check your cap or video root (!) --')
        exit(0)
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('-- (!) Check your cap or video root (!) --')
            cap.release()
            writer.release()

        detected.detectAndDisplay_yolo_df(frame,encoding_df)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            break

    cv2.destroyAllWindows()



# 실행
# CleanData 폴더에서 실행할것.
# python catchV/main.py
if __name__ == '__main__':

    if platform.system() == 'Windows':
        # rb = embeded_file(datapath='.\\data')
        pro_rb = proActorEmbeded(datapath='.\\data')
        encoding_file = '.\\data\\dataset.pkl'
        pro_encoding = '.\\data\\pro_dataset.pkl'
    else:
        # rb = embeded_file(datapath='../../data')
        pro_rb = proActorEmbeded(datapath='./data')
        encoding_file = '../../data/dataset.pkl'
        pro_encoding = '../../data/pro_dataset.pkl'

    data = pickle.loads(open(encoding_file, "rb").read())
    pro_data = pickle.loads(open(pro_encoding, "rb").read())

    print(len(pro_data))

    # people_candidates = []
    # for ppl in data.keys():
    #     emb_lst = data[ppl]
    #     for emb in emb_lst:
    #         ppl_row = []
    #         ppl_row.append(ppl)
    #         ppl_row.append(np.array(emb))
    #         people_candidates.append(ppl_row)
    # encoding_df = pd.DataFrame(people_candidates, columns=['candidate', 'embedding'])
    #
    # display_df(file_name=0, encoding_df=encoding_df)
