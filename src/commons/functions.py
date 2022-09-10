import os
import pickle
import platform
import cv2
import numpy as np
import pandas as pd
import platform
import time

from deepface import DeepFace
from tqdm import tqdm
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from glob import glob

from .yoloface.face_detector import YoloDetector


def make_pickle(emb_dic, datapath, pro):
    """
    :parameter:
        :param:emb_dic(dictionary): 딕셔너리 형태로 데이터가 들어옴.
        :param:datapath(string): 데이터의 경로.
        :param:pro(bool): 전문 배우의 pkl
    :return:
        rb(dictionary): emb_dic을 다시 불러와줌.
    """
    pickle_name = 'dataset.pkl'
    if pro:
        pickle_name = 'pro_dataset.pkl'
    try:  # 기존 pickle이 존재 덮어 쓰기
        rb = pickle.loads(open("/Users/leegangju/Documents/GitHub/CleanData/src/data/dataset.pkl", 'rb').read())
        for d_name in emb_dic.keys():
            if d_name in rb.keys():  # 기존 이름 있으면 데이터 추가
                rb[d_name].extend(emb_dic[d_name])
            else:
                rb[d_name] = emb_dic[d_name]
        with open("/Users/leegangju/Documents/GitHub/CleanData/src/data/dataset.pkl", 'wb') as f:
            pickle.dump(rb, f)
    except: # pickle 파일이 존재 안할때
        with open("/Users/leegangju/Documents/GitHub/CleanData/src/data/dataset.pkl", 'wb') as f:
            pickle.dump(emb_dic, f)

    rb = pickle.loads(open("/Users/leegangju/Documents/GitHub/CleanData/src/data/dataset.pkl", 'rb').read())
    return rb
#datapath + "/" + pickle_name, 'rb'
#"/Users/leegangju/Documents/GitHub/CleanData/src/data/dataset.pkl"
def time_second(strtime):
    split_time=strtime.split(":")
    total_time = 0
    for i, time in enumerate(split_time[::-1]):
        total_time += (int(time) * (60**i))
    return total_time


def rm_dir(paths):  # 파일 삭제.
    """
    파일을 삭제해 주는 함수
    :parameter:
        paths (list): 빈 파일을 한번에 지워주기 위해서 만든 기능
    :return:
        None
    """
    for path in paths:
        path = "\\".join(path.split("\\")[:-1]) + "\\"
        try:
            os.rmdir(path)
        except:
            continue


def read_csv(path): # "./testURL_youtube.csv"
    df_youtube = pd.read_csv(path)
    df_youtube['upload_date'] = pd.to_datetime(df_youtube['upload_date'], format="%Y%m")
    df_youtube['total_len'] = df_youtube['total_len'].map(time_second)
    return df_youtube


def embeded_file(datapath, target, pro, model):
    """
    이미지를 embed 해주고 삭제해 주는 함수.
    :param
        datapath(string):기본 경로
    :return
        rb(dictionary):{id:[[emb_data1], [emb_data2], ...], id2:[[emb_data1], [emb_data2], ...], ...}
    """
    img_path_list = []
    img_path_list.extend(glob(os.path.join(datapath, f'{target}/*/*.*')))

    model_name = 'Facenet512'
    detector_backend = 'skip'

    emb_dic = {}

    for img_path in tqdm(img_path_list):
        if platform.system() == 'Windows':
            split_key = '\\'
        else:
            split_key = '/'
        img_path_split = img_path.split(split_key)
        d_name = img_path_split[-2]  # 디렉토리 이름
        # ----Face Detection with YOLO V5-----
        frame = cv2.imread(img_path)

        bboxes, _ = model.predict(frame)  # [[[432, 134, 545, 280], [72, 221, 171, 358], [209, 225, 314, 357],...]]
        frame_copy = np.array(frame)
        yb = bboxes


        if len(yb) >= 2:
            continue

        try:
            lx, ly, rx, ry = yb[0]
            img_crop = frame_copy[ly:ry + 1, lx:rx + 1, :]
            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
            # -----------------------------------
            embedding = DeepFace.represent(img_path=img_crop, model_name=model_name, detector_backend=detector_backend,
                                           enforce_detection=False)  # img_path 변경
        except:  # 얼굴 인식 못할시 원본, mtcnn
            embedding = DeepFace.represent(img_path=img_path, model_name=model_name, detector_backend='mtcnn',
                                           enforce_detection=False)  # img_path 변경
        os.remove(img_path)
        if d_name not in emb_dic.keys():
            emb_dic[d_name] = [embedding]
        else:
            emb_dic[d_name].append(embedding)
    rm_dir(img_path_list)
    rb = make_pickle(emb_dic, datapath, pro=pro)

    return rb


def default_set(os_name='Windows', start_date='2018-01-01', avi_length=60*60*2):

    if os_name == 'Windows':
        datapath = '.\\src\\data'
        gpu_name = 0
    else:
        datapath = './src/data'
        gpu_name = 'mps'

    try: # gpu
        model = YoloDetector(weights_name='yolov5n_state_dict.pt', config_name='yolov5n.yaml',
                             target_size=480, gpu=gpu_name)
    except: # cpu
        model = YoloDetector(weights_name='yolov5n_state_dict.pt', config_name='yolov5n.yaml',
                             target_size=480, gpu=-1)

    rb = embeded_file(datapath=datapath, target="target", pro=False, model=model) # add model
    pro_rb = embeded_file(datapath=datapath, target="AvList", pro=True, model=model) # add model

    data = rb
    pro_data = pro_rb

    people_candidates, people_candidates2 = [], []

    for ppl in data.keys():
        emb_lst = data[ppl]
        for emb in emb_lst:
            ppl_row = []
            ppl_row.append(ppl)
            ppl_row.append(np.array(emb))
            people_candidates.append(ppl_row)

    for ppl2 in pro_data.keys():
        emb_lst = pro_data[ppl2]
        for emb in emb_lst:
            ppl_row2 = []
            ppl_row2.append(ppl2)
            ppl_row2.append(np.array(emb))
            people_candidates2.append(ppl_row2)

    encoding_df = pd.DataFrame(people_candidates, columns=['candidate', 'embedding'])
    pro_encoding_df = pd.DataFrame(people_candidates2, columns=['candidate2', 'embedding2'])

    start_date = start_date
    df = read_csv('/Users/leegangju/Documents/GitHub/CleanData/data/testURL_youtube.csv')  ## 수정되어야 되는 부분.



    tmp_df1 = df.loc[
        (df['upload_date'] >= start_date) & (df['pro_actor'] == False)
        & (df['total_len'] <= avi_length),:]
    tmp_df1.reset_index(drop=True, inplace=True)

    ext_file2 = '/Users/leegangju/Documents/GitHub/CleanData/src/cjpalhdlnbpafiamejdnhcphjbkeiagm.crx'
    
    if platform.system() =='Windows':
        executable_path = '/Users/leegangju/Documents/GitHub/CleanData/src/chromedriver'
    else:
        executable_path = '/Users/leegangju/Documents/GitHub/CleanData/src/chromedriver'

    options = Options()

    options.add_extension(ext_file2)

    driver = webdriver.Chrome(executable_path=executable_path, chrome_options=options)

    return tmp_df1, driver, model, encoding_df, pro_encoding_df


