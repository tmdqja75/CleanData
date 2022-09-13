import os
import pickle
import platform
import cv2
import numpy as np
import pandas as pd
import platform

from deepface import DeepFace
from tqdm import tqdm
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from glob import glob

from .yoloface.face_detector import YoloDetector

os_name = platform.system()

root = "/"

if os_name == 'Windows':
    root = "\\"

cleandata = root.join(os.getcwd().split(root)[:-1])               # D:\src\CleanData
print("function.py cleandata", cleandata)
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
        rb = pickle.loads(open(datapath + "/" + pickle_name, 'rb').read())
        for d_name in emb_dic.keys():
            if d_name in rb.keys():  # 기존 이름 있으면 데이터 추가
                rb[d_name].extend(emb_dic[d_name])
            else:
                rb[d_name] = emb_dic[d_name]
        with open(datapath + "/" + pickle_name, 'wb') as f:
            pickle.dump(rb, f)
    except: # pickle 파일이 존재 안할때
        with open(datapath + "/" + pickle_name, 'wb') as f:
            pickle.dump(emb_dic, f)

    rb = pickle.loads(open(datapath + "/" + pickle_name, 'rb').read())
    return rb


def time_second(strtime):
    """
    :param: strtime(string)
    str to int \n
    ex) str(12:34) -> int(754)
    :return: total_time(int)
    """
    split_time=strtime.split(":")
    total_time = 0
    for i, time in enumerate(split_time[::-1]):
        total_time += (int(time) * (60**i))
    return total_time


def rm_dir(paths):  # 파일 삭제.
    """
    파일을 삭제해 주는 함수
    :param paths(list): empty dir list

    파일 경로를 받아 해당 파일이 비어있으면 지워준다. \n
    파일 안에 아직 문서가 있으면

    :return
        no_dir(list) no empty dir list
    """
    no_dir = []
    for path in paths:
        path = root.join(path.split(root)[:-1]) + root
        try:
            os.rmdir(path)
        except:
            no_dir.append(path)
    return no_dir

def read_csv(path): # "./testURL_youtube.csv"
    """
    :param path: url 경로
    DataFrame 및 날짜 format 변경
    :return:
    """
    df_youtube = pd.read_csv(path)
    df_youtube['upload_date'] = pd.to_datetime(df_youtube['upload_date'], format="%Y%m")
    df_youtube['total_len'] = df_youtube['total_len'].map(time_second)
    return df_youtube


def embeded_file(datapath, model, target="target", pro=False):
    """
    :param datapath: main directory root
    :param model: model
    :param target: pro_act = "AvList", user = "target"
    :param pro: pro_act = True, user = False

    경로에 있는 사진을 받아와 file name 으로 dictionary 형태로 반환해주는 함수.

    :return:
        rb(dictionary): {d_name:{embedding, ...}}
    """
    img_path_list = []
    img_path_list.extend(glob(os.path.join(datapath, f'{target}/*/*.*')))

    model_name = 'Facenet512'
    detector_backend = 'skip'

    emb_dic = {}

    for img_path in tqdm(img_path_list):
        img_path_split = img_path.split(root)
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
    """
    :param os_name: platform.system() 으로 받아온 os name
    :param start_date: user가 입력한 datetime
    :param avi_length: 영상의 길이

    os flatform 및 gpu 유,무로 환경 setting

    :return:
        tmp_df1: time, pro_actor, length로 정제된 DataFrame <br/>
         driver: Webdriver<br/>
          model: Yolov5 model<br/>
       encoding_df: target DataFrame<br/>
        pro_encoding_df: pro_actor DataFrame
    """
    datapath = cleandata + '/data'
    gpu_name = 'mps'

    if os_name == 'Windows':
        datapath = cleandata + '\\data'
        gpu_name = 0

    try: # gpu
        model = YoloDetector(weights_name='yolov5n_state_dict.pt', config_name='yolov5n.yaml',
                             target_size=480, gpu=gpu_name)
    except: # cpu
        model = YoloDetector(weights_name='yolov5n_state_dict.pt', config_name='yolov5n.yaml',
                             target_size=480, gpu=-1)

    rb = embeded_file(datapath=datapath, target="target", pro=False, model=model) # add deteted_model
    pro_rb = embeded_file(datapath=datapath, target="AvList", pro=True, model=model) # add deteted_model

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
    df = read_csv(cleandata + '/data/testURL_youtube.csv')  ## 수정되어야 되는 부분.



    tmp_df1 = df.loc[
        (df['upload_date'] >= start_date) & (df['pro_actor'] == False)
        & (df['total_len'] <= avi_length),:]
    tmp_df1.reset_index(drop=True, inplace=True)

    ext_file2 = cleandata + '/src/driver_tools/cjpalhdlnbpafiamejdnhcphjbkeiagm.crx'
    
    if platform.system() =='Windows':
        executable_path = cleandata + '/src/driver_tools/chromedriver.exe'
    else:
        executable_path = cleandata + '/src/driver_tools/chromedriver'

    options = Options()

    options.add_extension(ext_file2)

    print(executable_path)

    driver = webdriver.Chrome(executable_path=executable_path, chrome_options=options)

    return tmp_df1, driver, model, encoding_df, pro_encoding_df


