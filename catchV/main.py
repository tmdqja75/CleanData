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
        # -----------------------------------
        try:
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
        os.remove(img_path)
        if d_name not in emb_dic.keys():
            emb_dic[d_name] = [embedding]
        else:
            emb_dic[d_name].append(embedding)
    functions.rm_dir(img_path_list)
    rb = functions.make_pickle(emb_dic, datapath, pro=False)

    return rb


def proActorEmbeded(datapath):
    """
        전문 배우 이미지를 embed 해주고 삭제해 주는 함수.
        :param
            datapath(string):기본 경로
        :return
            rb(dictionary):{id:[[emb_data1], [emb_data2], ...], id2:[[emb_data1], [emb_data2], ...], ...}
    """
    img_path_list = []
    img_path_list.extend(glob(os.path.join(datapath, 'AvList/*/*.*')))

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
        try:
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
        os.remove(img_path)
        if d_name not in emb_dic.keys():
            emb_dic[d_name] = [embedding]
        else:
            emb_dic[d_name].append(embedding)
    functions.rm_dir(img_path_list)

    rb = functions.make_pickle(emb_dic, datapath, pro=True)

    return rb

###########################
import pyautogui
from time import time
import time as ti
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys



def crawling_path(driver,url):

    driver.get(url)
    # 기다림

    ti.sleep(5)
    # wait = WebDriverWait(driver, 10)
    # wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'html5-video-container')))

    try:
        player_btn = driver.find_element(by=By.XPATH, value='//button[@class="ytp-large-play-button ytp-button"]')
        player_btn.send_keys(Keys.SPACE)
    except:
        pass
    # 총 동영상 길이에 따른 필터링
    total_time = driver.find_element(By.XPATH, '//span[@class="ytp-time-duration"]')
    dateString = total_time.text  # '4:46' '10:46' '1:00:02'

    # if len(dateString) == 4:
    #     tmp_idx = 1
    # elif len(dateString) == 5:
    #     tmp_idx = 2
    surplus = 120
    # second = (int(dateString[:tmp_idx]) * 60) + int(dateString[-2:])
    # total_time = round(time() + second)
    raw_second = 300

    raw_totalTime = round(time() + raw_second + surplus)
    # if total_time > raw_totalTime:
    #     print("동일한 영상이 아닙니다(영상길이)")

    # 전체화면
    fullScreen_btn = driver.find_elements(By.XPATH, '//button[@class="ytp-fullscreen-button ytp-button"]')
    fullScreen_btn[0].click()

    setting_btn = driver.find_elements(By.XPATH, '//button[@class="ytp-button ytp-settings-button"]')
    setting_btn_hd = driver.find_elements(By.XPATH,
                                          '//button[@class="ytp-button ytp-settings-button ytp-hd-quality-badge"]')

    if setting_btn:
        setting_btn[0].click()
    elif setting_btn_hd:
        setting_btn_hd[0].click()

    id, is_vitim = display_df(driver=driver, url_name=url, dateString=dateString,
                          encoding_df=encoding_df, pro_encoding_df=pro_encoding_df)

    return id, is_vitim



#################################
def display_df(driver, url_name, dateString, encoding_df, pro_encoding_df):
    """

    :parameter
        :param file_name: VideoCapture 실행 시킬 경로.
        :param encoding_df: encoding 된 dataframe 형태
        :param pro_encoding_df: encoding 된 전문 배우
    :return:
        :param
    """
    print("display_df")  ###########################
    match_cnt = 0
    while True:
        ing_video = driver.find_element(By.XPATH, '//span[@class="ytp-time-current"]').get_attribute('innerText')

        screen = pyautogui.screenshot(region=(0, 0, 1920, 1080))
        screen = np.array(screen)
        src = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        frameUnderTest = np.array(src)
        if ing_video==dateString:
            return None, False

        id, pro_act = detected.detectAndDisplay_yolo_df(frameUnderTest, encoding_df, pro_encoding_df)
        if id and not pro_act:
            match_cnt += 1
            print("!!!catch!!!")
            if match_cnt == 15:
                return id, False

        elif pro_act:
            print(f'this pro actor url {url_name}')
            return id, True
        else:
            pass
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None, False

    cv2.destroyAllWindows()
    return None, False



# 실행
# CleanData 폴더에서 실행할것.
# python ./catchV/main.py
if __name__ == '__main__':

    if platform.system() == 'Windows':
        # rb = embeded_file(datapath='.\\data')
        # pro_rb = proActorEmbeded(datapath='.\\data')
        encoding_file = '.\\data\\dataset.pkl'
        pro_encoding = '.\\data\\pro_dataset.pkl'
    else:
        rb = embeded_file(datapath='../../data')
        pro_rb = proActorEmbeded(datapath='../../data')
        encoding_file = '../../data/dataset.pkl'
        pro_encoding = '../../data/pro_dataset.pkl'

    data = pickle.loads(open(encoding_file, "rb").read())
    pro_data = pickle.loads(open(pro_encoding, "rb").read())

    print(len(data))

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

    start_date = '2018-09-01' # 날짜 고쳐야 하는 부분
    df_youtube = functions.read_youtube_csv('./catchV/testURL_youtube.csv')
    urls = df_youtube.loc[(df_youtube['upload_date'] >= start_date) & (df_youtube['pro_actor'] == False), 'link'].to_list()

    tmp_df1 = df_youtube.loc[
        (df_youtube['upload_date'] >= start_date) & (df_youtube['pro_actor'] == False), :]


    ext_file2 = './catchV/cjpalhdlnbpafiamejdnhcphjbkeiagm.crx'
    executable_path = './catchV/chromedriver.exe'
    options = Options()
    # options.add_argument('headless')
    # options.add_argument('lang=ko_KR')
    options.add_extension(ext_file2)

    driver = webdriver.Chrome(executable_path=executable_path, chrome_options=options)

    for i, url in enumerate(urls):
        id, is_vitim = crawling_path(driver=driver, url=url) # 이름, 전문배우 (맞으면 True, 틀리면 False)
        if is_vitim:
            tmp_df1["pro_actor"][i] = is_vitim
            tmp_df1["id"][i] = id
        else:
            tmp_df1["id"][i] = id
    tmp_df1.to_csv('./test.csv', encoding='UTF8')




