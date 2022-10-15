import os
import platform
import datetime
import pandas as pd
import pyautogui
import cv2
import numpy as np
import time as ti

from .commons import functions, detected

from tkinter import Tk
from selenium import webdriver

from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

pwd = os.getcwd()

os_name = platform.system()
root = "/"

if os_name == 'Windows':
    root = "\\"

pwd = os.getcwd()
cleandata = root.join(pwd.split(root)[:-1])

tk = Tk()
monitor_height = tk.winfo_screenheight()
monitor_width = tk.winfo_screenwidth()
print("main.py pwd", pwd)
print("main.py cleandata", cleandata)
def cfg_set():
    chromedriver_path = os.path.join(pwd,'driver_tools\\chromedriver.exe')
    options = Options()
    ext_file1 = os.path.join(pwd,"driver_tools\\bihmplhobchoageeokmgbdihknkjbknd.crx")  # vpn 확장자
    ext_file2 = os.path.join(pwd,"driver_tools\\cjpalhdlnbpafiamejdnhcphjbkeiagm.crx")  # 광고 방지 확장자
    options.add_argument()
    options.add_extension(ext_file1)
    options.add_extension(ext_file2)
    hub_driver = webdriver.Chrome(executable_path=chromedriver_path, chrome_options=options)
    return hub_driver

cfg = {'youtube':
               {'player_btn': '//button[@class="ytp-large-play-button ytp-button"]',
                'fullScreen_btn': '//button[@class="ytp-fullscreen-button ytp-button"]',
                'auto_bool': '//div[@aria-checked="true"]',
                'auto_btn': '//button[@data-tooltip-target-id="ytp-autonav-toggle-button"]',
                'setting_btn': '//button[@class="ytp-button ytp-settings-button"]',
                'setting_btn_hd': '//button[@class="ytp-button ytp-settings-button ytp-hd-quality-badge"]',
                'ing_video': '//span[@class="ytp-time-current"]'},
           'pornhub':
               {'setting_btn': '//div[@class="mgp_options"]',
                'ing_video': '//span[@class="mgp_elapsed"]',
                'fullScreen_btn': '//div[@class="mgp_fullscreen"]'}
           }

def time2str(seconds):
    seconds = int(seconds)
    str_time = str(datetime.timedelta(seconds=seconds))
    split_time = str_time.split(':')
    print(split_time)
    ans = []
    for i, st in enumerate(split_time):
        print(i, len(ans))
        if i == 0 and st =='0':
            pass
        elif i > 0 and len(ans) >= 1:
            ans.append(st)
        elif i==0 and st!='0':
            ans.append(st)
        else:
            ans.append(str(int(split_time[1])))
    output = ':'.join(ans)
    return output

def crawling_path(driver, url, model, encoding_df, pro_encoding_df, total_time):
    """
    :param total_time: time
    :param driver: chromedriver
    :param url: url
    :param model: YOLO v5
    :param encoding_df: DataFrame
    :param pro_encoding_df: DataFrame

    auto web surfing, crop and connect display function

    :return:
        id: user_email <br/>
        isvitim: bool
    """
    site = url.split('.')[1]
    site_cfg = cfg[site]

    if site == 'pornhub':
        driver = cfg_set()

    while True:
        driver.get(url)
        a = driver.current_url
        if a != "data:,":
            break
    ti.sleep(3)
    if site == 'youtube':
        try:
            player_btn = driver.find_element(by=By.XPATH, value=site_cfg['player_btn'])
            player_btn.send_keys(Keys.SPACE)
        except:
            pass
    # 총 동영상 길이에 따른 필터링
    dateString = time2str(total_time)  # '4:46' '10:46' '1:00:02'

    # 전체화면

    fullScreen_btn = driver.find_elements(By.XPATH, site_cfg['fullScreen_btn'])
    fullScreen_btn[0].click()

    if site == 'youtube':
    # 자동재생 off
        auto_bool = driver.find_elements(By.XPATH, site_cfg['auto_bool'])
        if auto_bool:
            auto_btn = driver.find_elements(By.XPATH, site_cfg['auto_btn'])
            auto_btn[0].click()
    setting_btn = driver.find_elements(By.XPATH, site_cfg['setting_btn'])
    if site == 'youtube':
        setting_btn_hd = driver.find_elements(By.XPATH,
                                          site_cfg['setting_btn_hd'])
    if setting_btn:
        setting_btn[0].click()
    elif setting_btn_hd and site == 'youtube':
        setting_btn_hd[0].click()

    id, is_vitim = display_df(driver=driver, url_name=url, dateString=dateString,site_cfg=site_cfg,
                          encoding_df=encoding_df, pro_encoding_df=pro_encoding_df, model=model)

    return id, is_vitim



#################################
def display_df(driver, url_name, dateString, site_cfg, encoding_df, pro_encoding_df, model):
    """
    :param driver: webdriver
    :param url_name: url link
    :param dateString: Total video time
    :param encoding_df: DataFrame
    :param pro_encoding_df: DataFrame
    :param model: YOLOv5

    화면을 캡처하여 detect 실시

    :return:
        id: 해당 동영상 얼굴에 매칭되는 id 값 <br/>
        bool: pro_actor에 해당되면 True
    """
    nomal_match_dit = {}
    pro_match_dit = {}
    while True:
        ing_video = driver.find_element(By.XPATH, '//span[@class="ytp-time-current"]').get_attribute('innerText')

        screen = pyautogui.screenshot(region=(0, 0, monitor_width, monitor_height))
        screen = np.array(screen)
        src = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        frameUnderTest = np.array(src)
        print(ing_video, ":",dateString, "<< check >>")
        if ing_video==dateString:
            return None, False

        match_list = detected.detectAndDisplay_yolo_df(frameUnderTest, encoding_df, pro_encoding_df, model)
        for id, pro_act in match_list:
            # 얼굴의 갯수가 11개가 될때 초기화가 된다.
            # 한 동영상에서 (결과 값이 아닌) 얼굴이 여러개 인식 될때 오인식 방지
            total_detect_count = len(nomal_match_dit.keys())+len(pro_match_dit.keys())
            print("total_detect_count = ", total_detect_count)
            if total_detect_count > 11:
                nomal_match_dit = {}
                pro_match_dit = {}
                print(total_detect_count,"11 초과로 dic 초기화!")

            if id and not pro_act:
                if id in nomal_match_dit.keys():
                    nomal_match_dit[id] += 1
                else :
                    nomal_match_dit[id] = 1
                print(nomal_match_dit[id], f"id={id}")
                if nomal_match_dit[id] == 15:
                    print("!!!catch!!!")
                    return id, False

            elif pro_act:
                if id in pro_match_dit.keys():
                    pro_match_dit[id] += 1
                else :
                    pro_match_dit[id] = 1
                print(pro_match_dit[id], f"ipro_id={id}")
                if pro_match_dit[id] == 25:
                    print(f'this pro actor url: {url_name} id={id}')
                    return id, True
            else:
                pass
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None, False

    cv2.destroyAllWindows()
    return None, False



# 실행
# CleanData 폴더에서 실행할것.
# python ./face_recognition/main.py
# if __name__ == '__main__':
def run(startDate, model):
    """
    :param startDate: 날짜 값을 받음
    :param model: YOLO Model
    startDate format is "2018-01-01"
    :return:
        None
    """
    # 유출된 날짜(start_date), 동영상 총 길이(avi_length)을 입력받는다.
    avi_length = 600
    tmp_df1, driver, encoding_df, pro_encoding_df = \
        functions.default_set(model=model, os_name=platform.system(), start_date=startDate, avi_length=avi_length)
    with open(cleandata+'/data/running.txt', 'w') as f:
        f.write('False') # True
    f.close()

    for i, (url, total_time) in enumerate(zip(tmp_df1['link'], tmp_df1['total_len'])):
        id, is_vitim = crawling_path(driver=driver, url=url, model=model,
                                     encoding_df=encoding_df, pro_encoding_df=pro_encoding_df, total_time=total_time) # 이름, 전문배우 (맞으면 True, 틀리면 False)
        if is_vitim:
            print(is_vitim, id)
            tmp_df1.loc[i, "pro_actor"] = is_vitim
            tmp_df1.loc[i, "id"] = id
        else:
            tmp_df1.loc[i, "id"] = id
    
    with open(cleandata+'/data/running.txt', 'w') as f:
        f.write('False')
    f.close()    
    # return tmp_df1
    df = pd.read_csv(cleandata + '/data/testURL_youtube.csv')
    tmp_df = pd.concat([df,tmp_df1]).drop_duplicates(['link'], keep='last')
    # testURL UPDATE
    tmp_df.to_csv(cleandata+'/data/testURL_youtube_update.csv', index=False, encoding='utf-8-sig')

    tmp_df1.to_csv(cleandata+'/result/answer.csv', index=False, encoding='UTF8')




