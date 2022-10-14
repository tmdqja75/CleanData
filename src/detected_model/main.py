import os
import platform
import pyautogui
import cv2
import numpy as np
import time as ti

from .commons import functions, detected

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

pwd = os.getcwd()

os_name = platform.system()
root = "/"

if os_name == 'Windows':
    root = "\\"

pwd = os.getcwd()
cleandata = root.join(pwd.split(root)[:-1])
print("main.py pwd", pwd)
print("main.py cleandata", cleandata)

def crawling_path(driver, url, model, encoding_df, pro_encoding_df):
    """
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
    while True:
        driver.get(url)
        a = driver.current_url
        print(a)
        if a != "data:,":
            break
    ti.sleep(3)

    try:
        player_btn = driver.find_element(by=By.XPATH, value='//button[@class="ytp-large-play-button ytp-button"]')
        player_btn.send_keys(Keys.SPACE)
    except:
        pass

    # 총 동영상 길이에 따른 필터링
    total_time = driver.find_element(By.XPATH, '//span[@class="ytp-time-duration"]')
    dateString = total_time.get_attribute('innerText')  # '4:46' '10:46' '1:00:02'

    # 전체화면
    fullScreen_btn = driver.find_elements(By.XPATH, '//button[@class="ytp-fullscreen-button ytp-button"]')
    fullScreen_btn[0].click()

    # 자동재생 off
    auto_bool = driver.find_elements(By.XPATH, '//div[@aria-checked="true"]')
    if auto_bool:
        auto_btn = driver.find_elements(By.XPATH, '//button[@data-tooltip-target-id="ytp-autonav-toggle-button"]')
        auto_btn[0].click()
    #
    # setting_btn = driver.find_elements(By.XPATH, '//button[@class="ytp-button ytp-settings-button"]')
    # setting_btn_hd = driver.find_elements(By.XPATH,
    #                                       '//button[@class="ytp-button ytp-settings-button ytp-hd-quality-badge"]')
    #
    # if setting_btn:
    #     setting_btn[0].click()
    # elif setting_btn_hd:
    #     setting_btn_hd[0].click()

    # detected person face
    id, is_vitim = display_df(driver=driver, url_name=url, dateString=dateString,
                          encoding_df=encoding_df, pro_encoding_df=pro_encoding_df, model=model)

    return id, is_vitim



#################################
def display_df(driver, url_name, dateString, encoding_df, pro_encoding_df, model):
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

        screen = pyautogui.screenshot(region=(0, 0, 1920, 1080))
        screen = np.array(screen)
        src = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        frameUnderTest = np.array(src)
        print(ing_video, ":",dateString, "<< check >>")
        if ing_video==dateString:
            return None, False

        match_list = detected.detectAndDisplay_yolo_df(frameUnderTest, encoding_df, pro_encoding_df, model)
        for id, pro_act in match_list:
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
        f.write('True')
    f.close()
    
    for i, url in enumerate(tmp_df1['link']):
        id, is_vitim = crawling_path(driver=driver, url=url, model=model,
                                     encoding_df=encoding_df, pro_encoding_df=pro_encoding_df) # 이름, 전문배우 (맞으면 True, 틀리면 False)
        if is_vitim:
            print(is_vitim, id)
            tmp_df1.loc[i, "pro_actor"] = is_vitim
            tmp_df1.loc[i, "id"] = id
        else:
            tmp_df1.loc[i, "id"] = id
    print(tmp_df1)
    
    with open(cleandata+'/data/running.txt', 'w') as f:
        f.write('False')
    f.close()    
    # return tmp_df1
    tmp_df1.to_csv(cleandata+'/result/answer.csv', index=False, encoding='UTF8')




