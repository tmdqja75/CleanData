import platform
import pyautogui
import cv2
import numpy as np

import time as ti

from commons import functions, detected
from time import time

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

def crawling_path(driver, url, model):

    driver.get(url)
    # 기다림

    ti.sleep(3)
    # wait = WebDriverWait(driver, 10)
    # wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'html5-video-container')))

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

    setting_btn = driver.find_elements(By.XPATH, '//button[@class="ytp-button ytp-settings-button"]')
    setting_btn_hd = driver.find_elements(By.XPATH,
                                          '//button[@class="ytp-button ytp-settings-button ytp-hd-quality-badge"]')

    if setting_btn:
        setting_btn[0].click()
    elif setting_btn_hd:
        setting_btn_hd[0].click()

    id, is_vitim = display_df(driver=driver, url_name=url, dateString=dateString,
                          encoding_df=encoding_df, pro_encoding_df=pro_encoding_df, model=model)

    return id, is_vitim



#################################
def display_df(driver, url_name, dateString, encoding_df, pro_encoding_df, model):
    """

    :parameter
        :param file_name: VideoCapture 실행 시킬 경로.
        :param encoding_df: encoding 된 dataframe 형태
        :param pro_encoding_df: encoding 된 전문 배우
    :return:
        :param
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

        id, pro_act = detected.detectAndDisplay_yolo_df(frameUnderTest, encoding_df, pro_encoding_df, model)
        if id and not pro_act:
            if id in nomal_match_dit.keys():
                nomal_match_dit[id] += 1
            else :
                nomal_match_dit[id] = 1
            print(nomal_match_dit[id])
            if nomal_match_dit[id] == 15:
                print("!!!catch!!!")
                return id, False

        elif pro_act:
            if id in pro_match_dit.keys():
                pro_match_dit[id] += 1
            else :
                pro_match_dit[id] = 1
            print(pro_match_dit[id])
            if pro_match_dit[id] == 15:
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
# python ./catchV/main.py
if __name__ == '__main__':

    # 유출된 날짜(start_date), 동영상 총 길이(avi_length)을 입력받는다.
    avi_length = 600
    tmp_df1, driver, model,  encoding_df, pro_encoding_df = \
        functions.default_set(os_name=platform.system(), start_date='2022-07-01', avi_length=avi_length)

    for i, url in enumerate(tmp_df1['link']):

        id, is_vitim = crawling_path(driver=driver, url=url, model=model) # 이름, 전문배우 (맞으면 True, 틀리면 False)
        if is_vitim:
            print(is_vitim, id)
            tmp_df1.loc[i, "pro_actor"] = is_vitim
            tmp_df1.loc[i, "id"] = id
        else:
            tmp_df1.loc[i, "id"] = id
    print(tmp_df1)
    tmp_df1.to_csv('./test.csv', index=False, encoding='UTF8')




