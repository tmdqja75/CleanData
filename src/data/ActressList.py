import os
from time import sleep

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

options = webdriver.ChromeOptions()
options.add_argument("headless")
options.add_argument("lang=ko_KR")
chromedriver_path = ".\chromedriver.exe"
##############################################################  ############
##################### variable related selenium ##########################
##########################################################################
driver = webdriver.Chrome(executable_path=chromedriver_path)


def main():
    tmp = "https://www.avdbs.com"
    path = "D:/ITStudy/CatchV/AvList/"
    url_tmp = "https://www.avdbs.com/menu/actor.php?actor_idx="
    for i in range(1555, 9999):
        if i % 5 == 0:
            sleep(10)
        url = url_tmp + "{0:04}".format(i)
        try:
            driver.get(url)

            wait = WebDriverWait(driver, 5)
            element = wait.until(EC.element_to_be_clickable((By.ID, "contants")))

            kr_name = driver.find_element(
                By.XPATH, '//span[@class="inner_name_kr"]'
            ).text
            cn_name = driver.find_element(
                By.XPATH, '//span[@class="inner_name_cn"]'
            ).text
            en_name = driver.find_element(
                By.XPATH, '//span[@class="inner_name_en"]'
            ).text
            img_lst = []

            p_img = driver.find_element(
                By.XPATH, '//p[@class="profile_gallery"]/img'
            ).get_dom_attribute("src")
            img_lst.append(p_img)

            f_img = driver.find_elements(
                By.XPATH, '//div[@class="other_photo_list"]/ul/li[1]/a/img'
            )
            if f_img:
                img_lst.append(f_img[0].get_dom_attribute("src"))
            etc_imgs = driver.find_elements(
                By.XPATH, '//img[@alt="{0}"]'.format(kr_name)
            )
            for img in etc_imgs:
                img_lst.append(img.get_dom_attribute("src"))

            newpath = path + en_name.replace(" ", "_")
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            for idx, img in enumerate(img_lst):
                img_data = requests.get(img).content
                with open(
                    newpath + "/" + en_name.replace(" ", "_") + "_" + str(idx) + ".jpg",
                    "wb",
                ) as handler:
                    handler.write(img_data)
        except:
            tmp = driver.find_elements(By.XPATH, '//p[@class="confirm"]')
            if tmp:
                sleep(30)


if __name__ == "__main__":
    main()
