from time import sleep

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

options = webdriver.ChromeOptions()
options.add_argument("headless")
options.add_argument("lang=ko_KR")
chromedriver_path = ".\chromedriver.exe"

from selenium.webdriver.chrome.options import Options

options = Options()
ext_file1 = "./bihmplhobchoageeokmgbdihknkjbknd.crx"  # vpn 확장자
ext_file2 = "./cjpalhdlnbpafiamejdnhcphjbkeiagm.crx"  # 광고 방지 확장자
options.add_extension(ext_file1)
options.add_extension(ext_file2)
driver = webdriver.Chrome(executable_path=chromedriver_path, chrome_options=options)
url = "chrome-extension://bihmplhobchoageeokmgbdihknkjbknd/panel/index.html"
driver.get(url)

wait = WebDriverWait(driver, 10)
element = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "staticBackground")))

vpn_btn = driver.find_element(By.XPATH, '//div[@class="staticBackground"]')
vpn_btn.click()
wait = WebDriverWait(driver, 10)
element = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "postConnection")))


def pornhub(input1):
    if input1 == "asian":  # 카테고리가 asian이 맞다면
        cat = "video?c=1"  # cat 코드 생성
    else:
        cat = ""
    url = "https://www.pornhub.com/" + cat
    link_list = []
    time_list = []
    drtn_list = []
    cnt = 1
    next_page = [1]
    while next_page:  # 다음 페이지가 없을 때까지 순회
        driver.get(url)

        # 수집할 태그들이 다 뜰 때까지 기다림
        wait = WebDriverWait(driver, 10)
        element = wait.until(EC.element_to_be_clickable((By.ID, "js-networkBar")))

        # url,업로드날짜, 영상길이 list를 받아옴
        hrefs = driver.find_elements(By.XPATH, '//div[@class="phimage"]/a')
        srcs = driver.find_elements(By.XPATH, '//div[@class="phimage"]/a/img')
        drtns = driver.find_elements(By.XPATH, '//var[@class="duration"]')

        for h, s, d in zip(hrefs, srcs, drtns):
            link_list.append(
                "https://www.pornhub.com" + h.get_dom_attribute("href")
            )  # 본 페이지 + url 주소
            time_list.append(
                s.get_dom_attribute("data-thumb_url").split("/")[4]
            )  # 연도와 날짜만 있는 str 축출
            drtn_list.append(d.get_attribute("innerText"))  # 영상길이

        cnt += 1
        url = (
            "https://www.pornhub.com/" + cat + "&page=" + "{}".format(cnt)
        )  # 다음 페이지 주소
        next_page = driver.find_elements(
            By.XPATH, '//a[@class="orangeButton"]'
        )  # 다음페이지 버튼 유무
    df = pd.DataFrame(
        (zip(link_list, time_list, drtn_list)),
        columns=["link", "upload_date", "total_len"],
    )
    return df


def gotporn(cat):

    url = "https://www.gotporn.com/categories/" + cat
    link_list = []
    time_list = []
    drtn_list = []
    cnt = 1
    next_page = [1]
    while next_page:  # next_page:   # next_page
        driver.get(url)

        # 기다림
        wait = WebDriverWait(driver, 10)
        element = wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, '//*[@id="wrapper"]/header/div[1]/div[1]/a/img')
            )
        )
        # btnAdult =  driver.find_elements(By.XPATH, "/html/body/section/div/div[2]/div")
        # if btnAdult:
        # btnAdult[0].click()

        # url,업로드날짜, 영상길이 list를 받아옴
        hrefs = driver.find_elements(
            By.XPATH, '//li[@class="video-item  poptrigger "]/a'
        )
        srcs = driver.find_elements(
            By.XPATH, '//li[@class="video-item  poptrigger "]/a/span/span[2]/img'
        )
        drtns = driver.find_elements(By.XPATH, '//span[@class="duration"]')

        for h, s, d in zip(hrefs, srcs, drtns):
            """
            타이틀을 받아올까 했지만 무의미하여 주석처리
            try:
                title_list.append(h.get_dom_attribute('data-title'))
            except:
                title_list.append('None')
            """
            link_list.append(h.get_dom_attribute("href"))
            time_list.append(
                "".join(s.get_dom_attribute("data-default-src").split("/")[3:5])
            )
            drtn_list.append(d.get_attribute("innerText"))

        cnt += 1
        url = (
            "https://www.gotporn.com/"
            + "categories/"
            + cat
            + "?page="
            + "{}".format(cnt)
        )
        next_page = driver.find_elements(
            By.XPATH, '//a[@class="btn btn-secondary paginate-show-more"]'
        )
    df = pd.DataFrame(
        (zip(link_list, time_list, drtn_list)),
        columns=["link", "upload_date", "total_len"],
    )  # title_list
    return df


def spankwire(cat):
    cat = cat
    url = "https://www.redtube.com/redtube/" + cat
    link_list = []
    time_list = []
    drtn_list = []
    cnt = 1
    next_page = [1]
    while next_page:  # (cnt != 5)
        driver.get(url)

        # 기다림
        wait = WebDriverWait(driver, 10)
        element = wait.until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="redtube_logo_image"]'))
        )
        try:
            # url,업로드날짜, 영상길이 list를 받아옴
            hrefs = driver.find_elements(
                By.XPATH,
                '//div[@class="video_block_wrapper js_mediaBookBounds "]/span/a',
            )
            srcs = driver.find_elements(
                By.XPATH,
                '//div[@class="video_block_wrapper js_mediaBookBounds "]/span/a/picture/img',
            )
            drtns = driver.find_elements(By.XPATH, '//span[@class="duration"]')
            for h, s, d in zip(hrefs, srcs, drtns):
                try:
                    link_list.append(
                        "https://www.redtube.com" + h.get_dom_attribute("href")
                    )
                    time_list.append(
                        "".join(s.get_dom_attribute("data-path").split("/")[4])
                    )
                    drtn_list.append(d.get_attribute("innerText")[-5:])
                except:
                    pass
        except:
            pass

        cnt += 1
        url = (
            "https://www.redtube.com/" + "redtube/" + cat + "?page=" + "{}".format(cnt)
        )
        next_page = driver.find_elements(By.XPATH, '//a[@id="wp_navNext"]')

    df = pd.DataFrame(
        (zip(link_list, time_list, drtn_list)),
        columns=["link", "upload_date", "total_len"],
    )  # title_list
    return df


def main(cat, str1):
    df1 = pornhub(cat)
    df2 = gotporn(cat)
    df3 = spankwire(cat)

    all_df = pd.concat([df1, df2, df3])  # df3
    all_df = all_df.drop_duplicates(["link"])
    all_df["pro_actor"] = False
    all_df.to_csv("{0}.csv".format(str1), index=False, encoding="utf-8-sig")
    return all_df


if __name__ == "__main__":
    cat = "asian"
    df = main(cat, "url_day2")
