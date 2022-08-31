import os
import pickle
import pandas as pd

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
        f = open(datapath + "/" + pickle_name, 'rb')
        rb = pickle.load(f)
        f.close()
        for d_name in emb_dic.keys():
            if d_name in emb_dic.keys():  # 기존 이름 있으면 데이터 추가
                rb[d_name].extend(emb_dic[d_name])
            else:
                rb[d_name] = emb_dic[d_name]
        f = open(datapath + "/" + pickle_name, 'wb')
        pickle.dump(rb, f)
        f.close
    except: # pickle 파일이 존재 안할때
        f = open(datapath + "/" + pickle_name, 'wb')
        pickle.dump(emb_dic, f)
        f.close

    with open(datapath + "/" + pickle_name, 'rb') as f:
        rb = pickle.load(f)
        f.close
    return rb


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


def read_youtube_csv(path): # "./testURL_youtube.csv"
    df_youtube = pd.read_csv(path)
    df_youtube['upload_date'] = pd.to_datetime(df_youtube['upload_date'], format="%Y%m")
    return df_youtube