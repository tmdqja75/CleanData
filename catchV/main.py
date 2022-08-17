from commons import functions, detected

from glob import glob
from deepface import DeepFace
from tqdm import tqdm

import os
import pickle
import cv2


def embeded_file(datapath):
    """
    이미지를 embed 해주고 삭제해 주는 함수.
    :param
        datapath(string):기본 경로
    :return
        rb(dictionary):{id:[[emb_data1], [emb_data2], ...], id2:[[emb_data1], [emb_data2], ...], ...}
    """
    img_path_list = []
    img_path_list.extend(glob(os.path.join(datapath, '*/*.jpg')))

    model_name = 'Facenet512'
    detector_backend = 'mtcnn'

    emb_dic = {}
    print(img_path_list)
    for img_path in tqdm(img_path_list):
        img_path_split = img_path.split('\\')
        d_name = img_path_split[-2] # 디렉토리 이름
        embedding = DeepFace.represent(img_path=img_path, model_name=model_name, detector_backend=detector_backend) # img_path 변경
        os.remove(img_path)
        if d_name not in emb_dic.keys():
            emb_dic[d_name] = [embedding]
        else:
            emb_dic[d_name].append(embedding)
    functions.rm_dir(img_path_list)

    rb = functions.make_pickle(emb_dic, datapath)

    return rb


def display(file_name, encoding_file):
    """
    :parameter
        :param file_name: VideoCapture 실행 시킬 경로.
        :param encoding_file: encoding 된 데이터 들
    :return:
    """
    cap = cv2.VideoCapture(file_name)
    writer = None
    data = pickle.loads(open(encoding_file, "rb").read())

    if not cap.isOpened:
        print('-- (!) Check your cap or video root (!) --')
        exit(0)
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('-- (!) Check your cap or video root (!) --')
            cap.release()
            writer.release()

        detected.detectAndDisplay(frame, id, data)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            break

    cv2.destroyAllWindows()


### 실행
if __name__ == '__main__':
    rb=embeded_file(datapath='.\\data') # embed
    display(file_name=0, encoding_file='.\\data\\dataset.pkl') # detected
