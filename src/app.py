import json
import os
import uuid
from datetime import datetime

import pandas as pd
import requests
import werkzeug
from flask import (Flask, abort, current_app, request, send_file,
                   send_from_directory)
from flask_restful import Api, Resource, reqparse
from werkzeug.utils import secure_filename

from main import run

app = Flask(__name__)
app.secret_key = "secret key"

api = Api(app)

# app.config에 사진 저장할 data 폴더 경로 저장
# UPLOAD_FOLDER 경로 없으면 만들기
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 처리 가능한 파일확장자만 거를 수 있는 함수
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg", "gif", "mov", "mp4", "mpeg"])
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@api.representation('multipart/form-data')
def output_file(data, code, headers):
    filepath = os.path.join(data["directory"], data["filename"])

    response = send_file(
        filename_or_fp=filepath,
        mimetype="application/octet-stream",
        as_attachment=True,
        attachment_filename=data["filename"]
    )
    return response


@app.route('/image/api', methods=['POST'])
def image_api():
    
    # 딥러닝 모델이 돌고 있으면, 모델이 돌고 있다는 값 반환하고 post 함수 exit.
    with open('./src/running.txt', 'r') as f:
        running = f.readline().strip()
    f.close()
    
    if running=='True':
        # return {"message": "Face Detection is in Progress"}
        requests.post('http://localhost:8080/image/receive', "Face Detection is in Progress")
        return
        
    # 사진 소유자 email과 영상 생성 시점 받이오기
    img_owner = request.form['email']
    # 영상 생성 시점이 있는지 없는지 확인
    try:
        vid_date = request.form['startDate'] # form: 'Fri Sep 09 2022 15:00:12 GMT+0900'
    except:
        vid_date = None
        
    # 사진 담을 directory 생성하고 만들기
    file_path = os.path.join(app.config["UPLOAD_FOLDER"],img_owner)
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    
    # 영상 생성 시점 'yyyy-mm-dd'형식으로 formatting
    if vid_date is not None:
        vid_date = datetime.strptime(vid_date[:15], '%a %b %d %Y').strftime('%Y-%m-%d')
        print("VID_DATE:", vid_date)
    
        # 피해자 원본 영상 생성 시점(startDate)를 txt 파일에 기록하기
        with open(f'{app.config["UPLOAD_FOLDER"]}/startDate.txt', 'a') as f_date:
            f_date.write(vid_date)
        f_date.close()
    
    # 사진 파일들을 새 uuid 이름을 지정한 뒤위에 생성된 directory에 저장
    for f in list(request.files.listvalues())[0]:
        if allowed_file(f.filename):
            original_filename = f.filename
            extension = original_filename.rsplit(".", 1)[1].lower()
            filename = str(uuid.uuid1()) + "." + extension
            f.save(os.path.join(file_path, filename))
        
    # return {"message": "File(s) successfully uploaded"}
    requests.post('http://localhost:8080/image/receive', "File(s) successfully uploaded")


@app.route('/process')
def send_csvfile():
    # 피해자 리스트 중에 영상 시작일이 가장 일찍인 영상 시점 찾기
    datelist = []
    f_date = open(os.path.join(app.config['UPLOAD_FOLDER'], 'startDate.txt'))
    for line in f_date:
        line = line.strip()
        dateline = datetime.strptime(line, '%Y-%m-%d')
        datelist.append(dateline)
    earliest_date = min(datelist)
    f_date.close()
    
    # 딥러닝 모델 작동
    run(earliest_date)
    
    # ---결과 읽어오기---
    PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result", 'answer.csv')
    df_result = pd.read_csv(PATH)
    result = df_result[df_result['등장인물']=='정답영상']
    
    list_client = list(result.id.unique())
    #-----------------
    
    # ---결과값 json formatting 결과 dict형식으로 준비---
    return_dict = dict()
    return_dict["total_inspected_video_count"] = len(df_result)
    return_dict['result'] = []
    
    print(list_client)
    for client in list_client:
        client_dict = dict()
        client_dict['requested_user_email'] = client
        client_dict['urls'] = result[result['id']==client]['link'].to_list()
        
        return_dict['result'].append(client_dict)
    #------------------------------------------------
    
    # 내보낼 결과값 json 형식으로 변형
    return_json = json.dumps(return_dict, indent=2)
    return return_json


if __name__ == "__main__":
    app.run(port=5001, debug=True)
    
# 서버에서 돌릴 시
# if __name__=="__main__":
#    app.run(host='0.0.0.0', port=8080)
 