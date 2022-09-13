# facedetect

## v1.1.0
1. zip 파일 [다운로드](https://drive.google.com/file/d/1xtowuN7dZRAzPMpLzpb2lmwInq6zkrBD/view?usp=sharing)
  - 모든 과정을 확인
    - data 폴더에 AvList와 target 압축 해제후 해당 경로처럼 파일 넣기
    ```
    data
      ㄴ AvList
          ㄴ Id값1
              ㄴ file.jpg
              ㄴ ...
          ㄴ Id값2
              ㄴ file.jpg
              ㄴ ...
      ㄴ target
          ㄴ Id값1
              ㄴ file.jpg
              ㄴ ...
          ㄴ Id값2
              ㄴ file.jpg
              ㄴ ... 
    ```
    - data 폴더에 testURL_youtube.csv 파일 넣기
    - embeding 생략
    - 폴더에 사진 넣지 말고 진행
    - data 폴더에 testURL_youtube.csv 파일 넣기
    - data 폴더에 dataset.pkl 및 pro_dataset.pkl 넣기

2. 실행방법
    1. https://github.com/gangfunction/catchvfrontnext.git
       - develop branch 에서 실행
       - npm rum dev
       - http://localhost:3000
    2. https://github.com/gangfunction/catchvbackend.git
       - develop branch 에서 실행
       - 실행하기.
       - 따로 페이지 띄울 필요없음.
    3. 프로그램 시작
       - pip install -r requirements.txt
       - app.py 를 실행 시킬때 작업 디렉터리 변경
         - ./CleanData/src
       - 실행.
# 주의사항
1. login 후 service에서 사진 업로드
2. 자동으로 딕텍션 시작.

# 현재
1. user만 사진 업로드 가능