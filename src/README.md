# facedetect

## v1.0.0
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
   - pip install -r requirements.txt
   - python .\src\main.py

# 주의사항
1. 더블 모니터에서 확인해야 결과를 확인가능
2. 1번 모니터에서 캡처를 계속 진행하여 작업 불가
3. 2번 모니터에서 프로그램 실행
4. imshow 진행되면 웹 실행 반대 모니터로 이동.