# Pedestrian-Detection
Repo for Pedestrian-Detection part of Autonomous Driving Project

- 사용 데이터 = https://www.kaggle.com/datasets/karthika95/pedestrian-detection 
( 데이터 압축 파일 다운로드 경로 = https://drive.google.com/file/d/1U0QhjjhvuhT28uxxy_E_dgyu1cRlO15-/view?usp=sharing )
- 사용 모델 = faster rcnn
- 현재 collate_fn부분 오류 발생
- 이후 `python preprocess.py`실행 (이때 해당 파일 상단의 split='Train'부분을 변경해 데이터를 전처리)
- `python main.py`를 통해 모델 학습

*이후 F2DNet으로 재시도해볼 예정...

## References
- https://dacon.io/competitions/official/236107/codeshare/8321?page=1&dtype=recent
- https://www.kaggle.com/code/a0121543/pedestrian-detection-with-pytorch