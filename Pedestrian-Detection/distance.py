import cv2
import torch
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Fast R-CNN 모델 로드
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# GPU 사용 여부 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 비디오 로드
video = cv2.VideoCapture('testVideo.mp4') ### 비디오 파일명 바꿔야함!

width, height = 1280, 720
fps = video.get(cv2.CAP_PROP_FPS)

# 비디오 저장을 위한 VideoWriter 객체 생성
_output1 = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# 폴리곤 ROI 정의
roi_points = np.array([[400, 720], [400, 400], [870, 400], [870, 720]], np.int32)

# 실제 객체의 폭과 카메라 초점 거리 정의
object_width = 50  # 센티미터 단위
focal_length = 1000  # 픽셀 단위

# CWS 거리 범위 정의
dist_ = 12

# 비디오 프레임 처리
while True:
    success, frame = video.read()
    frame = cv2.resize(frame, (1280, 720))
    if not success:
        break

    # 프레임을 텐서로 변환 및 정규화
    frame_tensor = F.to_tensor(frame).unsqueeze(0).to(device)

    # 객체 감지 수행
    with torch.no_grad():
        detections = model(frame_tensor)[0]

    # 폴리곤 ROI 그리기
    cv2.polylines(frame, [roi_points], True, (0, 200, 0), 2)

    # ROI 내부의 세 개의 수평선 y-좌표 계산
    line_y1 = 600
    line_gap = 10
    line_ys = [line_y1 + i * line_gap for i in range(dist_)]

    # 라인 색상 및 교차된 라인 수 초기화
    line_colors = [(255, 0, 0) for _ in range(dist_)]
    crossed_lines = []

    # 프레임의 중심선 그리기
    height, width, _ = frame.shape
    cv2.line(frame, (width // 2, 0), (width // 2, height), (160, 160, 160), 2)

    # ROI 내부에 있는 객체의 중심점 확인
    for i in range(len(detections["boxes"])):
        xmin, ymin, xmax, ymax = detections["boxes"][i]
        score = detections["scores"][i]

        # 점수 임계값 설정
        if score >= 0.4:
            centroid_x = int((xmin + xmax) / 2)
            centroid_y = int((ymin + ymax) / 2)

            # 객체 중심점이 ROI 내부에 있는지 확인
            if cv2.pointPolygonTest(roi_points, (centroid_x, centroid_y), False) > 0:
                cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)

                # 경계 상자가 선을 교차하는지 확인
                for i, line_y in enumerate(line_ys):
                    if ymax >= line_y:
                        line_colors[i] = (0, 0, 255)
                        crossed_lines.append(i + 1)

                # 객체와의 거리 계산
                distance = (object_width * focal_length) / (xmax - xmin)

                # 경계 상자 그리기 및 거리 표시
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                cv2.putText(frame, f"Dist: {distance:.2f} cm", (int(xmin), int(ymin) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            else:
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)

    # 라인 그리기
    for line_y, color in zip(line_ys, line_colors):
        cv2.line(frame, (roi_points[0][0], line_y), (roi_points[2][0], line_y), color, 2)

    # 교차된 라인 번호 표시 및 경고 메시지 출력
    if crossed_lines:
        cv2.putText(frame, 'BRAKE', (1000 - 25, 100 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        crossed_lines_str = ', '.join(str(line_num) for line_num in crossed_lines)
        cv2.putText(frame, str(max(crossed_lines)), (1000, 100 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if max(crossed_lines) == 4 and max(crossed_lines) <= 5:
            cv2.putText(frame, 'FORWARD COLLISION WARNING', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        elif max(crossed_lines) == 6 and max(crossed_lines) <= 8:
            cv2.putText(frame, 'COLLISION WARNING SEVERE', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        elif max(crossed_lines) >= 9 and max(crossed_lines) <= 11:
            cv2.putText(frame, 'PAY ATTENTION & TAKE CONTROL', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    # 비디오 프레임 저장
    _output1.write(frame)

# 비디오 릴리스 및 저장
video.release()
_output1.release()
cv2.destroyAllWindows()
