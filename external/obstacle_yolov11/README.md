# obstacle_yolov11

이 폴더는 MORAI AI 미션용 장애물 인식을 위해 YOLOv11 모델을 학습·관리하는 전용 작업 공간입니다. ROS 패키지와는 독립적으로 유지되므로, `catkin_ws/external/obstacle_yolov11` 위치에 그대로 두고 학습 스크립트만 실행하면 됩니다.

## 폴더 구조
- `raw/` : 증강 전 원본 이미지·라벨
- `dataset/` : 증강 후 데이터(`augment_dataset.py` 실행 시 자동 생성, Git에서는 무시)
- `weights/` : 학습된 가중치 보관 (`obstacle_best.pt`만 Git 추적)
- `runs/` : Ultralytics 학습 결과 (대용량이라 Git 무시)
- `augment_dataset.py` : 원본 → 증강 데이터 생성
- `split_dataset.py` : 증강 데이터(train/val/test 분할)
- `train_yolov11.py` : Ultralytics YOLO 학습 스크립트
- `yolo11n.pt` : 기본 사전학습 가중치 (필요 시 교체 가능)

## 환경 설정
```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell이면 .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  # GPU 없으면 cpu 버전
pip install ultralytics albumentations scikit-image scikit-learn opencv-python-headless PyYAML
```

## 데이터 증강
`raw/images`, `raw/labels`에 YOLO 형식 라벨을 준비한 뒤 다음 명령을 실행합니다. 이미지당 기본 10개의 증강 샘플이 생성됩니다.
```bash
python augment_dataset.py \
    --images raw/images \
    --labels raw/labels \
    --output dataset \
    --image-format jpg
```

## 데이터 분할
증강된 데이터를 학습/검증/테스트 세트로 나눕니다.
```bash
python split_dataset.py \
    --images dataset/images \
    --labels dataset/labels \
    --output dataset \
    --train-ratio 0.8 \
    --val-ratio 0.2 \
    --seed 42
```
> `dataset/` 전체는 `.gitignore`로 제외되므로 필요한 경우 수동으로 백업하세요.

## 학습
현재 라벨 순서는 `cone, wall1, wall2, barrel, box, red, red2, red3, orange, white` 입니다. 아래 명령으로 학습을 실행하면 `runs/train/obstacle_yolov11/` 폴더에 결과가 저장되고, 최종 가중치는 `weights/obstacle_best.pt`로 복사해 둡니다.
```bash
python train_yolov11.py \
    --dataset-root dataset \
    --model yolo11n.pt \
    --epochs 30 \
    --img-size 640 \
    --batch 16 \
    --device cpu \
    --project runs/train \
    --name obstacle_yolov11 \
    --class-names cone wall1 wall2 barrel box red red2 red3 orange white

cp runs/train/obstacle_yolov11/weights/best.pt weights/obstacle_best.pt
```

## ROS 연동
학습된 `weights/obstacle_best.pt`는 `perception_pkg`의 `obstacle_detection_node.py`에서 자동으로 로드됩니다. 런치 시 `obstacle_enabled:=true`와 `obstacle_model:=<가중치 경로>`를 지정하면 `/perception/obstacles_2d`, `/perception/obstacle_bias` 토픽을 통해 장애물 회피 조향이 적용됩니다.

## 기타
- `dataset/`, `runs/`는 용량이 크므로 Git에 올리지 않습니다.
- 새로운 클래스 순서를 사용하려면 라벨 파일 ID와 `--class-names` 순서를 동일하게 조정해야 합니다.
- 가중치 공유가 필요하면 `weights/` 폴더 대신 별도의 스토리지나 릴리스를 이용하는 것을 권장합니다.

