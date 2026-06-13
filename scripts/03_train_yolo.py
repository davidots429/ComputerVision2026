import os
import yaml
import glob
import time
import argparse
import numpy as np
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO11 학습 및 성능 평가")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 사이즈 (기본값: 16)")
    parser.add_argument("--lr", type=float, default=0.01, help="초기 학습률 (Learning Rate, 기본값: 0.01)")
    parser.add_argument("--epochs", type=int, default=300, help="최대 학습 에폭 (기본값: 300)")
    parser.add_argument("--patience", type=int, default=30, help="Early Stopping (기본값: 30)")
    return parser.parse_args()

def load_gt_boxes(label_path, img_size=640):
    """YOLO 포맷의 정답 라벨을 절대 좌표(x1, y1, x2, y2)로 변환"""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                _, xc, yc, w, h = map(float, parts)
                x1 = int((xc - w/2) * img_size)
                y1 = int((yc - h/2) * img_size)
                x2 = int((xc + w/2) * img_size)
                y2 = int((yc + h/2) * img_size)
                boxes.append([x1, y1, x2, y2])
    return boxes

def calculate_iou(box1, box2):
    """두 바운딩 박스 간의 IoU(교집합/합집합) 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def main():
    args = parse_args()

    # 환경 설정 파일(yaml) 자동 생성
    yaml_data = {
        'path': os.path.abspath('./data/yolo_format'),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'open', 1: 'short', 2: 'mousebite', 3: 'spur', 4: 'copper', 5: 'pin-hole'}
    }
    yaml_path = './data/yolo_format/dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    print(f"YAML 설정 파일 생성 완료. | 파일 위치: {yaml_path}")

    # ==========================================
    # YOLO11 모델 학습
    # ==========================================
    print(f"YOLO11s 모델 로드 및 학습 시작...")
    print(f"설정값: Epochs={args.epochs}, Patience={args.patience}, Batch={args.batch_size}, LR={args.lr}")
    
    model = YOLO('yolo11s.pt')
    
    model.train(
        data=yaml_path,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=640,
        batch=args.batch_size,
        lr0=args.lr,
        device=0,
        project='./models/yolo_results',
        name='pcb_defect',
        exist_ok=True,
        workers=4
    )
    print("YOLO11 학습 완료. 성능 평가를 시작합니다.")

    # ==========================================
    # 평가지표 산출 (비교용)
    # ==========================================
    
    # 학습된 최고 성능 가중치 로드
    best_model_path = './runs/detect/models/yolo_results/pcb_defect/weights/best.pt'
    if not os.path.exists(best_model_path):
        print("모델 가중치 파일이 없습니다. 평가를 건너뜁니다.")
        return
        
    eval_model = YOLO(best_model_path)
    
    VAL_IMG_DIR = "./data/yolo_format/images/val"
    VAL_LBL_DIR = "./data/yolo_format/labels/val"
    test_images = glob.glob(os.path.join(VAL_IMG_DIR, "*.jpg"))
    
    total_tp, total_fp, total_fn = 0, 0, 0
    inference_times = []
    
    for img_path in test_images:
        base_name = os.path.basename(img_path)
        
        # 1. 추론 및 시간 측정
        start_time = time.time()
        # verbose=False로 설정하여 화면에 불필요한 로그 출력 방지
        results = eval_model.predict(source=img_path, verbose=False) 
        end_time = time.time()
        inference_times.append((end_time - start_time) * 1000) # ms 단위 변환
        
        # 2. 예측 바운딩 박스 추출
        pred_boxes = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            # GPU에 있는 좌표 데이터를 CPU로 가져와 Numpy 배열로 변환
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                pred_boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
                
        # 3. 정답 라벨 로드 및 IoU 기반 매칭 평가
        label_path = os.path.join(VAL_LBL_DIR, base_name.replace(".jpg", ".txt"))
        gt_boxes = load_gt_boxes(label_path)
        
        matched_gt = set()
        matched_pred = set()
        
        for g_idx, gt in enumerate(gt_boxes):
            for p_idx, pred in enumerate(pred_boxes):
                # 동일한 조건 테스트를 위해 IoU > 0.1 적용
                if calculate_iou(gt, pred) > 0.1: 
                    matched_gt.add(g_idx)
                    matched_pred.add(p_idx)
                    
        tp = len(matched_gt)
        fn = len(gt_boxes) - tp
        fp = len(pred_boxes) - len(matched_pred)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # 종합 지표 계산
    mean_time = np.mean(inference_times)
    fps = 1000 / mean_time if mean_time > 0 else 0
    accuracy = (total_tp / (total_tp + total_fp + total_fn)) * 100 if (total_tp + total_fp + total_fn) > 0 else 0
    
    print("\n[YOLO11 최종 성능 평가 결과]")
    print(f"· 평균 추론 시간 (Mean Inference Time): {mean_time:.2f} ms")
    print(f"· 초당 처리 속도 (Throughput): {fps:.2f} FPS")
    print(f"· 결함 탐지 정확도 (Detection Accuracy): {accuracy:.2f} %")
    print(f"  - True Positive(정상 검출): {total_tp}개")
    print(f"  - False Positive(과검/오탐): {total_fp}개")
    print(f"  - False Negative(미검/놓침): {total_fn}개")

if __name__ == '__main__':
    main()
