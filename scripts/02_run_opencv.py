import os
import cv2
import numpy as np
import time
import argparse
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="OpenCV Image Subtraction Baseline")
    parser.add_argument("--threshold", type=int, default=50, help="픽셀 차이 임계값 (기본값: 50)")
    parser.add_argument("--kernel_size", type=int, default=3, help="노이즈 제거용 커널 크기 (기본값: 3)")
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
    """두 바운딩 박스 간의 IoU 계산"""
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
    
    VAL_IMG_DIR = "./data/yolo_format/images/val"
    VAL_LBL_DIR = "./data/yolo_format/labels/val"
    TEMPLATE_DIR = "./data/anomalib_format/normal"
    RESULT_DIR = "./results/opencv_results"
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    test_images = glob.glob(os.path.join(VAL_IMG_DIR, "*.jpg"))
    
    total_tp, total_fp, total_fn = 0, 0, 0
    inference_times = []
    
    print(f"=== OpenCV 이미지 차분 검증 시작 (Threshold: {args.threshold}, Kernel: {args.kernel_size}) ===")
    
    for img_path in test_images:
        base_name = os.path.basename(img_path)
        img_id = base_name.replace("_test.jpg", "")
        
        # 1. 이미지 로드 (검사 대상 및 정상 템플릿)
        test_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        temp_path = os.path.join(TEMPLATE_DIR, f"{img_id}_temp.jpg")
        temp_img = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
        
        if test_img is None or temp_img is None:
            continue
            
        # 2. 이미지 차분 및 추론 시간 측정
        start_time = time.time()
        
        diff = cv2.absdiff(temp_img, test_img)
        _, thresh = cv2.threshold(diff, args.threshold, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((args.kernel_size, args.kernel_size), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        end_time = time.time()
        inference_times.append((end_time - start_time) * 1000) # ms 단위
        
        # 3. 예측 바운딩 박스 추출
        pred_boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 10: # 미세 노이즈 면적 필터링
                x, y, w, h = cv2.boundingRect(cnt)
                pred_boxes.append([x, y, x+w, y+h])
                
        # 4. 정답 라벨 로드 및 매칭 평가 (IoU 기준)
        label_path = os.path.join(VAL_LBL_DIR, base_name.replace(".jpg", ".txt"))
        gt_boxes = load_gt_boxes(label_path)
        
        matched_gt = set()
        matched_pred = set()
        
        for g_idx, gt in enumerate(gt_boxes):
            for p_idx, pred in enumerate(pred_boxes):
                if calculate_iou(gt, pred) > 0.1: # 흑백 차분 특성상 느슨한 IoU 기준 적용
                    matched_gt.add(g_idx)
                    matched_pred.add(p_idx)
                    
        tp = len(matched_gt)
        fn = len(gt_boxes) - tp
        fp = len(pred_boxes) - len(matched_pred)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # 5. 최종 종합 지표 산출
    mean_time = np.mean(inference_times)
    fps = 1000 / mean_time if mean_time > 0 else 0
    accuracy = (total_tp / (total_tp + total_fp + total_fn)) * 100 if (total_tp + total_fp + total_fn) > 0 else 0
    
    print("\n[최종 성능 평가 결과]")
    print(f"· 평균 추론 시간 (Mean Inference Time): {mean_time:.2f} ms")
    print(f"· 초당 처리 속도 (Throughput): {fps:.2f} FPS")
    print(f"· 결함 탐지 정확도 (Detection Accuracy): {accuracy:.2f} %")
    print(f"  - True Positive(정상 검출): {total_tp}개")
    print(f"  - False Positive(오탐지): {total_fp}개")
    print(f"  - False Negative(미검출): {total_fn}개")

if __name__ == "__main__":
    main()
