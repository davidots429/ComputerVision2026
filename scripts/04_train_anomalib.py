import os
import cv2
import time
import glob
import argparse
import numpy as np
import torch
from torchvision import transforms
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

def parse_args():
    parser = argparse.ArgumentParser(description="Anomalib PatchCore 학습 파이프라인")
    parser.add_argument("--coreset_ratio", type=float, default=0.1, 
                        help="메모리 뱅크 압축 비율 (기본값: 0.1)")
    return parser.parse_args()

def load_gt_boxes(label_path, img_size=640):
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
    print(f"Anomalib PatchCore 파이프라인 시작 (Coreset Ratio: {args.coreset_ratio})")
    
    datamodule = Folder(
        name="pcb_defect",
        root=os.path.abspath("./data/anomalib_format"),
        normal_dir="normal",
        abnormal_dir="abnormal",
        train_batch_size=32,
        eval_batch_size=32
    )

    model = Patchcore(
        backbone="wide_resnet50_2",  
        pre_trained=True,            
        coreset_sampling_ratio=args.coreset_ratio
    )

    engine = Engine(
        default_root_dir="./models/anomalib_results", 
        accelerator="gpu",           
        devices=1
    )

    print("메모리 뱅크 저장 시작.")
    engine.fit(datamodule=datamodule, model=model)
    
    print("\n기본 성능 테스트 진행 중...")
    engine.test(datamodule=datamodule, model=model)

    print("\n[사용자 정의 성능 평가 진행 중...]")
    
    VAL_IMG_DIR = "./data/yolo_format/images/val"
    VAL_LBL_DIR = "./data/yolo_format/labels/val"
    test_images = glob.glob(os.path.join(VAL_IMG_DIR, "*.jpg"))
    
    total_tp, total_fp, total_fn = 0, 0, 0
    inference_times = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        th_value = model.pixel_threshold.value.item()
    except:
        th_value = 0.5 

    for img_path in test_images:
        base_name = os.path.basename(img_path)
        
        img = cv2.imread(img_path)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = transform(img_rgb).unsqueeze(0).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            output = model(input_tensor)
            
            # 버전에 따른 출력 형태 완벽 대응 (InferenceBatch 객체 속성 추출)
            if hasattr(output, 'anomaly_map'):
                anomaly_map = output.anomaly_map
            elif isinstance(output, dict) and "anomaly_map" in output:
                anomaly_map = output["anomaly_map"]
            else:
                anomaly_map = output
                
        anomaly_map = anomaly_map.squeeze().cpu().numpy()
        anomaly_map_resized = cv2.resize(anomaly_map, (640, 640))
        mask = (anomaly_map_resized > th_value).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        end_time = time.time()
        
        inference_times.append((end_time - start_time) * 1000)
        
        pred_boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 10: 
                x, y, w, h = cv2.boundingRect(cnt)
                pred_boxes.append([x, y, x+w, y+h])
                
        label_path = os.path.join(VAL_LBL_DIR, base_name.replace(".jpg", ".txt"))
        gt_boxes = load_gt_boxes(label_path)
        
        matched_gt = set()
        matched_pred = set()
        
        for g_idx, gt in enumerate(gt_boxes):
            for p_idx, pred in enumerate(pred_boxes):
                if calculate_iou(gt, pred) > 0.1: 
                    matched_gt.add(g_idx)
                    matched_pred.add(p_idx)
                    
        tp = len(matched_gt)
        fn = len(gt_boxes) - tp
        fp = len(pred_boxes) - len(matched_pred)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn

    mean_time = np.mean(inference_times)
    fps = 1000 / mean_time if mean_time > 0 else 0
    accuracy = (total_tp / (total_tp + total_fp + total_fn)) * 100 if (total_tp + total_fp + total_fn) > 0 else 0
    
    print(f"\n[PatchCore (Ratio: {args.coreset_ratio}) 최종 성능 평가 결과]")
    print(f"· 평균 추론 시간 (Mean Inference Time): {mean_time:.2f} ms")
    print(f"· 초당 처리 속도 (Throughput): {fps:.2f} FPS")
    print(f"· 결함 탐지 정확도 (Detection Accuracy): {accuracy:.2f} %")
    print(f"  - True Positive(정상 검출): {total_tp}개")
    print(f"  - False Positive(과검/오탐): {total_fp}개")
    print(f"  - False Negative(미검/놓침): {total_fn}개")

if __name__ == '__main__':
    main()
