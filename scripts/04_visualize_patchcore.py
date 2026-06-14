import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from anomalib.models import Patchcore
import glob

def main():
    print("PatchCore 시각화 (노이즈 필터 제거 & 가로 병합)...")
    
    SAVE_DIR = "./results/patchcore_visuals"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model = Patchcore(backbone="wide_resnet50_2", pre_trained=False, coreset_sampling_ratio=0.1)
    
    # 가중치 로드
    ckpt_path = glob.glob("./models/anomalib_results/Patchcore/pcb_defect/latest/weights/lightning/*.ckpt")
    if not ckpt_path: 
        print("에러: 가중치 파일(.ckpt)을 찾지 못했습니다.")
        return
        
    model.load_state_dict(torch.load(ckpt_path[0], map_location=device, weights_only=False)['state_dict'])
    model.to(device)
    model.eval()
    
    # 이미지 전처리 파이프라인
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_images = glob.glob("./data/yolo_format/images/val/*.jpg")[:5]
    
    for idx, img_path in enumerate(test_images):
        base_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        input_tensor = transform(img_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            if hasattr(output, 'anomaly_map'):
                anomaly_map = output.anomaly_map
            elif isinstance(output, dict) and "anomaly_map" in output:
                anomaly_map = output["anomaly_map"]
            else:
                anomaly_map = output
                
        # 1. 이상치 맵 후처리 및 0~255 정규화
        anomaly_map = anomaly_map.squeeze().cpu().numpy()
        anomaly_map_resized = cv2.resize(anomaly_map, (640, 640))
        
        # Min-Max 스케일링으로 0~255 변환 (시각화 및 고정 임계값 적용 용이)
        anomaly_norm = cv2.normalize(anomaly_map_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 2. 고정 임계값(Threshold) 적용 (예: 255 중 150 이상만 불량으로 판단)
        THRESH_VALUE = 150 
        _, mask = cv2.threshold(anomaly_norm, THRESH_VALUE, 255, cv2.THRESH_BINARY)
        
        # 3. 모폴로지 연산(Morphology)으로 자잘한 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # 노이즈 점 제거
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # 끊어진 결함 영역 연결
        
        # 4. 바운딩 박스 추출 (최소 면적 필터 다시 적용)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result_img = img.copy()
        for cnt in contours:
            # 면적이 50 픽셀 이상인 유의미한 결함만 박스 표시
            if cv2.contourArea(cnt) > 50: 
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
        # 컬러 히트맵 생성
        heatmap_color = cv2.applyColorMap(anomaly_norm, cv2.COLORMAP_JET)
        
        # 원본 | 히트맵 | 박스 결과 가로 병합
        combined_img = cv2.hconcat([img, heatmap_color, result_img])
        
        save_path = os.path.join(SAVE_DIR, f"debug_{base_name}")
        cv2.imwrite(save_path, combined_img)
        print(f"[{base_name}] 저장 완료 (임계값: {THRESH_VALUE})")
        
    print(f"\n시각화 완료! 결과가 {SAVE_DIR} 에 저장되었습니다.")

if __name__ == '__main__':
    main()