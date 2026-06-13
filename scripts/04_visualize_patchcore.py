import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from anomalib.models import Patchcore
import glob

def main():
    print("PatchCore 딥다이브 시각화 (노이즈 필터 제거 & 가로 병합)...")
    
    SAVE_DIR = "./results/patchcore_visuals"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    device = torch.device('cpu') 
    model = Patchcore(backbone="wide_resnet50_2", pre_trained=False, coreset_sampling_ratio=0.1)
    
    ckpt_path = glob.glob("./models/anomalib_results/Patchcore/pcb_defect/latest/weights/lightning/*.ckpt")
    if not ckpt_path: 
        print("에러: 가중치 파일(.ckpt)을 찾지 못했습니다.")
        return
        
    model.load_state_dict(torch.load(ckpt_path[0], map_location=device, weights_only=False)['state_dict'])
    model.to(device)
    model.eval()
    
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
                
        anomaly_map = anomaly_map.squeeze().cpu().numpy()
        anomaly_map_resized = cv2.resize(anomaly_map, (640, 640))
        
        # [디버깅] 터미널에 모델이 계산한 최대/최소/상위1% 수치 출력
        max_val = anomaly_map_resized.max()
        min_val = anomaly_map_resized.min()
        th_value = np.percentile(anomaly_map_resized, 99.0)
        print(f"[{base_name}] Anomaly Score - 최고: {max_val:.2f}, 최저: {min_val:.2f}, 상위1% 기준: {th_value:.2f}")
        
        # 1. 원본 사이즈 히트맵 (컬러 매핑)
        heatmap_norm = cv2.normalize(anomaly_map_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        # 2. 박스 추출 (면적 필터 삭제 - 1픽셀짜리도 무조건 네모 쳐줌)
        mask = (anomaly_map_resized > th_value).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result_img = img.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 2) # 두께 2의 빨간 박스
                
        # 3. [원본] | [순수 히트맵] | [박스 결과] 가로 병합
        combined_img = cv2.hconcat([img, heatmap_color, result_img])
        
        save_path = os.path.join(SAVE_DIR, f"debug_{base_name}")
        cv2.imwrite(save_path, combined_img)
        
    print(f"\n딥다이브 시각화 완료! 결과가 {SAVE_DIR} 에 저장되었습니다.")

if __name__ == '__main__':
    main()