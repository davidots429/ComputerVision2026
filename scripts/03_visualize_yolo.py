import os
import glob
import cv2
from ultralytics import YOLO

def main():
    print("YOLO11 예측 결과 시각화 이미지 생성 중...")
    
    # 우리가 학습한 최고 성능 가중치 로드
    best_model_path = './runs/detect/models/yolo_results/pcb_defect/weights/best.pt'
    if not os.path.exists(best_model_path):
        print("학습된 YOLO 가중치를 찾을 수 없습니다. 경로를 확인해주세요.")
        return
        
    model = YOLO(best_model_path)
    
    VAL_IMG_DIR = "./data/yolo_format/images/val"
    SAVE_DIR = "./results/yolo_visuals"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    test_images = glob.glob(os.path.join(VAL_IMG_DIR, "*.jpg"))[:5] # 5장만 추출
    
    for img_path in test_images:
        base_name = os.path.basename(img_path)
        
        # YOLO 추론 실행
        results = model.predict(source=img_path, conf=0.25, verbose=False)
        
        # YOLO가 자체적으로 렌더링한 이미지(numpy array) 추출
        plotted_img = results[0].plot() 
        
        # 저장
        save_path = os.path.join(SAVE_DIR, f"yolo_{base_name}")
        cv2.imwrite(save_path, plotted_img)
        
    print(f"YOLO11 시각화 완료! 결과가 {SAVE_DIR} 에 저장되었습니다.")

if __name__ == "__main__":
    main()