import os
import cv2
import glob
import numpy as np

def main():
    print("OpenCV 시각화 이미지 생성 중...")
    
    VAL_IMG_DIR = "./data/yolo_format/images/val"
    TEMPLATE_DIR = "./data/anomalib_format/normal"
    SAVE_DIR = "./results/opencv_visuals"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    test_images = glob.glob(os.path.join(VAL_IMG_DIR, "*.jpg"))[:5] # 5장만 추출
    threshold_value = 50 # 이전 평가에서 사용한 기본 임계값
    
    for img_path in test_images:
        base_name = os.path.basename(img_path)
        img_id = base_name.replace("_test.jpg", "")
        
        test_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        temp_path = os.path.join(TEMPLATE_DIR, f"{img_id}_temp.jpg")
        temp_img = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
        
        if test_img is None or temp_img is None: continue
            
        # 1. 차분 및 이진화 (마스크 생성)
        diff = cv2.absdiff(temp_img, test_img)
        _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # 2. 바운딩 박스 그리기용 컬러 이미지 변환
        result_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 10:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 2) # 빨간 박스
                
        # 3. 4단계를 가로로 이어 붙이기 (비교 시각화)
        # 흑백 이미지들을 컬러로 변환하여 채널 수를 맞춰줌
        temp_color = cv2.cvtColor(temp_img, cv2.COLOR_GRAY2BGR)
        test_color = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
        mask_color = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
        
        # cv2.hconcat으로 가로로 병합: [템플릿] | [테스트] | [마스크] | [결과]
        combined_img = cv2.hconcat([temp_color, test_color, mask_color, result_img])
        
        save_path = os.path.join(SAVE_DIR, f"opencv_{base_name}")
        cv2.imwrite(save_path, combined_img)
        
    print(f"✅ OpenCV 시각화 완료! 결과가 {SAVE_DIR} 에 저장되었습니다.")

if __name__ == "__main__":
    main()