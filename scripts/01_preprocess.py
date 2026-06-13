import os
import glob
import shutil
import random

# --- Paths ---
RAW_DIR = './data/raw_data/PCBData'
YOLO_DIR = './data/yolo_format'
ANOMALIB_DIR = './data/anomalib_format'

IMG_SIZE = 640 # DeepPCB 기본 해상도

def create_dirs():
    """YOLO와 Anomalib 학습을 위한 폴더 구조 생성"""
    dirs = [
        f"{YOLO_DIR}/images/train", f"{YOLO_DIR}/images/val",
        f"{YOLO_DIR}/labels/train", f"{YOLO_DIR}/labels/val",
        f"{ANOMALIB_DIR}/normal", f"{ANOMALIB_DIR}/abnormal"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def convert_to_yolo(bbox):
    """DeepPCB 라벨(x1, y1, x2, y2, class)을 YOLO 포맷(class, x_center, y_center, w, h)으로 변환"""
    x1, y1, x2, y2, class_id = map(float, bbox)
    class_id = int(class_id) - 1 # DeepPCB는 1~6, YOLO는 0~5
    
    x_center = ((x1 + x2) / 2.0) / IMG_SIZE
    y_center = ((y1 + y2) / 2.0) / IMG_SIZE
    width = (x2 - x1) / IMG_SIZE
    height = (y2 - y1) / IMG_SIZE
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_data():
    create_dirs()
    
    print("데이터 파일들을 검색하고 있습니다. 잠시만 기다려주세요...")
    
    # 1. 모든 하위 폴더의 txt 파일 검색
    txt_files = glob.glob(os.path.join(RAW_DIR, '**', '*.txt'), recursive=True)
    
    all_pairs = []
    for txt_path in txt_files:
        filename = os.path.basename(txt_path)
        
        # trainval.txt 등의 메타데이터 파일은 제외하고 순수 라벨 파일만 처리
        if not filename[0].isdigit():
            continue
            
        base_name = filename.replace('.txt', '')
        
        # txt 파일이 있는 'groupXXXXX' 최상위 폴더 경로 추출
        parent_group_dir = os.path.dirname(os.path.dirname(txt_path))
        
        # 해당 그룹 폴더 내에서 템플릿(정상) 이미지와 테스트(검사) 이미지 검색
        test_img_list = glob.glob(os.path.join(parent_group_dir, '**', f"{base_name}_test.jpg"), recursive=True)
        temp_img_list = glob.glob(os.path.join(parent_group_dir, '**', f"{base_name}_temp.jpg"), recursive=True)
        
        # 파일이 모두 존재하는 완벽한 쌍(Pair)만 리스트에 추가
        if test_img_list and temp_img_list:
            all_pairs.append((temp_img_list[0], test_img_list[0], txt_path))

    if len(all_pairs) == 0:
        print("오류: 데이터를 하나도 찾지 못했습니다.")
        return

    # Train 80%, Validation 20% 무작위 분할
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * 0.8)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    print(f"발견된 데이터 쌍: 전체 {len(all_pairs)}개 | Train: {len(train_pairs)}개 | Val: {len(val_pairs)}개")
    print("파일 복사 및 라벨 변환 시작.")

    # 데이터 복사 및 라벨 변환 로직
    for phase, pairs in [('train', train_pairs), ('val', val_pairs)]:
        for temp_img, test_img, txt_path in pairs:
            base_name = os.path.basename(test_img)
            
            # 1. Anomalib용 데이터 세팅
            shutil.copy(temp_img, os.path.join(ANOMALIB_DIR, 'normal', base_name.replace('_test', '_temp')))
            shutil.copy(test_img, os.path.join(ANOMALIB_DIR, 'abnormal', base_name))
            
            # 2. YOLO용 데이터 세팅
            shutil.copy(test_img, os.path.join(YOLO_DIR, 'images', phase, base_name))
            
            # 3. YOLO용 라벨 변환 및 저장
            yolo_txt_path = os.path.join(YOLO_DIR, 'labels', phase, base_name.replace('.jpg', '.txt'))
            with open(txt_path, 'r') as f_in, open(yolo_txt_path, 'w') as f_out:
                for line in f_in:
                    bbox = line.strip().split(' ')
                    if len(bbox) == 5:
                        yolo_line = convert_to_yolo(bbox)
                        f_out.write(yolo_line + '\n')

    print("모든 전처리 완료")

if __name__ == '__main__':
    process_data()
