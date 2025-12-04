import os
import torch
from ultralytics import YOLO

class ObjectDetector:
    """
    YOLO 모델을 로드하고, 입력된 이미지에서 객체를 찾아내는 클래스
    하드웨어 환경(CPU/GPU)에 따라 최적의 모델 포맷(.onnx / .engine)을 자동으로 선택
    """
    def __init__(self, model_path='best.pt'):
        # ---------------------------------------------------------
        # 1. 하드웨어 가속 확인
        # ---------------------------------------------------------
        # torch.cuda.is_available()을 통해 GPU 사용 가능 여부를 확인합니다.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️ [Vision] 하드웨어 가속 상태: {self.device.upper()}")

        self.model_path = model_path
        
        # 환경에 맞는 모델(TensorRT 또는 ONNX)을 로드합니다.
        self.model = self._load_model()
        self.class_names = self.model.names

        # ---------------------------------------------------------
        # 2. 거리 계산을 위한 상수 설정
        # ---------------------------------------------------------
        # FOCAL_LENGTH_PX: 카메라의 초점 거리 (사전에 캘리브레이션 된 값 가정)
        self.FOCAL_LENGTH_PX = 1000
        
        # KNOWN_WIDTH_M: 각 클래스(객체)의 실제 물리적 가로 너비 (단위: 미터)
        # 이 값을 기준으로 화면상 픽셀 크기와 비교해 거리를 추정합니다.
        self.KNOWN_WIDTH_M = {
            0: 1.6,   # 아군 (Blue)
            1: 14.4,  # 자동차 (Car)
            2: 1.6,   # 적군 (Red)
            3: 15.2,  # 바위 (Rock)
            4: 13.7   # 탱크 (Tank)
        }
        
        # 200m 이상인 물체는 화면에 표시하지 않음 (너무 멀면 오차가 커짐)
        self.MAX_DRAW_DISTANCE_M = 200.0

    def _load_model(self):
        """
        [모델 최적화 로드 로직]
        - GPU 환경 (.engine): TensorRT 엔진 사용. 없으면 .pt에서 변환. (가장 빠름)
        - CPU 환경 (.onnx): ONNX 런타임 사용. 없으면 .pt에서 변환. (기본 .pt보다 2~3배 빠름)
        """
        # 현재 파일 위치 기준으로 절대 경로를 계산하여 경로 문제 방지
        base_path = os.path.dirname(os.path.abspath(__file__))
        pt_path = os.path.join(base_path, self.model_path)
        
        # 경로 보정: 만약 계산된 경로에 없으면 입력받은 상대 경로 그대로 사용
        if not os.path.exists(pt_path): pt_path = self.model_path
        
        # 각 포맷별 파일 경로 정의
        engine_path = pt_path.replace('.pt', '.engine')
        onnx_path = pt_path.replace('.pt', '.onnx')

        # ---------------------------------------------------------
        # CASE 1: GPU 환경 (TensorRT 사용)
        # ---------------------------------------------------------
        if self.device == 'cuda':
            # 이미 변환된 엔진 파일이 있으면 즉시 로드
            if os.path.exists(engine_path):
                print(f"🚀 [Vision] TensorRT 엔진 발견! 로드 중: {os.path.basename(engine_path)}")
                return YOLO(engine_path, task='detect')
            
            # 없으면 변환 시작
            print("⚡ [Vision] GPU 발견! TensorRT(.engine) 변환을 시작합니다 (3~5분 소요)...")
            try:
                temp_model = YOLO(pt_path)
                # half=True: 16-bit 부동소수점 사용 (속도 2배 향상, 정확도 유지)
                temp_model.export(format='engine', device=0, half=True, verbose=False)
                print("✅ [Vision] Engine 변환 완료! 다음부터는 즉시 실행됩니다.")
                return YOLO(engine_path, task='detect')
            except Exception as e:
                print(f"⚠️ [Vision] Engine 변환 실패 (기본 .pt 사용): {e}")
                return YOLO(pt_path, task='detect')

        # ---------------------------------------------------------
        # CASE 2: CPU 환경 (ONNX 사용)
        # ---------------------------------------------------------
        else:
            # 이미 변환된 ONNX 파일이 있으면 즉시 로드
            if os.path.exists(onnx_path):
                print(f"🚀 [Vision] ONNX 모델 발견! CPU 최적화 로드: {os.path.basename(onnx_path)}")
                # ONNX 로드 시 task='detect' 명시 권장
                return YOLO(onnx_path, task='detect')
            
            # 없으면 변환 시작
            print("⚡ [Vision] CPU 환경 감지. 속도 향상을 위해 ONNX 변환을 시작합니다...")
            try:
                temp_model = YOLO(pt_path)
                # CPU용 ONNX 변환 (GPU 없이도 가능)
                temp_model.export(format='onnx', verbose=False)
                print("✅ [Vision] ONNX 변환 완료!")
                return YOLO(onnx_path, task='detect')
            except Exception as e:
                print(f"⚠️ [Vision] ONNX 변환 실패 (기본 .pt 사용): {e}")
                return YOLO(pt_path, task='detect')

    def calculate_sim_distance(self, cls_id, x1, x2):
        """
        [거리 추정 공식]
        거리 = (실제 너비 * 초점 거리) / 화면상 픽셀 너비
        """
        pixel_width = x2 - x1
        real_width = self.KNOWN_WIDTH_M.get(cls_id, 1.5) # 등록되지 않은 객체는 1.5m로 가정
        
        if pixel_width > 0:
            return (real_width * self.FOCAL_LENGTH_PX) / pixel_width
        return 999.9

    def detect(self, frame):
        """
        [추론 실행]
        입력된 프레임에 대해 YOLO 모델을 실행하고 결과를 반환합니다.
        """
        # TensorRT/ONNX 모델은 입력 이미지 크기가 고정(640)되어야 성능이 최적화됨
        results = self.model(frame, verbose=False, conf=0.6, iou=0.45, imgsz=640)
        
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])   # 박스 좌표 추출
            cls_id = int(box.cls[0])                 # 클래스 ID 추출
            cls_name = self.class_names.get(cls_id, 'unknown')
            
            # 시각적 거리 계산
            sim_dist = self.calculate_sim_distance(cls_id, x1, x2)

            # 너무 먼 거리는 무시 (노이즈 제거)
            if sim_dist > self.MAX_DRAW_DISTANCE_M: continue
            
            # 탐지 정보를 딕셔너리로 구조화하여 반환
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'cls_name': cls_name,
                'sim_dist': sim_dist,
                'matched_map_obj': None # 추후 맵 데이터와 매칭될 공간 (detect.py에서 채움)
            })
        
        return detections