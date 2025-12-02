import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
import math
import json
import os
import time
import requests
import threading
import torch

# GUI 딜레이 해결
# pip install tensorrt

class ScreenDetector:
    def __init__(self, model_path='best.pt', map_path='best.map', server_url='http://127.0.0.1:5000'):
        # ---------------------------------------------------------

        # 1. 하드웨어 및 모델 최적화 (TensorRT 자동 변환)

        # ---------------------------------------------------------
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️ 하드웨어 가속 상태: {self.device.upper()}")


        if self.device == 'cpu':
            print("⚠️ 경고: GPU가 없어 TensorRT 가속을 사용할 수 없습니다.")
            final_model_path = model_path
            self.use_half = False
        else:
            # GPU가 있을 경우 TensorRT(.engine) 변환 시도
            base_path = os.path.dirname(os.path.abspath(__file__))
            pt_path = os.path.join(base_path, model_path)
            if not os.path.exists(pt_path): pt_path = model_path # 경로 재확인

            # 엔진 파일 경로 (.pt -> .engine)
            engine_path = pt_path.replace('.pt', '.engine')

            if os.path.exists(engine_path):
                print(f"🚀 최적화된 엔진 파일 발견! 로드 중: {os.path.basename(engine_path)}")
                final_model_path = engine_path
            else:
                print("⚡ 성능 최적화를 위해 TensorRT(.engine) 변환을 시작합니다.")
                print("⏳ 처음 한 번은 3~5분 정도 걸립니다. 멈춘 게 아니니 기다려주세요...")
                try:
                    # 변환 실행
                    temp_model = YOLO(pt_path)
                    # half=True로 변환하여 속도 극대화
                    temp_model.export(format='engine', device=0, half=True, verbose=False)
                    print("✅ 변환 완료! 다음 실행부터는 즉시 실행됩니다.")
                    final_model_path = engine_path
                except Exception as e:
                    print(f"⚠️ 변환 실패 (기본 .pt 사용): {e}")
                    final_model_path = pt_path

            self.use_half = True # TensorRT는 기본적으로 half 추론

        # 모델 로드
        self.model = YOLO(final_model_path, task='detect')
        self.class_names = self.model.names

        # ---------------------------------------------------------
        # 2. 기본 설정 (서버, 해상도, 맵)
        # ---------------------------------------------------------
        self.server_url = server_url
        self.player_pos = [0.0, 0.0, 0.0]
        self.last_fire_count = -1

        self.screen_width = 1920
        self.screen_height = 1080
        self.monitor = {"top": 0, "left": 0, "width": self.screen_width, "height": self.screen_height}
        self.sct = mss()

        # 상수 설정
        self.FOCAL_LENGTH_PX = 1000
        self.KNOWN_WIDTH_M = {0: 1.6, 1: 14.4, 2: 1.6, 3: 15.2, 4: 13.7}
        self.MAX_DRAW_DISTANCE_M = 200.0

        self.TARGET_FILE_PATH = map_path
        self.map_data_cache = self.load_target_coordinates()

        # 리로딩 설정
        self.is_reloading = False
        self.reload_start_time = 0
        self.RELOAD_DURATION = 6.5

        # 스레드 시작
        self.running = True
        self.thread = threading.Thread(target=self.server_polling_thread)
        self.thread.daemon = True
        self.thread.start()

    def load_target_coordinates(self):
        if not os.path.exists(self.TARGET_FILE_PATH): return []
        try:
            with open(self.TARGET_FILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                targets = []
                for obj in data.get('obstacles', []):
                    pos = obj.get('position', {})
                    targets.append({
                        'prefabName': str(obj.get('prefabName', '')).lower(),
                        'x': float(pos.get('x', 0)), 'y': float(pos.get('y', 0)), 'z': float(pos.get('z', 0))
                    })
                return targets
        except: return []

    def calculate_sim_distance(self, cls_id, x1, x2):
        w = x2 - x1
        return (self.KNOWN_WIDTH_M.get(cls_id, 1.5) * self.FOCAL_LENGTH_PX) / w if w > 0 else 999.9

    def calculate_real_distance(self, p_pos, t_obj):
        return math.sqrt((p_pos[0]-t_obj['x'])**2 + (p_pos[1]-t_obj['y'])**2 + (p_pos[2]-t_obj['z'])**2)

    def get_fixed_color(self, cls_name):
        n = cls_name.lower()
        if n in ['red', 'tank']: return (0, 0, 255)
        elif n == 'blue': return (255, 0, 0)
        elif n in ['car', 'rock']: return (128, 128, 128)
        return (255, 255, 255)

    def analyze_battlefield(self, counts):
        red = sum(v for k,v in counts.items() if 'red' in k.lower())
        tank = sum(v for k,v in counts.items() if 'tank' in k.lower())
        blue = sum(v for k,v in counts.items() if 'blue' in k.lower())
        if tank > 0: return f"WARNING: {tank} TANKS!"
        elif red > 0: return f"Enemy: {red}"
        elif blue > 0: return f"Friendly: {blue}"
        return "Scouting..." if counts else "Searching..."

    def server_polling_thread(self):
        while self.running:
            try:
                resp = requests.get(f"{self.server_url}/info", timeout=0.5)
                if resp.status_code == 200:
                    data = resp.json()
                    pos = data.get("pos", {})
                    self.player_pos = [float(pos.get("x", 0)), 0.0, float(pos.get("z", 0))]
                    shots = int(data.get("fire_count", 0))
                    if self.last_fire_count != -1 and shots > self.last_fire_count:
                        self.is_reloading = True
                        self.reload_start_time = time.time()
                    self.last_fire_count = shots
                time.sleep(0.05)
            except: time.sleep(0.1)

    def run_detection(self):
        print("🚀 [최종] TensorRT 실시간 모드 시작 (프레임 제한 해제)")

        CAR_FILTERS = ['car001', 'car002', 'car003', 'car004', 'car005']
        ROCK_FILTERS = ['rock001', 'rock002']
        TANK_FILTERS = ['tank001']

        while True:
            try:
                # 1. 캡처
                img_mss = self.sct.grab(self.monitor)
                img = np.array(img_mss)
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                # 2. YOLO 탐지 (TensorRT 엔진은 자동으로 half 등 최적화 적용됨)
                # imgsz=640은 모델이 학습된 사이즈에 맞추는 게 가장 빠름

                results = self.model(frame, verbose=False, conf=0.6, iou=0.45, imgsz=640)


                current_boxes = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.class_names.get(cls_id, 'unknown')
                    sim_dist = self.calculate_sim_distance(cls_id, x1, x2)

                    if sim_dist > self.MAX_DRAW_DISTANCE_M: continue
                    current_boxes.append({'bbox': (x1, y1, x2, y2), 'cls_name': cls_name, 'sim_dist': sim_dist, 'matched_map_obj': None})

                # 3. 맵 매칭
                unique_classes = set(b['cls_name'] for b in current_boxes)
                counts = {}

                for cls_name in unique_classes:
                    cls_boxes = [b for b in current_boxes if b['cls_name'] == cls_name]
                    counts[cls_name] = len(cls_boxes)
                    cls_lower = cls_name.lower()
                    relevant = []
                    for m in self.map_data_cache:
                        p = m['prefabName']
                        match = False
                        if cls_lower == 'red' and 'human003' in p: match = True
                        elif cls_lower == 'blue' and 'human002' in p: match = True
                        elif cls_lower == 'car' and any(f in p for f in CAR_FILTERS): match = True
                        elif cls_lower == 'tank' and any(f in p for f in TANK_FILTERS): match = True
                        elif cls_lower == 'rock' and any(f in p for f in ROCK_FILTERS): match = True
                        if match:
                            d = self.calculate_real_distance(self.player_pos, m)
                            if d <= 500.0:
                                m_copy = m.copy()
                                m_copy['real_dist'] = d
                                relevant.append(m_copy)

                    candidates = []
                    for bi, box in enumerate(cls_boxes):
                        for mi, m_obj in enumerate(relevant):
                            diff = abs(box['sim_dist'] - m_obj['real_dist'])
                            candidates.append({'diff': diff, 'bi': bi, 'mi': mi})
                    candidates.sort(key=lambda x: x['diff'])
                    used_b, used_m = set(), set()
                    for c in candidates:
                        if c['bi'] not in used_b and c['mi'] not in used_m:
                            cls_boxes[c['bi']]['matched_map_obj'] = relevant[c['mi']]
                            used_b.add(c['bi'])
                            used_m.add(c['mi'])
                    for bi, box in enumerate(cls_boxes):
                        if box['matched_map_obj'] is None:
                            best, min_d = None, float('inf')
                            for m in relevant:
                                diff = abs(box['sim_dist'] - m['real_dist'])
                                if diff < min_d: min_d, best = diff, m
                            if best: box['matched_map_obj'] = best

                # 4. 그리기 (OpenCV)
                for box in current_boxes:
                    x1, y1, x2, y2 = box['bbox']
                    c_name = box['cls_name']
                    color = self.get_fixed_color(c_name)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    m_obj = box['matched_map_obj']
                    label_top = c_name.capitalize()
                    label_bottom = f"{m_obj['real_dist']:.1f}m" if m_obj else f"({box['sim_dist']:.0f}m)"

                    cv2.rectangle(frame, (x1, y1-20), (x1+100, y1), color, -1)
                    cv2.putText(frame, label_top, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(frame, label_bottom, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # 5. HUD
                overlay = frame.copy()
                cv2.rectangle(overlay, (5, 50), (350, 200), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

                sit_txt = self.analyze_battlefield(counts)

                cv2.putText(frame, "BATTLEFIELD STATUS", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(frame, f"My Pos: {self.player_pos[0]:.1f}, {self.player_pos[2]:.1f}", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.putText(frame, f"Status: {sit_txt}", (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,255), 1)

                summary = " | ".join([f"{k}:{v}" for k,v in counts.items()])
                cv2.putText(frame, f"Total: {summary}", (15, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

                # 6. 리로딩 (텍스트 점멸)
                if self.is_reloading:
                    elapsed = time.time() - self.reload_start_time
                    if elapsed > self.RELOAD_DURATION:
                        self.is_reloading = False
                    else:
                        if int(elapsed * 5) % 2 == 0:
                            cv2.putText(frame, "!!! RELOADING !!!", (self.screen_width//2 - 200, self.screen_height//2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

                cv2.imshow("Smart Map ID Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
            except Exception as e:
                print(f"Error: {e}")
                break
        self.close()

    def close(self):
        cv2.destroyAllWindows()

if __name__ == "__main__":

    MODEL_FILE_PATH = 'detector_gui/weights/5cls_v7_case6_best.pt'
    MAP_FILE_PATH = 'flask_server/map/11_28_notree.map'

    d = ScreenDetector(model_path=MODEL_FILE_PATH, map_path=MAP_FILE_PATH)
    d.run_detection()
