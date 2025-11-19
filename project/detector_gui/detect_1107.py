import sys
import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication 
import math 
import json 
import os
import time
import threading
import logging
from flask import Flask, request

# Flask 로그 끄기 (콘솔이 너무 지저분해지지 않도록)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
detector_instance = None  # Flask 라우트에서 접근할 전역 변수

@app.route('/info', methods=['POST'])
def info():
    """게임에서 좌표 정보를 받아 Detector의 player_pos를 갱신"""
    global detector_instance
    if detector_instance is None:
        return "Detector not ready", 503

    try:
        data = request.get_json(force=True)
        player_pos_data = data.get('playerPos', {})
        
        # 좌표 추출 (None일 경우 0.0 처리)
        x = float(player_pos_data.get('x', 0.0))
        y = float(player_pos_data.get('y', 0.0)) # 게임 높이
        z = float(player_pos_data.get('z', 0.0)) 

        # [중요] 좌표계 변환
        # 이전 코드(load_target_coordinates)의 로직에 맞춤:
        # 게임 X -> 맵 X (index 0)
        # 게임 Z -> 맵 Y (index 1, 지도상 세로 위치)
        # 게임 Y -> 맵 Z (index 2, 고도)
        detector_instance.player_pos = [x, z, y]
        
        # 디버깅용 출력 (필요시 주석 해제)
        # print(f"Update Player Pos: {detector_instance.player_pos}")
        
        return "OK", 200
    except Exception as e:
        print(f"Data Error: {e}")
        return "Error", 400

class ScreenDetector:
    def __init__(self, model_path='5cls_v5_2_case2_best.pt'):
        # 1. 화면 해상도 감지
        try:
            self.app = QApplication.instance() or QApplication(sys.argv)
            geometry = self.app.primaryScreen().geometry()
            self.screen_width = geometry.width()
            self.screen_height = geometry.height()
        except Exception:
            print("경고: PyQt5 감지 실패. 기본 해상도 1920x1080 사용.")
            self.screen_width = 1920
            self.screen_height = 1080

        print(f"감지된 화면 해상도: {self.screen_width}x{self.screen_height}")

        # 2. mss 설정
        self.monitor = {"top": 0, "left": 0, "width": self.screen_width, "height": self.screen_height}
        self.sct = mss()

        # 3. YOLO 모델 로드
        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
            full_model_path = os.path.join(base_path, model_path)
            if not os.path.exists(full_model_path):
                full_model_path = model_path
                
            self.model = YOLO(full_model_path)
            self.class_names = self.model.names
            print(f"YOLO 모델 로드 완료: {full_model_path}")
        except Exception as e:
            print(f"오류: YOLO 모델 로드 실패. 경로를 확인하세요.")
            sys.exit(1)
            
        self.line_width = 3
        self.font_size = 1.0
        self.FOCAL_LENGTH_PX = 1000 
        
        self.KNOWN_WIDTH_M = {
            0: 1.0, 1: 1.8, 2: 1.0, 3: 0.8, 4: 1.92
        }
        
        self.CLASS_COLOR_MAP = {}
        for id, name in self.class_names.items():
            name_lower = name.lower()
            if name_lower in ['red', 'tank']:
                self.CLASS_COLOR_MAP[id] = (0, 0, 255)
            elif name_lower == 'blue':
                self.CLASS_COLOR_MAP[id] = (255, 0, 0)
            elif name_lower in ['car', 'rock']:
                self.CLASS_COLOR_MAP[id] = (168, 168, 168)
            else:
                self.CLASS_COLOR_MAP[id] = (255, 255, 255) 

        self.MAX_DRAW_DISTANCE_M = 120.0 
        self.MAP_SIZE_INFO = "300 x 300"
        
        # 맵 파일 경로 설정
        base_path = os.path.dirname(os.path.abspath(__file__))
        # [수정 필요] 본인의 환경에 맞는 맵 파일 경로로 변경하세요
        self.TARGET_FILE_PATH = r'C:\test\tankchallenge\test.map'
        
        self.player_pos = [0.0, 0.0, 0.0] # [x, z, y] 순서 (Flask에서 업데이트됨)
        self.map_data_cache = self.load_target_coordinates() # 맵 데이터 미리 로드

    def load_target_coordinates(self):
        """ 맵 파일 로드 및 좌표 변환 """
        if not os.path.exists(self.TARGET_FILE_PATH):
            print(f"[오류] 맵 파일을 찾을 수 없습니다: {self.TARGET_FILE_PATH}")
            return []
            
        try:
            with open(self.TARGET_FILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                obstacles = data.get('obstacles', [])
                
                standardized_targets = []
                for obj in obstacles:
                    if 'prefabName' in obj and 'position' in obj:
                        pos = obj['position']
                        # Unity 좌표(x,y,z) -> 2D 지도 좌표계로 변환
                        standardized_targets.append({
                            'prefabName': obj['prefabName'],
                            'x': pos.get('x', 0.0),
                            'y': pos.get('z', 0.0), # Game Z -> Map Y
                            'z': pos.get('y', 0.0)  # Game Y -> Map Z (Height)
                        })
                return standardized_targets
        except Exception as e:
            print(f"[오류] 맵 로딩 실패: {e}")
            return []

    def calculate_distance(self, cls_id, x1, x2):
        pixel_width = x2 - x1
        known_width_m = self.KNOWN_WIDTH_M.get(cls_id, 1.0)
        if pixel_width > 0:
            distance_m = (known_width_m * self.FOCAL_LENGTH_PX) / pixel_width
        else:
            distance_m = 999.9 
        return min(distance_m, 200.0) 

    def calculate_real_distance(self, player_pos, target_pos_xyz):
        px, py, pz = player_pos # [MapX, MapY(GameZ), MapZ(GameY)]
        tx, ty, tz = target_pos_xyz
        return math.sqrt((tx - px)**2 + (ty - py)**2 + (tz - pz)**2)

    def run_detection(self):
        print("객체 탐지 및 서버 시작. 'q' 종료.")
        summary_start_x, summary_start_y = 20, 40
        line_height = 35
        
        while True:
            # player_pos는 Flask 스레드에 의해 실시간으로 변경됨
            
            # 실제 거리 매칭 맵 생성
            closest_real_dist = {}
            
            for target in self.map_data_cache:
                t_pos = (target['x'], target['y'], target['z'])
                dist = self.calculate_real_distance(tuple(self.player_pos), t_pos)
                
                t_name = target['prefabName'].lower()
                matched_cls = None
                
                for _, cls_name in self.class_names.items():
                    if cls_name.lower() in t_name:
                        matched_cls = cls_name.lower()
                        break
                
                if not matched_cls and 'tank' in t_name: matched_cls = 'tank'
                if not matched_cls and 'car' in t_name: matched_cls = 'car'
                
                if matched_cls:
                    current = closest_real_dist.get(matched_cls, {'dist': float('inf')})
                    if dist < current['dist']:
                        closest_real_dist[matched_cls] = {'name': target['prefabName'], 'dist': dist}

            try:
                img_mss = self.sct.grab(self.monitor)
                img = np.array(img_mss)
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                results = self.model(frame, verbose=False, conf=0.5, iou=0.38)

                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.class_names.get(cls_id, 'unknown').lower()
                    
                    sim_dist = self.calculate_distance(cls_id, x1, x2)
                    if sim_dist > self.MAX_DRAW_DISTANCE_M: continue

                    color = self.CLASS_COLOR_MAP.get(cls_id, (255, 255, 255))
                    
                    real_info = closest_real_dist.get(cls_name)
                    if real_info and real_info['dist'] <= self.MAX_DRAW_DISTANCE_M * 1.5:
                        label = f"{cls_name.upper()} R:{real_info['dist']:.1f}m / S:{sim_dist:.1f}m"
                    else:
                        label = f"{cls_name.upper()} S:{sim_dist:.1f}m"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)

                # HUD 그리기
                overlay_lines = [
                    f"Player Pos: {self.player_pos[0]:.1f}, {self.player_pos[1]:.1f}, {self.player_pos[2]:.1f}",
                    "- Real Distances (Map) -"
                ]
                for c_name, info in closest_real_dist.items():
                    overlay_lines.append(f"{c_name.upper()}: {info['dist']:.1f}m")
                
                h = summary_start_y + len(overlay_lines) * line_height
                cv2.rectangle(frame, (10, 10), (350, h + 10), (0,0,0), -1)
                
                y = summary_start_y
                for line in overlay_lines:
                    cv2.putText(frame, line, (summary_start_x, y), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1)
                    y += line_height

                cv2.imshow("YOLO Distance Sim", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

            except Exception as e:
                print(f"Detection Loop Error: {e}")
                break
                
    def close(self):
        cv2.destroyAllWindows()

def run_flask():
    """Flask 서버를 별도 스레드에서 실행"""
    # host='0.0.0.0'으로 해야 외부/로컬에서 접속 가능
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

if __name__ == "__main__":
    MODEL_FILE_PATH = '5cls_v5_2_case2_best.pt' 
    
    # 1. Detector 인스턴스 생성
    detector_instance = ScreenDetector(model_path=MODEL_FILE_PATH)
    
    # 2. Flask 서버를 데몬 스레드로 시작 (메인 프로그램 종료 시 같이 종료됨)
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    print("Flask Server started on port 5000...")

    # 3. 메인 스레드에서 YOLO 탐지 루프 실행
    try:
        detector_instance.run_detection()
    except Exception as e:
        print(f"실행 오류: {e}")
    finally:
        if detector_instance: detector_instance.close()
        
"""
### 변경 및 적용된 핵심 사항

1.  **스레딩(Threading) 적용**:
    * `app.run()`은 코드를 멈추게(Block) 하므로, 별도의 스레드(`flask_thread`)에서 실행시켰습니다.
    * 메인 스레드는 계속해서 `detector_instance.run_detection()`을 돌며 화면을 인식합니다.

2.  **데이터 연동 (`detector_instance`)**:
    * `detector_instance`라는 전역 변수를 사용하여, Flask의 `/info` 함수가 데이터를 받을 때마다 `detector_instance.player_pos` 값을 직접 수정합니다.

3.  **좌표계 변환 (매우 중요)**:
    * 이전 코드(`load_target_coordinates`)에서 Unity 좌표계와 맵 좌표계를 맞추기 위해 `z`와 `y`를 바꿨던 로직을 그대로 적용했습니다.
    * **받는 데이터**: `x`, `y`(높이), `z`(깊이)
    * **저장 데이터**: `[x, z, y]` (이렇게 해야 `calculate_real_distance` 함수가 맵 데이터와 올바르게 거리를 잴 수 있습니다.)

4.  **랜덤 이동 제거**:
    * 기존의 `update_player_position` 함수(랜덤 워크)는 삭제했습니다. 이제 실제 HTTP 요청으로만 위치가 바뀝니다.

### 실행 방법

1.  이 파이썬 스크립트를 실행합니다. (객체 탐지 창이 뜹니다.)
2.  게임이나 테스트 클라이언트에서 다음 주소로 데이터를 보냅니다.
    * **URL**: `http://127.0.0.1:5000/info`
    * **Method**: `POST`
    * **Body (JSON)**:
        ```json
        {
            "playerPos": {
                "x": 150.5,
                "y": 10.0,
                "z": 150.5
            }
        }

"""