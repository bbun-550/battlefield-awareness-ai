import cv2
import numpy as np
import time
import threading
import requests
import json
import os
import math
from mss import mss

from detect_module import ObjectDetector, Visualizer

class MainDetector:
    """
    메인 클래스
    - vision에게 탐지를 시킴
    - 서버에서 데이터를 받아옴
    - 맵 데이터와 탐지 결과를 매칭함
    - gui에게 그리라고 시킴
    """
    def __init__(self, model_path, map_path):
        # 1. 기본 변수 설정
        self.server_url = 'http://127.0.0.1:5000'
        self.player_pos = [0.0, 0.0, 0.0]
        self.last_fire_count = -1 # 서버의 발사 횟수 기록 (변경 감지용)
        self.running = True

        # 2. 모듈 초기화
        # ObjectDetector: YOLO 모델 로드 및 추론 담당
        self.detector = ObjectDetector(model_path)
        # Visualizer: 화면 해상도 1920x1080 설정
        self.visualizer = Visualizer(1920, 1080)
        
        # 3. 화면 캡처 라이브러리(mss) 설정
        self.sct = mss()
        self.monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}

        # 4. 맵 데이터 로드 (실제 거리 계산용)
        self.map_data = self._load_map_data(map_path)
        
        # 5. 서버 통신 스레드 시작
        # (GUI가 멈추지 않게 하기 위해 통신은 백그라운드 스레드에서 수행)
        self.thread = threading.Thread(target=self._server_polling)
        self.thread.daemon = True # 메인 프로그램 종료 시 같이 종료됨
        self.thread.start()

    def _load_map_data(self, map_path):
        """ 
        JSON 맵 파일에서 장애물(Tank, Rock 등)의 실제 좌표를 읽어옴.
        경로 문제 방지를 위해 절대 경로/상대 경로를 모두 체크함.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # 1차 시도: 현재 폴더 기준
        full_path = os.path.join(base_dir, map_path)
        
        # 1차 실패 시: 상위 폴더 기준 (flask_server 폴더 구조 대응)
        if not os.path.exists(full_path):
            full_path = os.path.join(os.path.dirname(base_dir), map_path)

        targets = []
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for obj in data.get('obstacles', []):
                    pos = obj.get('position', {})
                    # 필요한 정보만 추출하여 저장
                    targets.append({
                        'prefabName': str(obj.get('prefabName', '')).lower(),
                        'x': float(pos.get('x', 0)), 
                        'y': float(pos.get('y', 0)), 
                        'z': float(pos.get('z', 0))
                    })
            print(f"🗺️ [Main] 맵 데이터 로드 완료: {len(targets)}개 객체")
        except Exception as e:
            print(f"⚠️ [Main] 맵 로드 실패 (파일 경로 확인 필요): {e}")
        return targets

    def _server_polling(self):
        """ 
        [백그라운드 스레드] 0.05초마다 서버(/info)에 접속하여 정보 갱신
        """
        while self.running:
            try:
                resp = requests.get(f"{self.server_url}/info", timeout=0.2)
                if resp.status_code == 200:
                    data = resp.json()
                    # 1. 내 탱크 위치 업데이트
                    pos = data.get("pos", {})
                    self.player_pos = [float(pos.get("x", 0)), 0.0, float(pos.get("z", 0))]
                    
                    # 2. 발사 카운트 확인 -> 리로딩 트리거 작동
                    shots = int(data.get("fire_count", 0))
                    # 이전보다 카운트가 늘어났으면 "발사했다"는 의미
                    if self.last_fire_count != -1 and shots > self.last_fire_count:
                        self.visualizer.trigger_reload() # GUI에게 알림
                    self.last_fire_count = shots
                time.sleep(0.05)
            except: 
                # 서버가 꺼져있거나 통신 에러 시 무시하고 재시도
                time.sleep(0.1)

    def _match_map_objects(self, detections):
        """ 
        [핵심 로직] YOLO가 찾은 객체(화면상 거리)와 맵 데이터(실제 거리)를 매칭
        - 가장 가까운 거리에 있는 실제 객체를 찾아냄
        """
        # 매칭할 키워드 정의
        FILTERS = {
            'car': ['car'], 'rock': ['rock'], 'tank': ['tank'],
            'red': ['human003'], 'blue': ['human002']
        }

        unique_classes = set(b['cls_name'] for b in detections)
        counts = {}

        for cls_name in unique_classes:
            cls_boxes = [b for b in detections if b['cls_name'] == cls_name]
            counts[cls_name] = len(cls_boxes)
            cls_lower = cls_name.lower()
            
            # 1. 후보군 필터링: 내 주변 500m 이내이고, 이름이 일치하는 맵 객체만 추림
            relevant = []
            for m in self.map_data:
                pname = m['prefabName']
                is_match = False
                
                for key, keywords in FILTERS.items():
                    if cls_lower == key and any(k in pname for k in keywords):
                        is_match = True
                        break
                
                if is_match:
                    dist = math.sqrt((self.player_pos[0]-m['x'])**2 + (self.player_pos[2]-m['z'])**2)
                    if dist <= 500.0:
                        m_copy = m.copy()
                        m_copy['real_dist'] = dist
                        relevant.append(m_copy)

            # 2. 거리 오차 기반 매칭: (화면상 추정 거리 - 실제 거리) 차이가 가장 적은 것끼리 짝지음
            candidates = []
            for bi, box in enumerate(cls_boxes):
                for mi, m_obj in enumerate(relevant):
                    diff = abs(box['sim_dist'] - m_obj['real_dist'])
                    candidates.append({'diff': diff, 'bi': bi, 'mi': mi})
            
            candidates.sort(key=lambda x: x['diff']) # 오차가 적은 순서대로 정렬
            used_b, used_m = set(), set() # 이미 짝지어진 것 체크용
            
            for c in candidates:
                if c['bi'] not in used_b and c['mi'] not in used_m:
                    cls_boxes[c['bi']]['matched_map_obj'] = relevant[c['mi']]
                    used_b.add(c['bi'])
                    used_m.add(c['mi'])
            
            # 3. 짝을 못 찾은 박스는 남은 것 중 가장 가까운 걸로 강제 할당 (근사값)
            for box in cls_boxes:
                if box['matched_map_obj'] is None and relevant:
                    box['matched_map_obj'] = min(relevant, key=lambda m: abs(box['sim_dist'] - m['real_dist']))

        return counts

    def run(self):
        print("🚀 [Main] GUI 탐지기 시작 (TensorRT + 모듈화 적용)")
        
        while True:
            try:
                # 1. 화면 캡처 (가장 빠른 mss 라이브러리 사용)
                img_mss = self.sct.grab(self.monitor)
                frame = cv2.cvtColor(np.array(img_mss), cv2.COLOR_BGRA2BGR)

                # 2. Vision 모듈에게 탐지 요청
                detections = self.detector.detect(frame)

                # 3. 맵 데이터와 매칭 (누가 누구인지 식별)
                counts = self._match_map_objects(detections)

                # 4. GUI 모듈에게 그리기 요청 (박스, HUD, 리로딩 텍스트 등)
                frame = self.visualizer.draw(frame, detections, self.player_pos, counts)

                # 5. 화면 출력
                cv2.imshow("Smart Map ID Tracker", frame)
                
                # 'q' 키를 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break

            except Exception as e:
                print(f"❌ Main Loop Error: {e}")
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 모델 경로와 맵 파일 경로 지정
    MODEL_PATH = 'detector_gui/weights/5cls_v7.pt'
    MAP_PATH = 'flask_server/map/scenario_v5.map'
    
    app = MainDetector(model_path=MODEL_PATH, map_path=MAP_PATH)
    app.run()