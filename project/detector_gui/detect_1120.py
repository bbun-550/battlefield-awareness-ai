import sys
import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
import math 
import json 
import os
import time
import requests
# import threading
# import logging
# from flask import Flask, request
from PIL import ImageFont, ImageDraw, Image

# Flask 로그 끄기
# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)

# app = Flask(__name__)
# detector_instance = None

# @app.route('/info', methods=['POST'])
# def info():
#     global detector_instance
#     if detector_instance is None:
#         return "Detector not ready", 503

#     try:
#         data = request.get_json(force=True)
#         player_pos_data = data.get('playerPos', {})
        
#         x = float(player_pos_data.get('x', 0.0))
#         y = float(player_pos_data.get('y', 0.0))
#         z = float(player_pos_data.get('z', 0.0)) 

#         detector_instance.player_pos = [x, y, z] 
        
#         return "OK", 200
#     except Exception as e:
#         print(f"Data Error: {e}")
#         return "Error", 400

class ScreenDetector:
    def __init__(self, model_path='5cls_v5_2_case2_best.pt', server_url='http://127.0.0.1:5000'):
        self.server_url = server_url
        self.player_pos = [0.0, 0.0, 0.0]
        
        # 화면 해상도 설정 (기본 FHD)
        self.screen_width = 1920
        self.screen_height = 1080
        
        # ======================
        # self.sct = mss()
        # mon = self.sct.monitors[1]   # 실제 물리 모니터

        # self.screen_width = mon["width"]
        # self.screen_height = mon["height"]

        # self.monitor = {
        #     "top": mon["top"],
        #     "left": mon["left"],
        #     "width": mon["width"],
        #     "height": mon["height"]
        # }        
        self.monitor = {"top": 0, "left": 0, "width": self.screen_width, "height": self.screen_height}
        self.sct = mss()
        # ========================

        base_path = os.path.dirname(os.path.abspath(__file__))
        full_model_path = os.path.join(base_path, model_path)
        if not os.path.exists(full_model_path):
            full_model_path = model_path
            
        self.model = YOLO(full_model_path)
        self.class_names = self.model.names
        
        self.FOCAL_LENGTH_PX = 1000 
        
        # [최종 보정값]
        self.KNOWN_WIDTH_M = {
            0: 1.6,   # Red (사람)
            1: 14.4,  # Car
            2: 1.6,   # Blue (사람)
            3: 15.2,  # Rock
            4: 13.7   # Tank
        }
        
        self.MAX_DRAW_DISTANCE_M = 200.0 
        
        self.TARGET_FILE_PATH = 'flask_server/map/11_20.map'
        self.player_pos = [0.0, 0.0, 0.0] 
        self.map_data_cache = self.load_target_coordinates()
        
        # [한글 폰트 설정]
        self.font_path = "C:/Windows/Fonts/malgun.ttf"
        self.bold_font_path = "C:/Windows/Fonts/malgunbd.ttf" 
        
        try:
            self.font = ImageFont.truetype(self.font_path, 20) 
            if os.path.exists(self.bold_font_path):
                self.font_bold = ImageFont.truetype(self.bold_font_path, 20)
            else:
                self.font_bold = ImageFont.truetype(self.font_path, 20)
        except:
            print("한글 폰트 로드 실패. 기본 폰트를 사용합니다.")
            self.font = None
            self.font_bold = None

    def load_target_coordinates(self):
        """ 맵 파일 로드 """
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
                        full_name = obj['prefabName']
                        
                        standardized_targets.append({
                            'id': full_name, 
                            'prefabName': full_name.lower(),
                            'x': float(pos.get('x', 0.0)),
                            'y': float(pos.get('y', 0.0)), 
                            'z': float(pos.get('z', 0.0))
                        })
                print(f"맵 데이터 로드 완료: {len(standardized_targets)}개 객체")
                return standardized_targets
        except Exception as e:
            print(f"[오류] 맵 로딩 실패: {e}")
            return []

    def calculate_sim_distance(self, cls_id, x1, x2):
        pixel_width = x2 - x1
        known_width = self.KNOWN_WIDTH_M.get(cls_id, 1.5)
        if pixel_width > 0:
            return (known_width * self.FOCAL_LENGTH_PX) / pixel_width
        return 999.9

    def calculate_real_distance(self, p_pos, t_obj):
        return math.sqrt(
            (p_pos[0] - t_obj['x'])**2 + 
            (p_pos[1] - t_obj['y'])**2 + 
            (p_pos[2] - t_obj['z'])**2
        )

    def get_fixed_color(self, cls_name):
        name = cls_name.lower()
        if name == 'red' or name == 'tank':
            return (0, 0, 255) # 빨강
        elif name == 'blue':
            return (255, 0, 0) # 파랑
        elif name == 'car' or name == 'rock':
            return (128, 128, 128) # 회색
        else:
            return (255, 255, 255)

    def run_detection(self):
        print("객체 탐지 시작... (HUD 총합 표시 및 자동 크기 조절)")
        
        CAR_FILTERS = ['car001', 'car002', 'car003', 'car004', 'car005']
        ROCK_FILTERS = ['rock001', 'rock002']
        TANK_FILTERS = ['tank001']

        while True:
            try:
                # 서버에서 플레이어 위치 갱신
                self.update_player_pos_from_server()

                img_mss = self.sct.grab(self.monitor)
                img = np.array(img_mss)
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                results = self.model(frame, verbose=False, conf=0.5, iou=0.45)
                
                current_frame_boxes = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.class_names.get(cls_id, 'unknown')
                    sim_dist = self.calculate_sim_distance(cls_id, x1, x2)
                    
                    if sim_dist > self.MAX_DRAW_DISTANCE_M: continue
                    
                    current_frame_boxes.append({
                        'bbox': (x1, y1, x2, y2),
                        'cls_id': cls_id,
                        'cls_name': cls_name,
                        'sim_dist': sim_dist,
                        'matched_map_obj': None 
                    })

                unique_classes = set(b['cls_name'] for b in current_frame_boxes)
                
                # 개수 카운팅 (탐지된 박스 기준)
                total_counts = {} 

                for cls_name in unique_classes:
                    cls_boxes = [b for b in current_frame_boxes if b['cls_name'] == cls_name]
                    total_counts[cls_name] = len(cls_boxes)
                    
                    cls_lower = cls_name.lower()
                    relevant_map_objs = []
                    for m_obj in self.map_data_cache:
                        map_prefab = m_obj['prefabName']
                        is_match = False

                        if cls_lower == 'red':
                            if 'human003' in map_prefab: is_match = True
                        elif cls_lower == 'blue':
                            if 'human002' in map_prefab: is_match = True
                        elif cls_lower == 'car':
                            if any(f in map_prefab for f in CAR_FILTERS): is_match = True
                        elif cls_lower == 'tank':
                            if any(f in map_prefab for f in TANK_FILTERS): is_match = True
                        elif cls_lower == 'rock':
                             if any(f in map_prefab for f in ROCK_FILTERS): is_match = True
                        
                        if is_match:
                            dist = self.calculate_real_distance(self.player_pos, m_obj)
                            if dist <= 500.0:
                                obj_with_dist = m_obj.copy()
                                obj_with_dist['real_dist'] = dist
                                relevant_map_objs.append(obj_with_dist)

                    # Best-Fit 1:1 매칭
                    match_candidates = []
                    for box_idx, box in enumerate(cls_boxes):
                        for map_idx, map_obj in enumerate(relevant_map_objs):
                            diff = abs(box['sim_dist'] - map_obj['real_dist'])
                            match_candidates.append({
                                'diff': diff,
                                'box_idx': box_idx,
                                'map_idx': map_idx
                            })
                    
                    match_candidates.sort(key=lambda x: x['diff'])
                    
                    used_boxes = set()
                    used_maps = set()
                    
                    for cand in match_candidates:
                        b_idx = cand['box_idx']
                        m_idx = cand['map_idx']
                        
                        if b_idx not in used_boxes and m_idx not in used_maps:
                            cls_boxes[b_idx]['matched_map_obj'] = relevant_map_objs[m_idx]
                            used_boxes.add(b_idx)
                            used_maps.add(m_idx)

                    # N:1 매칭
                    for b_idx, box in enumerate(cls_boxes):
                        if box['matched_map_obj'] is None:
                            best_approx = None
                            min_diff = float('inf')
                            for map_obj in relevant_map_objs:
                                diff = abs(box['sim_dist'] - map_obj['real_dist'])
                                if diff < min_diff:
                                    min_diff = diff
                                    best_approx = map_obj
                            if best_approx:
                                box['matched_map_obj'] = best_approx

                hud_text_list = []
                
                # 1. 박스 그리기 (OpenCV)
                for box in current_frame_boxes:
                    x1, y1, x2, y2 = box['bbox']
                    cls_name = box['cls_name']
                    color = self.get_fixed_color(cls_name)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 2. 텍스트 및 HUD 그리기 (PIL)
                if self.font_bold is not None:
                    # OpenCV BGR -> PIL RGB 변환
                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    # 투명도 처리를 위해 RGBA 모드로 변환
                    img_pil = img_pil.convert("RGBA")
                    
                    # 투명 레이어 생성 (여기다가 반투명 박스 그림)
                    overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
                    draw = ImageDraw.Draw(overlay)
                    
                    # 객체 라벨링 (이전과 동일)
                    current_frame_boxes.sort(key=lambda x: x['sim_dist'])
                    counters = {} 

                    for box in current_frame_boxes:
                        x1, y1, x2, y2 = box['bbox']
                        cls_name = box['cls_name']
                        
                        if cls_name not in counters: counters[cls_name] = 1
                        simple_name = f"{cls_name.capitalize()} {counters[cls_name]}"
                        counters[cls_name] += 1

                        bgr = self.get_fixed_color(cls_name)
                        name_color_rgb = (bgr[2], bgr[1], bgr[0])
                        
                        map_obj = box['matched_map_obj']
                        
                        if map_obj:
                            real_d = map_obj['real_dist']
                            label_name = simple_name
                            label_dist = f"R : {real_d:.1f}m"
                            
                            id_str = f"{simple_name} : {real_d:.1f}m"
                            if id_str not in hud_text_list:
                                hud_text_list.append(id_str)

                            # 이름
                            draw.text((x1, y1-25), label_name, font=self.font_bold, fill=name_color_rgb)
                            
                            # 거리 배경 및 텍스트
                            text_bbox = draw.textbbox((x1, y2 + 5), label_dist, font=self.font_bold)
                            # draw.rectangle(text_bbox, fill=(200, 200, 200)) 
                            draw.text((x1, y2 + 5), label_dist, font=self.font_bold, fill=(0, 0, 0))
                        else:
                            label = f"{simple_name} (추정:{box['sim_dist']:.1f}m)"
                            draw.text((x1, y1-25), label, font=self.font_bold, fill=(180, 180, 180))
                    
                    # --- HUD 그리기 (투명도 70% 적용 & 위치 이동) ---
                    
                    hud_text_list.sort()
                    display_list = hud_text_list[:20] 
                    
                    summary_text = " | ".join([f"{k.capitalize()}: {v}개" for k, v in total_counts.items()])
                    
                    # 위치 조정 (y=10 -> y=50)
                    base_x = 10
                    base_y = 50  
                    
                    line_height = 20
                    header_gap = 40   
                    list_height = len(display_list) * line_height
                    summary_height = 40
                    
                    total_box_height = header_gap + list_height + summary_height
                    
                    max_text_width = 300 
                    for txt in display_list:
                        w = draw.textlength(txt, font=self.font_bold)
                        if w > max_text_width: max_text_width = int(w)
                    w_sum = draw.textlength(summary_text, font=self.font_bold)
                    if w_sum > max_text_width: max_text_width = int(w_sum)
                    
                    box_width = max_text_width + 40 
                    
                    # [수정] 반투명 검정 박스 (Alpha 180 ≈ 70%)
                    # (R, G, B, Alpha) -> (0, 0, 0, 180)
                    draw.rectangle(
                        [(base_x, base_y), (base_x + box_width, base_y + total_box_height)], 
                        fill=(0, 0, 0, 180) 
                    )
                    
                    # 텍스트 그리기 (텍스트는 불투명)
                    text_start_x = base_x + 10
                    text_start_y = base_y + 5
                    
                    # 1. 내 위치
                    draw.text((text_start_x, text_start_y), f"내 위치: {self.player_pos[0]:.1f}, {self.player_pos[2]:.1f}", font=self.font_bold, fill=(0, 255, 0))
                    
                    # 2. 리스트
                    for i, txt in enumerate(display_list):
                        draw.text((text_start_x, text_start_y + header_gap + i*line_height), txt, font=self.font_bold, fill=(200, 200, 200))
                    
                    # 3. 총합
                    sum_y = text_start_y + header_gap + list_height + 10
                    draw.text((text_start_x, sum_y), summary_text, font=self.font_bold, fill=(255, 255, 0))
                    
                    # [합성] 원본 이미지와 반투명 오버레이 합성
                    out = Image.alpha_composite(img_pil, overlay)
                    
                    # 다시 OpenCV 포맷으로 변환 (RGBA -> RGB -> BGR)
                    frame = cv2.cvtColor(np.array(out.convert('RGB')), cv2.COLOR_RGB2BGR)
                
                cv2.imshow("Smart Map ID Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

            except Exception as e:
                print(f"Loop Error: {e}")
                break
        
        self.close()

    def close(self):
        cv2.destroyAllWindows()

    def update_player_pos_from_server(self):
        """Flask 서버(sinario_1120.py)의 /info에서 플레이어 위치를 가져옴"""
        try:
            resp = requests.get(f"{self.server_url}/info", timeout=0.2)
            if resp.status_code != 200:
                return
            data = resp.json()
            pos = data.get("pos")
            if isinstance(pos, dict):
                x = float(pos.get("x", 0.0))
                z = float(pos.get("z", 0.0))
                # y는 시뮬레이터 상에서 크게 의미 없어서 0.0으로 고정
                self.player_pos = [x, 0.0, z]
        except Exception as e:
            # 서버 미응답 시 그냥 이전 위치 유지
            # print(f"[WARN] 서버에서 위치 받기 실패: {e}")
            pass

if __name__ == "__main__":
    MODEL_FILE_PATH = 'detector_gui/weights/5cls_v5_2_case2_best.pt' 
    detector = ScreenDetector(model_path=MODEL_FILE_PATH)
    detector.run_detection()

# def run_flask():
#     app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# if __name__ == "__main__":
#     MODEL_FILE_PATH = '5cls_v5_2_case2_best.pt' 
#     detector_instance = ScreenDetector(model_path=MODEL_FILE_PATH)
    
#     flask_thread = threading.Thread(target=run_flask)
#     flask_thread.daemon = True
#     flask_thread.start()
    
#     detector_instance.run_detection()