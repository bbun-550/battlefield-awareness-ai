import cv2
import numpy as np
from mss import mss
<<<<<<< HEAD
from ultralytics import YOLO
=======
>>>>>>> 6ea758d (원격환경 merge)
import math
import json
import os
import time
import requests
from PIL import ImageFont, ImageDraw, Image, ImageSequence


<<<<<<< HEAD
=======

>>>>>>> 6ea758d (원격환경 merge)
class ScreenDetector:
    def __init__(self, model_path='best.pt', map_path='best.map', server_url='http://127.0.0.1:5000'):
        # ---------------------------------------------------------
        # [초기화] 화면 캡처, 모델 로드, 설정값 정의
        # ---------------------------------------------------------
        self.server_url = server_url
        self.player_pos = [0.0, 0.0, 0.0]

        # 서버의 포격 카운트를 추적하기 위한 변수
        self.last_fire_count = -1

        # 화면 해상도 설정 (기본 FHD)
        self.screen_width = 1920
        self.screen_height = 1080

        # mss: 초고속 화면 캡처 라이브러리 설정
        self.monitor = {"top": 0, "left": 0, "width": self.screen_width, "height": self.screen_height}
        self.sct = mss()

        # YOLO 모델 경로 설정 및 로드
        base_path = os.path.dirname(os.path.abspath(__file__))
        full_model_path = os.path.join(base_path, model_path)
        if not os.path.exists(full_model_path):
            full_model_path = model_path
<<<<<<< HEAD
            
=======
        
>>>>>>> 6ea758d (원격환경 merge)
        self.model = YOLO(full_model_path)
        self.class_names = self.model.names
        
        # 거리 계산을 위한 초점 거리 상수 (임의 설정값)
        self.FOCAL_LENGTH_PX = 1000 
        
        # [거리 추정용] 각 클래스별 실제 너비 (단위: 미터)
        self.KNOWN_WIDTH_M = {
            0: 1.6,   # Red (사람)
            1: 14.4,  # Car
            2: 1.6,   # Blue (사람)
            3: 15.2,  # Rock
            4: 13.7   # Tank
        }
        
        # 화면에 그릴 최대 거리 제한
        self.MAX_DRAW_DISTANCE_M = 200.0 
        
        # 맵 파일(장애물 좌표) 로드
        self.TARGET_FILE_PATH = map_path
        self.player_pos = [0.0, 0.0, 0.0]
        self.map_data_cache = self.load_target_coordinates()
        
        # [폰트 설정] 한글 출력을 위한 폰트 로드
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

        # ---------------------------------------------------------
        # [GIF 설정] Reload 애니메이션을 위한 변수 및 파일 로드
        # ---------------------------------------------------------
        self.gif_path = "detector_gui/image/reload.gif"
        self.gif_frames = []        # GIF 프레임들을 저장할 리스트
        self.is_reloading = False   # 현재 리로딩 중인지 상태 플래그
        self.reload_start_time = 0  # 리로딩 시작 시간
        self.RELOAD_DURATION = 6.5  # 리로딩 지속 시간 (7초)
        
        self.load_gif_frames()      # GIF 미리 로드 실행

    def load_gif_frames(self):
        """ 
        [최적화] 실행 중에 파일을 읽으면 렉이 걸리므로,
        시작할 때 GIF의 모든 프레임을 미리 메모리에 로드해둡니다.
        """
        abs_path = os.path.abspath(self.gif_path)
        print(f"GIF 파일 찾는 위치: {abs_path}")

        if not os.path.exists(self.gif_path):
            print(f"[!!! 경고 !!!] '{self.gif_path}' 파일을 찾을 수 없습니다.")
            return

        try:
            with Image.open(self.gif_path) as im:
                # GIF의 모든 프레임을 순회하며 RGBA(투명도 포함)로 변환해 저장
                for frame in ImageSequence.Iterator(im):
                    frame_rgba = frame.convert("RGBA")
                    self.gif_frames.append(frame_rgba)
            print(f"Reload GIF 로드 성공: 총 {len(self.gif_frames)} 프레임")
        except Exception as e:
            print(f"[오류] GIF 로딩 실패: {e}")
            self.gif_frames = []

    def load_target_coordinates(self):
        """ JSON 맵 파일에서 장애물(Target) 좌표들을 읽어옵니다. """
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
        """ [시각적 거리] 화면상 박스 크기를 이용해 거리를 추정합니다. """
        pixel_width = x2 - x1
        known_width = self.KNOWN_WIDTH_M.get(cls_id, 1.5)
        if pixel_width > 0:
            return (known_width * self.FOCAL_LENGTH_PX) / pixel_width
        return 999.9

    def calculate_real_distance(self, p_pos, t_obj):
        """ [실제 거리] 플레이어 좌표와 맵 객체 좌표 간의 3차원 거리를 계산합니다. """
        return math.sqrt(
            (p_pos[0] - t_obj['x'])**2 + 
            (p_pos[1] - t_obj['y'])**2 + 
            (p_pos[2] - t_obj['z'])**2
        )

    def get_fixed_color(self, cls_name):
        """ 클래스 이름에 따라 고정된 색상을 반환합니다 (BGR 포맷). """
        name = cls_name.lower()
        if name == 'red' or name == 'tank':
            return (0, 0, 255) # 빨강
        elif name == 'blue':
            return (255, 0, 0) # 파랑
        elif name == 'car' or name == 'rock':
            return (128, 128, 128) # 회색
        else:
            return (255, 255, 255)

    def analyze_battlefield(self, counts):
        """ 탐지된 객체 수를 바탕으로 전장 상황 멘트를 생성합니다. """
        red_cnt = 0
        tank_cnt = 0
        blue_cnt = 0
        # 대소문자 구분 없이 카운트 합산
        for k, v in counts.items():
            kl = k.lower()
            if 'red' in kl: red_cnt += v
            elif 'tank' in kl: tank_cnt += v
            elif 'blue' in kl: blue_cnt += v
        
        enemies = red_cnt + tank_cnt
        if enemies > 0:
            if tank_cnt > 0:
                return f" 위험! 적 전차 {tank_cnt}대 식별!"
            else:
                return f" 경고! 적군 {red_cnt}명 접근 중"
        elif blue_cnt > 0:
            return f" 아군 {blue_cnt}명과 합류 가능"
        elif len(counts) > 0:
            return "전방 안전 / 중립 물체 식별"
        else:
            return "탐색 중..."

    def run_detection(self):
        print("객체 탐지 시작... (Spacebar: Reload GIF 출력 - Global Key 감지)")
        
        # 맵 데이터 매칭을 위한 필터 키워드
        CAR_FILTERS = ['car001', 'car002', 'car003', 'car004', 'car005']
        ROCK_FILTERS = ['rock001', 'rock002']
        TANK_FILTERS = ['tank001']

        while True:
            try:
                # 서버에서 플레이어 위치 갱신
                self.update_player_pos_from_server()
                
                # 1. 화면 캡처 (MSS)
                img_mss = self.sct.grab(self.monitor)
                img = np.array(img_mss)
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) # RGBA -> BGR 변환
                
<<<<<<< HEAD
                # 2. YOLO 객체 탐지 수행
=======
                # 2. ONNX 객체 탐지 수행
>>>>>>> 6ea758d (원격환경 merge)
                results = self.model(frame, verbose=False, conf=0.7, iou=0.38)
                
                # 3. 탐지된 박스 데이터 정리
                current_frame_boxes = []
<<<<<<< HEAD
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.class_names.get(cls_id, 'unknown')
                    sim_dist = self.calculate_sim_distance(cls_id, x1, x2)
                    
                    # 너무 먼 거리(200m 이상)는 무시
                    if sim_dist > self.MAX_DRAW_DISTANCE_M: continue
                    
                    current_frame_boxes.append({
                        'bbox': (x1, y1, x2, y2),
                        'cls_id': cls_id,
                        'cls_name': cls_name,
                        'sim_dist': sim_dist,
                        'matched_map_obj': None 
=======
                for det in results:
                    x1, y1, x2, y2 = det["bbox"]
                    cls_id = det["cls"]
                    cls_name = det["cls_name"]
                    conf = det["score"]

                    sim_dist = self.calculate_sim_distance(cls_id, x1, x2)
                    if sim_dist > self.MAX_DRAW_DISTANCE_M:
                        continue

                    current_frame_boxes.append({
                        "bbox": (x1, y1, x2, y2),
                        "cls_id": cls_id,
                        "cls_name": cls_name,
                        "sim_dist": sim_dist,
                        "matched_map_obj": None
>>>>>>> 6ea758d (원격환경 merge)
                    })

                # 4. 맵 데이터와 탐지된 객체 매칭 (Matching Logic)
                unique_classes = set(b['cls_name'] for b in current_frame_boxes)
                total_counts = {}

                for cls_name in unique_classes:
                    # 해당 클래스의 박스들만 추출
                    cls_boxes = [b for b in current_frame_boxes if b['cls_name'] == cls_name]
                    total_counts[cls_name] = len(cls_boxes)
                    
                    # 맵 데이터에서 해당 클래스와 관련된 객체 필터링
                    cls_lower = cls_name.lower()
                    relevant_map_objs = []
                    for m_obj in self.map_data_cache:
                        map_prefab = m_obj['prefabName']
                        is_match = False
                        # 이름 매칭 규칙
                        if cls_lower == 'red' and 'human003' in map_prefab: is_match = True
                        elif cls_lower == 'blue' and 'human002' in map_prefab: is_match = True
                        elif cls_lower == 'car' and any(f in map_prefab for f in CAR_FILTERS): is_match = True
                        elif cls_lower == 'tank' and any(f in map_prefab for f in TANK_FILTERS): is_match = True
                        elif cls_lower == 'rock' and any(f in map_prefab for f in ROCK_FILTERS): is_match = True
                        
                        if is_match:
                            # 거리 계산 (플레이어 위치 기준)
                            dist = self.calculate_real_distance(self.player_pos, m_obj)
                            if dist <= 500.0: # 500m 이내 후보만
                                obj_with_dist = m_obj.copy()
                                obj_with_dist['real_dist'] = dist
                                relevant_map_objs.append(obj_with_dist)

                    # [알고리즘] 시각적 거리와 실제 거리 차이가 가장 적은 것끼리 매칭
                    match_candidates = []
                    for box_idx, box in enumerate(cls_boxes):
                        for map_idx, map_obj in enumerate(relevant_map_objs):
                            diff = abs(box['sim_dist'] - map_obj['real_dist'])
                            match_candidates.append({'diff': diff, 'box_idx': box_idx, 'map_idx': map_idx})
                    
                    match_candidates.sort(key=lambda x: x['diff'])
                    
                    used_boxes = set()
                    used_maps = set()
                    
                    # 1:1 매칭 우선 수행
                    for cand in match_candidates:
                        b_idx = cand['box_idx']
                        m_idx = cand['map_idx']
                        if b_idx not in used_boxes and m_idx not in used_maps:
                            cls_boxes[b_idx]['matched_map_obj'] = relevant_map_objs[m_idx]
                            used_boxes.add(b_idx)
                            used_maps.add(m_idx)
                    
                    # 매칭되지 않은 박스들에 대해 가장 가까운 근사값 매칭 (N:1)
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

                # 5. 박스 그리기 (OpenCV 사용)
                for box in current_frame_boxes:
                    x1, y1, x2, y2 = box['bbox']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.get_fixed_color(box['cls_name']), 2)

                # 6. 텍스트 및 HUD 그리기 (PIL 사용 - 한글 및 투명도 지원)
                if self.font_bold is not None:
                    # OpenCV(BGR) -> PIL(RGBA) 변환
                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    img_pil = img_pil.convert("RGBA")
                    overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0)) # 투명 레이어
                    draw = ImageDraw.Draw(overlay)
                    
                    hud_text_list = []
                    current_frame_boxes.sort(key=lambda x: x['sim_dist'])
                    counters = {} 

                    # 각 객체 위에 이름과 거리 텍스트 표시
                    for box in current_frame_boxes:
                        x1, y1, x2, y2 = box['bbox']
                        cls_name = box['cls_name']
                        if cls_name not in counters: counters[cls_name] = 1
                        simple_name = f"{cls_name.capitalize()} {counters[cls_name]}"
                        counters[cls_name] += 1
                        bgr = self.get_fixed_color(cls_name)
                        map_obj = box['matched_map_obj']
                        
                        if map_obj:
                            real_d = map_obj['real_dist']
                            label_dist = f"R : {real_d:.1f}m"
                            hud_text_list.append(f"{simple_name} : {real_d:.1f}m")
                            # 객체 이름 (색상 적용)
                            draw.text((x1, y1-25), simple_name, font=self.font_bold, fill=(bgr[2], bgr[1], bgr[0]))
                            # 거리 정보 (검정색)
                            text_bbox = draw.textbbox((x1, y2 + 5), label_dist, font=self.font_bold)
                            draw.text((x1, y2 + 5), label_dist, font=self.font_bold, fill=(0, 0, 0))
                        else:
                            # 매칭 실패 시 추정 거리 표시
                            label = f"{simple_name} (추정:{box['sim_dist']:.1f}m)"
                            draw.text((x1, y1-25), label, font=self.font_bold, fill=(180, 180, 180))
                    
                    # 7. 좌측 상단 HUD 패널 그리기
                    hud_text_list.sort()
                    display_list = hud_text_list[:20] 
                    situation_text = self.analyze_battlefield(total_counts)
                    summary_text = " | ".join([f"{k.capitalize()}: {v}개" for k, v in total_counts.items()])
                    
                    base_x, base_y = 5, 50
                    line_height = 25
                    max_text_width = 300
                    all_texts = ["전장상황인식", situation_text, summary_text] + display_list
                    
                    # HUD 박스 너비 자동 계산
                    for txt in all_texts:
                        w = draw.textlength(txt, font=self.font_bold)
                        if w > max_text_width: max_text_width = int(w)
                    
                    total_box_height = 35 + 30 + 30 + (len(display_list) * line_height) + 40
                    
                    # 반투명 검정 배경 박스 그리기
                    draw.rounded_rectangle([(base_x, base_y), (base_x + max_text_width + 110, base_y + total_box_height)], fill=(0, 0, 0, 200), radius=10)
                    
                    # HUD 내용 텍스트 그리기
                    tx, ty = base_x + 10, base_y + 5
                    draw.text((tx, ty), "전장상황인식", font=self.font_bold, fill=(255, 255, 255))
                    draw.text((tx, ty+35), f"현재 내 좌표: X : {self.player_pos[0]:.2f}, Z :{self.player_pos[2]:.2f}", font=self.font_bold, fill=(0, 255, 0))
                    draw.text((tx, ty+65), f"상황: {situation_text}", font=self.font_bold, fill=(255, 100, 100))
                    
                    ty_list = ty + 95
                    for txt in display_list:
                        draw.text((tx, ty_list), txt, font=self.font_bold, fill=(200, 200, 200))
                        ty_list += line_height
                    draw.text((tx, ty_list+5), f'총합 : {summary_text}', font=self.font_bold, fill=(255, 255, 0))

                    # ---------------------------------------------------------
                    # 8. GIF 오버레이
                    # ---------------------------------------------------------
                    # [GIF 그리기] 리로딩 상태일 때만 실행
                    if self.is_reloading:
                        elapsed = time.time() - self.reload_start_time
                        
                        if elapsed > self.RELOAD_DURATION:
                            self.is_reloading = False # 7초 지나면 종료
                        
                        elif self.gif_frames:
                            # 경과 시간에 맞춰 프레임 인덱스 계산 (애니메이션 속도 조절)
                            total_frames = len(self.gif_frames)
                            # 0.1초당 1프레임 기준 (*10)
                            frame_idx = int((elapsed * 10) % total_frames)
                            gif_frame = self.gif_frames[frame_idx]
                            
                            # 화면 중앙 좌표 계산
                            cx = (self.screen_width - gif_frame.width) // 2
                            cy = (self.screen_height - gif_frame.height) // 2
                            
                            # 투명도 유지하며 합성 (Alpha Composite)
                            overlay.alpha_composite(gif_frame, dest=(cx, cy))
                    
                    # PIL 이미지 합성 후 다시 OpenCV 포맷으로 변환
                    out = Image.alpha_composite(img_pil, overlay)
                    frame = cv2.cvtColor(np.array(out.convert('RGB')), cv2.COLOR_RGB2BGR)
                
                # 9. 최종 화면 출력
                cv2.imshow("Smart Map ID Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

            except Exception as e:
                print(f"Loop Error: {e}")
                break
        
        self.close()

    def close(self):
        cv2.destroyAllWindows()

    def update_player_pos_from_server(self):
        """Flask 서버의 /info에서 플레이어 위치 및 포격 정보 가져오기"""
        try:
            resp = requests.get(f"{self.server_url}/info", timeout=0.2)
            if resp.status_code != 200:
                return
            
            data = resp.json()

            # 1. 위치 정보 업데이트
            pos = data.get("pos")
            if isinstance(pos, dict):
                x = float(pos.get("x", 0.0))
                z = float(pos.get("z", 0.0))
                # y는 시뮬레이터 상에서 크게 의미 없어서 0.0으로 고정
                self.player_pos = [x, 0.0, z]

            # 2. 포격(Fire) 감지 로직
            current_total_shots = int(data.get("fire_count", 0))

            # 처음 연결 시(-1)에는 현재 값으로 동기화만 하고 넘어감
            if self.last_fire_count == -1:
                self.last_fire_count = current_total_shots

            # 카운트가 증가했다면 -> "발사됨" -> GIF 재생 시작
            elif current_total_shots > self.last_fire_count:
                # print(f">> 포격 감지! (Count: {self.last_fire_count} -> {current_fire_count}) GIF 재생")
                self.is_reloading = True
                self.reload_start_time = time.time()
                self.last_fire_count = current_total_shots

        except Exception as e:
            # 서버 미응답 시 그냥 이전 위치 유지
            # print(f"[WARN] 서버에서 위치 받기 실패: {e}")
            pass


if __name__ == "__main__":
<<<<<<< HEAD
    MODEL_FILE_PATH = 'detector_gui/weights/5cls_v6_case9_best.pt'
=======
    MODEL_FILE_PATH = 'detector_gui/weights/5cls_v6_case9_best.onnx'
>>>>>>> 6ea758d (원격환경 merge)
    MAP_FILE_PATH = 'flask_server/map/11_28.map'
    detector_instance = ScreenDetector(model_path=MODEL_FILE_PATH, map_path=MAP_FILE_PATH)
    detector_instance.run_detection()
    