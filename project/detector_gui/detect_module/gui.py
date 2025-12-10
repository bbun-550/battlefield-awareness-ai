import cv2
import time

class Visualizer:
    """
    탐지된 정보와 상태를 화면에 그려주는 클래스
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        # [상태 관리] 리로딩 중인가? (GIF 대신 텍스트 점멸을 위해 필요)
        self.is_reloading = False
        self.reload_start_time = 0
        self.RELOAD_DURATION = 6.5 # 재장전 소요 시간 (초)

    def get_color(self, cls_name):
        """ 클래스 이름에 따라 고정된 색상(BGR) 반환 """
        n = cls_name.lower()
        if n in ['red', 'tank']: return (0, 0, 255)         # 적군: 빨강
        elif n == 'blue': return (255, 0, 0)                # 아군: 파랑
        elif n in ['car', 'rock']: return (128, 128, 128)   # 장애물: 회색
        return (255, 255, 255)                              # 기타: 흰색

    def trigger_reload(self):
        """ 
        [이벤트] 서버에서 발사 신호가 오면 호출됨.
        리로딩 상태를 True로 켜고, 타이머를 시작함.
        """
        self.is_reloading = True
        self.reload_start_time = time.time()

    def draw(self, frame, detections, player_pos, counts):
        """ 
        [메인 그리기] 
        1. 바운딩 박스 그리기
        2. HUD 정보창 그리기
        3. 리로딩 상태 표시 (텍스트 점멸)
        """
        
        # 1. 객체 바운딩 박스 및 정보 표시
        for box in detections:
            x1, y1, x2, y2 = box['bbox']
            cls_name = box['cls_name']
            color = self.get_color(cls_name)
            
            # 박스 테두리
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 라벨 텍스트 (맵 데이터와 매칭된 경우 실제 거리를 표시)
            m_obj = box['matched_map_obj']
            label_top = cls_name.capitalize()
            # 매칭 성공 시: 실제 거리 / 실패 시: 시각적 추정 거리
            label_dist = f"{m_obj['real_dist']:.1f}m" if m_obj else f"({box['sim_dist']:.0f}m)"

            # 가독성을 위해 텍스트 배경 (검은색 박스) 추가
            cv2.rectangle(frame, (x1, y1-20), (x1+100, y1), color, -1)
            cv2.putText(frame, label_top, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, label_dist, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # 2. 좌측 상단 HUD (Heads-Up Display) 배경 그리기
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 50), (350, 200), (0, 0, 0), -1)
        # 반투명 효과 적용 (배경이 살짝 비치게)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # 텍스트 정보 구성
        sit_txt = self._analyze_battlefield(counts)
        summary = " | ".join([f"{k}:{v}" for k,v in counts.items()])

        # HUD 내용 출력
        cv2.putText(frame, "BATTLEFIELD STATUS", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"[My Pos] {player_pos[0]:.1f}, {player_pos[2]:.1f}", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(frame, f"[Status] {sit_txt}", (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,255), 1)
        cv2.putText(frame, f"[Total] {summary}", (15, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        # 3. 리로딩 알림 (GIF 제거 -> 텍스트 점멸)
        if self.is_reloading:
            elapsed = time.time() - self.reload_start_time
            # 6.5초가 지났으면 상태 해제
            if elapsed > self.RELOAD_DURATION:
                self.is_reloading = False
            else:
                # 깜빡임 효과: 경과 시간에 5를 곱해 정수로 만들고 짝수일 때만 그림
                if int(elapsed * 5) % 2 == 0: 
                    text = "!!! RELOADING !!!"
                    # 화면 중앙 계산
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                    text_x = (self.width - text_size[0]) // 2
                    text_y = (self.height + text_size[1]) // 2
                    
                    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        
        return frame

    def _analyze_battlefield(self, counts):
        """ 탐지된 객체 수를 바탕으로 현재 상황을 문자열로 반환 """
        tank = sum(v for k,v in counts.items() if 'tank' in k.lower())
        red = sum(v for k,v in counts.items() if 'red' in k.lower())
        blue = sum(v for k,v in counts.items() if 'blue' in k.lower())
        
        if tank > 0: return f"WARNING: {tank} TANKS!"
        elif red > 0: return f"Enemy: {red}"
        elif blue > 0: return f"Friendly: {blue}"
        return "Scouting..." if counts else "Searching..."