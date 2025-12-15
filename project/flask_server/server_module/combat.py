import math, os, json
import pandas as pd
import numpy as np

class Gunner:
    def __init__(self, map_file):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.csv_file = os.path.join(BASE_DIR, "log_data", "output.csv")

        # 탄도학 상수 (포탄 속도 58m/s, 중력 9.81)
        self.v_init = 58.0
        self.g = 9.81
        self.h_offset = 2.1             # 포탑이 바닥보다 2.1m 위에 있음
        self.max_range = 130.0          # 최대 사거리 제한

        self.targets = []
        self._load_targets(map_file)    # 적 위치 로딩
    
    # 맵 파일에서 tank 이름을 가진 적만 
    def _load_targets(self, map_file):
        if not os.path.exists(map_file): 
            print("맵 파일을 찾을 수 없습니다. {map_file}")
            return
        
        with open(map_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for ob in data.get("obstacles", []):
            if "tank" in str(ob.get("prefabName", "")).lower():
                p = ob.get("position", {})
                self.targets.append({'x': float(p['x']), 'y': float(p['y']), 'z': float(p['z'])})
        print(f"포격 준비 완료: 적군 {len(self.targets)}대 확인")

    # 내 위치 기준으로 가까운 순서대로 index번째 적을 반환
    def get_target(self, px, pz, index=0):
        if not self.targets:
            return None, 9999
        
        # 거리순 정렬 (람다식 사용)
        sorted_t = sorted(self.targets, key=lambda t: math.hypot(t['x']-px, t['z']-pz))

        # index가 범위를 넘어가면 마지막 적 선택
        safe_idx = min(index, len(sorted_t)-1)
        target = sorted_t[safe_idx]
        dist = math.hypot(target['x']-px, target['z']-pz)
        return target, dist

    # 물리 공식을 이용해 포신 각도 계산
    def calculate_solution(self, px, py, pz, tx, ty, tz):
        # 내 포탑 높이(h_offset)를 고려해 상대 좌표 계산
        dx, dy, dz = tx - px, ty - (py + self.h_offset), tz - pz
        dist_h = math.hypot(dx, dz) # 수평 거리

        # 좌우(Yaw) 계산: 아크탄젠트 사용
        yaw = math.degrees(math.atan2(dx, dz))
        yaw = (yaw + 180.0) % 360.0 - 180.0

        # 상하(pitch) 계산: 포뮬선 방정식 
        v2 = self.v_init ** 2
        # 판별식: 이 값이 음수면 사거리 밖
        term = v2**2 - self.g * (self.g * dist_h**2 + 2 * dy * v2)

        # 해가 있으면 물리 공식 적용 (낮은 탄도 선택)
        if term >= 0:
            pitch = math.degrees(math.atan((v2 - math.sqrt(term)) / (self.g * dist_h)))
        # 해가 없으면
        else:
            pitch = self._get_pitch_from_csv(math.hypot(dx, dy, dz))
            if pitch is None: # CSV도 없으면 그냥 적을 바라보게 (직사)
                pitch = math.degrees(math.atan2(dy, dist_h))
        
        return {"ok": True, "yaw": yaw, "pitch": max(-30.0, min(10.0, pitch))}
    
    # CSV 파일에서 거리에 따른 각도를 찾아 보간
    def _get_pitch_from_csv(self, dist):
        if not os.path.exists(self.csv_file):
            return None
        try:
            df = pd.read_csv(self.csv_file)
            arr = df.to_numpy()
            # 3번이 컬럼이 거리, 0번 컬럼이 각도
            z_vals = arr[:, 3]
            angles = arr[:, 0]
            # np.interp 거리 55m면 50m와 60m 데이터 사이값으로 추정
            return float(np.interp(dist, z_vals, angles))
        except Exception as e:
            # 파일이 없거나 읽기 실패시 에러 출력
            print(f"CSV 읽기 오류: {e} (경로: {self.csv_file})")
            return None
        
    # 현재 각도와 목표 각도의 차이를 계산해서 Q,E,R,F 명령 
    def get_turret_control(self, curr_yaw, curr_pitch, target_yaw, target_pitch):
        def norm(a): return (a + 180.0) % 360.0 - 180.0
        dyaw = norm(target_yaw - curr_yaw)
        dpitch = target_pitch - curr_pitch
        cmd_y = "E" if dyaw > 0 else "Q"
        cmd_p = "R" if dpitch > 0 else "F"
        
        return {
            # 오차가 클수록 속도를 1.0으로, 작으면 천천히
            "turretQE": {"command": cmd_y, "weight": min(abs(dyaw)*0.02, 1.0)},
            "turretRF": {"command": cmd_p, "weight": min(abs(dpitch)*0.1, 1.0)},
            "aimed": abs(dyaw) < 3.0 and abs(dpitch) < 3.0 # 오차가 3도 미만이면 조준 완료 판정
        }
    