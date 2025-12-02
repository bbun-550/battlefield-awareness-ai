import math, json, heapq, os

class Navigator:
    def __init__(self, map_file_path):
        self.grid_size = 1.0            # 1m 단위 격자
        self.obstacle_margin = 7.0      #  장애물 회피 거리
        self.obstacles = []             # 장애물 좌표 
        self.current_path = []          # 현재 계산된 경로
        self._load_map(map_file_path)   # 시작하자마자 맵을 로딩
    
    def _load_map(self, path):
        # 멥 파일이 없을때
        if not os.path.exists(path):
            print(f"맵 파일이 없습니다.{path}")
            return
        # 맵 파일 json파일로 읽어서 파이썬 딕셔너리로 변환
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 장애물로 인식할 키워드 지정
        keywords = ["tank", "car", "rock"]
        for ob in data.get("obstacles", []):
            name = str(ob.get("prefabName", "")).lower()
            if any(k in name for k in keywords):
                p = ob.get("position", {})
                # 필요한 x, z좌표만 뽑아서 저장
                self.obstacles.append({'x': float(p.get('x', 0)), 'z':float(p.get('z', 0))})
        print(f"A* 경로 준비 완료: 장애물 {len(self.obstacles)}개 업로드 완료")

    # A* 알고리즘 헬퍼 함수
    def _world_to_grid(self, x, z):
        # 실수형 좌표(10.5)를 격자 인덱스(11)로 반올림(A*는 격자 기반으로 정수가 필요)
        return int(round(x / self.grid_size)), int(round(z / self.grid_size))
    
    def _grid_to_world(self, r, c):
        # 정수 격자(11)를 다시 실제 월드 좌표(11.0)로 반환 (유니티의 정밀한 좌표계를 A*가 이해할 수 있는 단순한 좌표로 바꾸기 위해서)
        return float(r) * self.grid_size, float(c) * self.grid_size
    
    def _get_blocked_cells(self):
        # A*에서 사용할 지나갈 수 없는 칸들의 합 만들기
        blocked = set()
        # 안전거리가 격자로 몇 칸인지 계산
        steps = int(math.ceil(self.obstacle_margin / self.grid_size))

        for ob in self.obstacles:
            gr, gc = self._world_to_grid(ob['x'], ob['z'])
            # 장애물 중심 주변을 탐색
            for r in range(gr - steps, gr + steps + 1):
                for c in range(gc - steps, gc + steps + 1):
                    # 격자를 실제 좌표로 바꿔서 정밀 계산
                    wx, wz = self._grid_to_world(r, c)
                    #실제 거리가 안전거리 이내면 위험 구역
                    if math.hypot(wx - ob['x'], wz - ob['z']) <= self.obstacle_margin:
                        blocked.add((r, c))
        return blocked
    
    # 더 정밀한 주행을 위해서 경로 평탄화
    def _is_los_clear(self, p1, p2, blocked_cells):
        # p1, p2 사이가 뚫려있는지 확인
        x1, z1 = p1
        x2, z2 = p2
        dist = math.hypot(x2 - x1, z2 - z1)
        # 거리가 1칸보다 가까우면 뚫려있다고 간주
        if dist < self.grid_size:
            return True
        
        # 0.5m 간격으로 점을 찍어서 벽에 닿는지 검사
        steps = int(dist / (self.grid_size * 0.5))
        for i in range(1, steps + 1):
            t = i / steps   # 0 ~ 1 사이 비율
            lx = x1 + (x2 - x1) * t
            lz = z1 + (z2 - z1) * t
            # 중간 점이 벽에 포함되면 실패
            if self._world_to_grid(lx, lz) in blocked_cells:
                return False
        return True # 벽에 안 닿으면 성공
    
    # A*의 계단식 경로를 직선으로 펴주는 함수
    def _smooth_path(self, path, blocked_cells):
        if len(path) < 3:   # 점이 2개면 펼 필요가 없음
            return path
        
        smoothed = [path[0]]    # 시작점을 넣어주고 시작
        current_idx = 0

        while current_idx < len(path) - 1:
            # 내 위치에서 멀리 있는 점부터 확인
            for i in range(len(path) - 1, current_idx, -1):
                # 현재 위치에서 제일 멀리 있는 점까지 직선으로 갈 수 있는지 확인
                if self._is_los_clear(path[current_idx], path[i], blocked_cells):
                    smoothed.append(path[i])    # 중간 점들 다 생략하고 바로 연결
                    current_idx = i     # 내 위치로 이동
                    break
        return smoothed
    
    # 현재 위치에서 목표까지 최적의 경로 계산
    def update_path(self, start_pos, end_pos):
        print(f"경로 탐색: {start_pos} -> {end_pos}")
        start_node = self._world_to_grid(*start_pos)
        end_node = self._world_to_grid(*end_pos)
        blocked = self._get_blocked_cells()   # 위험구역 게산

        # A* 알고리즘 시작 (우선순위)
        open_set = []
        heapq.heappush(open_set, (0, start_node))
        came_from = {}      # 경로 추적용
        g_score = {start_node: 0}   # 시작점부터의 비용

        while open_set:
            _, current = heapq.heappop(open_set)

            # 목표 지점 근처(2칸) 도착시 종료
            if math.hypot(current[0]-end_node[0], current[1]-end_node[1]) < 2.0:
                # 목표 지점에 도착하면 역추적해서 경로 완성하기
                path = []

                while current in came_from:
                    path.append(self._grid_to_world(*current))
                    current = came_from[current]

                path.append(start_pos)
                path.reverse()  # 역추적으로 정보가 거꾸로 있어서 반대로 뒤집기
                path.append(end_pos)

                    # 경로를 정밀화해서 저장
                self.current_path = self._smooth_path(path, blocked)
                return self.current_path
                
                # 8방향으로 탐색
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in blocked:
                    continue    # 벽으로 막혀있으면 통과

                # 대각선은 1.4 직선은 1.0
                cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                tentative_g = g_score[current] + cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    # 휴리스틱(남은 거리 추정) 더해서 큐에 넣기
                    f = tentative_g + math.hypot(neighbor[0]-end_node[0], neighbor[1]-end_node[1])
                    heapq.heappush(open_set, (f, neighbor))
                
        print("A* 실패 장애물 무시하고 직선 경로 반환")
        self.current_path = [start_pos, end_pos]
        return self.current_path
    
    # 주행 pure pursuit & 조향 제어
    def get_pure_pursuit_target(self, px, pz, lookahead=6.0):
        if not self.current_path:
            return (px, pz)
        # 경로 리스트를 거꾸로 돌면서 lookahead 거리보다 먼 첫 점 찾기
        for i in range(len(self.current_path)-1, -1, -1):
            nx, nz = self.current_path[i]
            if math.hypot(nx - px, nz - pz) <= lookahead:
                # 찾은 덤의 바로 다음 점을 타겟으로 
                target_idx = min(i + 1, len(self.current_path) - 1)
                return self.current_path[target_idx]
        return self.current_path[-1]    # 못 찾으면 마지막 점으로
        
    # 모드에 따라 W,A,S,D 명령 생성
    def get_motor_control(self, px, pz, body_yaw, target_x, target_z, mode="NORMAL"):
        dx, dz = target_x - px, target_z - pz
        target_angle = math.degrees(math.atan2(dx, dz))

        # 각도를 -180 ~ 180 사이로 보정해주는 함수
        def normalize(a):
            return (a + 180.0) % 360.0 - 180.0
        
        # 1. 후진모드 (회피기동)
        if mode == "REVERSE":
            back_yaw = normalize(body_yaw + 180.0)      # 탱크 뒷부분 방향
            diff = normalize(target_angle - back_yaw)   # 목표와 뒷부분 각도 차이
            # 뒷부분이 너무 틀어져 있으면 멈춰서 방향부터 잡음
            if abs(diff) > 40.0:
                return {"moveWS": {"command": "STOP", "weight": 1}, "moveAD": {"command": "D" if diff > 0 else "A", "weight": 0.8}}
            # 적당하면 후진 하면서 핸들 조작
            return {"moveWS": {"command": "S", "weight": 0.5}, "moveAD": {"command": "D" if diff > 0 else "A", "weight": min(1.0, abs(diff)*0.05)}}
        
        # 2. 전진모드 (일반 주행)
        diff = normalize(target_angle - body_yaw)
        abs_diff = abs(diff)

        # 모드별 튜닝값 설정 (드리프트는 각도 커도 그냥 달림, 정밀은 멈춤)
        if mode == "DRIFT":
            pivot_limit, min_throttle, steer_gain = 180.0, 0.6, 0.06
        elif mode == "PRECISION": # 정밀 모드는 그대로 (정확해야 하니까)
            pivot_limit, min_throttle, steer_gain = 55.0, 0.0, 0.04
        else: # [NORMAL 모드 튜닝]
            pivot_limit = 150.0  # 웬만하면 멈추지 말고 돌아라 (기존 120)
            min_throttle = 0.35  # 코너에서도 속도 유지해라 (기존 0.2)
            steer_gain = 0.025

        # 제자리 회전
        if abs_diff > pivot_limit:
            return {"moveWS": {"command": "STOP", "weight": 1}, "moveAD": {"command": "D" if diff > 0 else "A", "weight": 0.6}}
        
        # 조향 및 속도 계산
        steer_cmd = "D" if diff > 0 else "A"
        steer_weight = min(1.0, abs_diff * steer_gain) if abs_diff > 3.0 else 0
        # 각도가 클수록 속도를 줄임 
        throttle = min_throttle if mode == "DRIFT" else min(0.7, max(min_throttle, 1.0 - (abs_diff / 120.0)))
        return {"moveWS": {"command": "W", "weight": throttle}, "moveAD": {"command": steer_cmd, "weight": steer_weight}}