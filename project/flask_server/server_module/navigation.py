import math, os, json, heapq

class Navigator:
    def __init__(self, map_file):
        self.map_file = map_file
        self.grid_size = 1.0        # 지도를 1m 단위 격자로 쪼개기
        self.obstacle_margin = 7.0  # 장애물로부터 7m 떨어지게 설정
        self.obstacles = []         # 장애물 좌표 리스트
        self.blocked_cells = set()  # 못 지나가는 격자를 표시
        self.final_path = []        # 최종 계산된 경로
        
        self._load_map()            # 맵 파일을 읽어서 장애물 위치 파악
        self._build_obstacle_map()  # 장애물 주변을 위험구역으로

    def _load_map(self): # json 파일에서 내가 지정한 값만 골라서 장애물로 등록
        if not os.path.exists(self.map_file): return
        with open(self.map_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        obstacle_keywords = ["tank", "car", "rock"]
        for ob in data.get("obstacles", []):
            name = str(ob.get("prefabName", "")).lower()
            if any(k in name for k in obstacle_keywords):
                pos = ob.get("position", {})
                self.obstacles.append({'x': float(pos.get('x',0)), 'z': float(pos.get('z',0))})
        print(f"A*주행 준비완료: 장애물 {len(self.obstacles)}개 로드됨")

    # 실수 좌표를 A*가 알 수 있게 정수 격자로 변환
    def _world_to_grid(self, x, z):
        return int(round(x / self.grid_size)), int(round(z / self.grid_size))
    
    # 정수 격자를 다시 실수 좌표로 변환
    def _grid_to_world(self, r, c):
        return float(r) * self.grid_size, float(c) * self.grid_size
    
    # 장애물 주변 7m를 전부 벽으로 등록
    def _build_obstacle_map(self):
        margin_steps = int(math.ceil(self.obstacle_margin / self.grid_size))
        for ob in self.obstacles:
            gr, gc = self._world_to_grid(ob['x'], ob['z'])
            # 장애물을 중심으로 범위를 검사
            for r in range(gr - margin_steps, gr + margin_steps + 1):
                for c in range(gc - margin_steps, gc + margin_steps + 1):
                    wx, wz = self._grid_to_world(r, c)
                    # 실제 거리가 7m 이내면 벽으로 간주
                    if math.hypot(wx - ob['x'], wz - ob['z']) <= self.obstacle_margin:
                        self.blocked_cells.add((r, c))

    # 목표까지 남은 직선 거리를 계산 
    def _heuristic(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # 두 지점 사이에 장애물이 없는지 확인 
    def _is_los_clear(self, p1, p2):
        x1, z1 = p1; x2, z2 = p2
        dist = math.hypot(x2 - x1, z2 - z1)
        if dist < self.grid_size: 
            return True # 너무 가까우면 통과
        
        # 0.5m 간격으로 점을 찍어가며 벽이 있는지 검사
        steps = int(dist / (self.grid_size * 0.5))
        for i in range(1, steps + 1):
            t = i / steps
            lx, lz = x1 + (x2 - x1) * t, z1 + (z2 - z1) * t
            if self._world_to_grid(lx, lz) in self.blocked_cells: 
                return False    # 벽이 있을때
        return True             # 벽이 없을때

    # [1단계] 경로를 직선으로 펴주기 (경로 평탄화)
    def _smooth_path(self, path):
        if len(path) < 3: return path
        smoothed = [path[0]]
        current_idx = 0
        while current_idx < len(path) - 1:
            for i in range(len(path) - 1, current_idx, -1):
                if self._is_los_clear(path[current_idx], path[i]):
                    smoothed.append(path[i])
                    current_idx = i
                    break
        return smoothed

    # [2단계] 직선으로 펴진 경로 위에 1m 간격으로 점을 찍어주는 함수
    def _expand_path(self, path):
        if len(path) < 2: return path
        expanded = []
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i+1]
            dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            steps = int(dist / 1.0) # 1m 간격으로 점 추가

            # p1과 p2 사이에 점 추가
            for s in range(steps):
                t = s / steps
                ex = p1[0] + (p2[0] - p1[0]) * t
                ez = p1[1] + (p2[1] - p1[1]) * t
                expanded.append((ex, ez))
        expanded.append(path[-1])   # 마지막 점 추가
        return expanded
    
    # [A*알고리즘] 길찾기 로직
    def _a_star(self, start, end):
        def heuristic(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
        start_node = self._world_to_grid(*start)
        end_node = self._world_to_grid(*end)
        
        open_set = []
        heapq.heappush(open_set, (0, start_node))   # 우선순위
        came_from, g_score = {}, {start_node: 0}    # 경로 기록용
        neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

        while open_set:
            _, current = heapq.heappop(open_set) # 가장 좋은 노드 꺼내기
            
            # 목표 지점 근처(2m)에 도착하면 종료
            if heuristic(current, end_node) < 2.0:
                path = []
                while current in came_from: # 경로 역추적
                    path.append(self._grid_to_world(*current))
                    current = came_from[current]
                path.append(start); path.reverse(); path.append(end)
                
                # 평탄화 후 -> 다시 점을 촘촘하게 채움
                return self._expand_path(self._smooth_path(path))

            for dx, dy in neighbors:
                neighbor = (current[0]+dx, current[1]+dy)
                if neighbor in self.blocked_cells: 
                    continue # 장애물이면 패스
                # 대각선 이동 시 장애물 긁기 방지
                if dx!=0 and dy!=0:
                    if (current[0]+dx, current[1]) in self.blocked_cells or \
                       (current[0], current[1]+dy) in self.blocked_cells: continue
                
                # 이동 비용 계산 (대각선은 1.4, 직선은 1.0)
                tentative_g = g_score[current] + (1.414 if dx!=0 and dy!=0 else 1.0)
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    heapq.heappush(open_set, (tentative_g + heuristic(neighbor, end_node), neighbor))
        
        # 길을 못 찾으면 직선 경로라도 반환
        return [start, end]
    
    # 시작점과 끝점을 주면 경로를 만들어줌
    def generate_path(self, start_pos, end_pos):
        self.final_path = self._a_star(start_pos, end_pos)
        return self.final_path
    
    # [Pure Pursuit]현재 내 위치에서 6m 앞의 목표점을 찾는 함수
    def get_lookahead_target(self, px, pz, lookahead=6.0):
        if not self.final_path: return (px, pz)
        closest_idx, min_dist = 0, 9999.0
        # 촘촘해진 경로에서 가장 가까운 점 찾기
        for i, (nx, nz) in enumerate(self.final_path):
            d = math.hypot(nx - px, nz - pz)
            if d < min_dist: min_dist, closest_idx = d, i
        
        # 거기서부터 6m 앞을 봄 
        for i in range(closest_idx, len(self.final_path)):
            nx, nz = self.final_path[i]
            if math.hypot(nx - px, nz - pz) >= lookahead: return (nx, nz)
        return self.final_path[-1]

    # 목표 좌표를 보고 W/S/A/D 명령을 계산하는 함수
    def get_drive_control(self, px, pz, body_yaw, target_x, target_z, is_retreating=False, drift_mode=False, is_combat=False):
        def normalize(a): return (a + 180.0) % 360.0 - 180.0
        dx, dz = target_x - px, target_z - pz
        target_angle = math.degrees(math.atan2(dx, dz))
        
        # 후진 모드
        if is_retreating:
            back_yaw = normalize(body_yaw + 180.0)
            back_diff = normalize(target_angle - back_yaw)
            if abs(back_diff) > 40.0:
                 return {"moveWS": {"command": "STOP", "weight": 1}, "moveAD": {"command": "D" if back_diff > 0 else "A", "weight": 0.8}}
            else:
                 return {"moveWS": {"command": "S", "weight": 0.5}, "moveAD": {"command": "D" if back_diff > 0 else "A", "weight": min(1.0, abs(back_diff) * 0.05)}}
        
        # 전진 모드
        diff = normalize(target_angle - body_yaw)
        abs_diff = abs(diff)
        
        # 주행 모드에 따른 설정값 (전투, 부드러운 커브, 일반)
        if is_combat: pivot_limit, min_throttle, steer_gain = 55.0, 0.0, 0.04
        elif drift_mode: pivot_limit, min_throttle, steer_gain = 180.0, 0.6, 0.06
        else: pivot_limit, min_throttle, steer_gain = 120.0, 0.2, 0.04

        # 제자리 회전
        if abs_diff > pivot_limit:
            return {"moveWS": {"command": "STOP", "weight": 1}, "moveAD": {"command": "D" if diff > 0 else "A", "weight": 0.6}}

        # 조향 명령 
        steer_cmd = "D" if diff > 0 else "A" if abs_diff > 3.0 else ""
        steer_weight = min(1.0, abs_diff * steer_gain) if steer_cmd else 0.0
        #가속 명령
        fwd = min_throttle if drift_mode else min(0.7, max(min_throttle, 1.0 - (abs_diff / 120.0)))
        
        return {"moveWS": {"command": "W", "weight": fwd}, "moveAD": {"command": steer_cmd, "weight": steer_weight}}