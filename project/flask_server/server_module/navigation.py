import math, os, json, heapq

class Navigator:
    def __init__(self, map_file):
        self.map_file = map_file
        self.grid_size = 1.0
        self.obstacle_margin = 7.0 
        self.obstacles = []
        self.blocked_cells = set()
        self.final_path = []
        
        self._load_map()
        self._build_obstacle_map()

    def _load_map(self):
        if not os.path.exists(self.map_file): return
        with open(self.map_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        obstacle_keywords = ["tank", "car", "rock"]
        for ob in data.get("obstacles", []):
            name = str(ob.get("prefabName", "")).lower()
            if any(k in name for k in obstacle_keywords):
                pos = ob.get("position", {})
                self.obstacles.append({'x': float(pos.get('x',0)), 'z': float(pos.get('z',0))})
        print(f"네비게이션 준비: 장애물 {len(self.obstacles)}개 로드됨")

    def _world_to_grid(self, x, z):
        return int(round(x / self.grid_size)), int(round(z / self.grid_size))

    def _grid_to_world(self, r, c):
        return float(r) * self.grid_size, float(c) * self.grid_size

    def _build_obstacle_map(self):
        margin_steps = int(math.ceil(self.obstacle_margin / self.grid_size))
        for ob in self.obstacles:
            gr, gc = self._world_to_grid(ob['x'], ob['z'])
            for r in range(gr - margin_steps, gr + margin_steps + 1):
                for c in range(gc - margin_steps, gc + margin_steps + 1):
                    wx, wz = self._grid_to_world(r, c)
                    if math.hypot(wx - ob['x'], wz - ob['z']) <= self.obstacle_margin:
                        self.blocked_cells.add((r, c))

    def _is_los_clear(self, p1, p2):
        x1, z1 = p1; x2, z2 = p2
        dist = math.hypot(x2 - x1, z2 - z1)
        if dist < self.grid_size: return True
        steps = int(dist / (self.grid_size * 0.5))
        for i in range(1, steps + 1):
            t = i / steps
            lx, lz = x1 + (x2 - x1) * t, z1 + (z2 - z1) * t
            if self._world_to_grid(lx, lz) in self.blocked_cells: return False
        return True

    # [1단계] 경로 펴기 (ZigZag 제거)
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

    # [2단계] ★핵심★ 직선 경로 위에 점 다시 찍기 (Resampling)
    # 이걸 해야 점을 지나쳤을 때 뒤로 돌아가는 현상을 막습니다.
    def _expand_path(self, path):
        if len(path) < 2: return path
        expanded = []
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i+1]
            dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            steps = int(dist / 1.0) # 1m 간격으로 점 추가
            for s in range(steps):
                t = s / steps
                ex = p1[0] + (p2[0] - p1[0]) * t
                ez = p1[1] + (p2[1] - p1[1]) * t
                expanded.append((ex, ez))
        expanded.append(path[-1])
        return expanded

    def _a_star(self, start, end):
        def heuristic(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
        start_node = self._world_to_grid(*start)
        end_node = self._world_to_grid(*end)
        
        open_set = []
        heapq.heappush(open_set, (0, start_node))
        came_from, g_score = {}, {start_node: 0}
        neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

        while open_set:
            _, current = heapq.heappop(open_set)
            
            if heuristic(current, end_node) < 2.0:
                path = []
                while current in came_from:
                    path.append(self._grid_to_world(*current))
                    current = came_from[current]
                path.append(start); path.reverse(); path.append(end)
                
                # 평탄화 후 -> 다시 점을 촘촘하게 채움
                return self._expand_path(self._smooth_path(path))

            for dx, dy in neighbors:
                neighbor = (current[0]+dx, current[1]+dy)
                if neighbor in self.blocked_cells: continue
                if dx!=0 and dy!=0:
                    if (current[0]+dx, current[1]) in self.blocked_cells or \
                       (current[0], current[1]+dy) in self.blocked_cells: continue

                tentative_g = g_score[current] + (1.414 if dx!=0 and dy!=0 else 1.0)
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    heapq.heappush(open_set, (tentative_g + heuristic(neighbor, end_node), neighbor))
        
        return [start, end]

    def generate_path(self, start_pos, end_pos):
        self.final_path = self._a_star(start_pos, end_pos)
        return self.final_path

    def get_lookahead_target(self, px, pz, lookahead=6.0):
        if not self.final_path: return (px, pz)
        closest_idx, min_dist = 0, 9999.0
        # 촘촘해진 경로에서 가장 가까운 점 찾기
        for i, (nx, nz) in enumerate(self.final_path):
            d = math.hypot(nx - px, nz - pz)
            if d < min_dist: min_dist, closest_idx = d, i
        
        # 거기서부터 6m 앞을 봄 (점이 많으므로 안전함)
        for i in range(closest_idx, len(self.final_path)):
            nx, nz = self.final_path[i]
            if math.hypot(nx - px, nz - pz) >= lookahead: return (nx, nz)
        return self.final_path[-1]

    def get_drive_control(self, px, pz, body_yaw, target_x, target_z, is_retreating=False, drift_mode=False, is_combat=False):
        def normalize(a): return (a + 180.0) % 360.0 - 180.0
        dx, dz = target_x - px, target_z - pz
        target_angle = math.degrees(math.atan2(dx, dz))
        
        if is_retreating:
            back_yaw = normalize(body_yaw + 180.0)
            back_diff = normalize(target_angle - back_yaw)
            if abs(back_diff) > 40.0:
                 return {"moveWS": {"command": "STOP", "weight": 1}, "moveAD": {"command": "D" if back_diff > 0 else "A", "weight": 0.8}}
            else:
                 return {"moveWS": {"command": "S", "weight": 0.5}, "moveAD": {"command": "D" if back_diff > 0 else "A", "weight": min(1.0, abs(back_diff) * 0.05)}}

        diff = normalize(target_angle - body_yaw)
        abs_diff = abs(diff)
        
        if is_combat: pivot_limit, min_throttle, steer_gain = 55.0, 0.0, 0.04
        elif drift_mode: pivot_limit, min_throttle, steer_gain = 180.0, 0.6, 0.06
        else: pivot_limit, min_throttle, steer_gain = 120.0, 0.2, 0.04

        if abs_diff > pivot_limit:
            return {"moveWS": {"command": "STOP", "weight": 1}, "moveAD": {"command": "D" if diff > 0 else "A", "weight": 0.6}}

        steer_cmd = "D" if diff > 0 else "A" if abs_diff > 3.0 else ""
        steer_weight = min(1.0, abs_diff * steer_gain) if steer_cmd else 0.0
        fwd = min_throttle if drift_mode else min(0.7, max(min_throttle, 1.0 - (abs_diff / 120.0)))
        
        return {"moveWS": {"command": "W", "weight": fwd}, "moveAD": {"command": steer_cmd, "weight": steer_weight}}