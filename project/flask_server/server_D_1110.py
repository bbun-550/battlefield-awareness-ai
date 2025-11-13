# =============================================================================
# 사용방법
# 코드에서는 PATH 경로를 본인에게 맞춰 수정

# 시뮬레이션:
# .map 파일은 default로 커스텀하게 만들어서 사용하심됩니다.
# 초반  restart -> start , tracking edit mode에서 깃발꽂을때 적 탱크쪽에.. (포탄 확인하기 위함)
# =============================================================================

# =============================================================================
# Tank Challenge - 자율주행 서버 (D* Lite 경로 계획 + 자동 조준/사격)
# =============================================================================
#
# 주요 기능:
# 1. D* Lite 알고리즘으로 목적지까지 최적 경로 계획
# 2. 라이다 센서로 동적 장애물 실시간 감지
# 3. Unity에서 받은 정적 장애물(.map 데이터) 처리
# 4. 자동 포탑 조준 및 발사 제어
# 5. 긴급 회피 로직 (전방 장애물 감지 시 자동 회피)
#
# =============================================================================

from __future__ import annotations

from flask import Flask, request, jsonify
import os, math, threading, time
from typing import Optional, Tuple, Dict, Set, List, Any

# -----------------------------
# 포탑 조준/사격 외부 모듈
# -----------------------------
from tank_aim_app_1105 import TankAimer
aimer = TankAimer()  # 탄도 계산 및 포탑 제어 담당

def _merge_aim_response(cmd_ws: dict, cmd_ad: dict, dbg: dict | None) -> Dict[str, Any]:
    """주행 명령 + 포탑 조준/발사 명령 통합

    Args:
        cmd_ws: 전후진 명령 (W/S 키)
        cmd_ad: 좌우 회전 명령 (A/D 키)
        dbg: 디버깅 정보

    Returns:
        통합 명령 (moveWS, moveAD, turretQE, turretRF, fire)
    """
    try:
        req_json = safe_get_json()
        turret = (req_json.get("turret") or {}) if isinstance(req_json, dict) else {}
        tx = float(turret.get("x", 0.0))
        ty = float(turret.get("y", 0.0))
        aim = aimer.get_action_dict(tx, ty)
        return {
            "moveWS": cmd_ws,
            "moveAD": cmd_ad,
            "turretQE": aim.get("turretQE", {"command": "", "weight": 0.0}),
            "turretRF": aim.get("turretRF", {"command": "", "weight": 0.0}),
            "fire": bool(aim.get("fire", False)),
            "debug": {**(dbg or {}), "aim": aim.get("debug", {})},
        }
    except Exception as e:
        return {
            "moveWS": cmd_ws,
            "moveAD": cmd_ad,
            "turretQE": {"command": "", "weight": 0.0},
            "turretRF": {"command": "", "weight": 0.0},
            "fire": False,
            "debug": {**(dbg or {}), "aim_error": str(e)},
        }

# -----------------------------
# Flask & 경로
# -----------------------------
app = Flask(__name__)

SAVE_DIR  = r"C:\Users\SeYun\anaconda3\envs\tf\TC\Tank Challenge"
DEST_FILE = os.path.join(SAVE_DIR, "last_destination.txt")
os.makedirs(SAVE_DIR, exist_ok=True)

# model = load_yolo_model()

# =============================================================================
# 핵심 파라미터 설정
# =============================================================================

# --- 맵 설정 ---
MAP_MIN_X, MAP_MAX_X = 0.0, 300.0  # 맵 X축 범위 (미터)
MAP_MIN_Z, MAP_MAX_Z = 0.0, 300.0  # 맵 Z축 범위 (미터)

GRID_RES_M = 2.0                    # 그리드 해상도 (1칸 = 2m × 2m)
GRID_ORIGIN_XZ = (MAP_MIN_X, MAP_MIN_Z)

# --- 경로 계획 설정 ---
ALLOW_DIAGONAL = True               # 대각선 이동 허용
NO_CORNER_CUT = True                # 모서리 자르기 방지 (대각선 이동 시 벽 통과 방지)

SAFETY_CLEARANCE_M = 3.0            # 장애물 안전 거리 (3m)
INFLATE_CELLS = int(math.ceil(SAFETY_CLEARANCE_M / GRID_RES_M))  # = 2칸 버퍼

DYN_TTL_S = 2.0                     # 동적 장애물 유지 시간 (2초 후 자동 삭제)
ARRIVE_RADIUS_M = 2.5               # 목적지 도착 판정 거리 (2.5m 이내면 도착)

# --- 주행 제어 ---
ANGLE_DEADZONE_DEG   = 5.0          # 각도 오차 무시 범위 (±5도 이내면 직진)
ROTATE_IN_PLACE_DEG  = 45.0         # 제자리 회전 시작 각도 (45도 이상이면 정지 후 회전)
EVADE_CONE_DEG       = 30.0         # 전방 위험 감지 범위 (±30도 부채꼴)

LOOKAHEAD_CELLS = 2                 # 경로 추종 시 미리 보는 거리 (2칸 = 4m)

# --- 긴급 회피 ---
EVADE_DURATION_S = 1.0              # 회피 동작 유지 시간 (1초간 회피 행동 지속)

TURN_LEFT_KEY  = "A"
TURN_RIGHT_KEY = "D"
FORWARD_KEY    = "W"
BACKWARD_KEY   = "S"
FORCE_AD_FLIP  = False  # 좌우 키 반전 필요시 True

# Unity → 표준 yaw 변환 기준
YAW_ZERO_BASIS = "Z0"
YAW_SIGN = +1

# =============================================================================
# 전역 상태 (멀티스레드 안전)
# =============================================================================
state_lock = threading.Lock()
state: Dict[str, Any] = {
    # --- 탱크 위치/방향 ---
    "pos": None,              # 현재 위치 (x, z)
    "last_pos": None,         # 이전 위치 (속도 계산용)
    "goal": None,             # 목적지 (x, z)
    "yaw_deg": None,          # 현재 방향 (도, 0° = 북쪽)
    "raw": {"body_x": None, "body_y": None, "body_z": None},  # Unity 원본 각도

    # --- 장애물 관리 (3종류) ---
    "static_cells": set(),    # 정적 장애물 (벽, 바위 등 - /update_obstacle로 받음)
    "manual_cells": set(),    # 수동 추가 장애물 (/update_occupancy로 추가/삭제)
    "dynamic_cells": {},      # 동적 장애물 (라이다로 감지, TTL 있음)

    # --- 경로 계획 ---
    "path_world": [],         # D* Lite가 계산한 경로 (월드 좌표 리스트)
    "last_cell": None,        # 이전 그리드 셀 (경로 재계산 트리거용)
    "need_replan": True,      # 경로 재계획 필요 여부
    "last_replan_ts": 0.0,    # 마지막 재계획 시각

    # --- 라이다 데이터 집계 (전방 위험도) ---
    "ahead_min_dist": float("inf"),  # 전방 최소 거리
    "ahead_left_hits": 0,            # 좌측 전방 장애물 개수
    "ahead_right_hits": 0,           # 우측 전방 장애물 개수
    "last_lidar_ts": 0.0,            # 마지막 라이다 수신 시각

    # --- 긴급 회피 상태 ---
    "evade_until_ts": 0.0,   # 회피 종료 시각 (현재 시각 < evade_until_ts 이면 회피 중)
    "evade_dir": 0,          # 회피 방향 (1: 우회전, -1: 좌회전)

    # --- 타이머 ---
    "start_time": None,      # 시작 시각 (도착 시간 계산용)
}

# =============================================================================
# 유틸리티 함수들
# =============================================================================

def safe_get_json() -> dict:
    """Flask 요청에서 JSON 안전하게 추출 (에러 시 빈 dict 반환)"""
    return request.get_json(silent=True, force=False) or {}

def normalize_angle(angle: float) -> float:
    """각도를 -180° ~ +180° 범위로 정규화

    예: 270° → -90° (시계방향 90도)
        -200° → 160° (반시계방향 160도)
    """
    angle = angle % 360.0
    return angle - 360.0 if angle > 180.0 else angle

def calculate_angle_to_target(current_x: float, current_z: float,
                              target_x: float, target_z: float) -> float:
    """현재 위치에서 목표까지의 방향 계산 (도)

    Returns:
        0° = 북쪽(+Z), 90° = 동쪽(+X), -90° = 서쪽(-X), ±180° = 남쪽(-Z)
    """
    dx = target_x - current_x
    dz = target_z - current_z
    return math.degrees(math.atan2(dx, dz))

def convert_unity_angle_to_standard(unity_angle: float) -> float:
    angle_radians = math.radians(YAW_SIGN * unity_angle)
    if YAW_ZERO_BASIS == "Z0":
        direction_x = math.sin(angle_radians)
        direction_z = math.cos(angle_radians)
    else:
        direction_x = math.cos(angle_radians)
        direction_z = math.sin(angle_radians)
    return math.degrees(math.atan2(direction_x, direction_z))

def update_player_position(data: dict) -> None:
    """들어오는 payload 내 포맷 다양성을 고려해 위치 파싱."""
    player_position = None
    pos_data = data.get("position") or data.get("playerPos")
    if isinstance(pos_data, dict):
        x = pos_data.get("x"); z = pos_data.get("z")
        if x is not None and z is not None:
            player_position = (float(x), float(z))
    if player_position is None and ("x" in data and "z" in data):
        player_position = (float(data["x"]), float(data["z"]))
    if player_position is None:
        px = data.get("Player_Pos_X"); pz = data.get("Player_Pos_Z")
        if px is not None and pz is not None:
            player_position = (float(px), float(pz))
    if player_position is not None:
        state["last_pos"] = state["pos"]
        state["pos"] = player_position

def update_player_direction(data: dict) -> None:
    """여러 키 후보에서 Unity 기반 yaw를 찾아 표준 yaw로 환산."""
    unity_angle = None
    for key_name in ("Player_Body_X", "playerBodyX", "bodyX", "Player_Bodyx", "yaw", "heading", "playerYaw", "bodyYawDeg"):
        if any(k in data for k in ("Player_Body_X","playerBodyX","bodyX","Player_Bodyx","yaw","heading","playerYaw","bodyYawDeg")):
            try:
                unity_angle = float(data[key_name]); break
            except: pass
    if unity_angle is None:
        for key_name in ("yaw", "heading", "playerYaw", "bodyYawDeg"):
            if key_name in data:
                try:
                    unity_angle = float(key_name and data[key_name]); break
                except: pass
    if unity_angle is None:
        for key_name in ("Player_Body_Y", "playerBodyY", "bodyY",
                         "Player_Body_Z", "playerBodyZ", "bodyZ"):
            if key_name in data:
                try:
                    unity_angle = float(data[key_name]); break
                except: pass
    if unity_angle is not None:
        state["raw"]["body_x"] = unity_angle
        state["yaw_deg"] = convert_unity_angle_to_standard(unity_angle)

    # 이동 벡터로 보정
    if state["yaw_deg"] is None and state["last_pos"] and state["pos"]:
        (ox, oz), (nx, nz) = state["last_pos"], state["pos"]
        mvx, mvz = (nx - ox), (nz - oz)
        if abs(mvx) + abs(mvz) > 1e-4:
            state["yaw_deg"] = math.degrees(math.atan2(mvx, mvz))

def is_cell_inside_map(cell_x: int, cell_z: int) -> bool:
    world_x = cell_x * GRID_RES_M + GRID_ORIGIN_XZ[0]
    world_z = cell_z * GRID_RES_M + GRID_ORIGIN_XZ[1]
    return ((MAP_MIN_X - 1e-6) <= world_x <= (MAP_MAX_X + 1e-6)
            and (MAP_MIN_Z - 1e-6) <= world_z <= (MAP_MAX_Z + 1e-6))

def world_to_grid(world_x: float, world_z: float) -> Tuple[int, int]:
    origin_x, origin_z = GRID_ORIGIN_XZ
    return (int(round((world_x - origin_x) / GRID_RES_M)),
            int(round((world_z - origin_z) / GRID_RES_M)))

def grid_to_world(grid_x: int, grid_z: int) -> Tuple[float, float]:
    origin_x, origin_z = GRID_ORIGIN_XZ
    return (grid_x * GRID_RES_M + origin_x,
            grid_z * GRID_RES_M + origin_z)

def add_dynamic_world(x: float, z: float, now_ts: float, ttl: float = 2.0, r: int = INFLATE_CELLS) -> None:
    """동적 장애물 추가 (라이다 감지 점)

    Args:
        x, z: 장애물 위치 (월드 좌표)
        now_ts: 현재 시각
        ttl: 유지 시간 (초) - 이 시간 후 자동 삭제
        r: 버퍼 반경 (셀 개수)

    동작:
        장애물 위치 주변 r칸을 모두 막힌 것으로 표시
        예: r=2이면 5×5 = 25칸 차단 (안전 거리 확보)
    """
    ix, iz = world_to_grid(x, z)
    exp = now_ts + ttl  # 만료 시각
    for dx in range(-r, r + 1):
        for dz in range(-r, r + 1):
            cx, cz = ix + dx, iz + dz
            if is_cell_inside_map(cx, cz):
                state["dynamic_cells"][(cx, cz)] = exp

def blocked_cells(now_ts: Optional[float] = None) -> set:
    """현재 막힌 모든 셀 반환 (정적 + 동적 + 수동)

    Returns:
        set of (ix, iz): 막힌 그리드 셀 좌표

    동작:
        1. 동적 장애물: TTL 만료된 것은 자동 삭제
        2. 정적 장애물: 벽, 바위 등 (계속 유지)
        3. 수동 장애물: 사용자가 추가한 것 (계속 유지)
    """
    if now_ts is None:
        now_ts = time.time()

    # 동적 장애물 중 유효한 것만 선택
    dyn = set()
    for cell, exp in list(state["dynamic_cells"].items()):
        if exp >= now_ts:
            dyn.add(cell)
        else:
            state["dynamic_cells"].pop(cell, None)  # 만료된 것 삭제

    # 3종류 장애물 통합
    return set(state["static_cells"]) | set(state["manual_cells"]) | dyn

def inflate_add_cell(cells_set: set, ix: int, iz: int, r: int = INFLATE_CELLS) -> None:
    """셀 주변에 버퍼 추가"""
    for dx in range(-r, r + 1):
        for dz in range(-r, r + 1):
            cx, cz = ix + dx, iz + dz
            if is_cell_inside_map(cx, cz):
                cells_set.add((cx, cz))

def remove_inflated_cell(cells_set: set, ix: int, iz: int, r: int = INFLATE_CELLS) -> None:
    """셀 주변 버퍼 제거"""
    for dx in range(-r, r + 1):
        for dz in range(-r, r + 1):
            cx, cz = ix + dx, iz + dz
            cells_set.discard((cx, cz))

def replace_static_from_rects(rects: list, r: int = INFLATE_CELLS) -> None:
    """Unity에서 받은 사각형 장애물을 그리드로 변환"""
    new_set = set()
    for rect in rects:
        try:
            x0 = float(rect["x_min"]); x1 = float(rect["x_max"])
            z0 = float(rect["z_min"]); z1 = float(rect["z_max"])
            xmin, xmax = sorted([x0, x1])
            zmin, zmax = sorted([z0, z1])
        except:
            continue
        c0 = world_to_grid(xmin, zmin)
        c1 = world_to_grid(xmax, zmax)
        ix0, iz0 = min(c0[0], c1[0]), min(c0[1], c1[1])
        ix1, iz1 = max(c0[0], c1[0]), max(c0[1], c1[1])
        for ix in range(ix0 - r, ix1 + r + 1):
            for iz in range(iz0 - r, iz1 + r + 1):
                if is_cell_inside_map(ix, iz):
                    new_set.add((ix, iz))
    state["static_cells"] = new_set


def check_immediate_danger(player_pos_w: Tuple[float, float], player_yaw_deg: float) -> dict:
    """전방 긴급 위험 감지 (3방향 체크)

    Args:
        player_pos_w: 탱크 현재 위치 (x, z)
        player_yaw_deg: 탱크 현재 방향 (도)

    Returns:
        {'front': bool, 'front_left': bool, 'front_right': bool}
        - front: 정면 (0도)
        - front_left: 좌측 전방 (-30도)
        - front_right: 우측 전방 (+30도)

    동작:
        1. 현재 방향 기준 3방향을 4.5m 앞까지 체크
        2. 막힌 셀이 있으면 True 반환
        3. 긴급 회피 로직에서 사용 (3방향 모두 막히면 후진)
    """
    danger = {'front': False, 'front_left': False, 'front_right': False}
    all_blocked = blocked_cells(time.time())  # 모든 장애물 가져오기
    if not all_blocked:
        return danger  # 장애물 없으면 즉시 반환

    player_yaw_rad = math.radians(player_yaw_deg)
    check_dist = 4.5  # 4.5m 앞까지 체크 (너무 멀면 불필요한 회피, 너무 가까우면 충돌)
    angles_to_check_deg = {'front': 0, 'front_left': -30, 'front_right': 30}

    for direction, angle_offset_deg in angles_to_check_deg.items():
        # 체크할 방향 계산
        check_angle_rad = player_yaw_rad + math.radians(angle_offset_deg)

        # 체크할 위치 계산 (4.5m 앞)
        check_pos_w = (
            player_pos_w[0] + check_dist * math.sin(check_angle_rad),
            player_pos_w[1] + check_dist * math.cos(check_angle_rad)
        )

        # 그리드 좌표로 변환
        check_pos_g = world_to_grid(check_pos_w[0], check_pos_w[1])

        # 막힌 셀에 있으면 위험!
        if check_pos_g in all_blocked:
            danger[direction] = True

    return danger

# =============================================================================
# D* Lite 경로 계획 알고리즘
# =============================================================================
# 동적 장애물 환경에서 효율적으로 경로를 재계산하는 알고리즘
# - 장애물이 변해도 전체를 다시 계산하지 않고 변경된 부분만 업데이트
# - A* 알고리즘의 동적 버전
# =============================================================================

class DStarLite:
    def __init__(self, blocked: Set[Tuple[int,int]]):
        self.blocked: Set[Tuple[int,int]] = set(blocked)
        self.g: Dict[Tuple[int,int], float] = {}
        self.rhs: Dict[Tuple[int,int], float] = {}
        self.U: List[Tuple[Tuple[float,float], int, Tuple[int,int]]] = []
        self.open_keys: Dict[Tuple[int,int], Tuple[float,float]] = {}
        self.s_start: Tuple[int,int] = (0,0)
        self.s_goal: Tuple[int,int] = (0,0)
        self.s_last: Tuple[int,int] = (0,0)
        self.Km: float = 0.0
        self.counter: int = 0
        self.INF = float('inf')
        self.EPS = 1e-9

    def free(self, s: Tuple[int,int]) -> bool:
        return is_cell_inside_map(*s) and (s not in self.blocked)

    def neighbors(self, s: Tuple[int,int]):
        ix, iz = s
        dirs4 = [(1,0),(-1,0),(0,1),(0,-1)]
        neigh = dirs4[:]
        if ALLOW_DIAGONAL:
            neigh += [(1,1),(1,-1),(-1,1),(-1,-1)]
        for dx, dz in neigh:
            ns = (ix + dx, iz + dz)
            if not self.free(ns):
                continue
            if ALLOW_DIAGONAL and NO_CORNER_CUT and (dx != 0 and dz != 0):
                if (not self.free((ix + dx, iz))) or (not self.free((ix, iz + dz))):
                    continue
            yield ns

    def step_cost(self, a: Tuple[int,int], b: Tuple[int,int]) -> float:
        if (not self.free(a)) or (not self.free(b)):
            return self.INF
        ax, az = a; bx, bz = b
        diag = (ax != bx and az != bz)
        return (math.sqrt(2.0) if diag else 1.0) * GRID_RES_M

    def h(self, a: Tuple[int,int], b: Tuple[int,int]) -> float:
        ax, az = a; bx, bz = b
        dx = abs(ax - bx); dz = abs(az - bz)
        dmin = min(dx, dz); dmax = max(dx, dz)
        return (math.sqrt(2.0) * dmin + (dmax - dmin)) * GRID_RES_M

    def calculate_key(self, s: Tuple[int,int]) -> Tuple[float, float]:
        gv = self.g.get(s, self.INF)
        rv = self.rhs.get(s, self.INF)
        m = min(gv, rv)
        return (m + self.h(self.s_start, s) + self.Km, m)

    def top_key(self) -> Optional[Tuple[float, float]]:
        return self.U[0][0] if self.U else None

    def update_vertex(self, u: Tuple[int,int]) -> None:
        if u != self.s_goal:
            pred = list(self.neighbors(u))
            self.rhs[u] = min([self.step_cost(u, s_) + self.g.get(s_, self.INF) for s_ in pred]) if pred else self.INF
        if u in self.open_keys:
            old = self.open_keys[u]
            self.U = [(k, c, n) for (k, c, n) in self.U if n != u]
            import heapq as _hq; _hq.heapify(self.U)
            del self.open_keys[u]
        gv, rv = self.g.get(u, self.INF), self.rhs.get(u, self.INF)
        if abs(gv - rv) > self.EPS:
            key = self.calculate_key(u)
            self.counter += 1
            import heapq as _hq; _hq.heappush(self.U, (key, self.counter, u))
            self.open_keys[u] = key

    def compute_shortest_path(self) -> None:
        import heapq as _hq
        while self.U:
            k_old, _, u = _hq.heappop(self.U)
            if u in self.open_keys:
                del self.open_keys[u]
            else:
                continue
            k_new = self.calculate_key(u)
            if k_old < k_new:
                self.counter += 1
                _hq.heappush(self.U, (k_new, self.counter, u))
                self.open_keys[u] = k_new
                continue
            gv, rv = self.g.get(u, self.INF), self.rhs.get(u, self.INF)
            if gv > rv:
                self.g[u] = rv
                for s in self.neighbors(u): self.update_vertex(s)
            else:
                self.g[u] = self.INF
                self.update_vertex(u)
                for s in self.neighbors(u): self.update_vertex(s)
            k_start = self.calculate_key(self.s_start)
            top_k = self.top_key()
            if top_k is None or k_start < top_k:
                break

    def initialize(self, start: Tuple[int,int], goal: Tuple[int,int]) -> None:
        self.s_start, self.s_goal, self.s_last = start, goal, start
        self.Km = 0.0
        self.g.clear(); self.rhs.clear(); self.U.clear(); self.open_keys.clear()
        self.counter = 0
        self.rhs[self.s_goal] = 0.0
        key = self.calculate_key(self.s_goal)
        import heapq as _hq; _hq.heappush(self.U, (key, self.counter, self.s_goal))
        self.open_keys[self.s_goal] = key

    def update_start(self, new_start: Tuple[int,int]) -> None:
        if new_start == self.s_start: return
        self.Km += self.h(self.s_last, new_start)
        self.s_last = new_start
        self.s_start = new_start

    def set_blocked(self, new_blocked: Set[Tuple[int,int]]) -> None:
        changed = (self.blocked ^ new_blocked)
        if not changed: return
        self.blocked = set(new_blocked)
        affected = set()
        for c in changed:
            affected.add(c)
            for nb in self.neighbors(c): affected.add(nb)
        for u in affected: self.update_vertex(u)

    def reconstruct_path(self) -> List[Tuple[int,int]]:
        if self.g.get(self.s_start, self.INF) == self.INF:
            return []
        path = [self.s_start]
        s = self.s_start
        visited = {s}
        max_len = 10000
        while s != self.s_goal and len(path) < max_len:
            best, best_val = None, self.INF
            for n in self.neighbors(s):
                if n in visited: continue
                c = self.step_cost(s, n)
                val = c + self.g.get(n, self.INF)
                if val < best_val:
                    best_val, best = val, n
            if best is None or best_val == self.INF:
                break
            path.append(best); visited.add(best); s = best
        return path if s == self.s_goal else []

dstar: Optional[DStarLite] = None
dstar_goal: Optional[Tuple[int,int]] = None

def ensure_path_dstar(now_ts: Optional[float] = None) -> None:
    global dstar, dstar_goal
    pos, goal = state["pos"], state["goal"]
    if pos is None or goal is None:
        return
    if now_ts is None: now_ts = time.time()
    s = world_to_grid(*pos)
    g = world_to_grid(*goal)
    moved_cell = (state["last_cell"] != s)
    if moved_cell: state["last_cell"] = s
    obs = blocked_cells(now_ts)
    if (dstar is None) or (dstar_goal != g) or state["need_replan"]:
        print(f"🔄 D* Lite 전체 초기화: start=({s[0]},{s[1]}) goal=({g[0]},{g[1]})")
        dstar = DStarLite(obs)
        dstar.initialize(s, g)
        dstar.compute_shortest_path()
        dstar_goal = g
        state["need_replan"] = False
        state["last_replan_ts"] = now_ts
    else:
        if moved_cell: dstar.update_start(s)
        dstar.set_blocked(obs)
        dstar.compute_shortest_path()
    cells = dstar.reconstruct_path()
    state["path_world"] = [grid_to_world(ix, iz) for (ix, iz) in cells]

# -----------------------------
# Flask REST
# -----------------------------

def parse_destination(data: dict) -> tuple[float, float]:
    gx = gz = None
    if "destination" in data:
        dest = data["destination"]
        if isinstance(dest, str):
            parts = [s.strip() for s in dest.split(",")]
            if len(parts) >= 2:
                gx = float(parts[0]); gz = float(parts[-1])
        elif isinstance(dest, dict):
            gx = float(dest.get("x")); gz = float(dest.get("z"))
    if gx is None or gz is None:
        if "x" in data and "z" in data:
            gx = float(data["x"]); gz = float(data["z"])
    if gx is None or gz is None:
        raise ValueError("Missing x/z")
    return gx, gz

@app.route('/set_destination', methods=['POST'])
def set_destination():
    try:
        gx, gz = parse_destination(safe_get_json())
    except Exception as e:
        return jsonify({"status":"ERROR","message":f"Invalid destination: {e}"}), 400
    with state_lock:
        state["goal"] = (gx, gz)
        state["need_replan"] = True
    print(f"🎯 Destination set: ({gx:.3f}, {gz:.3f})")
    return jsonify({"status":"OK","destination":{"x":gx,"z":gz}})

@app.route('/clear_destination', methods=['POST'])
def clear_destination():
    """목표를 비운다 (단일 모드용)."""
    with state_lock:
        state["goal"] = None
        state["need_replan"] = True
    print("🧹 Destination cleared")
    return jsonify({"status":"OK"})

@app.route('/info', methods=['POST'])
def info():
    data = safe_get_json()
    now_ts = float(data.get("time", time.time()))
    with state_lock:
        update_player_position(data)
        update_player_direction(data)
        px, pz = (state["pos"] or (None, None))
        cy = state["yaw_deg"]

        pts = data.get("lidarPoints") or []
        for p in pts:
            try:
                if not p.get("isDetected", False): continue
                pos = p.get("position") or {}
                x = float(pos.get("x")); z = float(pos.get("z"))
            except Exception:
                continue
            add_dynamic_world(x, z, now_ts, ttl=DYN_TTL_S, r=INFLATE_CELLS)
            if px is not None and cy is not None:
                d = math.hypot(x - px, z - pz)
                bearing = calculate_angle_to_target(px, pz, x, z)
                rel = normalize_angle(bearing - cy)
                # 전방 위험 집계
                if abs(rel) <= EVADE_CONE_DEG:
                    state["ahead_min_dist"] = min(state["ahead_min_dist"], d)
                    if rel > 0: state["ahead_right_hits"] += 1
                    else:       state["ahead_left_hits"]  += 1

        state["last_lidar_ts"] = now_ts
    return jsonify({"status":"OK"})

@app.route('/get_action', methods=['POST'])
def get_action():
    """메인 주행 제어 엔드포인트 (Unity가 매 프레임 호출)

    동작 흐름:
        1. 위치/방향 업데이트
        2. 도착 여부 확인 → 도착했으면 정지
        3. 긴급 위험 체크 → 전방 막혔으면 회피
        4. 회피 중인지 확인 → 회피 타임윈도우 유지
        5. D* Lite로 경로 계획
        6. 경로 따라 주행 명령 생성
        7. 포탑 조준/발사 명령 추가

    Returns:
        {moveWS, moveAD, turretQE, turretRF, fire, debug}
    """
    data = safe_get_json()
    now_ts = float(data.get("time", time.time()))

    # 상태 업데이트 (thread-safe)
    with state_lock:
        update_player_position(data)
        update_player_direction(data)
        pos, goal, cur_yaw = state["pos"], state["goal"], state["yaw_deg"]
        ahead_min_dist   = state["ahead_min_dist"]
        left_hits        = state["ahead_left_hits"]
        right_hits       = state["ahead_right_hits"]
        last_lidar_ts    = state["last_lidar_ts"]
        evade_until_ts   = state["evade_until_ts"]
        evade_dir        = state["evade_dir"]

    # === 1단계: 목적지/위치 확인 ===
    if pos is None or goal is None:
        dbg = {
            "reason": "no_goal_or_pos",
            "pos": pos,
            "goal": goal,
            "yaw": cur_yaw,
            "last_lidar_ts": last_lidar_ts
        }
        return jsonify(_merge_aim_response(
            {"command": "STOP", "weight": 1.0},
            {"command": "",     "weight": 0.0},
            dbg
        ))

    px, pz = pos
    gx, gz = goal
    dist_goal = math.hypot(gx - px, gz - pz)

    # === 2단계: 도착 확인 ===
    if dist_goal <= ARRIVE_RADIUS_M:  # 2.5m 이내면 도착
        with state_lock:
            state["evade_until_ts"] = 0.0
            start_time = state.get("start_time")

        if start_time is not None:
            elapsed = time.time() - start_time
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            print(f"✅✅✅ Arrived in {minutes}m {seconds:.2f}s (single-dest)")
        else:
            print(f"✅ Arrived. pos=({px:.2f},{pz:.2f}) goal=({gx:.2f},{gz:.2f}) [single-dest]")

        dbg = {"reason": "arrived", "pos": [px, pz], "goal": [gx, gz]}
        return jsonify(_merge_aim_response(
            {"command": "STOP", "weight": 1.0},
            {"command": "",     "weight": 0.0},
            dbg
        ))

    # === 3단계: 긴급 위험 회피 (최우선) ===
    if cur_yaw is not None:
        danger_info = check_immediate_danger(pos, cur_yaw)
        if danger_info['front']:  # 전방이 막혔다!
            ws_cmd_str, ad_cmd_str = FORWARD_KEY, ""

            # 회피 방향 결정 (우선순위: 우측 → 좌측 → 후진)
            if not danger_info['front_right']:
                # 우측이 뚫렸으면 우회전
                ad_cmd_str = TURN_RIGHT_KEY
            elif not danger_info['front_left']:
                # 좌측이 뚫렸으면 좌회전
                ad_cmd_str = TURN_LEFT_KEY
            else:
                # 3방향 모두 막혔으면 후진 + 회전
                ws_cmd_str = BACKWARD_KEY
                ad_cmd_str = TURN_RIGHT_KEY if (left_hits >= right_hits) else TURN_LEFT_KEY

            with state_lock:
                state["evade_until_ts"] = now_ts + EVADE_DURATION_S
                state["evade_dir"]      = 1 if ad_cmd_str == TURN_RIGHT_KEY else -1

            dbg = {
                "reason": "immediate_front_blocked",
                "ahead_min": ahead_min_dist,
                "left_hits": left_hits,
                "right_hits": right_hits,
                "evade_until": state["evade_until_ts"]
            }
            return jsonify(_merge_aim_response(
                {"command": ws_cmd_str, "weight": 1.0},
                {"command": ad_cmd_str, "weight": 1.0},
                dbg
            ))

    # === 4단계: 회피 타임윈도우 (회피 동작 유지) ===
    if evade_until_ts > now_ts:  # 아직 회피 중
        ad_cmd_str = TURN_RIGHT_KEY if evade_dir > 0 else TURN_LEFT_KEY
        dbg = {
            "reason": "evade_window",
            "until": evade_until_ts,
            "dir": evade_dir,
            "ahead_min": ahead_min_dist
        }
        return jsonify(_merge_aim_response(
            {"command": FORWARD_KEY, "weight": 0.5},
            {"command": ad_cmd_str,  "weight": 1.0},
            dbg
        ))

    # === 5단계: 경로 계획 (D* Lite) ===
    try:
        ensure_path_dstar(now_ts)  # 시작점에서 목표까지 최적 경로 계산
    except Exception as e:
        print(f"[WARN] ensure_path_dstar failed: {e}")

    with state_lock:
        path_world = list(state.get("path_world") or [])

    # === 6단계: 경로 추종 주행 ===
    # 경로가 없으면 목표 방향으로 회전 + 전진
    if not path_world:
        if cur_yaw is None:
            dbg = {"reason": "no_path_and_no_yaw"}
            return jsonify(_merge_aim_response(
                {"command": "STOP", "weight": 1.0},
                {"command": "",     "weight": 0.0},
                dbg
            ))

        tgt_deg = calculate_angle_to_target(px, pz, gx, gz)
        diff    = normalize_angle(tgt_deg - cur_yaw)
        left_key, right_key = (
            (TURN_RIGHT_KEY, TURN_LEFT_KEY)
            if FORCE_AD_FLIP else
            (TURN_LEFT_KEY, TURN_RIGHT_KEY)
        )
        turn_key = right_key if diff > 0 else left_key

        dbg = {
            "reason": "no_path_turn_to_goal",
            "tgt_deg": tgt_deg,
            "yaw": cur_yaw
        }
        return jsonify(_merge_aim_response(
            {"command": FORWARD_KEY, "weight": 0.35},
            {"command": turn_key,    "weight": 0.65},
            dbg
        ))

    # Lookahead 방식: 지정 칸 수 앞을 목표로 주행
    look_idx = min(LOOKAHEAD_CELLS, len(path_world) - 1)
    tgt_x, tgt_z = path_world[look_idx]                 # 목표 지점
    tgt_deg = calculate_angle_to_target(px, pz, tgt_x, tgt_z)  # 목표 방향

    # yaw 정보를 아직 못 받았으면 우선 전진
    if cur_yaw is None:
        dbg = {"reason": "no_yaw_forward", "tgt": [tgt_x, tgt_z]}
        return jsonify(_merge_aim_response(
            {"command": FORWARD_KEY, "weight": 0.5},
            {"command": "",          "weight": 0.0},
            dbg
        ))

    # === 7단계: 각도 기반 주행 제어 (RL 완전 제거) ===
    diff = normalize_angle(tgt_deg - cur_yaw)  # 목표 방향 - 현재 방향
    abs_diff = abs(diff)
    left_key, right_key = (
        (TURN_RIGHT_KEY, TURN_LEFT_KEY)
        if FORCE_AD_FLIP else
        (TURN_LEFT_KEY, TURN_RIGHT_KEY)
    )
    turn_key = right_key if diff > 0 else left_key

    # 7-1. 45도 이상 차이 → 제자리 회전
    if abs_diff > ROTATE_IN_PLACE_DEG:
        cmd_ws = {"command": "",        "weight": 0.0}
        cmd_ad = {"command": turn_key,  "weight": 1.0}
        dbg = {"reason": "rotate_in_place", "deg": diff, "tgt": [tgt_x, tgt_z]}

    # 7-2. 5도 이하 → 직진만
    elif abs_diff <= ANGLE_DEADZONE_DEG:
        cmd_ws = {"command": FORWARD_KEY, "weight": 1.0}
        cmd_ad = {"command": "",          "weight": 0.0}
        dbg = {"reason": "forward_deadzone", "deg": diff, "tgt": [tgt_x, tgt_z]}

    # 7-3. 5~45도 사이 → 전진 + 회전 혼합
    else:
        turn_weight = min(1.0, 0.2 + abs_diff / 90.0)  # 각도 클수록 회전 비중 증가
        fwd_weight  = max(0.2, 1.0 - turn_weight)
        cmd_ws = {"command": FORWARD_KEY, "weight": fwd_weight}
        cmd_ad = {"command": turn_key,    "weight": turn_weight}
        dbg = {
            "reason": "mix",
            "deg": diff,
            "tw": turn_weight,
            "fw": fwd_weight,
            "tgt": [tgt_x, tgt_z]
        }

    return jsonify(_merge_aim_response(cmd_ws, cmd_ad, dbg))

@app.route('/debug_state', methods=['GET'])
def debug_state():
    with state_lock:
        p = state["pos"]; g = state["goal"]
        path_w = [{"x": x, "z": z} for (x, z) in state["path_world"][:200]]
        now_ts = time.time()
        obs = blocked_cells(now_ts)
        dstar_info = {}
        if dstar is not None:
            dstar_info = {"g_size": len(dstar.g), "rhs_size": len(dstar.rhs), "open_size": len(dstar.U), "km": dstar.Km}
        return jsonify({
            "pos": None if p is None else {"x": p[0], "z": p[1]},
            "goal": None if g is None else {"x": g[0], "z": g[1]},
            "yaw_deg": state["yaw_deg"],
            "raw": state["raw"],
            "path_world_head": path_w,
            "num_static": len(state["static_cells"]),
            "num_manual": len(state["manual_cells"]),
            "num_dynamic": len([1 for _,exp in state["dynamic_cells"].items() if exp >= now_ts]),
            "num_blocked_now": len(obs),
            "last_replan_ts": state["last_replan_ts"],
            "safety_clearance_m": SAFETY_CLEARANCE_M,
            "grid_res_m": GRID_RES_M,
            "algorithm": "D* Lite",
            "dstar": dstar_info
        })

@app.route('/init', methods=['GET'])
def init():
    config = {
        "startMode": "pause",
        "blStartX": 60, "blStartY": 10, "blStartZ": 27.23,
        "rdStartX": 59, "rdStartY": 10, "rdStartZ": 280,
        "detactMode": False,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": True,
        "saveLidarData": True,
        "destroyObstaclesOnHit": True,
        "trackingMode": True,
        "logMode": True,
        "lux": 30000
    }
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    # 단일 목적지 버전: 타이머만 시작. 목표는 사용자가 /set_destination 으로 설정.
    with state_lock:
        state["start_time"] = time.time()
        print("⏱️ Timer started.")
    print("🚀 /start - (single destination mode)")
    return jsonify({"status":"OK"})

@app.route('/collision', methods=['POST'])
def collision():
    data = safe_get_json()
    pos = data.get("position") or {}
    x = pos.get("x", data.get("x"))
    z = pos.get("z", data.get("z"))
    try:
        x = float(x); z = float(z)
    except Exception:
        return jsonify({"status":"ERROR","message":"position {x,z} required"}), 400
    r_cells = int(data.get("radius_cells", max(1, INFLATE_CELLS)))
    ttl_s   = float(data.get("ttl", 10.0))
    now_ts = time.time()
    with state_lock:
        add_dynamic_world(x, z, now_ts, ttl=ttl_s, r=r_cells)
        state["need_replan"] = True
    print(f"💥 Collision Registered at ({x:.2f},{z:.2f}) r={r_cells} ttl={ttl_s}s → replan")
    return jsonify({"status":"OK","registered":{"x":x,"z":z},"radius_cells":r_cells,"ttl":ttl_s})

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    """Unity에서 맵의 정적 장애물 수신 (.map 파일 데이터)

    Unity가 게임 시작 시 한 번 호출:
        POST /update_obstacle
        {
            "obstacles": [
                {"x_min": 10, "x_max": 20, "z_min": 30, "z_max": 40},
                ...
            ]
        }

    동작:
        각 사각형 장애물을 그리드로 변환하여 static_cells에 저장
        백그라운드 스레드에서 처리 (응답 빠르게 반환)
    """
    data = safe_get_json()
    rects = data.get("obstacles", [])
    print(f"📦 /update_obstacle: {len(rects)}개 장애물 수신 중...")

    def process_obstacles_async():
        time.sleep(0.1)
        with state_lock:
            print(f"🔧 장애물 처리 시작...")
            replace_static_from_rects(rects, r=INFLATE_CELLS)
            state["need_replan"] = True
            print(f"✅ {len(state['static_cells'])}개 정적 셀 생성 완료!")

    threading.Thread(target=process_obstacles_async, daemon=True).start()
    return jsonify({
        "status": "OK",
        "message": "Processing obstacles in background",
        "obstacle_count": len(rects)
    })

@app.route('/update_occupancy', methods=['POST'])
def update_occupancy():
    """수동으로 장애물 추가/제거 (디버깅/테스트용)

    사용 예:
        POST /update_occupancy
        {
            "blocked_world": [{"x": 100, "z": 150}],  # 이 위치 차단
            "clear_world": [{"x": 50, "z": 75}]       # 이 위치 해제
        }
    """
    data = safe_get_json()
    with state_lock:
        blocked = data.get("blocked", [])
        clear = data.get("clear", [])
        blocked_world = data.get("blocked_world", [])
        clear_world = data.get("clear_world", [])

        for it in blocked:
            try:
                cell = (int(it[0]), int(it[1]))
                inflate_add_cell(state["manual_cells"], cell[0], cell[1], r=INFLATE_CELLS)
            except:
                pass

        for it in clear:
            try:
                cell = (int(it[0]), int(it[1]))
                remove_inflated_cell(state["manual_cells"], cell[0], cell[1], r=INFLATE_CELLS)
            except:
                pass

        for it in blocked_world:
            try:
                cx, cz = world_to_grid(float(it["x"]), float(it["z"]))
                inflate_add_cell(state["manual_cells"], cx, cz, r=INFLATE_CELLS)
            except:
                pass

        for it in clear_world:
            try:
                cx, cz = world_to_grid(float(it["x"]), float(it["z"]))
                remove_inflated_cell(state["manual_cells"], cx, cz, r=INFLATE_CELLS)
            except:
                pass

        state["need_replan"] = True
        return jsonify({
            "status": "OK",
            "num_manual": len(state["manual_cells"])
        })

# -----------------------------
# 실행
# -----------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Start ")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000)
