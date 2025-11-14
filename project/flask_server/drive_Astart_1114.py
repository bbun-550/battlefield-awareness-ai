from flask import Flask, request, jsonify
import os, math, threading, heapq, time
from typing import Optional, Tuple

# ====================== 테스트 로그 설정 ====================== #
import pandas as pd
import time
from pathlib import Path

# VERSION = "old"                 # 🔸 구버전
# SCENARIO_ID = "SCN03"
# MAP_ID = "NewMap_1107"
# RUN_ID = f"{VERSION}_{int(time.time())}"
# LOG_DIR = Path("./logs")
# LOG_DIR.mkdir(exist_ok=True)
# log_rows = []                   # 프레임별 로그 누적 리스트
# frame_id = 0

# ===== 좌표 안전 언팩 유틸 =====
def _unpack3(v):
    """(x,z) 또는 (x,y,z) 모두 허용. None/에러 시 0으로 보정."""
    if v is None:
        return 0.0, 0.0, 0.0
    try:
        # list/tuple 길이 판단
        if len(v) == 2:
            x, z = v
            return float(x), 0.0, float(z)
        x, y, z = v[:3]
        return float(x), float(y), float(z)
    except Exception:
        return 0.0, 0.0, 0.0

# ======================================================= #



app = Flask(__name__)
SAVE_DIR  = r"C:\Users\SeYun\OneDrive\문서\Tank Challenge"
DEST_FILE = os.path.join(SAVE_DIR, "last_destination.txt")
os.makedirs(SAVE_DIR, exist_ok=True)
MAP_MIN_X, MAP_MAX_X = 0.0, 300.0
MAP_MIN_Z, MAP_MAX_Z = 0.0, 300.0
GRID_RES_M        = 1.0
GRID_ORIGIN_XZ    = (MAP_MIN_X, MAP_MIN_Z)
ALLOW_DIAGONAL    = True
NO_CORNER_CUT     = True
SAFETY_CLEARANCE_M = 2.0
INFLATE_CELLS      = int(math.ceil(SAFETY_CLEARANCE_M / GRID_RES_M))
DYN_TTL_S           = 2.0
ARRIVE_RADIUS_M     = 2.5
ANGLE_DEADZONE_DEG  = 5.0
ROTATE_IN_PLACE_DEG = 45.0
FWD_MIN, FWD_MAX    = 0.25, 1.00
ROT_MIN, ROT_MAX    = 0.35, 1.00
LOOKAHEAD_CELLS     = 2
REPLAN_LOOKAHEAD_K  = 3
EARLY_EVADE          = True
EVADE_CONE_DEG       = 45.0
EVADE_TRIG_DIST      = 25.0
EVADE_KEEP_S         = 1.2
EVADE_MIN_GAP_S      = 0.8
EVADE_LATERAL_M      = 6.0
EVADE_FWD            = 0.7
EVADE_ROT            = 0.95
WALL_ARC_DEG_MIN     = 360.0 / 8.0
WALL_ARC_DIST_MAX    = 25.0
WALL_BIN_DEG         = 1.0
CHORD_RADIUS_M          = WALL_ARC_DIST_MAX
CHORD_BAND_DEG          = 75.0
CHORD_MIN_COUNT         = 8
CHORD_LEN_FRAC_OF_DIAM  = 1.0/8.0
CHORD_LEN_MIN           = 2.0*CHORD_RADIUS_M*CHORD_LEN_FRAC_OF_DIAM
TURN_LEFT_KEY   = "A"
TURN_RIGHT_KEY  = "D"
FORWARD_KEY     = "W"
BACKWARD_KEY    = "S"
FORCE_AD_FLIP   = False
YAW_ZERO_BASIS = "Z0"
YAW_SIGN       = +1
state_lock = threading.Lock()
state = {
    "pos": None, "last_pos": None, "goal": None,
    "yaw_deg": None,
    "raw": {"body_x": None, "body_y": None, "body_z": None},
    "static_cells": set(),
    "manual_cells": set(),
    "dynamic_cells": {},
    "path_cells": [], "path_world": [],
    "last_cell": None, "need_replan": True, "last_replan_ts": 0.0,
    "ahead_min_dist": float("inf"),
    "ahead_left_hits": 0,
    "ahead_right_hits": 0,
    "last_lidar_ts": 0.0,
    "ring_arc_deg": 0.0,
    "ring_arc_center_deg": 0.0,
    "ring_arc_min_dist": float("inf"),
    "chord_len": 0.0,
    "chord_center_s": 0.0,
    "evade_until_ts": 0.0,
    "evade_dir": 0,
    "last_evade_end_ts": 0.0,
    "start_time": None,
}
# # 테스트
# def save_logs_on_exit():
#     """매번 호출 시 고유 파일명으로 로그 저장"""
#     if log_rows:
#         # 매번 호출마다 새 타임스탬프를 생성 (밀리초 단위까지 포함)
#         timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
#         millis = int((time.time() % 1) * 1000)
#         log_path = LOG_DIR / f"log_{VERSION}_{timestamp_str}_{millis}.csv"

#         df = pd.DataFrame(log_rows)
#         df.to_csv(log_path, index=True, encoding="utf-8")

#         print(f"[LOG] Saved {len(log_rows)} frames to {log_path}")
#     else:
#         print("[LOG] No frames to save.")

def normalize_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v
def forward_weight(abs_err: float) -> float:
    t = min(1.0, abs_err / 90.0)
    return FWD_MAX * (1.0 - t) + FWD_MIN * t
def rotate_weight(abs_err: float) -> float:
    t = min(1.0, abs_err / 90.0)
    return ROT_MIN * (1.0 - t) + ROT_MAX * t
def desired_yaw_deg(px: float, pz: float, gx: float, gz: float) -> float:
    return math.degrees(math.atan2(gx - px, gz - pz))
def heading_from_raw_yaw(raw_deg: float):
    th = math.radians(YAW_SIGN * raw_deg)
    if YAW_ZERO_BASIS == "Z0":
        hx = math.sin(th); hz = math.cos(th)
    else:
        hx = math.cos(th); hz = math.sin(th)
    return hx, hz
def canonical_yaw_from_raw(raw_deg: float) -> float:
    hx, hz = heading_from_raw_yaw(raw_deg)
    return math.degrees(math.atan2(hx, hz))
def update_pos_from_payload(payload: dict):
    p = None
    pos = payload.get("position") or payload.get("playerPos")
    if isinstance(pos, dict):
        x = pos.get("x"); z = pos.get("z")
        if x is not None and z is not None:
            p = (float(x), float(z))
    if p is None and ("x" in payload and "z" in payload):
        p = (float(payload["x"]), float(payload["z"]))
    if p is None:
        px = payload.get("Player_Pos_X"); pz = payload.get("Player_Pos_Z")
        if px is not None and pz is not None:
            p = (float(px), float(pz))
    if p is not None:
        state["last_pos"] = state["pos"]
        state["pos"] = p
def update_yaw_from_payload(payload: dict):
    raw_x = None
    for k in ("Player_Body_X", "playerBodyX", "bodyX", "Player_Bodyx"):
        if k in payload:
            try: raw_x = float(payload[k]); break
            except: pass
    if raw_x is None:
        for k in ("yaw","heading","playerYaw","bodyYawDeg"):
            if k in payload:
                try: raw_x = float(payload[k]); break
                except: pass
    if raw_x is None:
        for k in ("Player_Body_Y","playerBodyY","bodyY","Player_Body_Z","playerBodyZ","bodyZ"):
            if k in payload:
                try: raw_x = float(payload[k]); break
                except: pass
    if raw_x is not None:
        state["raw"]["body_x"] = raw_x
        state["yaw_deg"] = canonical_yaw_from_raw(raw_x)
    if state["yaw_deg"] is None and state["last_pos"] and state["pos"]:
        (x0, z0), (x1, z1) = state["last_pos"], state["pos"]
        dx, dz = x1 - x0, z1 - z0
        if abs(dx) + abs(dz) > 1e-4:
            state["yaw_deg"] = math.degrees(math.atan2(dx, dz))
def in_bounds(ix: int, iz: int) -> bool:
    x = ix * GRID_RES_M + GRID_ORIGIN_XZ[0]
    z = iz * GRID_RES_M + GRID_ORIGIN_XZ[1]
    return (MAP_MIN_X - 1e-6) <= x <= (MAP_MAX_X + 1e-6) and (MAP_MIN_Z - 1e-6) <= z <= (MAP_MAX_Z + 1e-6)
def world_to_cell(x: float, z: float):
    ox, oz = GRID_ORIGIN_XZ
    ix = int(round((x - ox) / GRID_RES_M))
    iz = int(round((z - oz) / GRID_RES_M))
    return (ix, iz)
def cell_to_world(ix: int, iz: int):
    ox, oz = GRID_ORIGIN_XZ
    x = ix * GRID_RES_M + ox
    z = iz * GRID_RES_M + oz
    return (x, z)
def blocked_cells(now_ts: Optional[float] = None) -> set:
    dyn = set()
    if now_ts is None:
        now_ts = time.time()
    for cell, exp in list(state["dynamic_cells"].items()):
        if exp >= now_ts:
            dyn.add(cell)
        else:
            state["dynamic_cells"].pop(cell, None)
    return set(state["static_cells"]) | set(state["manual_cells"]) | dyn
def inflate_add_cell(cells_set: set, ix: int, iz: int, r: int = INFLATE_CELLS):

    for dx in range(-r, r + 1):
        for dz in range(-r, r + 1):
            cx, cz = ix + dx, iz + dz
            if in_bounds(cx, cz):
                cells_set.add((cx, cz))
def remove_inflated_cell(cells_set: set, ix: int, iz: int, r: int = INFLATE_CELLS):

    for dx in range(-r, r + 1):
        for dz in range(-r, r + 1):
            cx, cz = ix + dx, iz + dz
            cells_set.discard((cx, cz))
def add_dynamic_world(x: float, z: float, now_ts: float, ttl: float = DYN_TTL_S, r: int = INFLATE_CELLS):
    ix, iz = world_to_cell(x, z)
    exp = now_ts + ttl
    for dx in range(-r, r + 1):
        for dz in range(-r, r + 1):
            cx, cz = ix + dx, iz + dz
            if in_bounds(cx, cz):
                state["dynamic_cells"][(cx, cz)] = exp
def replace_static_from_rects(rects: list, r: int = INFLATE_CELLS):
    new_set = set()
    for rect in rects:
        try:
            x0 = float(rect["x_min"]); x1 = float(rect["x_max"])
            z0 = float(rect["z_min"]); z1 = float(rect["z_max"])
            xmin, xmax = sorted([x0, x1]); zmin, zmax = sorted([z0, z1])
        except:
            continue
        c0 = world_to_cell(xmin, zmin)
        c1 = world_to_cell(xmax, zmax)
        ix0, iz0 = min(c0[0], c1[0]), min(c0[1], c1[1])
        ix1, iz1 = max(c0[0], c1[0]), max(c0[1], c1[1])
        for ix in range(ix0 - r, ix1 + r + 1):
            for iz in range(iz0 - r, iz1 + r + 1):
                if in_bounds(ix, iz):
                    new_set.add((ix, iz))
    state["static_cells"] = new_set
def astar_search(start_cell, goal_cell, obstacles: set, allow_diagonal=True, no_corner_cut=True, max_nodes=200000):
    if start_cell == goal_cell:
        return [start_cell]
    def free(ix, iz) -> bool:
        return in_bounds(ix, iz) and ((ix, iz) not in obstacles)
    def neighbors(s):
        ix, iz = s
        dirs4 = [(1,0),(-1,0),(0,1),(0,-1)]
        neigh = dirs4[:]
        if allow_diagonal:
            neigh += [(1,1),(1,-1),(-1,1),(-1,-1)]
        for dx, dz in neigh:
            nx, nz = ix + dx, iz + dz
            if not free(nx, nz):
                continue
            if allow_diagonal and no_corner_cut and (dx != 0 and dz != 0):
                if (not free(ix + dx, iz)) or (not free(ix, iz + dz)):
                    continue
            yield (nx, nz)
    def step_cost(a, b):
        ax, az = a; bx, bz = b
        diag = (ax != bx and az != bz)
        base = math.sqrt(2.0) if diag else 1.0
        return base * GRID_RES_M
    def h(a, b):
        ax, az = a; bx, bz = b
        return math.hypot(ax - bx, az - bz) * GRID_RES_M
    if (start_cell in obstacles) or (goal_cell in obstacles) or (not in_bounds(*start_cell)) or (not in_bounds(*goal_cell)):
        return []
    open_h = []
    g = {start_cell: 0.0}
    f = {start_cell: h(start_cell, goal_cell)}
    came = {}
    counter = 0
    heapq.heappush(open_h, (f[start_cell], 0.0, counter, start_cell))
    in_open = {start_cell}
    visited = 0
    while open_h and visited < max_nodes:
        _, _, _, cur = heapq.heappop(open_h)
        in_open.discard(cur)
        visited += 1
        if cur == goal_cell:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path
        for nb in neighbors(cur):
            tentative = g[cur] + step_cost(cur, nb)
            if tentative < g.get(nb, float("inf")):
                came[nb] = cur
                g[nb] = tentative
                f[nb] = tentative + h(nb, goal_cell)
                if nb not in in_open:
                    counter += 1
                    heapq.heappush(open_h, (f[nb], g[nb], counter, nb))
                    in_open.add(nb)
    return []
def ensure_path_astar(now_ts: Optional[float] = None):
    pos = state["pos"]; goal = state["goal"]
    if pos is None or goal is None:
        return
    s = world_to_cell(pos[0], pos[1])
    g = world_to_cell(goal[0], goal[1])
    moved_cell = (state["last_cell"] != s)
    if moved_cell:
        state["last_cell"] = s
    if now_ts is None:
        now_ts = time.time()
    obs = blocked_cells(now_ts)
    if state["need_replan"] or moved_cell or not state["path_cells"]:
        cells = astar_search(s, g, obs, allow_diagonal=ALLOW_DIAGONAL, no_corner_cut=NO_CORNER_CUT)
        state["path_cells"] = cells
        state["path_world"] = [cell_to_world(ix, iz) for (ix, iz) in cells]
        state["need_replan"] = False
        state["last_replan_ts"] = now_ts
def pick_lookahead_target(px, pz):
    if not state["path_world"]:
        return state["goal"]
    best_i, best_d = 0, float("inf")
    for i, (wx, wz) in enumerate(state["path_world"]):
        d = (wx - px)**2 + (wz - pz)**2
        if d < best_d:
            best_d = d; best_i = i
    idx = min(best_i + LOOKAHEAD_CELLS, len(state["path_world"]) - 1)
    return state["path_world"][idx]
def path_blocked_ahead(now_ts: float) -> bool:
    if not state["path_cells"]:
        return True
    obs = blocked_cells(now_ts)
    k = min(REPLAN_LOOKAHEAD_K, len(state["path_cells"]))
    for i in range(k):
        if state["path_cells"][i] in obs:
            return True
    return False
def check_immediate_danger(player_pos_w: Tuple[float, float], player_yaw_deg: float) -> dict:
    danger = {'front': False, 'front_left': False, 'front_right': False}
    all_blocked = blocked_cells(time.time())
    if not all_blocked:
        return danger
    player_yaw_rad = math.radians(player_yaw_deg)
    check_dist = SAFETY_CLEARANCE_M + GRID_RES_M * 0.5
    angles_to_check_deg = {
        'front': 0, 'front_left': -30, 'front_right': 30
    }
    for direction, angle_offset_deg in angles_to_check_deg.items():
        check_angle_rad = player_yaw_rad + math.radians(angle_offset_deg)
        check_pos_w = (
            player_pos_w[0] + check_dist * math.sin(check_angle_rad),
            player_pos_w[1] + check_dist * math.cos(check_angle_rad)
        )
        check_pos_g = world_to_cell(check_pos_w[0], check_pos_w[1])
        if check_pos_g in all_blocked:
            danger[direction] = True
    return danger
@app.route('/set_destination', methods=['POST'])
def set_destination():
    data = request.get_json(force=True)
    gx = gz = None
    try:
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
            return jsonify({"status":"ERROR","message":"Missing x/z"}), 400
    except Exception as e:
        return jsonify({"status":"ERROR","message":f"Invalid destination: {e}"}), 400
    with state_lock:
        state["goal"] = (gx, gz)
        state["need_replan"] = True
        try:
            with open(DEST_FILE, "w", encoding="utf-8") as f:
                f.write(f"{gx},{gz}\n")
        except Exception as fe:
            print(f"⚠️ 목적지 파일 저장 실패: {fe}")
    print(f"🎯 Destination set: ({gx:.3f}, {gz:.3f})")
    return jsonify({"status":"OK","destination":{"x":gx,"z":gz}})
@app.route('/info', methods=['POST'])
def info():
    data = request.get_json(force=True) or {}
    now_ts = float(data.get("time", time.time()))
    with state_lock:
        update_pos_from_payload(data)
        update_yaw_from_payload(data)
        px, pz = (state["pos"] or (None, None))
        cy = state["yaw_deg"]
        state["ahead_min_dist"] = float("inf")
        state["ahead_left_hits"] = 0
        state["ahead_right_hits"] = 0
        bins = [False] * 360
        bin_min_dist = [float("inf")] * 360
        state["ring_arc_deg"] = 0.0
        state["ring_arc_center_deg"] = 0.0
        state["ring_arc_min_dist"] = float("inf")
        state["chord_len"] = 0.0
        state["chord_center_s"] = 0.0
        s_values = []
        pts = data.get("lidarPoints") or []
        for p in pts:
            try:
                if not p.get("isDetected", False):
                    continue
                pos = p.get("position") or {}
                x = float(pos.get("x")); z = float(pos.get("z"))
            except:
                continue
            add_dynamic_world(x, z, now_ts, ttl=DYN_TTL_S, r=INFLATE_CELLS)
            if px is not None and cy is not None:
                d = math.hypot(x - px, z - pz)
                bearing = desired_yaw_deg(px, pz, x, z)
                rel = normalize_deg(bearing - cy)
                if abs(rel) <= EVADE_CONE_DEG:
                    state["ahead_min_dist"] = min(state["ahead_min_dist"], d)
                    if rel > 0: state["ahead_right_hits"] += 1
                    else:       state["ahead_left_hits"]  += 1
                if d <= WALL_ARC_DIST_MAX:
                    bi = int(round((rel + 180.0))) % 360
                    bins[bi] = True
                    if d < bin_min_dist[bi]:
                        bin_min_dist[bi] = d
                if d <= CHORD_RADIUS_M and abs(rel) <= CHORD_BAND_DEG:
                    yaw_rad = math.radians(cy)
                    hx = math.sin(yaw_rad); hz = math.cos(yaw_rad)
                    rx, rz = x - px, z - pz
                    f =  rx*hx + rz*hz
                    s =  rx*hz - rz*hx
                    if f >= 0:
                        s_values.append(s)
        if any(bins):
            best_len = 0; best_start = 0
            cur_len = 0; cur_start = 0
            for i in range(360 * 2):
                if bins[i % 360]:
                    if cur_len == 0: cur_start = i
                    cur_len += 1
                    if cur_len > best_len:
                        best_len = cur_len; best_start = cur_start
                else:
                    cur_len = 0
            best_len = min(best_len, 360)
            arc_deg = best_len * WALL_BIN_DEG
            if arc_deg > 0:
                start_idx = best_start % 360
                end_idx   = (best_start + best_len - 1) % 360
                if start_idx <= end_idx:
                    center_idx = (start_idx + end_idx) / 2.0
                else:
                    span = (end_idx + 360) - start_idx
                    center_idx = (start_idx + span / 2.0) % 360
                center_rel = center_idx - 180.0
                min_d = float("inf")
                for k in range(best_len):
                    idx = (best_start + k) % 360
                    if bins[idx] and bin_min_dist[idx] < min_d:
                        min_d = bin_min_dist[idx]
                state["ring_arc_deg"] = arc_deg
                state["ring_arc_center_deg"] = center_rel
                state["ring_arc_min_dist"] = min_d
        if len(s_values) >= CHORD_MIN_COUNT:
            s_min = min(s_values); s_max = max(s_values)
            state["chord_len"] = (s_max - s_min)
            state["chord_center_s"] = (s_min + s_max) / 2.0
        state["last_lidar_ts"] = now_ts
        _ = blocked_cells(now_ts)
    return jsonify({"status":"OK"})
@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    data = request.get_json(force=True) or {}
    rects = data.get("obstacles", [])
    with state_lock:
        replace_static_from_rects(rects, r=INFLATE_CELLS)
        state["need_replan"] = True
        ensure_path_astar(time.time())
        return jsonify({
            "status": "OK",
            "num_static": len(state["static_cells"]),
            "path_len": len(state["path_world"])
        })
@app.route('/update_occupancy', methods=['POST'])
def update_occupancy():
    data = request.get_json(force=True) or {}
    with state_lock:
        blocked = data.get("blocked", [])
        clear = data.get("clear", [])
        blocked_world = data.get("blocked_world", [])
        clear_world = data.get("clear_world", [])
        for it in blocked:
            try:
                cell = (int(it[0]), int(it[1]))
                inflate_add_cell(state["manual_cells"], cell[0], cell[1], r=INFLATE_CELLS)
            except: pass
        for it in clear:
            try:
                cell = (int(it[0]), int(it[1]))
                remove_inflated_cell(state["manual_cells"], cell[0], cell[1], r=INFLATE_CELLS)
            except: pass
        for it in blocked_world:
            try:
                cx, cz = world_to_cell(float(it["x"]), float(it["z"]))
                inflate_add_cell(state["manual_cells"], cx, cz, r=INFLATE_CELLS)
            except: pass
        for it in clear_world:
            try:
                cx, cz = world_to_cell(float(it["x"]), float(it["z"]))
                remove_inflated_cell(state["manual_cells"], cx, cz, r=INFLATE_CELLS)
            except: pass
        state["need_replan"] = True
        ensure_path_astar(time.time())
        return jsonify({
            "status":"OK",
            "num_manual": len(state["manual_cells"]),
            "path_len": len(state["path_world"])
        })
@app.route('/get_action', methods=['POST'])
def get_action():
    data = request.get_json(force=True) or {}
    now_ts = float(data.get("time", time.time()))
    with state_lock:
        update_pos_from_payload(data)
        update_yaw_from_payload(data)
        pos, goal, cur_yaw = state["pos"], state["goal"], state["yaw_deg"]
        ahead_min_dist   = state["ahead_min_dist"]
        left_hits        = state["ahead_left_hits"]
        right_hits       = state["ahead_right_hits"]
        last_lidar_ts    = state["last_lidar_ts"]
        evade_until_ts   = state["evade_until_ts"]
        evade_dir        = state["evade_dir"]
        last_evade_end   = state["last_evade_end_ts"]
        ring_arc_deg     = state["ring_arc_deg"]
        ring_arc_center  = state["ring_arc_center_deg"]
        ring_arc_min_d   = state["ring_arc_min_dist"]
        chord_len        = state["chord_len"]
        chord_center_s   = state["chord_center_s"]
    if pos is None or goal is None:
        return jsonify({
            "moveWS": {"command":"STOP","weight":1.0},
            "moveAD": {"command":"","weight":0.0},
            "turretQE": {"command":"","weight":0.0},
            "turretRF": {"command":"","weight":0.0},
            "fire": False
        })
    px, pz = pos
    gx, gz = goal
    dist_goal = math.hypot(gx - px, gz - pz)
    if dist_goal <= ARRIVE_RADIUS_M:
        with state_lock:
            state["evade_until_ts"] = 0.0
            start_time = state.get("start_time")
        if start_time is not None:
            elapsed = time.time() - start_time
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            print(f"✅✅✅ !!!! 에이 스타 !!!! Arrived in {minutes}m {seconds:.2f}s ✅✅✅")
            print(f"✅✅✅✅✅✅✅ 1111111111 Arrived. 1111111111 pos=({px:.2f},{pz:.2f}) goal=({gx:.2f},{gz:.2f}) ✅✅✅✅✅✅")

        else:
            print(f"✅ 222222222 Arrived. 222222222 pos=({px:.2f},{pz:.2f}) goal=({gx:.2f},{gz:.2f})")

        return  jsonify({
            "moveWS": {"command":"STOP","weight":1.0},
            "moveAD": {"command":"","weight":0.0},
            "turretQE": {"command":"","weight":0.0},
            "turretRF": {"command":"","weight":0.0},
            "fire": False
        })
    if cur_yaw is not None:
        danger_info = check_immediate_danger(pos, cur_yaw)
        if danger_info['front']:
            ws_cmd_str, ad_cmd_str = FORWARD_KEY, ""
            if not danger_info['front_right']:
                ad_cmd_str = TURN_RIGHT_KEY
            elif not danger_info['front_left']:
                ad_cmd_str = TURN_LEFT_KEY
            else:
                ws_cmd_str = BACKWARD_KEY
                ad_cmd_str = TURN_RIGHT_KEY
            print(f"⚠️ Immediate Danger! Evading. Danger: {danger_info}")
            return jsonify({
                "moveWS": {"command": ws_cmd_str, "weight": FWD_MAX},
                "moveAD": {"command": ad_cmd_str, "weight": ROT_MAX},
                "turretQE": {"command":"", "weight":0.0},
                "turretRF": {"command":"", "weight":0.0},
                "fire": False
            })
    with state_lock:
        ensure_path_astar(now_ts)
        if path_blocked_ahead(now_ts):
            state["need_replan"] = True
            ensure_path_astar(now_ts)
        path_world = list(state["path_world"])
    evading = False
    chosen_dir = 0
    recent_lidar = (now_ts - last_lidar_ts) <= 1.0
    active = now_ts < evade_until_ts
    can_retrigger = (now_ts - last_evade_end) >= EVADE_MIN_GAP_S
    if EARLY_EVADE:
        if active:
            evading = True
            chosen_dir = evade_dir
        else:
            front_trigger = recent_lidar and (ahead_min_dist <= EVADE_TRIG_DIST)
            wall_trigger  = recent_lidar and (ring_arc_deg >= WALL_ARC_DEG_MIN) and (ring_arc_min_d <= WALL_ARC_DIST_MAX)
            chord_trigger = recent_lidar and (chord_len >= CHORD_LEN_MIN)

            if (front_trigger or wall_trigger or chord_trigger) and can_retrigger:
                if chord_trigger:
                    chosen_dir = -1 if chord_center_s > 0 else +1
                elif wall_trigger:
                    chosen_dir = -1 if ring_arc_center > 0 else +1
                else:
                    chosen_dir = +1 if right_hits <= left_hits else -1

                with state_lock:
                    state["evade_dir"] = chosen_dir
                    state["evade_until_ts"] = now_ts + EVADE_KEEP_S
                    state["need_replan"] = True
                evading = True
        if not evading and (evade_until_ts > 0.0) and (now_ts >= evade_until_ts):
            with state_lock:
                state["evade_until_ts"] = 0.0
                state["last_evade_end_ts"] = now_ts
    if evading and cur_yaw is not None:
        yaw_rad = math.radians(cur_yaw)
        hx = math.sin(yaw_rad); hz = math.cos(yaw_rad)
        lateral = max(EVADE_LATERAL_M, SAFETY_CLEARANCE_M)
        if chosen_dir >= 0:
            sx, sz = hz, -hx
        else:
            sx, sz = -hz, hx
        tgt_x = px + sx * lateral
        tgt_z = pz + sz * lateral
    else:
        if not path_world:
            tgt_x, tgt_z = gx, gz
        else:
            tgt_x, tgt_z = pick_lookahead_target(px, pz)
    tgt = desired_yaw_deg(px, pz, tgt_x, tgt_z)
    cur = cur_yaw if cur_yaw is not None else tgt
    diff = normalize_deg(tgt - cur)
    abs_diff = abs(diff)
    left_key, right_key = (TURN_RIGHT_KEY, TURN_LEFT_KEY) if FORCE_AD_FLIP else (TURN_LEFT_KEY, TURN_RIGHT_KEY)
    turn_key = right_key if (evading and chosen_dir > 0) or (not evading and diff > 0) else left_key
    if evading:
        rot_w = EVADE_ROT
        fwd_w = EVADE_FWD
    else:
        rot_w = rotate_weight(abs_diff)
        fwd_w = forward_weight(abs_diff)
    if evading:
        cmd_ws = {"command":FORWARD_KEY,"weight":fwd_w}
        cmd_ad = {"command":turn_key, "weight":rot_w}
    else:
        if abs_diff > ROTATE_IN_PLACE_DEG:
            cmd_ws = {"command":"STOP","weight":0.0}
            cmd_ad = {"command":turn_key, "weight":rot_w}
        elif abs_diff <= ANGLE_DEADZONE_DEG:
            cmd_ws = {"command":FORWARD_KEY,"weight":FWD_MAX}
            cmd_ad = {"command":"", "weight":0.0}
        else:
            cmd_ws = {"command":FORWARD_KEY,"weight":fwd_w}
            cmd_ad = {"command":turn_key, "weight":rot_w}
    dbg = {
        "pos": {"x": px, "z": pz},
        "goal": {"x": gx, "z": gz},
        "dist_goal": round(dist_goal, 3),
        "yaw_deg": None if cur_yaw is None else round(cur, 3),
        "lookahead_or_evade_target": {"x": tgt_x, "z": tgt_z},
        "path_len": len(path_world),
        "evading": evading,
        "evade_dir": chosen_dir,
        "ahead_min_dist": None if math.isinf(ahead_min_dist) else round(ahead_min_dist, 2),
        "safety_clearance_m": SAFETY_CLEARANCE_M,
        "inflate_cells": INFLATE_CELLS
    }
    # save_logs_on_exit()
    # print(f"🧭 pos=({px:.2f},{pz:.2f}) yaw={cur:.1f}°  "
    #       f"goal=({gx:.2f},{gz:.2f})  dist={dist_goal:.2f}  "
    #       f"evading={evading}  clear={SAFETY_CLEARANCE_M}m  infl={INFLATE_CELLS}")

# 테스트
    # 🧩 로그 추가 블록 (return 직전)
    # global frame_id

    # try:
    #     timestamp_ms = int(time.time() * 1000)

    #     # 안전 언팩
    #     pos_x, pos_y, pos_z       = _unpack3(state.get("pos"))
    #     start_x, start_y, start_z = _unpack3(state.get("start"))
    #     goal_x, goal_y, goal_z    = _unpack3(state.get("goal"))
    #     yaw_deg                   = float(state.get("yaw_deg") or 0.0)

    #     # 수치/상태 (state 없으면 기본값)
    #     speed_mps             = float(state.get("speed_mps") or 0.0)
    #     obstacle_count        = int(state.get("obstacle_count") or 0)
    #     min_obstacle_dist_m   = state.get("min_obstacle_dist_m")  # None 허용
    #     replanned             = bool(state.get("replanned") or False)
    #     replan_latency_ms     = int(state.get("replan_latency_ms") or 0)
    #     collision             = bool(state.get("collision") or False)
    #     collision_type        = state.get("collision_type")       # None 허용
    #     event_str             = state.get("event")                # None 허용
    #     seed_val              = state.get("seed")                 # None 허용

    #     # 액션 처리: dict 형식이면 command/weight 분리
    #     _aws = state.get("action_ws")
    #     _aad = state.get("action_ad")

    #     if isinstance(_aws, dict):
    #         action_ws        = _aws.get("command")
    #         action_ws_weight = float(_aws.get("weight", 0.0))
    #     else:
    #         action_ws        = _aws
    #         action_ws_weight = float(state.get("action_ws_weight") or 0.0)

    #     if isinstance(_aad, dict):
    #         action_ad        = _aad.get("command")
    #         action_ad_weight = float(_aad.get("weight", 0.0))
    #     else:
    #         action_ad        = _aad
    #         action_ad_weight = float(state.get("action_ad_weight") or 0.0)

    #     # 서버 타이밍 (없으면 0, RTT는 req+resp)
    #     server_req_ms   = int(state.get("server_req_ms") or 0)
    #     server_resp_ms  = int(state.get("server_resp_ms") or 0)
    #     server_rtt_ms   = int(state.get("server_rtt_ms") or (server_req_ms + server_resp_ms))
    #     fps_val         = float(state.get("fps") or 0.0)

    #     log_rows.append({
    #         "timestamp_ms":          timestamp_ms,
    #         "frame_id":              frame_id,
    #         "scenario_id":           SCENARIO_ID,
    #         "map_id":                MAP_ID,
    #         "version":               VERSION,
    #         "run_id":                RUN_ID,
    #         "seed":                  seed_val,

    #         "start_x":               start_x,
    #         "start_y":               start_y,
    #         "start_z":               start_z,
    #         "goal_x":                goal_x,
    #         "goal_y":                goal_y,
    #         "goal_z":                goal_z,
    #         "pos_x":                 pos_x,
    #         "pos_y":                 pos_y,
    #         "pos_z":                 pos_z,
    #         "yaw_deg":               yaw_deg,
    #         "speed_mps":             speed_mps,

    #         "action_ws":             action_ws,
    #         "action_ad":             action_ad,
    #         "action_ws_weight":      action_ws_weight,
    #         "action_ad_weight":      action_ad_weight,

    #         "obstacle_count":        obstacle_count,
    #         "min_obstacle_dist_m":   min_obstacle_dist_m,

    #         "replanned":             replanned,
    #         "replan_latency_ms":     replan_latency_ms,

    #         "collision":             collision,
    #         "collision_type":        collision_type,
    #         "event":                 event_str,

    #         "server_rtt_ms":         server_rtt_ms,
    #         "server_req_ms":         server_req_ms,
    #         "server_resp_ms":        server_resp_ms,

    #         "fps":                   fps_val,
    #     })

    #     frame_id += 1

    # except Exception as e:
    #     print(f"[WARN][LOG] Failed to append log: {e}")
    return jsonify({
        "moveWS": cmd_ws,
        "moveAD": cmd_ad,
        "turretQE": {"command":"", "weight":0.0},
        "turretRF": {"command":"", "weight":0.0},
        "fire": False,
        "debug": dbg
    })
@app.route('/debug_state', methods=['GET'])
def debug_state():
    with state_lock:
        p = state["pos"]; g = state["goal"]
        path_w = [{"x": x, "z": z} for (x, z) in state["path_world"][:200]]
        now_ts = time.time()
        obs = blocked_cells(now_ts)
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
            "grid_res_m": GRID_RES_M
        })
@app.route('/init', methods=['GET'])
def init():
    cfg = {
        "startMode": "pause",
        "blStartX": 60, "blStartY": 10, "blStartZ": 27.23,
        "rdStartX": 59, "rdStartY": 10, "rdStartZ": 280,
        "trackingMode": True,
        "detactMode": False,
        "logMode": True,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": True,
        "saveLidarData": True,
        "lux": 30000
    }
    print("🛠️ /init OK ->", cfg)
    return jsonify(cfg)
@app.route('/start', methods=['GET'])
def start():
    with state_lock:
     state["start_time"] = time.time()
    print("🚀 /start - Timer started!")
    return jsonify({"control": ""})
import atexit

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

