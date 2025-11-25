# -*- coding: utf-8 -*-
# ============================================================
# [통합] A* Pathfinding + Pure Pursuit + Event Handling
# ============================================================

from flask import Flask, request, jsonify
import math, os, time, json, heapq
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
app = Flask(__name__)
model = YOLO('5cls_v6_case2_best.pt')

# 파일 경로 (사용자 환경에 맞게 확인 필요)
LOG_FILE    = r"C:\Users\cheei\Documents\Tank Challenge\log_data\tank_info_log.txt"
OUTPUT_CSV  = r"C:\Users\cheei\Documents\Tank Challenge\log_data\output.csv"
MAP_FILE    = r"11_20.map"

# ------------------------------------------------------------
# WAYPOINT (주요 경유지)
# ------------------------------------------------------------
WAYPOINTS = [
    (66.08732, 45.9379),   # [0] 회전 + 3초 정지
    # (100.425, 106.330),    # [1]
    # (81.277, 99.007),      # [2]
    # (90.565, 130.413),     # [3]
    # (111.759, 172.892),    # [4]
    (120.389, 181.441),    # [5] → 포격모드 ON
    (139.722, 258.477),    # [6]
    (128.686, 291.084),    # [7]
    (35.982, 284.198)      # [8]
]

# ------------------------------------------------------------
# Global State Variables
# ------------------------------------------------------------
FINAL_PATH = []         # A*로 생성된 전체 경로 (점들의 리스트)
path_generated = False  # 경로 생성 여부 체크

current_key_wp_index = 0  # WAYPOINTS 리스트의 몇 번째를 향해 가고 있는지 (이벤트 체크용)
wait_start_time = None

# 포격 관련
FIRE_MODE       = False
FIRE_COUNT      = 0
RECENTER_TURRET = False
FIRE_AIM_START  = None

# 맵 장애물 정보
TANK_OBJS = []

# ------------------------------------------------------------
# A* Algorithm Implementation
# ------------------------------------------------------------
GRID_SIZE = 1.0       # 격자 크기 (1m)
OBSTACLE_MARGIN = 4.0 # 장애물 안전 거리 (탱크가 부딪히지 않게 여유 둠)

def world_to_grid(x, z):
    return int(round(x / GRID_SIZE)), int(round(z / GRID_SIZE))

def grid_to_world(r, c):
    return float(r) * GRID_SIZE, float(c) * GRID_SIZE

def get_blocked_cells(obstacles):
    blocked = set()
    margin_steps = int(math.ceil(OBSTACLE_MARGIN / GRID_SIZE))
    print(f"🛠️ Building Obstacle Map (Margin: {OBSTACLE_MARGIN}m)...")

    for ob in obstacles:
        ox, oz = ob['x'], ob['z']
        gr, gc = world_to_grid(ox, oz)
        # 장애물 주변 마킹
        for r in range(gr - margin_steps, gr + margin_steps + 1):
            for c in range(gc - margin_steps, gc + margin_steps + 1):
                wx, wz = grid_to_world(r, c)
                if math.hypot(wx - ox, wz - oz) <= OBSTACLE_MARGIN:
                    blocked.add((r, c))
    return blocked

def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def a_star_search(start_pos, end_pos, blocked_cells):
    start_node = world_to_grid(*start_pos)
    end_node   = world_to_grid(*end_pos)

    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    open_set = []
    heapq.heappush(open_set, (0, start_node))

    came_from = {}
    g_score = {start_node: 0}
    f_score = {start_node: heuristic(start_node, end_node)}

    best_node = start_node # 만약 길을 못 찾으면 가장 가까운 곳이라도 가기 위함
    min_dist_to_goal = heuristic(start_node, end_node)

    while open_set:
        _, current = heapq.heappop(open_set)

        # 목표 근처 도착 (2칸 이내)
        dist = heuristic(current, end_node)
        if dist < min_dist_to_goal:
            min_dist_to_goal = dist
            best_node = current

        if dist < 2.0:
            path = []
            while current in came_from:
                path.append(grid_to_world(*current))
                current = came_from[current]
            path.append(start_pos)
            path.reverse()
            path.append(end_pos)
            return path

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor in blocked_cells: continue

            move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
            tentative_g = g_score[current] + move_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, end_node)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    print("⚠️ A* Path Not Found to exact target. Using closest approach.")
    # 경로를 못 찾았으면 start -> ... -> best_node -> end (직선) 라도 반환
    path = []
    curr = best_node
    while curr in came_from:
        path.append(grid_to_world(*curr))
        curr = came_from[curr]
    path.append(start_pos)
    path.reverse()
    path.append(end_pos)
    return path

def generate_full_path(start_x, start_z):
    """ 전체 경로 생성기 """
    global FINAL_PATH, WAYPOINTS, TANK_OBJS

    print("🗺️ Generating Full A* Path...")
    blocked = get_blocked_cells(TANK_OBJS)
    full_path = [(start_x, start_z)]
    current_pos = (start_x, start_z)

    for i, wp in enumerate(WAYPOINTS):
        print(f"   Calculating Leg {i}: {current_pos} -> {wp}")
        segment = a_star_search(current_pos, wp, blocked)
        if full_path:
            full_path.extend(segment[1:])
        else:
            full_path.extend(segment)
        current_pos = wp

    FINAL_PATH = full_path
    print(f"✅ Path Generation Complete! Total Nodes: {len(FINAL_PATH)}")

# ------------------------------------------------------------
# MAP Load
# ------------------------------------------------------------
def load_map():
    global TANK_OBJS
    TANK_OBJS = []
    if not os.path.exists(MAP_FILE):
        print("MAP not found:", MAP_FILE)
        return
    with open(MAP_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    for ob in data.get("obstacles", []):
        name = str(ob.get("prefabName", "")).lower()
        if name.startswith("tank"):
            pos = ob.get("position", {})
            TANK_OBJS.append({
                "name": ob.get("prefabName", "Tank"),
                "x": float(pos.get("x", 0.0)),
                "y": float(pos.get("y", 0.0)),
                "z": float(pos.get("z", 0.0))
            })
    print(f"Loaded TANK obstacles: {len(TANK_OBJS)}")

load_map()

# ------------------------------------------------------------
# Util & Pure Pursuit Helper
# ------------------------------------------------------------
def normalize(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0

def read_body_yaw_from_log():
    if not os.path.exists(LOG_FILE): return None
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines: return None
        last = lines[-1]
        for token in last.replace("{", " ").replace("}", " ").split(","):
            if "Player_Body_X" in token:
                return float(token.split(":")[1])
    except: pass
    return None

def get_lookahead_target_from_path(px, pz, lookahead=6.0):
    """ A*로 생성된 FINAL_PATH에서 Lookahead 지점 찾기 """
    global FINAL_PATH
    if not FINAL_PATH: return (px, pz)

    closest_idx = 0
    min_dist = 9999.0

    # 1. 경로상에서 나와 가장 가까운 점 찾기 (검색 최적화를 위해 전체 탐색 대신 근처 탐색 가능하지만 여기선 전체 탐색)
    # 경로가 길어지면 성능 이슈가 있을 수 있으나 1000개 미만은 순식간임.
    for i, (nx, nz) in enumerate(FINAL_PATH):
        d = math.hypot(nx - px, nz - pz)
        if d < min_dist:
            min_dist = d
            closest_idx = i

    # 2. 거기서부터 lookahead 거리만큼 떨어진 점 찾기
    for i in range(closest_idx, len(FINAL_PATH)):
        nx, nz = FINAL_PATH[i]
        d = math.hypot(nx - px, nz - pz)
        if d >= lookahead:
            return (nx, nz)

    return FINAL_PATH[-1] # 끝까지 안 나오면 마지막 점 리턴

# ------------------------------------------------------------
# 포격 계산
# ------------------------------------------------------------
MIN_PITCH_CFG, MAX_PITCH_CFG = -30.0, 10.0
V_INIT, G, MAX_RANGE, H_OFFSET = 58.0, 9.81, 130.0, 2.1

def pick_closest(px, pz):
    best, bd = None, 9999.0
    for ob in TANK_OBJS:
        d = math.hypot(ob["x"] - px, ob["z"] - pz)
        if d < bd: bd, best = d, ob
    return best, bd

def ballistic_pitch(sx, sy, sz, tx, ty, tz):
    dx, dy, dz = tx - sx, ty - sy, tz - sz
    H = math.hypot(dx, dz)
    if H < 1e-6: return False, None
    v2 = V_INIT**2
    term = v2**2 - G * (G * H**2 + 2.0 * dy * v2)
    if term < 0: return False, None
    r = math.sqrt(term)
    return True, math.degrees(min(math.atan((v2 + r) / (G * H)), math.atan((v2 - r) / (G * H))))

def angle_from_csv(d):
    if not os.path.exists(OUTPUT_CSV): return False, None
    df = pd.read_csv(OUTPUT_CSV)
    arr = df.to_numpy()
    ang, Z = arr[:, 0], arr[:, 3]
    idx = np.argsort(Z)
    if d < Z[idx][0] or d > Z[idx][-1]: return False, None
    return True, max(MIN_PITCH_CFG, min(MAX_PITCH_CFG, float(np.interp(d, Z[idx], ang[idx]))))

def compute_solution(px, py, pz, tx, ty):
    tgt, dist = pick_closest(px, pz)
    if not tgt or dist > MAX_RANGE: return {"ok": False}
    dyaw = math.degrees(math.atan2(tgt["x"] - px, tgt["z"] - pz))
    ok, bp = ballistic_pitch(px, py + H_OFFSET, pz, tgt["x"], tgt["y"], tgt["z"])
    base = bp if ok else (angle_from_csv(dist)[1] if angle_from_csv(dist)[0] else math.degrees(math.atan2(tgt["y"] - (py + H_OFFSET), dist)))
    return {"ok": True, "yaw": normalize(dyaw), "pitch": max(MIN_PITCH_CFG, min(MAX_PITCH_CFG, base))}

def turret_ctrl(cx, cy, tx, ty):
    ex, ey = normalize(tx - cx), ty - cy
    return {"QE": {"command": "E" if ex > 0 else "Q" if ex < 0 else "", "weight": min(abs(ex) * 0.15, 1.0)},
            "RF": {"command": "R" if ey > 0 else "F" if ey < 0 else "", "weight": min(abs(ey) * 0.45, 1.0)}, "ex": ex, "ey": ey}

def aim_good_enough(ex, ey): return abs(ex) < 3.0 and abs(ey) < 3.0

# ------------------------------------------------------------
# GET_ACTION (MAIN LOGIC)
# ------------------------------------------------------------
FIRST_FIRE_DELAY = 0.6

@app.route("/get_action", methods=["POST"])
def get_action():
    global current_key_wp_index, FIRE_MODE, FIRE_COUNT, FINAL_PATH, path_generated
    global RECENTER_TURRET, wait_start_time, FIRE_AIM_START

    req    = request.get_json(force=True) or {}
    pos    = req.get("position", {})
    turret = req.get("turret", {})
    px, py, pz = float(pos.get("x", 0)), float(pos.get("y", 0)), float(pos.get("z", 0))
    tx, ty = float(turret.get("x", 0)), float(turret.get("y", 0))

    body_yaw = read_body_yaw_from_log()
    if body_yaw is None: body_yaw = tx

    # ---------------------------------------------
    # 0. 초기화: A* 경로 생성 (최초 1회)
    # ---------------------------------------------
    if not path_generated:
        generate_full_path(px, pz)
        path_generated = True
        current_key_wp_index = 0

    # ---------------------------------------------
    # 1. 포격 모드 (최우선)
    # ---------------------------------------------
    if FIRE_MODE:
        sol = compute_solution(px, py, pz, tx, ty)
        if not sol["ok"]: # 타겟 없으면 정지
            return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "moveAD": {"command": "", "weight": 0}, "turretQE": {"command": "", "weight": 0}, "turretRF": {"command": "", "weight": 0}, "fire": False})

        ctrl = turret_ctrl(tx, ty, sol["yaw"], sol["pitch"])
        fire = False
        if aim_good_enough(ctrl["ex"], ctrl["ey"]):
            if FIRE_AIM_START is None: FIRE_AIM_START = time.time()
            if time.time() - FIRE_AIM_START >= FIRST_FIRE_DELAY: fire = True
        else:
            FIRE_AIM_START = None

        return jsonify({
            "moveWS": {"command": "STOP", "weight": 1}, "moveAD": {"command": "", "weight": 0},
            "turretQE": ctrl["QE"], "turretRF": ctrl["RF"], "fire": fire
        })

    # ---------------------------------------------
    # 2. 포격 후 포탑 복귀
    # ---------------------------------------------
    if RECENTER_TURRET:
        yaw_err = normalize(body_yaw - tx)
        if abs(yaw_err) > 3.0:
            return jsonify({
                "moveWS": {"command": "STOP", "weight": 1}, "moveAD": {"command": "", "weight": 0},
                "turretQE": {"command": "E" if yaw_err > 0 else "Q", "weight": min(abs(yaw_err) * 0.15, 1.0)},
                "turretRF": {"command": "", "weight": 0}, "fire": False
            })
        RECENTER_TURRET = False
        print("🔄 Turret Recentered")
        return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "moveAD": {"command": "", "weight": 0}, "turretQE": {"command": "", "weight": 0}, "turretRF": {"command": "", "weight": 0}, "fire": False})

    # ---------------------------------------------
    # 3. 이벤트 체크 (주요 경유지 도착 확인)
    # ---------------------------------------------
    if current_key_wp_index < len(WAYPOINTS):
        key_wp = WAYPOINTS[current_key_wp_index]
        dist_to_key = math.hypot(key_wp[0] - px, key_wp[1] - pz)

        # 주요 지점 근처 (3.5m) 도착 시 이벤트 트리거
        if dist_to_key < 3.5:
            # [0]번 WP: 회전 + 대기
            if current_key_wp_index == 0:
                target_rot = 335.0
                diff = normalize(target_rot - body_yaw)
                if abs(diff) > 4.0:
                    return jsonify({
                        "moveWS": {"command": "STOP", "weight": 1},
                        "moveAD": {"command": "D" if diff > 0 else "A", "weight": min(0.4, max(0.1, abs(diff) * 0.02))},
                        "turretQE": {"command": "", "weight": 0}, "turretRF": {"command": "", "weight": 0}, "fire": False
                    })
                if wait_start_time is None: wait_start_time = time.time()
                if time.time() - wait_start_time < 3.0:
                    return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "moveAD": {"command": "", "weight": 0}, "turretQE": {"command": "", "weight": 0}, "turretRF": {"command": "", "weight": 0}, "fire": False})

                current_key_wp_index += 1
                wait_start_time = None

            # [1]번 WP (기존 5번에서 변경됨): 포격 모드
            elif current_key_wp_index == 1:
                FIRE_MODE = True
                FIRE_COUNT = 0
                FIRE_AIM_START = None
                print("🔥 START FIRE MODE (New Index)")
                return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "moveAD": {"command": "", "weight": 0}, "turretQE": {"command": "", "weight": 0}, "turretRF": {"command": "", "weight": 0}, "fire": False})

            # 일반 WP: 그냥 통과 (인덱스만 증가)
            else:
                current_key_wp_index += 1

    # ---------------------------------------------
    # 4. 주행 제어 (A* Path + Drift Driving)
    # ---------------------------------------------
    # 설정값 (드리프트 주행을 위한 세팅)
    MAX_SPEED = 0.6    # 최고 속도
    LOOK_DIST = 10.0   # 전방 주시 거리
    STEER_GAIN = 0.04  # 회전 민감도 (0.03 -> 0.04로 약간 상향)
    PIVOT_LIMIT = 45.0 # 제자리 회전 기준 완화 (20도 -> 45도)

    # A* 경로에서 목표점 찾기
    target_x, target_z = get_lookahead_target_from_path(px, pz, LOOK_DIST)

    dx, dz = target_x - px, target_z - pz
    if math.hypot(dx, dz) < 0.5 and current_key_wp_index >= len(WAYPOINTS):
         return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "moveAD": {"command": "", "weight": 0}, "fire": False})

    target_angle = math.degrees(math.atan2(dx, dz))
    diff = normalize(target_angle - body_yaw)
    abs_diff = abs(diff)

    # 1. 정말 급한 코너(45도 이상)만 제자리 회전
    if abs_diff > PIVOT_LIMIT:
        pivot_weight = min(1.0, max(0.3, abs_diff * 0.03))
        return jsonify({
            "moveWS":   {"command": "STOP", "weight": 1},
            "moveAD":   {"command": "D" if diff > 0 else "A", "weight": pivot_weight},
            "turretQE": {"command": "", "weight": 0}, "turretRF": {"command": "", "weight": 0}, "fire": False
        })

    # 2. 드리프트 주행 (멈추지 않고 돔)
    # 각도가 클수록 전진 속도는 줄이지만, 절대 멈추지는 않음 (최소 0.3 보장)
    raw_fwd = 1.0 - (abs_diff / 60.0)
    fwd_weight = min(MAX_SPEED, max(0.3, raw_fwd))

    turn_weight = min(1.0, max(0.0, abs_diff * STEER_GAIN))

    return jsonify({
        "moveWS":   {"command": "W", "weight": fwd_weight},
        "moveAD":   {"command": "D" if diff > 0 else "A", "weight": turn_weight},
        "turretQE": {"command": "", "weight": 0},
        "turretRF": {"command": "", "weight": 0},
        "fire":     False
    })

# ------------------------------------------------------------
# 착탄 처리
# ------------------------------------------------------------
@app.route("/update_bullet", methods=["POST"])
def update_bullet():
    global FIRE_MODE, FIRE_COUNT, current_key_wp_index, RECENTER_TURRET, FIRE_AIM_START
    data = request.get_json(force=True) or {}
    if not FIRE_MODE: return jsonify({"status": "ignored"})

    FIRE_COUNT += 1
    print(f"🔥 Fire Count: {FIRE_COUNT}/3")

    if FIRE_COUNT >= 3:
        FIRE_MODE = False
        FIRE_COUNT = 0
        RECENTER_TURRET = True
        current_key_wp_index = min(current_key_wp_index + 1, len(WAYPOINTS) - 1)
        print("🎯 Fire Done -> Recenter -> Resume")
        return jsonify({"status": "done"})

    return jsonify({"status": "ok", "count": FIRE_COUNT})

# ------------------------------------------------------------
# 기타 API
# ------------------------------------------------------------
@app.route('/detect', methods=['POST'])
def detect(): return jsonify([]) # 생략 (필요시 복구)
@app.route('/info', methods=['POST'])
def info(): return jsonify({"status": "success"})
@app.route('/update_obstacle', methods=['POST'])
def update_obstacle(): return jsonify({'status': 'success'})
@app.route('/collision', methods=['POST'])
def collision(): return jsonify({'status': 'success'})
@app.route('/init', methods=['GET'])
def init(): return jsonify({"startMode": "start", "blStartX": 5, "blStartY": 10, "blStartZ": 5, "trackingMode": True, "detactMode": False, "logMode": True, "enemyTracking": False, "saveSnapshot": False, "saveLog": True, "saveLidarData": False, "lux": 30000})
@app.route('/start', methods=['GET'])
def start(): return jsonify({"control": ""})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)