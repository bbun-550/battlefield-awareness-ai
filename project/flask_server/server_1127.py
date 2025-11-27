# ============================================================
# [통합] A* Pathfinding + Pure Pursuit + Shoot & Scoot (Full)
# ============================================================

from flask import Flask, request, jsonify
import math, os, time, json, heapq
import numpy as np
import pandas as pd

app = Flask(__name__)

# ------------------------------------------------------------
# 1. 파일 경로 및 기본 설정
# ------------------------------------------------------------
OUTPUT_CSV  = "log_data/output.csv"
MAP_FILE    = "map/11_27.map"

# ------------------------------------------------------------
# 2. 웨이포인트 (경유지)
# ------------------------------------------------------------
WAYPOINTS = [
    (66.08732, 45.9379),    # [0] 회전 + 3초 정지
    (120.389, 181.441),     # [1] 포격위치
    (119.07, 287.42),       # [2]
    (35.982, 284.198)       # [3]
]

# ------------------------------------------------------------
# 3. 전역 변수 (Global State)
# ------------------------------------------------------------
server_player_pos = [0.0, 0.0, 0.0] # 서버에서 받은 내 좌표
FINAL_PATH = []                     # A* 경로 리스트
path_generated = False              # 최초 경로 생성 여부

current_key_wp_index = 0  # 현재 목표 웨이포인트 인덱스
wait_start_time = None    # 대기 타이머

# [Shoot-and-Scoot 설정]
RETREAT_POS = (111.44, 154.72)  # 후퇴할 좌표 (은폐 엄폐)
IS_RETREATING = False           # 후퇴 중인가?
IS_RETURNING = False            # 복귀 중인가?
FIRING_POS = WAYPOINTS[1]       # 사격 위치 고정

# [포격 관련 설정]
FIRE_MODE = False
FIRE_COUNT = 0
RECENTER_TURRET = False
FIRE_AIM_START = None
CURRENT_BODY_YAW = None
LAST_FIRE_TIME = 0      # 마지막 발사 시간 (재장전 체크용)
TOTAL_SHOT_COUNT = 0    # 누적 발사 카운트

# [맵 데이터]
ALL_OBSTACLES = []  # A* 장애물
TARGET_TANKS  = []  # 포격 타겟

# ------------------------------------------------------------
# 4. A* 알고리즘 (길찾기)
# ------------------------------------------------------------
GRID_SIZE = 1.0       # 1m 단위 격자
OBSTACLE_MARGIN = 7.0 # 장애물 회피 거리

def world_to_grid(x, z):
    return int(round(x / GRID_SIZE)), int(round(z / GRID_SIZE))

def grid_to_world(r, c):
    return float(r) * GRID_SIZE, float(c) * GRID_SIZE

def get_blocked_cells(obstacles):
    blocked = set()
    margin_steps = int(math.ceil(OBSTACLE_MARGIN / GRID_SIZE))
    print(f"🛠️ Building Obstacle Map with {len(obstacles)} objects...")
    
    for ob in obstacles:
        ox, oz = ob['x'], ob['z']
        gr, gc = world_to_grid(ox, oz)
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
    end_node = world_to_grid(*end_pos)
    
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    open_set = []
    heapq.heappush(open_set, (0, start_node))
    
    came_from = {}
    g_score = {start_node: 0}
    f_score = {start_node: heuristic(start_node, end_node)}
    
    best_node = start_node
    min_dist_to_goal = heuristic(start_node, end_node)

    while open_set:
        _, current = heapq.heappop(open_set)

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
            
            if dx != 0 and dy != 0: # 대각선 벽 뚫기 방지
                if (current[0] + dx, current[1]) in blocked_cells or \
                   (current[0], current[1] + dy) in blocked_cells:
                    continue
            
            move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
            tentative_g = g_score[current] + move_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, end_node)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    print("⚠️ Path blocked! Going to closest accessible point.")
    path = []
    curr = best_node
    while curr in came_from:
        path.append(grid_to_world(*curr))
        curr = came_from[curr]
    path.append(start_pos)
    path.reverse()
    path.append(end_pos)
    return path

# [전체 경로 생성]
def generate_full_path(start_x, start_z):
    global FINAL_PATH, WAYPOINTS, ALL_OBSTACLES
    sx = round(start_x / GRID_SIZE) * GRID_SIZE
    sz = round(start_z / GRID_SIZE) * GRID_SIZE
    print("🗺️ Generating Initial Full Path...")
    blocked = get_blocked_cells(ALL_OBSTACLES)
    full_path = [(start_x, start_z)]
    current_pos = (start_x, start_z)

    for i, wp in enumerate(WAYPOINTS):
        segment = a_star_search(current_pos, wp, blocked)
        if full_path: full_path.extend(segment[1:])
        else: full_path.extend(segment)
        current_pos = wp

    FINAL_PATH = full_path
    print(f"✅ Full Path Created: {len(FINAL_PATH)} nodes")

# [임시 경로 생성 - 후퇴/복귀용]
def generate_temp_path(start_x, start_z, end_x, end_z):
    global FINAL_PATH, ALL_OBSTACLES
    sx = round(start_x / GRID_SIZE) * GRID_SIZE
    sz = round(start_z / GRID_SIZE) * GRID_SIZE
    print(f"🔄 Re-calculating Path: ({start_x:.1f}, {start_z:.1f}) -> ({end_x:.1f}, {end_z:.1f})")
    blocked = get_blocked_cells(ALL_OBSTACLES)
    path = a_star_search((start_x, start_z), (end_x, end_z), blocked)
    FINAL_PATH = path

# ------------------------------------------------------------
# 5. 맵 로드 & Pure Pursuit
# ------------------------------------------------------------
def load_map():
    global ALL_OBSTACLES, TARGET_TANKS
    ALL_OBSTACLES = []
    TARGET_TANKS = []
    if not os.path.exists(MAP_FILE):
        print("MAP not found:", MAP_FILE)
        return

    with open(MAP_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for ob in data.get("obstacles", []):
        name = str(ob.get("prefabName", "")).lower()
        pos = ob.get("position", {})

        obj_data = {
            "name": name, 
            "x": float(pos.get("x", 0.0)),
            "y": float(pos.get("y", 0.0)),
            "z": float(pos.get("z", 0.0))
        }

        # 장애물로 인식할 키워드
        OBSTACLE_KEYWORDS = ["tank", "car", "rock"]
        
        # 이동 장애물
        if any(k in name for k in OBSTACLE_KEYWORDS):
            ALL_OBSTACLES.append(obj_data)
        # 사격 타겟
        if "tank" in name:
            TARGET_TANKS.append(obj_data)

    print(f"✅ Map Loaded: Obstacles={len(ALL_OBSTACLES)}, Targets={len(TARGET_TANKS)}")

load_map()

def normalize(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0

def get_lookahead_target_from_path(px, pz, lookahead=6.0):
    global FINAL_PATH
    if not FINAL_PATH: return (px, pz) # 경로 없으면 제자리
    
    closest_idx = 0
    min_dist = 9999.0

    for i, (nx, nz) in enumerate(FINAL_PATH):
        d = math.hypot(nx - px, nz - pz)
        if d < min_dist:
            min_dist = d
            closest_idx = i

    for i in range(closest_idx, len(FINAL_PATH)):
        nx, nz = FINAL_PATH[i]
        d = math.hypot(nx - px, nz - pz)
        if d >= lookahead:
            return (nx, nz)
        
    return FINAL_PATH[-1]

# ------------------------------------------------------------
# 6. 포격 솔루션 (Ballistic)
# ------------------------------------------------------------
MIN_PITCH, MAX_PITCH = -30.0, 10.0
V_INIT, G, MAX_RANGE, H_OFFSET = 58.0, 9.81, 130.0, 2.1
FIRST_FIRE_DELAY = 1.5
RELOAD_COOLDOWN = 7.0

def pick_target_by_index(px, pz, idx):
    global TARGET_TANKS
    
    # 1. 적이 하나도 없으면 리턴
    if not TARGET_TANKS: return None, 9999.0

    # 2. 내 위치 기준으로 가까운 순서대로 정렬
    sorted_targets = sorted(TARGET_TANKS, key=lambda t: math.hypot(t['x'] - px, t['z'] - pz))
    
    # 3. 인덱스 안전장치 (적이 2명인데 3번째 쏘려고 하면 마지막 적 선택)
    safe_idx = idx
    if safe_idx >= len(sorted_targets):
        safe_idx = len(sorted_targets) - 1
        
    target = sorted_targets[safe_idx]
    dist = math.hypot(target['x'] - px, target['z'] - pz)
    
    return target, dist

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
    return True, max(MIN_PITCH, min(MAX_PITCH, float(np.interp(d, Z[idx], ang[idx]))))

def compute_solution(px, py, pz, tx, ty):
    global FIRE_COUNT, TARGET_TANKS, MAX_RANGE

    # 1. 우선 n번째 타겟을 시도해봄
    tgt, dist = pick_target_by_index(px, pz, FIRE_COUNT)
    
    # [안전장치] 만약 n번째 타겟이 없거나, 사거리(MAX_RANGE) 밖이라면?
    if not tgt or dist > MAX_RANGE:
        print(f"⚠️ {FIRE_COUNT}번째 타겟 사거리 밖(Dist:{dist:.1f})! 가장 가까운 적으로 변경합니다.")
        
        # 가장 가까운 적(0번째)으로 다시 시도
        tgt, dist = pick_target_by_index(px, pz, 0)
        
        # 그래도 쏠 놈이 없으면 포기
        if not tgt or dist > MAX_RANGE:
            print("❌ 사거리 내에 공격 가능한 적이 없습니다.")
            return {"ok": False}

    # 2. 탄도 계산
    dyaw = math.degrees(math.atan2(tgt["x"] - px, tgt["z"] - pz))
    ok, bp = ballistic_pitch(px, py + H_OFFSET, pz, tgt["x"], tgt["y"], tgt["z"])
    
    # 탄도 계산 실패 시(각도가 안나옴), 직사 or CSV 데이터 사용
    base = bp if ok else (angle_from_csv(dist)[1] if angle_from_csv(dist)[0] else math.degrees(math.atan2(tgt["y"] - (py + H_OFFSET), dist)))
    
    return {"ok": True, "yaw": normalize(dyaw), "pitch": max(MIN_PITCH, min(MAX_PITCH, base))}

def turret_ctrl(cx, cy, tx, ty):
    ex, ey = normalize(tx - cx), ty - cy
    return {"QE": {"command": "E" if ex > 0 else "Q" if ex < 0 else "", "weight": min(abs(ex) * 0.05, 1.0)},
            "RF": {"command": "R" if ey > 0 else "F" if ey < 0 else "", "weight": min(abs(ey) * 0.2, 1.0)}, "ex": ex, "ey": ey}

def aim_good_enough(ex, ey): return abs(ex) < 3.0 and abs(ey) < 3.0

# ------------------------------------------------------------
# 7. GET_ACTION (메인 로직)
# ------------------------------------------------------------
@app.route("/get_action", methods=["POST"])
def get_action():
    global current_key_wp_index, FIRE_MODE, FIRE_COUNT, FINAL_PATH, path_generated
    global RECENTER_TURRET, wait_start_time, FIRE_AIM_START, CURRENT_BODY_YAW
    global IS_RETREATING, IS_RETURNING, RETREAT_POS, FIRING_POS, server_player_pos, WAYPOINTS

    req = request.get_json(force=True) or {}
    pos = req.get("position", {})
    turret = req.get("turret", {})
    px, py, pz = float(pos.get("x", 0)), float(pos.get("y", 0)), float(pos.get("z", 0))
    tx, ty = float(turret.get("x", 0)), float(turret.get("y", 0))

    server_player_pos = [px, py, pz]
    body_yaw = CURRENT_BODY_YAW if CURRENT_BODY_YAW is not None else tx

    if px == 0.0 and pz == 0.0:
        return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})

    # 초기화
    if not path_generated:
        generate_full_path(px, pz)
        path_generated = True
        current_key_wp_index = 0

    # -----------------------------------
    # [1] 포격 모드
    # -----------------------------------
    if FIRE_MODE:
        sol = compute_solution(px, py, pz, tx, ty)
        if not sol["ok"]: 
            return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})

        ctrl = turret_ctrl(tx, ty, sol["yaw"], sol["pitch"])
        fire = False
        
        if aim_good_enough(ctrl["ex"], ctrl["ey"]):
            if FIRE_AIM_START is None: FIRE_AIM_START = time.time()
            time_since_aim = time.time() - FIRE_AIM_START
            time_since_fire = time.time() - LAST_FIRE_TIME
            
            if time_since_aim >= FIRST_FIRE_DELAY and time_since_fire >= RELOAD_COOLDOWN:
                fire = True
        else:
            FIRE_AIM_START = None

        return jsonify({
            "moveWS": {"command": "STOP", "weight": 1}, "moveAD": {"command": "", "weight": 0},
            "turretQE": ctrl["QE"], "turretRF": ctrl["RF"], "fire": fire
        })

    # -----------------------------------
    # [2] 포탑 정렬 (이동 중)
    # -----------------------------------
    if RECENTER_TURRET:
        yaw_err = normalize(body_yaw - tx)
        if abs(yaw_err) > 3.0:
            return jsonify({
                "moveWS": {"command": "STOP", "weight": 1}, "moveAD": {"command": "", "weight": 0},
                "turretQE": {"command": "E" if yaw_err > 0 else "Q", "weight": 0.5},
                "fire": False
            })
        RECENTER_TURRET = False

    # -----------------------------------
    # [3] 목표 좌표 설정 (Target Selection)
    # -----------------------------------
    target_x, target_z = 0, 0

    # CASE 0: 첫 번째 경유지
    if current_key_wp_index == 0:
        wp_target = WAYPOINTS[0]
        dist = math.hypot(wp_target[0] - px, wp_target[1] - pz)

        if dist < 3.5:
            target_rot = 335.0
            diff = normalize(target_rot - tx)
            if abs(diff) > 4.0:
                return jsonify({
                    "moveWS": {"command": "STOP", "weight": 1},
                    "moveAD": {"command": "", "weight": 0},
                    "turretQE": {"command": "E" if diff > 0 else "Q", "weight": 0.3},
                    "fire": False
                })
            
            if wait_start_time is None: wait_start_time = time.time()
            if time.time() - wait_start_time < 3.0:
                return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})
            
            wait_start_time = None
            RECENTER_TURRET = True
            current_key_wp_index = 1
            return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})
        
        target_x, target_z = get_lookahead_target_from_path(px, pz, 3.5)

    # CASE 1: Shoot & Scoot (후진 적용)
    elif current_key_wp_index == 1:
        
        # [A] 후퇴 중 (후진으로 이동!)
        if IS_RETREATING:
            target_x, target_z = get_lookahead_target_from_path(px, pz, 3.5)
            
            # 후퇴 완료 체크
            if math.hypot(RETREAT_POS[0] - px, RETREAT_POS[1] - pz) < 2.0:
                IS_RETREATING = False
                IS_RETURNING = True
                generate_temp_path(px, pz, FIRING_POS[0], FIRING_POS[1])
                print("↩️ Retreat Done -> Generating Return Path")
                return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})

        # [B] 복귀 중 (전진으로 이동)
        elif IS_RETURNING:
            target_x, target_z = get_lookahead_target_from_path(px, pz, 3.5)
            
            # 복귀 완료 체크
            if math.hypot(FIRING_POS[0] - px, FIRING_POS[1] - pz) < 1.5:
                IS_RETURNING = False
                FIRE_MODE = True
                print("🔫 Back at pos -> Fire!")
                return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})

        # [C] 최초 진입
        else:
            wp_target = WAYPOINTS[1]
            dist = math.hypot(wp_target[0] - px, wp_target[1] - pz)
            if dist < 4.0:
                FIRE_MODE = True
                print("🔥 Arrived at WP1 -> START FIRE")
                return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})
            
            target_x, target_z = get_lookahead_target_from_path(px, pz, 3.5)

    # CASE 2+: 일반 주행
    else:
        if current_key_wp_index >= len(WAYPOINTS):
            return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})
        
        wp_target = WAYPOINTS[current_key_wp_index]
        dist = math.hypot(wp_target[0] - px, wp_target[1] - pz)
        
        # [수정] 도착하면 인덱스 올리고, 다음 경로 생성!
        if dist < 3.5:
            current_key_wp_index += 1
            if current_key_wp_index < len(WAYPOINTS):
                next_wp = WAYPOINTS[current_key_wp_index]
                generate_temp_path(px, pz, next_wp[0], next_wp[1]) 
                print(f"🚀 Generating Path to WP {current_key_wp_index}")
        
        target_x, target_z = get_lookahead_target_from_path(px, pz, 3.5)
    # =========================================================
    # [4] 모터 제어 (후진 로직 추가됨)
    # =========================================================
    dx, dz = target_x - px, target_z - pz
    target_angle = math.degrees(math.atan2(dx, dz))

    # ★★★ [핵심 변경] 후퇴 중일 때는 'S'키 로직 사용 ★★★
    if IS_RETREATING:
        # 내 엉덩이(Back)가 목표를 바라보는 각도 계산
        back_yaw = normalize(body_yaw + 180.0)
        diff = normalize(target_angle - back_yaw)
        abs_diff = abs(diff)
        
        # [중요] 엉덩이 각도가 40도 이상 틀어져 있으면 -> 'S' 떼고 제자리 회전만!
        if abs_diff > 40.0:
            return jsonify({
                "moveWS": {"command": "STOP", "weight": 1}, 
                "moveAD": {"command": "D" if diff > 0 else "A", "weight": 0.8}, # 회전 속도 높임
                "fire": False
            })
            
        # 각도가 얼추 맞으면 -> 후진(S) 하면서 조향
        else:
            return jsonify({
                "moveWS": {"command": "S", "weight": 0.5}, # 속도 조금 줄임 (안전하게)
                "moveAD": {"command": "D" if diff > 0 else "A", "weight": min(1.0, abs_diff * 0.05)},
                "fire": False
            })

    # 일반 전진 주행 (W키)
    else:
        diff = normalize(target_angle - body_yaw)
        abs_diff = abs(diff)

        if abs_diff > 60.0: # 각도가 너무 크면 제자리 회전
            return jsonify({
                "moveWS":   {"command": "STOP", "weight": 1},
                "moveAD":   {"command": "D" if diff > 0 else "A", "weight": 0.5},
                "fire":     False
            })

        fwd = min(0.6, max(0.3, 1.0 - (abs_diff / 60.0)))
        return jsonify({
            "moveWS":   {"command": "W", "weight": fwd},
            "moveAD":   {"command": "D" if diff > 0 else "A", "weight": min(1.0, abs_diff * 0.04)},
            "fire":     False
        })
    
# ------------------------------------------------------------
# 8. 착탄 처리 (이벤트)
# ------------------------------------------------------------
@app.route("/update_bullet", methods=["POST"])
def update_bullet():
    global FIRE_MODE, FIRE_COUNT, current_key_wp_index, RECENTER_TURRET, LAST_FIRE_TIME
    global IS_RETREATING, IS_RETURNING, FINAL_PATH, server_player_pos, WAYPOINTS, RETREAT_POS
    global TOTAL_SHOT_COUNT

    px, pz = server_player_pos[0], server_player_pos[2]
    data = request.get_json(force=True) or {}
    
    if not FIRE_MODE: return jsonify({"status": "ignored"})

    FIRE_COUNT += 1
    TOTAL_SHOT_COUNT += 1
    LAST_FIRE_TIME = time.time()
    print(f"🔥 Fire Count: {FIRE_COUNT}/3 (Total: {TOTAL_SHOT_COUNT})")

    # 3발 발사 완료 -> 다음 미션(WP2)
    if FIRE_COUNT >= 3:
        FIRE_MODE = False
        FIRE_COUNT = 0
        IS_RETREATING = False
        IS_RETURNING = False
        RECENTER_TURRET = True
        
        current_key_wp_index += 1
        if current_key_wp_index < len(WAYPOINTS):
             generate_temp_path(px, pz, WAYPOINTS[current_key_wp_index][0], WAYPOINTS[current_key_wp_index][1])
             
        print("🎯 All Shots Fired -> Next WP")
        return jsonify({"status": "done"})

    # 1~2발 -> 후퇴
    else:
        FIRE_MODE = False
        IS_RETREATING = True
        IS_RETURNING = False
        generate_temp_path(px, pz, RETREAT_POS[0], RETREAT_POS[1])
        print(f"🔙 Shot Fired! Retreating...")
        return jsonify({"status": "retreating", "count": FIRE_COUNT})

# ------------------------------------------------------------
# 9. 기타 API
# ------------------------------------------------------------
@app.route('/info', methods=['POST'])
def info():
    global server_player_pos, CURRENT_BODY_YAW
    try:
        data = request.get_json(force=True) or {}
        if "playerBodyX" in data:
            CURRENT_BODY_YAW = float(data["playerBodyX"])
        pos = data.get('playerPos', {})
        server_player_pos = [float(pos.get('x', 0)), float(pos.get('y', 0)), float(pos.get('z', 0))]
        return "OK", 200
    except: return "Error", 400

@app.route('/info', methods=['GET'])
def info_get():
    return jsonify({
        "pos":{
            "x":server_player_pos[0],
            "y":server_player_pos[1],
            "z":server_player_pos[2]
        },
        "fire_count": TOTAL_SHOT_COUNT
    })

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
