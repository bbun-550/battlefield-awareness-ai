# 각 PC에 맞는 폴더경로로 수정
# -*- coding: utf-8 -*-
# ============================================================
# WAYPOINT 이동 + 특정 지점 포격모드 + 착탄 3회 → 이동 재개
# - 이동 방향(body yaw): log.txt 의 Player_Body_X 사용
# - 포격 후: 포탑을 몸체 정면 방향으로 정렬 후 다시 이동
# - 처음 조준 후 0.6초 동안은 에임 안정될 때까지 발사 금지
# ============================================================

from flask import Flask, request, jsonify
import math, os, time, json
import numpy as np
import pandas as pd
# from ultralytics import YOLO

# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
app = Flask(__name__)
# model = YOLO('5cls_v6_case2_best.pt')

# log / csv / map 파일 경로
LOG_FILE    = r"C:\Users\acorn\Documents\Tank Challenge\log_data\tank_info_log.txt"
OUTPUT_CSV  = r"C:\Users\acorn\Documents\Tank Challenge\log_data\output.csv"
MAP_FILE    = r"\map\11_20_tuning.map"

# server_player_pos 초기화
server_player_pos = [0.0, 0.0, 0.0]

# ------------------------------------------------------------
# WAYPOINT 목록
# ------------------------------------------------------------
WAYPOINTS = [
    (66.08732, 45.9379),   # [0] 회전 + 3초 정지
    (100.425, 106.330),    # [1]
    (81.277, 99.007),      # [2]
    (90.565, 130.413),     # [3]
    (111.759, 172.892),    # [4]
    (120.389, 181.441),    # [5] → 포격모드 ON
    (139.722, 258.477),    # [6] → 포격 후 이동 재개
    (128.686, 291.084),    # [7]
    (35.982, 284.198)      # [8]
]

current_wp_index = 0
wait_start_time  = None

# 포격 관련 플래그
FIRE_MODE          = False   # True 이면 조준/사격만 수행 (이동 STOP)
FIRE_COUNT         = 0       # /update_bullet 에서 증가
RECENTER_TURRET    = False   # 포격 후 포탑을 몸체 정면으로 재정렬
FIRE_AIM_START     = None    # 에임 안정화 시작 시간 (0.6초용)

# ------------------------------------------------------------
# MAP Load (Tank obstacles)
# ------------------------------------------------------------
TANK_OBJS = []

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
# Util
# ------------------------------------------------------------
def normalize(a: float) -> float:
    """ -180 ~ +180 으로 정규화 """
    return (a + 180.0) % 360.0 - 180.0

def read_body_yaw_from_log():
    """
    log.txt에서 Player_Body_X(= 몸체 yaw) 값을 읽는다.
    예시:
      "Player_Body_X": 123.45,
    """
    if not os.path.exists(LOG_FILE):
        return None

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            return None
        last = lines[-1]
    except:
        return None

    # 아주 단순한 파싱
    for token in last.replace("{", " ").replace("}", " ").split(","):
        if "Player_Body_X" in token:
            try:
                val = float(token.split(":")[1])
                return val
            except:
                pass
    return None

# ------------------------------------------------------------
# 포격 계산 (탄도, csv 보간)
# ------------------------------------------------------------
MIN_PITCH_CFG = -30.0
MAX_PITCH_CFG = 10.0
V_INIT        = 58.0
G             = 9.81
MAX_RANGE     = 130.0
H_OFFSET      = 2.1   # 포탑 높이 오프셋

def pick_closest(px, pz):
    """현재 위치(px,pz)에서 가장 가까운 Tank obstacle 선택"""
    best = None
    bd   = 9999.0
    for ob in TANK_OBJS:
        d = math.hypot(ob["x"] - px, ob["z"] - pz)
        if d < bd:
            bd = d
            best = ob
    return best, bd

def ballistic_pitch(sx, sy, sz, tx, ty, tz):
    """탄도 방정식 기반 pitch 계산"""
    dx, dy, dz = tx - sx, ty - sy, tz - sz
    H = math.hypot(dx, dz)
    if H < 1e-6:
        return False, None
    v2   = V_INIT * V_INIT
    term = v2 * v2 - G * (G * H * H + 2.0 * dy * v2)
    if term < 0:
        return False, None
    r  = math.sqrt(term)
    t1 = math.atan((v2 + r) / (G * H))
    t2 = math.atan((v2 - r) / (G * H))
    return True, math.degrees(min(t1, t2))

def angle_from_csv(d):
    """output.csv 기반 거리->각도 보간"""
    if not os.path.exists(OUTPUT_CSV):
        return False, None
    df  = pd.read_csv(OUTPUT_CSV)
    arr = df.to_numpy()
    ang = arr[:, 0]
    Z   = arr[:, 3]
    idx = np.argsort(Z)
    Zs  = Z[idx]
    Angs = ang[idx]
    if d < Zs[0] or d > Zs[-1]:
        return False, None
    v = float(np.interp(d, Zs, Angs))
    return True, max(MIN_PITCH_CFG, min(MAX_PITCH_CFG, v))

def compute_solution(px, py, pz, tx, ty):
    """현재 탱크 위치에서 가장 가까운 Tank obstacle로 조준각 계산"""
    tgt, dist = pick_closest(px, pz)
    if not tgt:
        return {"ok": False}
    if dist > MAX_RANGE:
        return {"ok": False}

    ex, ey, ez = tgt["x"], tgt["y"], tgt["z"]

    # yaw
    dyaw = math.degrees(math.atan2(ex - px, ez - pz))

    # pitch
    ok, bp = ballistic_pitch(px, py + H_OFFSET, pz, ex, ey, ez)
    if ok:
        base = bp
    else:
        ok2, p2 = angle_from_csv(dist)
        if ok2:
            base = p2
        else:
            base = math.degrees(math.atan2(ey - (py + H_OFFSET), dist))

    dpitch = max(MIN_PITCH_CFG, min(MAX_PITCH_CFG, base))

    return {"ok": True, "yaw": normalize(dyaw), "pitch": dpitch}

# ------------------------------------------------------------
# turret control
# ------------------------------------------------------------
def turret_ctrl(cx, cy, tx, ty):
    """포탑 P제어"""
    ex = normalize(tx - cx)
    ey = ty - cy
    wx = min(abs(ex) * 0.15, 1.0)
    wy = min(abs(ey) * 0.45, 1.0)

    cmdx = "E" if ex > 0 else "Q" if ex < 0 else ""
    cmdy = "R" if ey > 0 else "F" if ey < 0 else ""

    return {
        "QE": {"command": cmdx, "weight": wx},
        "RF": {"command": cmdy, "weight": wy},
        "ex": ex,
        "ey": ey
    }

def aim_good_enough(ex, ey):
    """조준 허용 오차"""
    return (abs(ex) < 3.0 and abs(ey) < 3.0)

# ------------------------------------------------------------
# GET_ACTION
# ------------------------------------------------------------
FIRST_FIRE_DELAY = 0.6  # 처음 조준 안정화 대기 시간(초)

@app.route("/get_action", methods=["POST"])
def get_action():
    global current_wp_index, FIRE_MODE, FIRE_COUNT
    global RECENTER_TURRET, wait_start_time, FIRE_AIM_START

    req    = request.get_json(force=True) or {}
    pos    = req.get("position", {})
    turret = req.get("turret", {})

    px = float(pos.get("x", 0.0))
    py = float(pos.get("y", 0.0))
    pz = float(pos.get("z", 0.0))

    tx = float(turret.get("x", 0.0))  # turret yaw
    ty = float(turret.get("y", 0.0))  # turret pitch

    # 몸체 yaw = Player_Body_X (log.txt)
    body_yaw = read_body_yaw_from_log()
    if body_yaw is None:
        body_yaw = tx   # fallback

    # =========================================================
    # 🔥 포격 모드
    # =========================================================
    if FIRE_MODE:
        sol = compute_solution(px, py, pz, tx, ty)
        if not sol["ok"]:
            return jsonify({
                "moveWS":  {"command": "STOP", "weight": 1},
                "moveAD":  {"command": "",     "weight": 0},
                "turretQE": {"command": "",    "weight": 0},
                "turretRF": {"command": "",    "weight": 0},
                "fire":    False
            })

        ctrl = turret_ctrl(tx, ty, sol["yaw"], sol["pitch"])

        # --- 에임 안정화 + 0.6초 대기 ---
        if not aim_good_enough(ctrl["ex"], ctrl["ey"]):
            # 에임이 틀어지면 타이머 리셋
            FIRE_AIM_START = None
            fire = False
        else:
            # 에임이 충분히 좋으면 타이머 시작
            if FIRE_AIM_START is None:
                FIRE_AIM_START = time.time()
            # 0.6초 동안 유지되면 발사 허용
            if time.time() - FIRE_AIM_START >= FIRST_FIRE_DELAY:
                fire = True
            else:
                fire = False

        return jsonify({
            "moveWS":   {"command": "STOP", "weight": 1},
            "moveAD":   {"command": "",     "weight": 0},
            "turretQE": ctrl["QE"],
            "turretRF": ctrl["RF"],
            "fire":     fire
        })

    # =========================================================
    # 🔄 포격 후 포탑 복귀 모드 (몸체 정면으로 맞추기)
    # =========================================================
    if RECENTER_TURRET:
        yaw_err = normalize(body_yaw - tx)

        # 아직 차이가 크면 포탑만 회전
        if abs(yaw_err) > 3.0:
            k = 0.15
            w = min(abs(yaw_err) * k, 1.0)
            cmd = "E" if yaw_err > 0 else "Q"

            return jsonify({
                "moveWS":   {"command": "STOP", "weight": 1},
                "moveAD":   {"command": "",     "weight": 0},
                "turretQE": {"command": cmd,    "weight": w},
                "turretRF": {"command": "",     "weight": 0},
                "fire":     False
            })

        # 정렬 완료
        RECENTER_TURRET = False
        print("🔄 포탑 복귀 완료 (body_yaw 정면)")

        return jsonify({
            "moveWS":   {"command": "STOP", "weight": 1},
            "moveAD":   {"command": "",     "weight": 0},
            "turretQE": {"command": "",     "weight": 0},
            "turretRF": {"command": "",     "weight": 0},
            "fire":     False
        })

    # =========================================================
    # 🚗 이동 모드
    # =========================================================
    if current_wp_index >= len(WAYPOINTS):
        return jsonify({
            "moveWS":   {"command": "STOP", "weight": 1},
            "moveAD":   {"command": "",     "weight": 0},
            "turretQE": {"command": "",     "weight": 0},
            "turretRF": {"command": "",     "weight": 0},
            "fire":     False
        })

    wx, wz = WAYPOINTS[current_wp_index]
    dist   = math.hypot(wx - px, wz - pz)

    # --------------------------- waypoint 도착 처리 ---------------------------
    if dist < 2.0:
        # 0번 웨이포인트: 회전 + 3초 정지
        if current_wp_index == 0:
            target_rot = 335.0
            diff = normalize(target_rot - body_yaw)
            # 회전 아직 덜 됨 → 회전만
            if abs(diff) > 5.0:
                return jsonify({
                    "moveWS":   {"command": "STOP", "weight": 1},
                    "moveAD":   {"command": "D" if diff > 0 else "A", "weight": 1},
                    "turretQE": {"command": "", "weight": 0},
                    "turretRF": {"command": "", "weight": 0},
                    "fire":     False
                })
            # 방향 맞음 → 3초 대기
            if wait_start_time is None:
                wait_start_time = time.time()
            if time.time() - wait_start_time < 3.0:
                return jsonify({
                    "moveWS":   {"command": "STOP", "weight": 1},
                    "moveAD":   {"command": "",     "weight": 0},
                    "turretQE": {"command": "",     "weight": 0},
                    "turretRF": {"command": "",     "weight": 0},
                    "fire":     False
                })
            # 대기 완료 → 다음 웨이포인트
            current_wp_index += 1
            wait_start_time = None

        else:
            # [5] 번 웨이포인트: 포격모드 ON
            if current_wp_index == 5:
                FIRE_MODE      = True
                FIRE_COUNT     = 0
                FIRE_AIM_START = None
                print("🔥 Enter Fire Mode (WP5)")
                return jsonify({
                    "moveWS":   {"command": "STOP", "weight": 1},
                    "moveAD":   {"command": "",     "weight": 0},
                    "turretQE": {"command": "",     "weight": 0},
                    "turretRF": {"command": "",     "weight": 0},
                    "fire":     False
                })
            # 그 외 웨이포인트는 그냥 다음으로
            current_wp_index += 1

    # --------------------------- 이동 제어 ---------------------------
    wx, wz = WAYPOINTS[current_wp_index]
    dx, dz = wx - px, wz - pz
    target_angle = math.degrees(math.atan2(dx, dz))
    diff = normalize(target_angle - body_yaw)

    if abs(diff) > 5.0:
        # 회전 먼저
        return jsonify({
            "moveWS":   {"command": "STOP", "weight": 1},
            "moveAD":   {"command": "D" if diff > 0 else "A", "weight": 1},
            "turretQE": {"command": "", "weight": 0},
            "turretRF": {"command": "", "weight": 0},
            "fire":     False
        })

    # 정면 맞으면 전진
    return jsonify({
        "moveWS":   {"command": "W", "weight": 1},
        "moveAD":   {"command": "",  "weight": 0},
        "turretQE": {"command": "",  "weight": 0},
        "turretRF": {"command": "",  "weight": 0},
        "fire":     False
    })

# ------------------------------------------------------------
# 착탄 처리 (/update_bullet)
# ------------------------------------------------------------
@app.route("/update_bullet", methods=["POST"])
def update_bullet():
    global FIRE_MODE, FIRE_COUNT, current_wp_index, RECENTER_TURRET, FIRE_AIM_START

    data = request.get_json(force=True) or {}
    hit  = data.get("hit", False)

    print(f"💥 /update_bullet: hit={hit}, FIRE_MODE={FIRE_MODE}, COUNT={FIRE_COUNT}")

    if not FIRE_MODE:
        return jsonify({"status": "ignored"})

    # 착탄 1회 → FIRE_COUNT++
    FIRE_COUNT += 1
    print(f"🔥 Fire Count: {FIRE_COUNT}/3")

    if FIRE_COUNT >= 3:
        # 포격 종료 → 이동모드로 전환 + 포탑 복귀 플래그
        FIRE_MODE       = False
        FIRE_COUNT      = 0
        FIRE_AIM_START  = None
        RECENTER_TURRET = True

        # 다음 웨이포인트로 진행
        current_wp_index = min(current_wp_index + 1, len(WAYPOINTS) - 1)
        print("🎯 Fire Done → Recenter Turret → Resume Movement")

        return jsonify({"status": "done", "next_wp": current_wp_index})

    return jsonify({"status": "ok", "count": FIRE_COUNT})

# ------------------------------------------------------------
# 기타 API
# ------------------------------------------------------------
# @app.route('/detect', methods=['POST'])
# def detect():
#     image = request.files.get('image')
#     if not image: return jsonify({"error": "No image"}), 400
#     image.save('temp_image.jpg')
#     results = model('temp_image.jpg')
#     detections = results[0].boxes.data.cpu().numpy()
#     target_classes = {0: "RED", 1: "Car", 2: "Blue", 3: "Rock", 4: "Tank"}
#     filtered_results = []
#     for box in detections:
#         cid = int(box[5])
#         if cid in target_classes:
#             filtered_results.append({
#                 'className': target_classes[cid],
#                 'bbox': [float(c) for c in box[:4]],
#                 'confidence': float(box[4])
#             })
#     return jsonify(filtered_results)

@app.route('/info', methods=['POST'])
def info():
    """
    게임에서 POST 요청으로 보내준 플레이어 좌표(x, y, z)를 수신하여
    탐지기 인스턴스의 player_pos 변수에 업데이트함.
    """
    global server_player_pos

    try:
        # JSON 데이터 파싱
        data = request.get_json(force=True)
        pos = data.get('playerPos', {})
                
        x = float(pos.get('x', 0.0))
        y = float(pos.get('y', 0.0))
        z = float(pos.get('z', 0.0)) 

        # 좌표 업데이트
        server_player_pos = [x, y, z]
        return "OK", 200
    except Exception as e:
        print(f"Data Error: {e}")
        return "Error", 400

@app.route('/info', methods=['GET'])
def info_get():
    return jsonify({
        "pos":{
            "x":server_player_pos[0],
            "y":server_player_pos[1],
            "z":server_player_pos[2]
        }
    })

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    return jsonify({'status': 'success'})

@app.route('/collision', methods=['POST'])
def collision():
    return jsonify({'status': 'success'})

@app.route('/init', methods=['GET'])
def init():
    config = {
        "startMode": "start",
        "blStartX": 15,
        "blStartY": 10,
        "blStartZ": 5,
        "trackingMode": True,
        "detactMode": False,
        "logMode": True,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": True,
        "saveLidarData": False,
        "lux": 30000
    }
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    return jsonify({"control": ""})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)