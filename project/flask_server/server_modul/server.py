from flask import Flask, request, jsonify
import math, time, logging

# 모듈 import
from combat import Gunner
from navigation import Navigator

app = Flask(__name__)

# [로그 설정 1] 불필요한 Flask 통신 로그 끄기 (에러만 표시)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- 설정 및 전역 변수 ---
MAP_FILE = "map/11_28.map"
WAYPOINTS = [
    (66.08732, 45.9379), (120.389, 181.441), (119.07, 287.42), (35.982, 284.198)
]
RETREAT_POS = (111.44, 154.72)
FIRING_POS = WAYPOINTS[1]

# 상태 변수
server_player_pos = [0, 0, 0]
current_key_wp_index = 0
fire_count = 0
total_shot_count = 0

# 플래그
is_fire_mode = False
is_retreating = False
is_returning = False
recenter_turret = False
wait_start_time = None
last_fire_time = 0
fire_aim_start = None
current_body_yaw = None

# [로그 설정 2] 출력 제한용 타이머
last_print_time = 0

# 모듈 인스턴스 생성
gunner = Gunner(MAP_FILE)
nav = Navigator(MAP_FILE)
path_generated = False
def normalize(a): return (a + 180.0) % 360.0 - 180.0

@app.route("/get_action", methods=["POST"])
def get_action():
    global current_key_wp_index, is_fire_mode, is_retreating, is_returning
    global recenter_turret, wait_start_time, path_generated, fire_aim_start
    global total_shot_count, last_fire_time, current_body_yaw, server_player_pos, fire_count
    global last_print_time # [로그 설정]

    req = request.get_json(force=True) or {}
    pos = req.get("position", {})
    turret = req.get("turret", {})
    px, py, pz = float(pos.get("x", 0)), float(pos.get("y", 0)), float(pos.get("z", 0))
    tx, ty = float(turret.get("x", 0)), float(turret.get("y", 0))
    server_player_pos = [px, py, pz]
    body_yaw = current_body_yaw if current_body_yaw is not None else tx

    if px == 0.0 and pz == 0.0:
        return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})

    if not path_generated:
        full_path = []
        curr = (px, pz)
        for wp in WAYPOINTS:
            seg = nav.generate_path(curr, wp)
            full_path.extend(seg)
            curr = wp
        nav.final_path = full_path
        path_generated = True

    # ---------------------------------------------------------
    # [A] 포격 모드
    # ---------------------------------------------------------
    if is_fire_mode:
        tgt, dist = gunner.get_target(px, pz, index=fire_count)
        sol = {"ok": False}
        if tgt: sol = gunner.calculate_solution(px, py, pz, tgt['x'], tgt['y'], tgt['z'])
        if not sol["ok"]:
            tgt, dist = gunner.get_target(px, pz, index=0)
            if tgt: sol = gunner.calculate_solution(px, py, pz, tgt['x'], tgt['y'], tgt['z'])
        
        if not sol["ok"]:
             return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})

        ctrl = gunner.get_turret_control(tx, ty, sol["yaw"], sol["pitch"])
        fire = False
        if ctrl["aimed"]:
            if fire_aim_start is None: fire_aim_start = time.time()
            if (time.time() - fire_aim_start >= 1.5) and (time.time() - last_fire_time >= 7.0):
                fire = True
                total_shot_count += 1
                last_fire_time = time.time()
                print(f"🔥 발사! (누적: {total_shot_count}발)") # 중요 이벤트는 출력
        else:
            fire_aim_start = None
            
        return jsonify({
            "moveWS": {"command": "STOP", "weight": 1}, "moveAD": {"command": "", "weight": 0},
            "turretQE": ctrl["turretQE"], "turretRF": ctrl["turretRF"], "fire": fire
        })

    # ---------------------------------------------------------
    # [B] 포탑 정렬
    # ---------------------------------------------------------
    if recenter_turret:
        yaw_err = normalize(body_yaw - tx)
        if abs(yaw_err) > 3.0:
            return jsonify({
                "moveWS": {"command": "STOP", "weight": 1}, "moveAD": {"command": "", "weight": 0},
                "turretQE": {"command": "E" if yaw_err > 0 else "Q", "weight": 0.5}, "fire": False
            })
        recenter_turret = False

    # ---------------------------------------------------------
    # [C] 상태 판단 및 [D] 제어
    # ---------------------------------------------------------
    drift_mode = False
    is_combat_approach = (current_key_wp_index == 1) or is_returning
    
    # 거리 계산용 변수
    dist_to_wp = 0.0

    if current_key_wp_index == 0:
        dist_to_wp = math.hypot(WAYPOINTS[0][0]-px, WAYPOINTS[0][1]-pz)
        if dist_to_wp < 3.5:
            target_rot = 335.0
            diff = normalize(target_rot - tx)
            if abs(diff) > 4.0:
                return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "moveAD": {"command": "", "weight": 0}, "turretQE": {"command": "E" if diff > 0 else "Q", "weight": 0.3}, "fire": False})
            if wait_start_time is None: 
                wait_start_time = time.time()
                print("⏳ 1번 포인트 도착 -> 3초 대기 시작")
            if time.time() - wait_start_time < 3.0:
                return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})
            wait_start_time = None; recenter_turret = True; current_key_wp_index = 1
            print("▶️ 대기 종료 -> 이동 시작")
            return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})
        target_x, target_z = nav.get_lookahead_target(px, pz, 6.0)

    elif current_key_wp_index == 1:
        if is_retreating:
            target_x, target_z = nav.get_lookahead_target(px, pz, 3.5)
            if math.hypot(RETREAT_POS[0]-px, RETREAT_POS[1]-pz) < 2.0:
                is_retreating = False; is_returning = True
                nav.generate_path((px, pz), FIRING_POS)
                print("↩️ 후퇴 완료 -> 복귀 시작")
                return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})
        elif is_returning:
            target_x, target_z = nav.get_lookahead_target(px, pz, 3.5)
            if math.hypot(FIRING_POS[0]-px, FIRING_POS[1]-pz) < 1.5:
                is_returning = False; is_fire_mode = True
                print("🔫 사격 위치 복귀 완료 -> 사격 모드")
                return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})
        else:
            dist_to_wp = math.hypot(WAYPOINTS[1][0]-px, WAYPOINTS[1][1]-pz)
            if dist_to_wp < 4.0: 
                is_fire_mode = True
                print("🔥 사격 위치 도착 -> 사격 모드")
                return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})
            target_x, target_z = nav.get_lookahead_target(px, pz, 3.5)

    else:
        if current_key_wp_index >= len(WAYPOINTS):
            return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})
        
        if math.hypot(WAYPOINTS[2][0]-px, WAYPOINTS[2][1]-pz) < 20.0: drift_mode = True
        
        wp_target = WAYPOINTS[current_key_wp_index]
        dist_to_wp = math.hypot(wp_target[0]-px, wp_target[1]-pz)
        
        if dist_to_wp < (15.0 if current_key_wp_index == 2 else 3.5):
            print(f"✅ WP {current_key_wp_index} 통과")
            current_key_wp_index += 1
            if current_key_wp_index < len(WAYPOINTS): nav.generate_path((px, pz), WAYPOINTS[current_key_wp_index])
        target_x, target_z = nav.get_lookahead_target(px, pz, 6.0)

    control = nav.get_drive_control(px, pz, body_yaw, target_x, target_z, is_retreating=is_retreating, drift_mode=drift_mode, is_combat=is_combat_approach)
    
    # [로그 설정 3] 1초에 한 번만 주행 상태 출력
    if time.time() - last_print_time > 1.0:
        mode_str = "일반주행"
        if is_retreating: mode_str = "후퇴중"
        elif is_returning: mode_str = "복귀중"
        elif drift_mode: mode_str = "드리프트"
        
        # WP까지 남은 거리와 현재 속도/조향 명령 표시
        print(f"[{mode_str}] 목표:WP{current_key_wp_index} | 남은거리:{dist_to_wp:.1f}m | 명령:{control['moveWS']['command']}({control['moveWS']['weight']:.1f})")
        last_print_time = time.time()

    control["fire"] = False
    return jsonify(control)

@app.route("/update_bullet", methods=["POST"])
def update_bullet():
    global is_fire_mode, fire_count, is_retreating, is_returning, recenter_turret, current_key_wp_index
    if not is_fire_mode: return jsonify({"status": "ignored"})
    fire_count += 1
    if fire_count >= 3:
        is_fire_mode = False; fire_count = 0; is_retreating = False; is_returning = False; recenter_turret = True
        current_key_wp_index += 1
        if current_key_wp_index < len(WAYPOINTS): nav.generate_path((server_player_pos[0], server_player_pos[2]), WAYPOINTS[current_key_wp_index])
        print("🎯 3발 명중 -> 다음 미션 이동")
        return jsonify({"status": "done"})
    else:
        is_fire_mode = False; is_retreating = True; is_returning = False
        nav.generate_path((server_player_pos[0], server_player_pos[2]), RETREAT_POS)
        print(f"💥 {fire_count}발 명중 -> Shoot & Scoot")
        return jsonify({"status": "retreating"})

@app.route('/info', methods=['POST', 'GET'])
def info():
    global server_player_pos, current_body_yaw
    if request.method == 'POST':
        try:
            data = request.get_json(force=True) or {}
            if "playerBodyX" in data: current_body_yaw = float(data["playerBodyX"])
            pos = data.get('playerPos', {})
            server_player_pos = [float(pos.get('x',0)), float(pos.get('y',0)), float(pos.get('z',0))]
            return "OK", 200
        except: return "Error", 400
    else:
        return jsonify({"pos": {"x":server_player_pos[0], "y":server_player_pos[1], "z":server_player_pos[2]}, "fire_count": total_shot_count})

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle(): return jsonify({'status': 'success'})
@app.route('/collision', methods=['POST'])
def collision(): return jsonify({'status': 'success'})
@app.route('/init', methods=['GET'])
def init():
    return jsonify({"startMode": "start", "blStartX": 5, "blStartY": 10, "blStartZ": 5, "trackingMode": True, "detactMode": False, "logMode": True, "enemyTracking": False, "saveSnapshot": False, "saveLog": True, "saveLidarData": False, "lux": 30000})
@app.route('/start', methods=['GET'])
def start(): return jsonify({"control": ""})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)