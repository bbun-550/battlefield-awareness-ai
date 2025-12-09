from flask import Flask, request, jsonify
import math, time, logging, os
from combat import Gunner
from navigation import Navigator

app = Flask(__name__)

# 붛필요한 로그를 없애기 
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- 설정 및 전역 변수 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

# 3. 절대 경로 생성
MAP_FILE = os.path.join(ROOT_DIR, "map", "12_06_obstacle_v2.map")
CSV_FILE = os.path.join(ROOT_DIR, "log_data", "output.csv")
WAYPOINTS = [
    (110, 5), # (1번째 포인트) 
    (120.389, 181.441),  # (2번째 사격 포인트)   
    (81.959, 273.179),    # (3번째 코너 포인트)
]
RETREAT_POS = (111.44, 154.72)  # 사격 후 회피기동할 지점
FIRING_POS = WAYPOINTS[1]       # 다시 복귀해서 삭격할 지점

# 상태 변수
server_player_pos = [0, 0, 0]   # 내 탱크의 현재 위치 
current_key_wp_index = 0        # 현재 목표로 하는 웨이포인트 번호
fire_count = 0                  # 맞춘 적 타겟 수 
total_shot_count = 0            # 전체 누적 발사 수

# 플래그
is_fire_mode = False      # True면 사격 모드
is_retreating = False     # True면 후퇴 
is_returning = False      # True면 복귀 (후퇴 후 다시 사격 위치로 전진)
recenter_turret = False   # True면 포탑을 정면으로 정렬 시도
wait_start_time = None    # 1번 포인트 도착 후 3초 대기 타이머
scenario_start_time = None
last_fire_time = 0        # 마지막 발사 시간 (재장전 쿨타임 체크용)
fire_aim_start = None     # 조준이 완료된 시점 기록 (정밀 조준 대기용)
current_body_yaw = None   # 탱크 차체의 현재 회전 각도
has_faced_290 = False

# 로그 출력 설정
last_print_time = 0

# 모듈 인스턴스 생성
gunner = Gunner(MAP_FILE)
nav = Navigator(MAP_FILE)
path_generated = False      # 전체 경로가 생성되었는지 확인

# 각도를 -180~180도 사이로 변환해주는 함수
def normalize(a): return (a + 180.0) % 360.0 - 180.0

# =========================================================
# [메인 로직] 
@app.route("/get_action", methods=["POST"])
def get_action():
    global current_key_wp_index, is_fire_mode, is_retreating, is_returning
    global recenter_turret, wait_start_time, path_generated, fire_aim_start
    global total_shot_count, last_fire_time, current_body_yaw, server_player_pos, fire_count
    global last_print_time, has_faced_290

    # 유니티에서 보낸 데이터
    req = request.get_json(force=True) or {}
    pos = req.get("position", {})
    turret = req.get("turret", {})
    px, py, pz = float(pos.get("x", 0)), float(pos.get("y", 0)), float(pos.get("z", 0))
    tx, ty = float(turret.get("x", 0)), float(turret.get("y", 0))
    
    # 내 위치 없데이트
    server_player_pos = [px, py, pz]
    # 차체 각도 없데이트
    body_yaw = current_body_yaw if current_body_yaw is not None else tx

    if px == 0.0 and pz == 0.0:
        return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})

    # 최초 실행: 전체 경로 생성
    if not path_generated:
        full_path = []
        curr = (px, pz)
        for wp in WAYPOINTS:
            seg = nav.generate_path(curr, wp)   # 구간별 경로 생성
            full_path.extend(seg)               # 전체 경로에 추가
            curr = wp   
        nav.final_path = full_path              
        path_generated = True

    # ---------------------------------------------------------
    # [A] 포격 모드
    if is_fire_mode:
        # Gunner 모듈에게 타겟팅 위임
        tgt, dist = gunner.get_target(px, pz, index=fire_count)
        
        # 탄도 계산 시도
        sol = {"ok": False}
        if tgt: sol = gunner.calculate_solution(px, py, pz, tgt['x'], tgt['y'], tgt['z'])
        
        # 만약 n번째 타겟이 사거리 밖이거나 계산 불가하면 가장 가까운 적 조준
        if not sol["ok"]:
            tgt, dist = gunner.get_target(px, pz, index=0)
            if tgt: sol = gunner.calculate_solution(px, py, pz, tgt['x'], tgt['y'], tgt['z'])
        
        # 쏠 수 있는 적이 없으면 정지
        if not sol["ok"]:
            return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})
        
        # 포탑 회전 명령 계산
        ctrl = gunner.get_turret_control(tx, ty, sol["yaw"], sol["pitch"])
        fire = False
        if ctrl["aimed"]:
            if fire_aim_start is None: fire_aim_start = time.time()
            # 조준 후 1.5초 대기 + 재장전 쿨타임(7초) 체크
            if (time.time() - fire_aim_start >= 3.0) and (time.time() - last_fire_time >= 7.0):
                fire = True
                total_shot_count += 1
                last_fire_time = time.time()
                print(f"🔥 발사! (누적: {total_shot_count}발)") # 중요 이벤트는 출력
        else:
            fire_aim_start = None   # 조준 풀리면 타이머 초기화
            
        return jsonify({
            "moveWS": {"command": "STOP", "weight": 1}, "moveAD": {"command": "", "weight": 0},
            "turretQE": ctrl["turretQE"], "turretRF": ctrl["turretRF"], "fire": fire
        })

    # ---------------------------------------------------------
    # [B] 포탑 정렬 (포탑 복귀)
    if recenter_turret:
        yaw_err = normalize(body_yaw - tx)
        if abs(yaw_err) > 3.0:      # 오차가 3도 이상이면 회전
            return jsonify({
                "moveWS": {"command": "STOP", "weight": 1}, "moveAD": {"command": "", "weight": 0},
                "turretQE": {"command": "E" if yaw_err > 0 else "Q", "weight": 0.3}, "fire": False
            })
        recenter_turret = False     # 정렬 완료되면 종료

    # ---------------------------------------------------------
    # [C] 주행 시나리오 (위치에 따라 행동 결정)
    drift_mode = False
    is_combat_approach = (current_key_wp_index == 1) or is_returning    # 전투 지역 진입 여부
    
    if current_key_wp_index < len(WAYPOINTS):
        target_x, target_z = WAYPOINTS[current_key_wp_index]
    else:
        # 모든 웨이포인트를 지났으면 마지막 좌표를 유지 (에러 방지)
        target_x, target_z = WAYPOINTS[-1]

    # 거리 계산용 변수
    dist_to_wp = 0.0

    # [시나리오 1] 1번 웨이포인트: 도착 후 335도 회전 + 3초 대기
    if current_key_wp_index == 0:
        dist_to_wp = math.hypot(WAYPOINTS[0][0]-px, WAYPOINTS[0][1]-pz)
        # 1. 포탑을 335도로 회전
        if dist_to_wp < 3.5:

            if not has_faced_290:
                target_rot = 290.0
                diff = normalize(target_rot - tx)

                # 각도가 맞춰지지 않았다면 돌리기
                if abs(diff) > 4.0:
                    return jsonify({"moveWS": {"command": "STOP", "weight": 1}, 
                                    "moveAD": {"command": "", "weight": 0}, 
                                    "turretQE": {"command": "E" if diff > 0 else "Q", "weight": 0.3}, "fire": False})
                else:
                    if wait_start_time is None: 
                        wait_start_time = time.time()
                        print("1차 정렬(335도) 완료 -> 5초 대기 시작")
                    
                    # 3초가 아직 안 지났으면 정지
                    if time.time() - wait_start_time < 3.0:
                        return jsonify({"moveWS": {"command": "STOP", "weight": 1}, 
                                        "fire": False})
                    
                    # 3초 지남 -> 1단계 완료 처리
                    else:
                        has_faced_290 = True       # 1단계 완료 플래그
                        wait_start_time = None     # 타이머 리셋 (다음 단계를 위해 필수)
                        print("3초 대기 끝 -> 2차 회전 시작")
            
            # [2단계] 1단계가 끝났으므로 -> 70도 조준
            target_rot = 70.0
            diff = normalize(target_rot - tx)
        
            if abs(diff) > 4.0:
                return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "moveAD": {"command": "", "weight": 0}, "turretQE": {"command": "E" if diff > 0 else "Q", "weight": 0.3}, "fire": False})
                # 2. 회전 완료 후 객체인식을 위해 3초 대기
            else:
                # [3단계] 70도 회전까지 완료됨 -> 3초 대기 시작
                # (이 부분은 else 안에 있어야 회전이 끝난 뒤에 실행됩니다)
                
                if wait_start_time is None: 
                    wait_start_time = time.time()
                    print("1번 포인트 도착 -> 70도 정렬 완료 -> 3초 대기 시작")
                
                if time.time() - wait_start_time < 5.0:
                    # 3초가 안 지났으면 정지 상태 유지
                    return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})
                
                # 3초 지남 -> 다음 웨이포인트로 변경 및 변수 초기화
                wait_start_time = None
                has_faced_290 = False  # (중요) 다음 바퀴를 위해 초기화 필요할 수 있음 (상황에 따라 결정)
                recenter_turret = True
                current_key_wp_index = 1
                
                print("▶객체인식 완료 -> 이동 시작")
                return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})

    # [시나리오 2] 2번 웨이포인트: 포를 쏘고 엄폐
    elif current_key_wp_index == 1:
        # 발사 후 회피기동
        if is_retreating:
            # 후퇴 지점과의 거리 계산
            dist_retreat = math.hypot(RETREAT_POS[0]-px, RETREAT_POS[1]-pz)
            
            # 후퇴 지점 도착 (2m 이내)
            if dist_retreat < 2.0:
                # 타이머 시작
                if wait_start_time is None:
                    wait_start_time = time.time()
                    print("🛡️ 후퇴 완료 -> 3초간 숨어서 대기...")

                # 3초가 안 지났으면 정지
                if time.time() - wait_start_time < 3.0:
                    return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})
                
                # 3초 지남 -> 복귀 모드로 전환
                else:
                    wait_start_time = None          # 타이머 초기화 (필수)
                    is_retreating = False
                    is_returning = True
                    nav.generate_path((px, pz), FIRING_POS)  # 복귀 경로 생성
                    
                    print("3초 대기 끝 -> 다시 사격 위치로 전진!")
                    return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})
            
            # 도착 전이면 계속 이동 목표 설정
            target_x, target_z = nav.get_lookahead_target(px, pz, 3.5)
        
        # 2. 다시 포격 위치로 복귀 중일 때
        elif is_returning:
            # 포격 위치 도착 확인
            if math.hypot(FIRING_POS[0]-px, FIRING_POS[1]-pz) < 1.5:
                is_returning = False
                is_fire_mode = True
                print("⚔️ 포격 위치 복귀 완료 -> 사격 개시")
                return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})
            
            target_x, target_z = nav.get_lookahead_target(px, pz, 3.5)

        # 3. 처음 포격 위치에 도착했을 때 (가장 처음 진입 시)
        else:
            dist_to_wp = math.hypot(WAYPOINTS[1][0]-px, WAYPOINTS[1][1]-pz)
            if dist_to_wp < 4.0: 
                is_fire_mode = True
                print("🔥 포격 위치 최초 도착 -> 사격 개시")
                return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})
            
            target_x, target_z = nav.get_lookahead_target(px, pz, 3.5)

    # [시나리오 3] 나머지 구간 주행
    else:
        if current_key_wp_index >= len(WAYPOINTS):
            target_body_yaw = 270.0
            yaw_diff = normalize(target_body_yaw - body_yaw)

            # 오차가 4도 이상이면 제자리 회전 (Pivot Turn)
            if abs(yaw_diff) > 4.0:
                turn_cmd = "D" if yaw_diff > 0 else "A"
                print(f"최종 도착 -> 90도 정렬 중 (현재: {body_yaw:.1f})")
                return jsonify({
                    "moveWS": {"command": "STOP", "weight": 1}, 
                    "moveAD": {"command": turn_cmd, "weight": 0.8}, # 회전 속도 조절 필요 시 weight 변경
                    "fire": False
                })
            
            # 90도 정렬 완료 시 완전 정지
            else:
                return jsonify({"moveWS": {"command": "STOP", "weight": 1}, "fire": False})
        
        # 3번 웨이포인트 근처 20m에서는 멈춰서 회전 방지를 위해 부드러운 커브
        if math.hypot(WAYPOINTS[2][0]-px, WAYPOINTS[2][1]-pz) < 20.0: drift_mode = True
        
        wp_target = WAYPOINTS[current_key_wp_index]
        dist_to_wp = math.hypot(wp_target[0]-px, wp_target[1]-pz)
        
        # 웨이포인트 통과 체크 
        if dist_to_wp < (15.0 if current_key_wp_index == 2 else 3.5):
            print(f"웨이포인트 {current_key_wp_index} 통과")
            current_key_wp_index += 1
            if current_key_wp_index < len(WAYPOINTS): nav.generate_path((px, pz), WAYPOINTS[current_key_wp_index])
        target_x, target_z = nav.get_lookahead_target(px, pz, 6.0)

    control = nav.get_drive_control(px, pz, body_yaw, target_x, target_z, is_retreating=is_retreating, drift_mode=drift_mode, is_combat=is_combat_approach)
    
    # 1초에 한 번만 주행 상태 출력
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

# =========================================================
# [이벤트] 탄환이 적중했을 때 호출되는 함수

@app.route("/update_bullet", methods=["POST"])
def update_bullet():
    global is_fire_mode, fire_count, is_retreating, is_returning, recenter_turret, current_key_wp_index
    
    # 사격 모드가 아닐 때 온 신호는 무시
    if not is_fire_mode: return jsonify({"status": "ignored"})
    fire_count += 1
    
    # 3발 명중 -> 해당 구역 클리어 -> 다음 웨이포인트로 이동
    if fire_count >= 3:
        is_fire_mode = False; fire_count = 0; is_retreating = False; is_returning = False; recenter_turret = True
        current_key_wp_index += 1
        if current_key_wp_index < len(WAYPOINTS): nav.generate_path((server_player_pos[0], server_player_pos[2]), WAYPOINTS[current_key_wp_index])
        print("🎯 3발 명중 -> 다음 미션 이동")
        return jsonify({"status": "done"})
    
    # 1~2발 명중 -> 후퇴
    else:
        is_fire_mode = False; is_retreating = True; is_returning = False
        nav.generate_path((server_player_pos[0], server_player_pos[2]), RETREAT_POS)
        print(f"💥 {fire_count}발 명중 -> Shoot & Scoot")
        return jsonify({"status": "retreating"})

# =========================================================
# [기본 API] 정보 제공 및 초기화용

@app.route('/info', methods=['POST', 'GET'])
def info():
    global server_player_pos, current_body_yaw
    if request.method == 'POST':    # 내 위치 업데이트
        try:
            data = request.get_json(force=True) or {}
            if "playerBodyX" in data: current_body_yaw = float(data["playerBodyX"])
            pos = data.get('playerPos', {})
            server_player_pos = [float(pos.get('x',0)), float(pos.get('y',0)), float(pos.get('z',0))]
            return "OK", 200
        except: return "Error", 400
    else:   # 현재 상태 조회
        return jsonify({"pos": {"x":server_player_pos[0], "y":server_player_pos[1], "z":server_player_pos[2]}, "fire_count": total_shot_count})

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle(): return jsonify({'status': 'success'})
@app.route('/collision', methods=['POST'])
def collision(): return jsonify({'status': 'success'})
@app.route('/init', methods=['GET'])
def init():
    return jsonify({"startMode": "start", "blStartX": 34, 
                    "blStartY": 10, "blStartZ": 8, "trackingMode": True, 
                    "detactMode": False, "logMode": True, 
                    "enemyTracking": False, "saveSnapshot": False, 
                    "saveLog": True, "saveLidarData": False, "lux": 30000})
@app.route('/start', methods=['GET'])
def start(): return jsonify({"control": ""})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)