from flask import Flask, request, jsonify
import time, math, os, logging
from navigation import Navigator
from combat import Gunner

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PARENT_DIR = os.path.dirname(CURRENT_DIR)
MAP_FILE = os.path.join(PARENT_DIR, "map", "11_28.map")
CSV_FILE = os.path.join(PARENT_DIR, "log_data", "output.csv")

# 모듈 조립
nav = Navigator(MAP_FILE)
boom = Gunner(MAP_FILE, csv_file=CSV_FILE)

# 웨이포인트
WAYPOINTS = [
    (66.08, 45.93),   # [0] 시작 후 첫 경유지
    (120.38, 181.44), # [1] 사격 위치 
    (119.07, 287.42), # [2] 코너링 구간 
    (35.98, 284.19)   # [3] 최종 목적지
]
# 회피기동 포인트
RETREAT_POS = (111.44, 154.72)

class Context:
    wp_idx = 0          # 웨이포인트 인덱스
    pos = [0, 0, 0]     # 내 위치
    yaw = 0.0           # 내 각도
    
    fire_count = 0      # 발사 횟수
    last_fire_time = 0  # 쿨타임 체크용
    aim_start_time = None # 조준 시간 체크용
    wait_start_time = None
    
    mode = "DRIVING"    # 현재 상태 (DRIVING, SHOOTING, RETREAT, RETURN)
    path_initialized = False # 경로 초기화 여부
    last_log_time = 0

ctx = Context()

def normalize_angle(a):
    """각도를 -180 ~ 180도로 보정해주는 함수"""
    return (a + 180.0) % 360.0 - 180.0

# 4. 메인 로직
@app.route("/get_action", methods=["POST"])
def get_action():
    # 1. 데이터 수신 및 파싱
    req = request.get_json(force=True) or {}
    p = req.get("position", {})
    t = req.get("turret", {})
    
    ctx.pos = [float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", 0))]
    turret_yaw, turret_pitch = float(t.get("x", 0)), float(t.get("y", 0))
    ctx.yaw = float(req.get("playerBodyX", turret_yaw))
    
    px, py, pz = ctx.pos

    # 2. 초기 경로 생성 (게임 시작 시 1회)
    if not ctx.path_initialized:
        nav.update_path((px, pz), WAYPOINTS[ctx.wp_idx])
        ctx.path_initialized = True
    response = {"fire": False}

    # [상태 머신] 모드별 행동 결정
    # [A] 사격 모드 (SHOOTING)
    if ctx.mode == "SHOOTING":
        # 타겟 확인
        tgt, dist = boom.get_target(px, pz, ctx.fire_count)
        
        if tgt and dist < boom.max_range:
            # ★ 여기서 CSV를 참고해서 정교한 각도를 계산해옵니다.
            sol = boom.calculate_solution(px, py, pz, tgt['x'], tgt['y'], tgt['z'])
            
            # 포탑 제어 명령
            turret_cmd = boom.get_turret_control(turret_yaw, turret_pitch, sol['yaw'], sol['pitch'])
            response.update(turret_cmd)
            
            # 조준 완료 후 발사 로직
            if turret_cmd['aimed']:
                if ctx.aim_start_time is None: ctx.aim_start_time = time.time()
                
                # 1.5초 안정화 + 7초 쿨타임
                if (time.time() - ctx.aim_start_time > 1.5) and (time.time() - ctx.last_fire_time > 7.0):
                    response['fire'] = True
                    ctx.last_fire_time = time.time()
            else:
                ctx.aim_start_time = None
        else:
            ctx.fire_count = 0 # 타겟 없으면 리셋
            
        response["moveWS"] = {"command": "STOP", "weight": 1}

    # [B] 후퇴 (RETREAT)
    elif ctx.mode == "RETREAT":
        dist = math.hypot(RETREAT_POS[0] - px, RETREAT_POS[1] - pz)
        if dist < 2.0:
            print("후퇴 완료 -> 복귀(RETURN)")
            ctx.mode = "RETURN"
            nav.update_path((px, pz), WAYPOINTS[1])
        else:
            tgt = nav.get_pure_pursuit_target(px, pz, 3.5)
            response.update(nav.get_motor_control(px, pz, ctx.yaw, tgt[0], tgt[1], mode="REVERSE"))

    # [C] 복귀 (RETURN)
    elif ctx.mode == "RETURN":
        dist = math.hypot(WAYPOINTS[1][0] - px, WAYPOINTS[1][1] - pz)
        if dist < 1.5:
            print("복귀 완료 -> 사격(SHOOTING)")
            ctx.mode = "SHOOTING"
            response["moveWS"] = {"command": "STOP", "weight": 1}
        else:
            tgt = nav.get_pure_pursuit_target(px, pz, 3.5)
            response.update(nav.get_motor_control(px, pz, ctx.yaw, tgt[0], tgt[1], mode="PRECISION"))

    # [D] 이동 (DRIVING)
    else:
        curr_wp = WAYPOINTS[ctx.wp_idx]
        dist = math.hypot(curr_wp[0] - px, curr_wp[1] - pz)
        arrival_dist = 15.0 if ctx.wp_idx == 2 else 3.5

        if ctx.wp_idx == 0:
                target_rot = 335.0
                diff = normalize_angle(target_rot - turret_yaw)
                
                # 1. 포탑 각도가 안 맞으면 회전부터
                if abs(diff) > 4.0:
                    response["moveWS"] = {"command": "STOP", "weight": 1}
                    response["turretQE"] = {"command": "E" if diff > 0 else "Q", "weight": 0.3}
                    print(f"⏳ WP0 도착: 포탑 정렬 중... (오차: {diff:.1f})")
                    return jsonify(response)
                
                # 2. 각도 맞으면 3초 대기
                if ctx.wait_start_time is None:
                    ctx.wait_start_time = time.time()
                    print("⏳ WP0 정렬 완료: 3초 대기 시작")
                
                if time.time() - ctx.wait_start_time < 3.0:
                    response["moveWS"] = {"command": "STOP", "weight": 1}
                    return jsonify(response)
                
                # 3. 대기 끝났으면 다음으로 이동
                print("✅ WP0 대기 완료 -> 출발!")
                
                ctx.wp_idx += 1
                nav.update_path((px, pz), WAYPOINTS[ctx.wp_idx])
        
        elif ctx.wp_idx == 1:
                print("🔥 사격 위치 도착 -> 전투 개시")
                ctx.mode = "SHOOTING"
                response["moveWS"] = {"command": "STOP", "weight": 1}
                return jsonify(response)
            
        else:
            ctx.wp_idx += 1
            if ctx.wp_idx < len(WAYPOINTS):
                print(f"✅ WP 통과 -> {ctx.wp_idx}번 목표 설정")
                nav.update_path((px, pz), WAYPOINTS[ctx.wp_idx])
        
        is_drifting = (ctx.wp_idx == 3 or (ctx.wp_idx == 2 and dist < 20.0))
        drive_mode = "DRIFT" if is_drifting else "NORMAL"
        
        tgt = nav.get_pure_pursuit_target(px, pz, 6.0)
        response.update(nav.get_motor_control(px, pz, ctx.yaw, tgt[0], tgt[1], mode=drive_mode))

    if time.time() - ctx.last_log_time > 1.0:
        print(f"🚀 [상태:{ctx.mode}] WP:{ctx.wp_idx} | Pos:({px:.1f}, {pz:.1f}) | Fire:{ctx.fire_count}")
        ctx.last_log_time = time.time()
    return jsonify(response)

# ============================================================
# 5. 이벤트 핸들러
# ============================================================
@app.route("/update_bullet", methods=["POST"])
def update_bullet():
    if ctx.mode == "SHOOTING":
        ctx.fire_count += 1
        print(f"명중! ({ctx.fire_count}/3)")
        
        if ctx.fire_count >= 3:
            print("미션 완료 -> 이동 재개")
            ctx.mode = "DRIVING"
            ctx.fire_count = 0
            ctx.wp_idx += 1
            if ctx.wp_idx < len(WAYPOINTS):
                nav.update_path(tuple(ctx.pos[0::2]), WAYPOINTS[ctx.wp_idx])
            return jsonify({"status": "done"})
        else:
            print("Shoot & Scoot -> 회피기동")
            ctx.mode = "RETREAT"
            nav.update_path(tuple(ctx.pos[0::2]), RETREAT_POS)
            return jsonify({"status": "retreating"})
            
    return jsonify({"status": "ignored"})

@app.route('/info', methods=['POST', 'GET'])
def info(): return jsonify({"status": "OK", "mode": ctx.mode})
@app.route('/init', methods=['GET'])
def init():
    return jsonify({
        "startMode": "start",
        "blStartX": 5,  
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
    })
@app.route('/start', methods=['GET'])
def start(): return jsonify({"control": "start"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)