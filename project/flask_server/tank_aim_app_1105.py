"""
탱크 자동 조준 시스템 (Tank Auto-Aiming System)

주요 기능:
1. 탄도학 계산 (Ballistic Physics): 포탄의 포물선 궤적 예측
2. 경사 보정 (Slope Compensation): 지형 경사와 높이 차이 자동 보정
3. PID 제어 (Proportional Control): 포탑 회전 속도 자동 조절
4. 적응형 발사 허용치 (Adaptive Fire Gate): 거리/지형에 따른 동적 조준 정확도

사용 방법:
    aimer = TankAimer()
    action = aimer.get_action_dict(current_yaw, current_pitch)
    # action["fire"] = True/False (발사 여부)
    # action["turretQE"] = 좌우 회전 명령
    # action["turretRF"] = 상하 회전 명령
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import os, math, threading, collections
import numpy as np
import pandas as pd

# ========================
# 설정 파라미터 클래스들
# (환경 변수로 외부에서 조정 가능, 기본값 제공)
# ========================
@dataclass
class AimPaths:
    """파일 경로 설정

    LOG_PATH: Unity에서 실시간으로 기록하는 탱크 위치/적 위치 로그 파일
    OUTPUT_CSV: 사전 측정한 거리별 발사각 데이터 (Fallback용)
    """
    LOG_PATH: str   = os.getenv("LOG_PATH",   r"C:\Users\SeYun\OneDrive\문서\Tank Challenge\log_data\tank_info_log.txt")
    OUTPUT_CSV: str = os.getenv("OUTPUT_CSV", r"C:\Users\SeYun\OneDrive\문서\Tank Challenge\log_data\output.csv")

@dataclass
class AimGeom:
    """탱크 기하학적 파라미터

    L: 포신 길이 (미터) - 포구 위치 계산에 사용
    H_OFFSET: 포탑 높이 (미터) - 지상에서 포탑까지의 수직 거리
    """
    L: float        = float(os.getenv("BARREL_LEN", "0.0"))
    H_OFFSET: float = float(os.getenv("H_OFFSET",  "2.1"))

@dataclass
class AimLimits:
    """조준 한계값 설정

    MAX_RANGE: 최대 사거리 (미터) - 이 거리를 넘으면 조준 불가
    MIN_PITCH_CFG: 포신 최저 각도 (도) - 아래로 내릴 수 있는 한계 (음수)
    MAX_PITCH_CFG: 포신 최고 각도 (도) - 위로 올릴 수 있는 한계 (양수)
    """
    MAX_RANGE: float     = float(os.getenv("MAX_RANGE", "130.0"))
    MIN_PITCH_CFG: float = float(os.getenv("MIN_PITCH", "-30.0"))
    MAX_PITCH_CFG: float = float(os.getenv("MAX_PITCH", "10.0"))

@dataclass
class AimPhysics:
    """탄도 물리 파라미터

    V_INIT: 포탄 초속 (m/s) - 발사 직후 속도
    G: 중력 가속도 (m/s²) - 지구 중력 (9.81)
    """
    V_INIT: float = float(os.getenv("V_INIT", "58"))
    G: float      = float(os.getenv("G", "9.81"))

@dataclass
class AimSlope:
    """경사 및 높이 차이 보정 계수

    K0: 기본 오프셋 (도)
    K_ROLL: 좌우 기울기(Roll) 보정 계수 (도/도) - 탱크가 옆으로 기울었을 때
    K_DH: 높이 차이(ΔH) 보정 계수 (도/미터) - 적이 높은/낮은 곳에 있을 때
    PITCH_BIAS_DEG: 추가 고정 보정값 (도)
    ROLL_THRESH_DEG: Roll 보정 적용 시작 임계값 (도) - 이 각도 이상 기울면 보정
    DH_THRESH_M: 높이 차이 보정 적용 시작 임계값 (미터)
    ROLL_SIGN: Roll 방향 부호 조정 (±1.0)
    """
    K0: float               = float(os.getenv("K0", "0.0"))
    K_ROLL: float           = float(os.getenv("K_ROLL", "0.0"))
    K_DH: float             = float(os.getenv("K_DH", "0.0"))
    PITCH_BIAS_DEG: float   = float(os.getenv("PITCH_BIAS_DEG", "0.0"))
    ROLL_THRESH_DEG: float  = float(os.getenv("ROLL_THRESH_DEG", "5.0"))
    DH_THRESH_M: float      = float(os.getenv("DH_THRESH_M", "1.0"))
    ROLL_SIGN: float        = float(os.getenv("ROLL_SIGN", "1.0"))

@dataclass
class AimSlew:
    """포탑 회전 속도 제어 (Slew Rate Control)

    FAST_SLEW_ENABLE: 고속 회전 모드 활성화 여부
    YAW_FAST_STEP_DEG: Yaw(좌우) 고속 회전 단위 (도)
    PITCH_DOWN_FAST_STEP_DEG: Pitch(상하) 고속 회전 단위 (도)
    FAST_SLEW_YAW_THRESH_DEG: 고속 회전 시작 임계값 (도)
    """
    FAST_SLEW_ENABLE: bool        = os.getenv("FAST_SLEW_ENABLE", "1") != "0"
    YAW_FAST_STEP_DEG: float      = float(os.getenv("YAW_FAST_STEP_DEG", "1.2"))
    PITCH_DOWN_FAST_STEP_DEG: float = float(os.getenv("PITCH_DOWN_FAST_STEP_DEG", "0.6"))
    FAST_SLEW_YAW_THRESH_DEG: float = float(os.getenv("FAST_SLEW_YAW_THRESH_DEG", "1.5"))

@dataclass
class AimFireGate:
    """발사 허용 조건 (Fire Permission Gate)

    발사는 조준 오차가 허용 범위 내에 있을 때만 승인됨

    FIRE_TOL_YAW_DEG_BASE: Yaw 기본 허용 오차 (도)
    FIRE_TOL_PITCH_DEG_BASE: Pitch 기본 허용 오차 (도)
    FIRE_TOL_YAW_MAX: Yaw 최대 허용 오차 (도)
    FIRE_TOL_PITCH_MAX: Pitch 최대 허용 오차 (도)
    MICRO_FIRE_ENABLE: 미세 조준 모드 활성화
    MICRO_FIRE_WINDOW: 오차 이력 윈도우 크기 (프레임 수)
    MICRO_FIRE_SOFT_DEG: 미세 조준 판정 기준 (도)
    """
    FIRE_TOL_YAW_DEG_BASE: float   = float(os.getenv("FIRE_TOL_YAW_DEG",   "1.3"))
    FIRE_TOL_PITCH_DEG_BASE: float = float(os.getenv("FIRE_TOL_PITCH_DEG", "1.3"))
    FIRE_TOL_YAW_MAX: float        = float(os.getenv("FIRE_TOL_YAW_MAX",   "2.5"))
    FIRE_TOL_PITCH_MAX: float      = float(os.getenv("FIRE_TOL_PITCH_MAX", "3.0"))
    MICRO_FIRE_ENABLE: bool        = os.getenv("MICRO_FIRE_ENABLE", "1") != "0"
    MICRO_FIRE_WINDOW: int         = int(float(os.getenv("MICRO_FIRE_WINDOW", "4")))
    MICRO_FIRE_SOFT_DEG: float     = float(os.getenv("MICRO_FIRE_SOFT_DEG", "3.0"))

# ========================
# 유틸 함수
# ========================

def normalize_deg(a: float) -> float:
    """각도를 -180 ~ +180 범위로 정규화

    예: 270° → -90°, 540° → 180°
    포탑 회전 방향 결정 시 필수 (반대편으로 돌면 비효율)
    """
    return (a + 180.0) % 360.0 - 180.0

# ========================
# 메인 조준 시스템 클래스
# ========================
class TankAimer:
    """탱크 포탑 자동 조준 시스템

    핵심 기능:
    1. 탄도학 계산 (발사각 산출)
    2. 경사/높이 차이 보정
    3. 포탑 회전 속도 제어 (Slew Rate)
    4. 발사 허가 판정 (Fire Gate)

    사용 흐름:
    1. __init__() - 설정 초기화
    2. compute() - 매 프레임 호출하여 조준 명령 계산
    3. update_bias() - 실시간 오차 보정 (옵션)
    """

    def __init__(self,
                 paths: AimPaths | None = None,
                 geom: AimGeom | None = None,
                 limits: AimLimits | None = None,
                 physics: AimPhysics | None = None,
                 slope: AimSlope | None = None,
                #  calib: AimCalib | None = None,
                 slew: AimSlew | None = None,
                 firegate: AimFireGate | None = None):
        """조준 시스템 초기화

        매개변수:
            paths: 로그 파일 경로 설정
            geom: 탱크 기하학적 치수
            limits: 조준 한계값 (사거리, 각도 제한)
            physics: 탄도 물리 파라미터
            slope: 경사 보정 계수
            slew: 포탑 회전 속도 제어
            firegate: 발사 허용 조건

        내부 상태:
            self.BIAS: 실시간 조준 오차 보정값 (Yaw/Pitch)
            self.ERR_HIST: 조준 오차 이력 (발사 판정용)
            self.LAST_STATE: 이전 프레임 상태 (변화 감지)
        """
        self.paths   = paths   or AimPaths()
        self.geom    = geom    or AimGeom()
        self.limits  = limits  or AimLimits()
        self.physics = physics or AimPhysics()
        self.slope   = slope   or AimSlope()
        # self.calib   = calib   or AimCalib()
        self.slew    = slew    or AimSlew()
        self.fg      = firegate or AimFireGate()

        self.lock = threading.Lock()  # 멀티스레드 동기화 (Flask 서버용)

        # 실시간 조준 오차 보정값 (사격 후 결과에 따라 업데이트)
        self.BIAS = {"yaw": float(os.getenv("BIAS_YAW_DEG", "0.0")),
                     "pitch": float(os.getenv("BIAS_PITCH_DEG", "0.0"))}

        # 조준 오차 이력 (발사 허가 판정용 - 최근 N프레임의 안정성 체크)
        self.ERR_HIST = collections.deque(maxlen=self.fg.MICRO_FIRE_WINDOW)

        # 유효 Pitch 범위 (지형에 따라 동적 조정 가능)
        self.MIN_PITCH_EFF = self.limits.MIN_PITCH_CFG
        self.MAX_PITCH_EFF = self.limits.MAX_PITCH_CFG

        # 이전 프레임 상태 (변화 감지 및 디버깅용)
        self.LAST_STATE = {"gx":None,"gy":None,"gz":None,
                           "ex":None,"ey":None,"ez":None,
                           "px":None,"py":None,"pz":None}

    # ---------- 내부 수학 함수들 ----------

    def _turret_muzzle_from_current(self, px, py, pz, turret_yaw_deg, turret_pitch_deg):
        """현재 포탑 각도에서 포구(발사점) 위치 계산

        매개변수:
            px, py, pz: 탱크 차체 위치 (미터)
            turret_yaw_deg: 현재 포탑 Yaw 각도 (도)
            turret_pitch_deg: 현재 포탑 Pitch 각도 (도)

        반환값:
            (gx, gy, gz): 포구 3D 위치

        계산 방법:
            1. 각도를 라디안으로 변환
            2. 포신 방향 벡터 계산 (cos/sin 이용)
            3. 탱크 위치 + 포신 길이 * 방향 벡터
            4. 높이에 H_OFFSET 추가 (포탑 높이)
        """
        yaw   = math.radians(turret_yaw_deg)
        pitch = math.radians(turret_pitch_deg)

        # 포신 방향 단위 벡터 (정규화된 3D 벡터)
        dx = math.cos(pitch) * math.cos(yaw)
        dy = math.sin(pitch)
        dz = math.cos(pitch) * math.sin(yaw)

        # 포구 위치 = 차체 위치 + 포신 길이 * 방향 벡터
        gx = px + self.geom.L * dx
        gy = py + self.geom.H_OFFSET + self.geom.L * dy  # 포탑 높이 고려
        gz = pz + self.geom.L * dz
        return gx, gy, gz

    def _ballistic_pitch_deg(self, sx, sy, sz, ex, ey, ez):
        """탄도학 기반 발사각 계산 (물리 시뮬레이션)

        매개변수:
            sx, sy, sz: 발사 지점 (포구 위치)
            ex, ey, ez: 목표 지점 (적 위치)

        반환값:
            발사각 (Pitch, 도) 또는 None (도달 불가)

        원리:
            탄도 방정식: y = tan(θ)·x - (g·x²)/(2·v²·cos²(θ))
            → 2차 방정식으로 변환하여 각도 θ 계산

        판별식(D)으로 해 존재 여부 확인:
            D ≥ 0: 해 존재 (도달 가능)
            D < 0: 해 없음 (사거리 초과, 불가능한 각도)
        """
        v = self.physics.V_INIT; g = self.physics.G
        dx = ex - sx; dz = ez - sz; dy = ey - sy
        d_horiz = math.hypot(dx, dz)  # 수평 거리 (√(dx² + dz²))

        if d_horiz < 1e-6:  # 목표가 발사 지점과 거의 동일 (0으로 나누기 방지)
            return False, None

        # 탄도 방정식의 2차 방정식 판별식
        # term = v⁴ - g(g·d² + 2·Δy·v²)
        v2 = v*v
        term = v2*v2 - g*(g*d_horiz*d_horiz + 2.0*dy*v2)

        if term < 0.0:  # 판별식이 음수면 해가 없음 (사거리 초과)
            return False, None

        # 두 해(high angle, low angle) 중 낮은 각도 선택 (빠른 탄도)
        root = math.sqrt(term)
        theta1 = math.atan((v2 + root) / (g * d_horiz))  # 높은 각도 (곡사)
        theta2 = math.atan((v2 - root) / (g * d_horiz))  # 낮은 각도 (직사)
        return True, math.degrees(min(theta1, theta2))  # Low angle 선택

    def _angle_from_output_csv(self, desired_range):
        """사전 측정 데이터에서 발사각 조회 (Fallback 방법)

        매개변수:
            desired_range: 목표까지의 거리 (미터)

        반환값:
            (성공여부, 발사각, 에러메시지)

        용도:
            물리 계산이 실패하거나 부정확할 때 사용
            실제 Unity 환경에서 측정한 거리-각도 데이터 참조

        동작:
            1. output.csv 로드 (Angle, X, Y, Z 컬럼)
            2. 거리 계산: √(X² + Z²)
            3. 요청 거리와 가장 가까운 행 찾기
            4. 해당 행의 Angle 반환
        """
        csv_path = self.paths.OUTPUT_CSV
        if not os.path.exists(csv_path):
            return False, None, "output.csv not found"

        df = pd.read_csv(csv_path, names=None)
        if set(["Angle","X","Y","Z"]).issubset(df.columns):
            angs = df["Angle"].to_numpy(); zs = df["Z"].to_numpy()
        else:  # 컬럼명이 없는 경우 (숫자 인덱스로 접근)
            arr = df.to_numpy()
            if arr.shape[1] < 4:
                return False, None, "output.csv unexpected columns"
            angs = arr[:,0].astype(float); zs = arr[:,3].astype(float)  # Z축 거리 사용

        # Z 거리 기준 정렬 (보간 준비)
        idx = np.argsort(zs); zs_s = zs[idx]; ang_s = angs[idx]

        # 요청 거리가 데이터 범위를 벗어나면 실패
        if desired_range < zs_s[0]-1e-8 or desired_range > zs_s[-1]+1e-8:
            return False, None, f"desired_range {desired_range:.2f} out [{zs_s[0]:.2f},{zs_s[-1]:.2f}]"

        # 선형 보간으로 각도 계산
        angle_interp  = float(np.interp(desired_range, zs_s, ang_s))
        angle_clamped = max(self.MIN_PITCH_EFF, min(self.MAX_PITCH_EFF, angle_interp))
        return True, angle_clamped, f"interp ok ({angle_interp:.3f} -> clamp {angle_clamped:.3f})"

    # ---------- 공개 메인 로직 ----------

    def compute_solution(self, use_ballistic: bool = True, use_data_fallback: bool = True) -> Dict[str, Any]:
        """조준 해(Solution) 계산 - 시스템의 핵심 함수

        매개변수:
            use_ballistic: 물리 기반 탄도 계산 사용 여부
            use_data_fallback: 물리 계산 실패 시 CSV 데이터 사용 여부

        반환값:
            Dict containing:
                ok: 계산 성공 여부
                reason: 실패 이유 (실패 시)
                desired_yaw: 목표 Yaw 각도
                desired_pitch: 목표 Pitch 각도
                cmd_yaw: Yaw 조정 명령 (0/±1)
                cmd_pitch: Pitch 조정 명령 (0/±1)
                fire: 발사 허가 여부

        계산 흐름:
            1. 로그 파일에서 최신 탱크/적 위치 읽기
            2. 사거리 체크 (MAX_RANGE 초과 시 실패)
            3. Yaw 각도 계산 (atan2로 방향 계산)
            4. Pitch 각도 계산 (3단계 시도):
                a) 탄도학 물리 계산
                b) CSV 데이터 조회 (Fallback)
                c) LOS (Line of Sight) 각도 (최종 Fallback)
            5. 경사/높이 보정 적용
            6. BIAS 추가 (실시간 조준 오차 보정)
            7. 조정 명령 생성 (현재 각도 → 목표 각도)
            8. 발사 허가 판정 (오차 범위 체크)
        """
        # 로그 파일 존재 확인
        if not os.path.exists(self.paths.LOG_PATH):
            raise FileNotFoundError(f"log not found: {self.paths.LOG_PATH}")

        df = pd.read_csv(self.paths.LOG_PATH)
        if df.shape[0] == 0:
            raise ValueError("log empty")
        last = df.iloc[-1]  # 최신 프레임 데이터

        # 자동 Pitch 한계값 업데이트 (로그에서 실제 사용 범위 추출)
        if os.getenv("AUTO_MIN_PITCH_FROM_LOG", "1") != "0" and "Player_Turret_Y" in df.columns:
            obs_min = float(df["Player_Turret_Y"].min())
            self.MIN_PITCH_EFF = min(self.limits.MIN_PITCH_CFG, obs_min - 0.5)
            obs_max = float(df["Player_Turret_Y"].max())
            self.MAX_PITCH_EFF = max(self.limits.MAX_PITCH_CFG, obs_max + 0.5)
        else:
            self.MIN_PITCH_EFF = self.limits.MIN_PITCH_CFG
            self.MAX_PITCH_EFF = self.limits.MAX_PITCH_CFG

        # 탱크 및 적 위치 읽기
        px = float(last["Player_Pos_X"]) ; py = float(last["Player_Pos_Y"]) ; pz = float(last["Player_Pos_Z"])
        cur_yaw   = float(last["Player_Turret_X"]) ; cur_pitch = float(last["Player_Turret_Y"])
        ex = float(last["Enemy_Pos_X"]) ; ey = float(last["Enemy_Pos_Y"]) ; ez = float(last["Enemy_Pos_Z"])

        # 탱크 차체 기울기 (Roll) 읽기
        try:
            body_roll_deg = self.slope.ROLL_SIGN * normalize_deg(float(last.get("Player_Body_Z", 0.0)))
        except Exception:
            body_roll_deg = 0.0

        # 발사 지점 (포탑 높이 고려)
        sx, sy, sz = px, (py + self.geom.H_OFFSET), pz
        dx = ex - sx; dz = ez - sz; dy = ey - sy
        horiz = math.hypot(dx, dz)  # 수평 거리

        # 사거리 체크
        if horiz > self.limits.MAX_RANGE:
            return {"ok": False, "reason": f"out_of_range (horiz {horiz:.2f} m > {self.limits.MAX_RANGE} m)"}

        # Yaw 계산 (목표 방향)
        desired_yaw = math.degrees(math.atan2(dx, dz))

        # Pitch 계산 (3단계 시도)
        note = []
        base_pitch = None

        # 1단계: 탄도학 물리 계산
        if use_ballistic:
            ok, bp = self._ballistic_pitch_deg(sx, sy, sz, ex, ey, ez)
            if ok:
                base_pitch = max(self.MIN_PITCH_EFF, min(self.MAX_PITCH_EFF, bp))
                note.append(f"ballistic {bp:.3f} -> clamp {base_pitch:.3f}")

        # 2단계: CSV 데이터 Fallback
        if base_pitch is None and use_data_fallback:
            data_ok, data_angle, data_info = self._angle_from_output_csv(horiz)
            if data_ok:
                base_pitch = data_angle
                note.append(f"data {data_info}")

        # 3단계: LOS (Line of Sight) 최종 Fallback
        if base_pitch is None:
            los_pitch = math.degrees(math.atan2(dy, horiz))
            base_pitch = max(self.MIN_PITCH_EFF, min(self.MAX_PITCH_EFF, los_pitch))
            note.append(f"los {los_pitch:.3f} -> clamp {base_pitch:.3f}")

        # 경사/높이 보정 계산
        corr_pitch = 0.0
        if abs(body_roll_deg) >= self.slope.ROLL_THRESH_DEG or abs(dy) >= self.slope.DH_THRESH_M:
            corr_pitch = self.slope.K0 + self.slope.K_ROLL*body_roll_deg + self.slope.K_DH*dy + self.slope.PITCH_BIAS_DEG
            note.append(f"corr={corr_pitch:.3f} (roll={body_roll_deg:.2f}°, Δh={dy:.2f}m)")
        else:
            note.append("corr skipped (flat-ish)")

        # BIAS 적용 (실시간 오차 보정)
        with self.lock:
            bias_yaw, bias_pitch = self.BIAS["yaw"], self.BIAS["pitch"]

        # 최종 목표 각도 계산
        desired_yaw = normalize_deg(desired_yaw + bias_yaw)
        desired_pitch_raw = base_pitch + corr_pitch + bias_pitch
        desired_pitch = max(self.MIN_PITCH_EFF, min(self.MAX_PITCH_EFF, desired_pitch_raw))

        if desired_pitch != desired_pitch_raw:
            note.append(f"pitch clamped to [{self.MIN_PITCH_EFF:.1f},{self.MAX_PITCH_EFF:.1f}]")

        # 현재 각도와의 차이 계산
        yaw_delta   = normalize_deg(desired_yaw - cur_yaw)
        pitch_delta = normalize_deg(desired_pitch - cur_pitch)

        # 현재 포구 위치 계산 및 상태 저장
        gx, gy, gz = self._turret_muzzle_from_current(px, py, pz, cur_yaw, cur_pitch)
        with self.lock:
            self.LAST_STATE.update({"gx": gx, "gy": gy, "gz": gz,
                                    "ex": ex, "ey": ey, "ez": ez,
                                    "px": px, "py": py, "pz": pz})

        # 결과 반환 (디버깅 정보 포함)
        return {
            "ok": True, "method": "+".join([s for s in ["ballistic" if "ballistic" in ";".join(note) else None, "data" if "interp" in ";".join(note) else None, "los" if "los" in ";".join(note) else None] if s]),
            "note": "; ".join(note),
            "desired_yaw": round(desired_yaw, 3), "desired_pitch": round(desired_pitch, 3),
            "yaw_delta": round(yaw_delta, 3), "pitch_delta": round(pitch_delta, 3),
            "corr_pitch": round(corr_pitch, 3), "dy": round(dy, 3), "roll_deg": round(body_roll_deg, 3),
            "R": round(horiz, 3), "bias_yaw": round(bias_yaw, 3), "bias_pitch": round(bias_pitch, 3),
            "min_pitch_eff": round(self.MIN_PITCH_EFF, 2), "max_pitch_eff": round(self.MAX_PITCH_EFF, 2)
        }

    def compute_turret_weights(self, current_x: float, current_y: float, target_x: float, target_y: float) -> Dict[str, Any]:
        """포탑 회전 가중치 및 명령 계산 (비례 제어)

        매개변수:
            current_x: 현재 Yaw 각도 (도)
            current_y: 현재 Pitch 각도 (도)
            target_x: 목표 Yaw 각도 (도)
            target_y: 목표 Pitch 각도 (도)

        반환값:
            Dict containing:
                cmd_x: Yaw 조정 명령 ('Q' 왼쪽, 'E' 오른쪽, '' 정지)
                cmd_y: Pitch 조정 명령 ('R' 위, 'F' 아래, '' 정지)
                weight_x: Yaw 회전 강도 (0.0~1.0)
                weight_y: Pitch 회전 강도 (0.0~1.0)
                error_x: Yaw 오차 (도)
                error_y: Pitch 오차 (도)

        동작 원리:
            비례 제어 (P-Control): 오차에 비례하여 회전 속도 조절
            Weight = min(|오차| × Kp, 1.0)

            Kp 값 조정:
                - KP_X: Yaw 비례 계수 (작을수록 천천히 회전)
                - KP_Y: Pitch 비례 계수
                - KP_MAX_X/Y: 최대 비례 계수 (안전장치)

            FAST SLEW:
                오차가 임계값을 넘으면 고속 회전 모드
                Weight를 제한하여 부드러운 감속
        """
        # 오차 계산
        error_x = normalize_deg(target_x - current_x)
        error_y = target_y - current_y  # +올림, -내림

        # 비례 제어 기본 계수 (Yaw 느리게, Pitch 빠르게)
        KP_X         = float(os.getenv("KP_X", "0.05"))
        KP_Y         = float(os.getenv("KP_Y", "0.3"))
        KP_MAX_X     = float(os.getenv("KP_MAX_X", "1"))
        KP_MAX_Y     = float(os.getenv("KP_MAX_Y", "0.6"))

        Kp_x = min(KP_X, KP_MAX_X)
        Kp_y = min(KP_Y, KP_MAX_Y)

        # 회전 방향 결정
        cmd_x = 'E' if error_x > 0 else 'Q' if error_x < 0 else ''
        cmd_y = 'R' if error_y > 0 else 'F' if error_y < 0 else ''

        # 회전 강도 계산 (비례 제어)
        weight_x = min(abs(error_x) * Kp_x, 1.0)
        weight_y = min(abs(error_y) * Kp_y, 1.0)

        # FAST SLEW 완화 (급격한 변화 방지)
        if self.slew.FAST_SLEW_ENABLE:
            # Pitch 고속 회전 (큰 오차 시)
            if abs(error_y) >= self.slew.PITCH_DOWN_FAST_STEP_DEG:
                weight_y = min(weight_y, 0.9)  # 최대 강도 제한
                cmd_y = 'R' if error_y > 0 else 'F'

            # Yaw 고속 회전 (매우 큰 오차 시)
            if abs(error_x) >= max(self.slew.FAST_SLEW_YAW_THRESH_DEG * 5.0, self.slew.YAW_FAST_STEP_DEG * 5.0):
                weight_x = max(weight_x, 0.5)  # 최소 강도 보장
                cmd_x = 'E' if error_x > 0 else 'Q'

        return {
            "turretQE": {"command": cmd_x, "weight": round(weight_x, 3)},
            "turretRF": {"command": cmd_y, "weight": round(weight_y, 3)},
            "error_x": round(error_x, 3),
            "error_y": round(error_y, 3)
        }

    def adaptive_fire_gate(self, err_x: float, err_y: float, R: float, roll_deg: float, dy: float, desired_pitch: float):
        """발사 허가 판정 (적응형 Fire Gate)

        매개변수:
            err_x: Yaw 오차 (도)
            err_y: Pitch 오차 (도)
            R: 목표까지의 거리 (미터)
            roll_deg: 탱크 Roll 각도 (도)
            dy: 높이 차이 (미터)
            desired_pitch: 목표 Pitch 각도 (도)

        반환값:
            (fire, reason, detail)
            - fire: 발사 허가 여부 (True/False)
            - reason: 허가/거부 이유
            - detail: 상세 정보 (디버깅용)

        판정 기준:
            1. 기본 허용 오차 체크
               - |err_x| ≤ yaw_tol
               - |err_y| ≤ pitch_tol

            2. 미세 조준 모드 (MICRO_FIRE_ENABLE)
               - 최근 N 프레임의 오차가 모두 작은지 체크
               - 오차 이력: self.ERR_HIST
               - 판정 기준: MICRO_FIRE_SOFT_DEG

            3. 적응형 허용 오차
               - 거리가 멀수록 허용 오차 증가 (먼 거리는 덜 정밀해도 OK)
               - 경사/높이 차이가 크면 허용 오차 증가

        미세 조준 모드의 이점:
            - 단발 명중률 향상
            - 움직이는 목표 추적 시 안정성 확보
            - 급격한 발사 금지로 탄약 절약
        """
        ax, ay = abs(err_x), abs(err_y)
        yaw_tol   = self.fg.FIRE_TOL_YAW_DEG_BASE
        pitch_tol = self.fg.FIRE_TOL_PITCH_DEG_BASE

        # 적응형 허용 오차 계산
        # 1. 거리 보정: 60m 이상부터 거리에 비례하여 증가
        yaw_tol   = min(self.fg.FIRE_TOL_YAW_MAX,   yaw_tol   + 0.0020*max(0.0, R-60.0) + (0.3 if abs(roll_deg) >= self.slope.ROLL_THRESH_DEG else 0.0))
        pitch_tol = min(self.fg.FIRE_TOL_PITCH_MAX, pitch_tol + 0.0025*max(0.0, R-60.0) + (0.3 if abs(dy) >= self.slope.DH_THRESH_M else 0.0))

        # 2. Pitch 하한 도달 시 추가 완화 (물리적 한계 근처)
        if desired_pitch <= self.MIN_PITCH_EFF + 1e-3:
            pitch_tol = min(self.fg.FIRE_TOL_PITCH_MAX, pitch_tol + 0.6)

        # 3. 미세 조준 모드 (오차 이력 기반)
        if self.fg.MICRO_FIRE_ENABLE and len(self.ERR_HIST) >= 3:
            # 이전 프레임들의 평균 오차 계산
            mean_prev_x = sum(e[0] for e in list(self.ERR_HIST)[:-1]) / (len(self.ERR_HIST)-1)
            mean_prev_y = sum(e[1] for e in list(self.ERR_HIST)[:-1]) / (len(self.ERR_HIST)-1)

            # 현재 오차가 충분히 작고, 이전보다 개선되었으면 보너스 허용 오차
            if (ax <= self.fg.MICRO_FIRE_SOFT_DEG and ay <= self.fg.MICRO_FIRE_SOFT_DEG
                and ax <= mean_prev_x + 0.05 and ay <= mean_prev_y + 0.05):
                yaw_tol   = min(self.fg.FIRE_TOL_YAW_MAX,   yaw_tol   + 0.3)
                pitch_tol = min(self.fg.FIRE_TOL_PITCH_MAX, pitch_tol + 0.3)

        # 최종 판정: 두 축 모두 허용 오차 이내여야 발사 허가
        return (ax <= yaw_tol) and (ay <= pitch_tol), yaw_tol, pitch_tol

    # ---------- 통합 API ----------

    def get_action_dict(self, turret_x: float, turret_y: float) -> Dict[str, Any]:
        """조준 시스템 통합 API (Flask 엔드포인트용)

        매개변수:
            turret_x: 현재 포탑 Yaw 각도 (도)
            turret_y: 현재 포탑 Pitch 각도 (도)

        반환값:
            Dict containing:
                moveWS: 차체 전후진 명령 (이 스크립트에서는 STOP 고정)
                moveAD: 차체 좌우 이동 명령 (사용 안 함)
                turretQE: Yaw 조정 명령 및 강도
                turretRF: Pitch 조정 명령 및 강도
                fire: 발사 허가 여부
                debug: 상세 디버깅 정보 (조준 계산 과정)

        호출 흐름:
            1. compute_solution() - 목표 각도 계산
            2. compute_turret_weights() - 회전 명령 생성
            3. adaptive_fire_gate() - 발사 허가 판정
            4. 결과 종합하여 반환

        사용 예시 (Flask):
            @app.route('/aim', methods=['POST'])
            def aim():
                data = request.json
                result = aimer.get_action_dict(
                    turret_x=data['turret_x'],
                    turret_y=data['turret_y']
                )
                return jsonify(result)
        """
        # 1. 목표 각도 계산
        aim = self.compute_solution(use_ballistic=True, use_data_fallback=True)
        if not aim.get("ok", False):
            return {"status": "error", "reason": aim.get("reason", "unknown")}

        target_x, target_y = aim["desired_yaw"], aim["desired_pitch"]

        # 2. 회전 명령 생성 (비례 제어)
        ctrl = self.compute_turret_weights(turret_x, turret_y, target_x=target_x, target_y=target_y)

        # 3. 오차 이력 업데이트
        ax, ay = abs(ctrl["error_x"]), abs(ctrl["error_y"])
        self.ERR_HIST.append((ax, ay))

        # 4. 발사 허가 판정
        fire_ready, yaw_tol, pitch_tol = self.adaptive_fire_gate(
            err_x=ctrl["error_x"], err_y=ctrl["error_y"],
            R=aim["R"], roll_deg=aim["roll_deg"], dy=aim["dy"], desired_pitch=aim["desired_pitch"]
        )

        # 5. 결과 반환
        return {
            "moveWS": {"command": "STOP", "weight": 0.0},  # 차체 이동은 메인 스크립트에서 제어
            "moveAD": {"command": "", "weight": 0.0},
            "turretQE": ctrl["turretQE"],
            "turretRF": ctrl["turretRF"],
            "fire": bool(fire_ready),
            "debug": {**aim, "yaw_tol": round(yaw_tol,2), "pitch_tol": round(pitch_tol,2)}
        }
