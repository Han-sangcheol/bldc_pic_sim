import numpy as np                  # 수치 계산 라이브러리
import matplotlib.pyplot as plt     # 그래프 시각화 라이브러리

# =================================================================
# 1. 시스템 주기 및 상수
# =================================================================
DT = 25e-6                          # 25us (40kHz Control Loop)
RPM_TO_RADS = 2 * np.pi / 60        # RPM을 rad/s로 변환하는 상수
RADS_TO_RPM = 60 / (2 * np.pi)      # rad/s를 RPM으로 변환하는 상수

# =================================================================
# 2. 고성능 PID 제어기 (Clamping Anti-Windup 적용)
# =================================================================
class PIDController:
    def __init__(self, kp, ki, dt, out_min, out_max):
        self.kp, self.ki = kp, ki                       # PID 게인
        self.dt = dt                                    # 제어 주기 [초] 
        self.out_min, self.out_max = out_min, out_max   # 출력 제한
        self.integral = 0.0                             # 적분 누적값       

    def compute(self, setpoint, measurement):           
        error = setpoint - measurement                  # 오차 계산
        p_out = self.kp * error                         # P 항 계산
        
        i_candidate = self.integral + (self.ki * error * self.dt)   # I 항 계산 (적분 누적값 후보)
        full_output = p_out + i_candidate               # 전체 PID 출력 계산            
        
        # Clamping: 출력이 포화되지 않았을 때만 적분값 업데이트
        if self.out_min <= full_output <= self.out_max:     # 출력이 제한 범위 내에 있을 때 적분값 업데이트
            self.integral = i_candidate                     # 적분값 업데이트   
            output = full_output                            # 전체 PID 출력 사용    
        else:
            output = np.clip(full_output, self.out_min, self.out_max)   # 출력이 제한 범위를 벗어날 때 출력 클램핑
            
        return output

# =================================================================
# 3. 센서리스 속도 관측기 (BEMF Observer) - 간단한 역기전력 모델 기반
# =================================================================
class BEMFObserver:
    def __init__(self, R, L, Ke):
        self.R = R
        self.L = L
        self.Ke = Ke

    def estimate_rpm(self, vq, iq, prev_iq, dt):
        """
        Vq = R*iq + L*(diq/dt) + BEMF 에서 BEMF를 역산하여 속도 추정
        실제 펌웨어에서는 ADC 노이즈 필터링이 추가로 필요합니다.
        """
        # 노이즈 필터링 추가
        iq_filtered = iq  # 간단한 필터링 예시 (실제 구현에서는 더 복잡한 필터 사용)    

        diq_dt_approx = (iq - prev_iq) / dt                                 # 전류 변화율 근사
        bemf = vq - (self.R * iq_filtered) - (self.L * diq_dt_approx)       # 역기전력 계산
        
        est_omega = bemf / self.Ke                              # 역기전력 상수로 각속도 추정
        return est_omega * RADS_TO_RPM                          # 추정된 각속도를 RPM으로 변환하여 반환

# =================================================================
# 4. 고속 모터 물리 엔진 (Plant)
# =================================================================
class HighSpeedMotor:
    def __init__(self):
        # 40,000 RPM 고속 덴탈 모터 사양 기반
        self.R = 0.38           # 저항 [Ohm]
        self.L = 0.12e-3        # 인덕턴스 [H]
        self.Ke = 0.0042        # 역기전력 상수 [V*s/rad]
        self.Kt = 0.0042        # 토크 상수 [Nm/A]
        self.J = 0.00003        # 관성 [kg*m^2]
        self.B = 0.000005       # 마찰 [Nm*s/rad]
        self.P = 1              # 극쌍수

    def update(self, iq, omega, vq, load_torque):
        diq_dt = (vq - self.R * iq - self.Ke * omega) / self.L
        te = 1.5 * self.P * self.Kt * iq
        dw_dt = (te - load_torque - self.B * omega) / self.J
        return diq_dt, dw_dt

# =================================================================
# 5. 마스터 시뮬레이션 통합 실행
# =================================================================
def run_master_simulation():
    motor = HighSpeedMotor()
    observer = BEMFObserver(motor.R, motor.L, motor.Ke)
    
    # 이중 루프 제어기 선언
    speed_ctrl = PIDController(kp=0.18, ki=2.0, dt=DT, out_min=-25, out_max=25)
    current_ctrl = PIDController(kp=2.5, ki=800.0, dt=DT, out_min=-36, out_max=36)

    # 상태 변수 초기화
    iq, prev_iq, omega = 0.0, 0.0, 0.0
    vq_cmd = 0.0
    cmd_rpm = 0.0
    target_rpm = 40000.0
    
    # --- 가속 프로파일 (S-Curve & No Overshoot) 변수 ---
    accel_step = 0.0          # 현재 RPM 가속도
    max_accel_step = 2.0      # 지침 최대 가속도 (2 RPM / 25us)
    jerk_step = 0.0015        # 슬로우 스타트 감도 (낮을수록 더 부드러움)
    
    history = {'t': [], 'cmd_rpm': [], 'act_rpm': [], 'est_rpm': [], 'iq': [], 'accel': []}
    
    total_steps = int(1.5 / DT) # 1.5초 시뮬레이션
    
    for step in range(total_steps):
        t = step * DT
        act_rpm = omega * RADS_TO_RPM
        
        # [기능 1] 센서리스 속도 추정 (Observer)
        # 물리 엔진에서 발생한 Vq와 iq를 관측기에 넣어 속도를 추정
        est_rpm = observer.estimate_rpm(vq_cmd, iq, prev_iq, DT)
        
        # [기능 2] S-Curve 및 오버슈트 제로 램프 제어
        if cmd_rpm < target_rpm:
            remaining_rpm = target_rpm - cmd_rpm
            stopping_dist = (accel_step**2) / (2 * jerk_step) # 브레이킹 거리 계산
            
            if remaining_rpm <= stopping_dist:
                # 목표 지점 도달 전 감속 (오버슈트 차단)
                accel_step = max(0.001, accel_step - jerk_step)
            else:
                # 슬로우 스타트 가속
                accel_step = min(max_accel_step, accel_step + jerk_step)
                
            cmd_rpm = min(target_rpm, cmd_rpm + accel_step)
            
        elif cmd_rpm > target_rpm:
            # 감속 지침 (10 RPM / 25us) 반영
            cmd_rpm = max(target_rpm, cmd_rpm - 10.0)

        # [기능 3] 피드백 제어 (추정된 est_rpm을 기반으로 제어)
        iq_ref = speed_ctrl.compute(cmd_rpm, est_rpm)
        vq_cmd = current_ctrl.compute(iq_ref, iq)
        
        # [기능 4] 물리 엔진 업데이트
        prev_iq = iq # 관측기를 위해 이전 전류 저장
        diq_dt, dw_dt = motor.update(iq, omega, vq_cmd, 0.005) # 5mNm 기본 부하
        
        iq += diq_dt * DT
        omega += dw_dt * DT
        
        # 데이터 샘플링 (그래프 출력용)
        if step % 200 == 0:
            history['t'].append(t)
            history['cmd_rpm'].append(cmd_rpm)
            history['act_rpm'].append(act_rpm)
            history['est_rpm'].append(est_rpm)
            history['iq'].append(iq)
            history['accel'].append(accel_step)
            
    return history

# =================================================================
# 6. 통합 시각화
# =================================================================
data = run_master_simulation()

plt.figure(figsize=(12, 10))

# 1. 속도 응답 (실제 속도 vs 관측기 추정 속도 vs 명령어)
plt.subplot(3, 1, 1)
plt.plot(data['t'], data['cmd_rpm'], 'r--', label='Command Profile (No Overshoot)', linewidth=2)
plt.plot(data['t'], data['act_rpm'], 'b', label='Actual RPM', linewidth=3, alpha=0.5)
plt.plot(data['t'], data['est_rpm'], 'g:', label='Observer Estimated RPM', linewidth=2)
plt.title("Master FOC Simulation: S-Curve, No-Overshoot, Sensorless Observer")
plt.ylabel("Speed [RPM]"); plt.legend(loc='lower right'); plt.grid(True)

# 2. 가속도 프로파일 (슬로우 스타트 및 제동 확인)
plt.subplot(3, 1, 2)
plt.plot(data['t'], data['accel'], 'm', label='Acceleration Rate (RPM per 25us)')
plt.title("Acceleration Profile (Jerk Control)")
plt.ylabel("Accel Rate"); plt.legend(); plt.grid(True)

# 3. 전류 응답 (안정성 확인)
plt.subplot(3, 1, 3)
plt.plot(data['t'], data['iq'], 'k', label='q-axis Current (iq)')
plt.title("Current Response")
plt.xlabel("Time [s]"); plt.ylabel("Current [A]"); plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()