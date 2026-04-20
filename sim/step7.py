import numpy as np
import matplotlib.pyplot as plt

# =================================================================
# 1. 시스템 파라미터 및 제어 주기 (25us)
# =================================================================
DT = 25e-6          # 제어 주기: 25us (40kHz)
RPM_TO_RADS = 2 * np.pi / 60
RADS_TO_RPM = 60 / (2 * np.pi)

# =================================================================
# 2. 고성능 PID 제어기 (Clamping Anti-Windup 포함)
# =================================================================
class PIDController:
    def __init__(self, kp, ki, dt, out_min, out_max):
        self.kp = kp
        self.ki = ki
        self.dt = dt
        self.out_min = out_min
        self.out_max = out_max
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, setpoint, measurement):
        error = setpoint - measurement
        
        # Proportional term
        p_out = self.kp * error
        
        # Integral term (Clamping Anti-Windup)
        # 먼저 적분을 수행한 후, 출력이 포화되면 적분 처리를 결정함
        i_term_next = self.integral + (self.ki * error * self.dt)
        output_full = p_out + i_term_next
        
        if self.out_min <= output_full <= self.out_max:
            self.integral = i_term_next
            output = output_full
        else:
            # 포화 시 적분을 멈추고 출력을 제한함
            output = np.clip(output_full, self.out_min, self.out_max)
            
        return output

# =================================================================
# 3. 모터 물리 엔진 및 시뮬레이션 환경
# =================================================================
class HighSpeedMotor:
    def __init__(self):
        # 40,000 RPM급 덴탈/고속 모터 파라미터 (예시)
        self.R = 0.4            # [Ohm]
        self.L = 0.15e-3        # [H]
        self.Ke = 0.0045        # [V*s/rad] (역기전력 상수)
        self.Kt = 0.0045        # [Nm/A] (토크 상수)
        self.J = 0.00004        # [kg*m^2] (관성)
        self.B = 0.000008       # [Nm*s/rad] (마찰)
        self.P = 1              # 극쌍수 (Pole Pairs)

    def dynamics(self, iq, omega, Vq, Load_Torque):
        # 전기적 모델: Vq = R*iq + L*(diq/dt) + Ke*omega
        diq_dt = (Vq - self.R * iq - self.Ke * omega) / self.L
        
        # 기계적 모델: Te = 1.5 * P * Kt * iq (BLDC/PMSM 근사)
        Te = 1.5 * self.P * self.Kt * iq
        dw_dt = (Te - Load_Torque - self.B * omega) / self.J
        
        return diq_dt, dw_dt

# =================================================================
# 4. 메인 시뮬레이션 루프
# =================================================================
def run_simulation():
    motor = HighSpeedMotor()
    
    # 제어기 튜닝 (40,000 RPM 고속 영역 최적화)
    # Speed Loop: RPM -> Iq_ref (Max 25A)
    speed_ctrl = PIDController(kp=0.12, ki=1.5, dt=DT, out_min=-25, out_max=25)
    # Current Loop: Iq -> Vq (Max 36V)
    current_ctrl = PIDController(kp=2.0, ki=500.0, dt=DT, out_min=-36, out_max=36)

    # 상태 변수 초기화
    iq = 0.0
    omega = 0.0
    cmd_rpm = 0.0
    target_rpm = 40000.0
    load_torque = 0.01  # [Nm] 정격 부하 가정
    
    # 데이터 기록
    history = {'t': [], 'cmd_rpm': [], 'act_rpm': [], 'iq': [], 'vq': []}
    
    total_steps = int(3.0 / DT) # 3초간 시뮬레이션
    
    for step in range(total_steps):
        t = step * DT
        act_rpm = omega * RADS_TO_RPM
        
        # [Step 1] Ramp Profile Logic
        # 25us당 증가 2 RPM, 감속 10 RPM 반영
        if cmd_rpm < target_rpm:
            cmd_rpm = min(target_rpm, cmd_rpm + 2.0)
        elif cmd_rpm > target_rpm:
            cmd_rpm = max(target_rpm, cmd_rpm - 10.0)
            
        # [Step 2] Cascade Control (Speed -> Current)
        iq_ref = speed_ctrl.compute(cmd_rpm, act_rpm)
        vq_cmd = current_ctrl.compute(iq_ref, iq)
        
        # [Step 3] Plant Dynamics (Numerical Integration)
        diq_dt, dw_dt = motor.dynamics(iq, omega, vq_cmd, load_torque)
        iq += diq_dt * DT
        omega += dw_dt * DT
        
        # [Step 4] 데이터 샘플링 (100 step마다 저장 - 약 2.5ms)
        if step % 100 == 0:
            history['t'].append(t)
            history['cmd_rpm'].append(cmd_rpm)
            history['act_rpm'].append(act_rpm)
            history['iq'].append(iq)
            history['vq'].append(vq_cmd)
            
    return history

# =================================================================
# 5. 결과 분석 및 시각화
# =================================================================
history = run_simulation()

plt.figure(figsize=(12, 10))

# Speed Plot
plt.subplot(3, 1, 1)
plt.plot(history['t'], history['cmd_rpm'], 'r--', label='Command RPM', alpha=0.7)
plt.plot(history['t'], history['act_rpm'], 'b', label='Actual RPM', linewidth=1.5)
plt.title("Motor Speed Response (40,000 RPM Ramp)")
plt.ylabel("Speed [RPM]"); plt.legend(); plt.grid(True)

# Current Plot
plt.subplot(3, 1, 2)
plt.plot(history['t'], history['iq'], 'g', label='q-axis Current (iq)')
plt.title("Current Loop Stability")
plt.ylabel("Current [A]"); plt.legend(); plt.grid(True)

# Voltage Plot
plt.subplot(3, 1, 3)
plt.plot(history['t'], history['vq'], 'm', label='Control Voltage (Vq)')
plt.title("Output Voltage (Saturation Check)")
plt.xlabel("Time [s]"); plt.ylabel("Voltage [V]"); plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()