import numpy as np                  # 수치 계산 라이브러리
import matplotlib.pyplot as plt     # 그래프 시각화 라이브러리

# =================================================================
# 1. 시스템 주기 및 상수
# =================================================================
DT = 25e-6                          # 25us (40kHz Control Loop)
RPM_TO_RADS = 2 * np.pi / 60        # RPM -> rad/s (기계적 각속도)
RADS_TO_RPM = 60 / (2 * np.pi)      # rad/s -> RPM
V_DC_LINK = 48.0                    # 실제 인버터 DC 링크 한계 전압

# =================================================================
# 2. 하드웨어 모사 유틸리티 (ADC & LPF Filter)
# =================================================================
class LowPassFilter:
    """ 1차 디지털 IIR 저역통과필터 (노이즈 억제용) """
    def __init__(self, cutoff_freq_hz, dt):
        rc = 1.0 / (2.0 * np.pi * cutoff_freq_hz)
        self.alpha = dt / (rc + dt)
        self.out_prev = 0.0

    def update(self, input_val):
        out = self.alpha * input_val + (1.0 - self.alpha) * self.out_prev
        self.out_prev = out
        return out

class ADCSimulator:
    """ 실제 전류 센서 노이즈 및 ADC 양자화(Saturation 포함) 모사 """
    def __init__(self, resolution_bits=12, max_range=50.0, noise_std=0.1):
        self.levels = 2 ** resolution_bits
        self.max_range = max_range
        self.noise_std = noise_std

    def read(self, true_value):
        # 1. 노이즈 인가
        noisy_value = true_value + np.random.normal(0, self.noise_std)
        # [오류 수정] 2. 실제 센서/OP-AMP의 물리적 포화(Saturation) 현상 반영
        noisy_value = np.clip(noisy_value, -self.max_range, self.max_range)
        # 3. 양자화(이산화)
        quantized = np.round((noisy_value / self.max_range) * (self.levels / 2)) 
        return (quantized / (self.levels / 2)) * self.max_range

# =================================================================
# 3. 고성능 PID 제어기 (Anti-Windup)
# =================================================================
class PIDController:
    def __init__(self, kp, ki, dt, out_min, out_max):
        self.kp, self.ki, self.dt = kp, ki, dt
        self.out_min, self.out_max = out_min, out_max
        self.integral = 0.0

    def compute(self, setpoint, measurement):
        error = setpoint - measurement
        p_out = self.kp * error
        i_candidate = self.integral + (self.ki * error * self.dt)
        full_output = p_out + i_candidate
        
        # Clamping 방식의 적분기 Anti-Windup 로직
        if self.out_min <= full_output <= self.out_max:
            self.integral = i_candidate
            output = full_output
        else:
            output = np.clip(full_output, self.out_min, self.out_max)
        return output

# =================================================================
# 4. 센서리스 속도 관측기 (BEMF Observer)
# =================================================================
class BEMFObserver:
    def __init__(self, R, L, FluxLinkage, P, dt):
        self.R, self.L = R, L
        self.FluxLinkage = FluxLinkage  # 쇄교자속 (lambda_m)
        self.P = P                      # 극쌍수 (Pole Pairs)
        self.dt = dt
        
        # 미분 폭주를 막기 위한 LPF 필터 필수 적용
        self.iq_lpf = LowPassFilter(cutoff_freq_hz=3000.0, dt=self.dt)
        self.bemf_lpf = LowPassFilter(cutoff_freq_hz=1500.0, dt=self.dt)
        self.prev_iq_filtered = 0.0

    def estimate_rpm(self, vq, iq_measured):
        iq_filt = self.iq_lpf.update(iq_measured)
        diq_dt_approx = (iq_filt - self.prev_iq_filtered) / self.dt
        self.prev_iq_filtered = iq_filt
        
        # BEMF = Vq - R*Iq - L*(dIq/dt)
        raw_bemf = vq - (self.R * iq_filt) - (self.L * diq_dt_approx)
        filtered_bemf = self.bemf_lpf.update(raw_bemf)
        
        # [오류 수정] BEMF = 전기적 각속도(omega_e) * FluxLinkage
        est_omega_e = filtered_bemf / self.FluxLinkage
        est_omega_m = est_omega_e / self.P  # 기계적 각속도로 변환
        
        return est_omega_m * RADS_TO_RPM

# =================================================================
# 5. 고속 모터 물리 엔진 (Plant) - 에너지 보존 법칙 준수
# =================================================================
class HighSpeedMotor:
    def __init__(self):
        self.R = 0.38                   # 저항 [Ohm]
        self.L = 0.12e-3                # 인덕턴스 [H]
        self.FluxLinkage = 0.0042       # 쇄교자속 (lambda_m) [V*s/rad(elec)]
        self.J = 0.00003                # 관성 [kg*m^2]
        self.B = 0.000005               # 마찰 [Nm*s/rad]
        self.P = 1                      # 극쌍수

    def update(self, iq, omega_m, vq_inverter, load_torque, dt):
        # [오류 수정] 기계적 각속도와 전기적 각속도를 엄밀히 분리
        omega_e = omega_m * self.P
        
        # 전기적 모델: Vq = R*iq + L*(diq/dt) + omega_e * FluxLinkage
        diq_dt = (vq_inverter - self.R * iq - omega_e * self.FluxLinkage) / self.L
        
        # 기계적 모델 (3상 dq 모델): Te = 1.5 * P * FluxLinkage * iq
        te = 1.5 * self.P * self.FluxLinkage * iq
        dw_dt = (te - load_torque - self.B * omega_m) / self.J
        
        iq_next = iq + diq_dt * dt
        omega_m_next = omega_m + dw_dt * dt
        
        return iq_next, omega_m_next

# =================================================================
# 6. 마스터 시뮬레이션 통합 실행 (터미널 출력 포함)
# =================================================================
def run_master_simulation():
    motor = HighSpeedMotor()
    observer = BEMFObserver(motor.R, motor.L, motor.FluxLinkage, motor.P, DT)
    adc_iq = ADCSimulator(resolution_bits=12, max_range=20.0, noise_std=0.08)
    
    speed_ctrl = PIDController(kp=0.08, ki=1.5, dt=DT, out_min=-15, out_max=15) 
    current_ctrl = PIDController(kp=1.5, ki=500.0, dt=DT, out_min=-V_DC_LINK, out_max=V_DC_LINK)

    true_iq, true_omega_m = 0.0, 0.0
    vq_cmd, cmd_rpm = 0.0, 0.0
    target_rpm = 40000.0
    
    accel_step, max_accel_step, jerk_step = 0.0, 1.5, 0.0010
    history = {'t': [], 'cmd_rpm': [], 'act_rpm': [], 'est_rpm': [], 'true_iq': [], 'sensed_iq': [], 'vq': [], 'load': []}
    
    total_time = 2.0
    total_steps = int(total_time / DT)
    
    # 터미널 출력 포맷 (UI 어긋남 버그 수정)
    print("=" * 96)
    print(f"| {'Time[s]':^8} | {'Cmd RPM':^9} | {'Act RPM':^9} | {'Est RPM':^9} | {'Iq(ADC)[A]':^10} | {'Vq(PWM)[V]':^10} | {'Load[Nm]':^12} |")
    print("-" * 96)
    print_interval = int(0.1 / DT) 

    for step in range(total_steps):
        t = step * DT
        act_rpm = true_omega_m * RADS_TO_RPM
        
        # 동적 부하 (1.2초 시점 인가)
        load_torque = 0.002
        if t > 1.2:
            load_torque = 0.015 
            
        # 센서 샘플링 및 속도 추정
        sensed_iq = adc_iq.read(true_iq)
        est_rpm = observer.estimate_rpm(vq_cmd, sensed_iq) # Vq_cmd는 실제 하드웨어처럼 한 스텝 지연된 값 사용
        
        # [오류 수정] S-Curve 가/감속 채터링 버그 픽스 (여유 거리 확보)
        if cmd_rpm < target_rpm:
            remaining_rpm = target_rpm - cmd_rpm
            stopping_dist = (accel_step**2) / (2 * jerk_step) + (accel_step * 1.5) # 진동 방지 마진 추가
            
            if remaining_rpm <= stopping_dist:
                accel_step = max(0.0005, accel_step - jerk_step)
            else:
                accel_step = min(max_accel_step, accel_step + jerk_step)
            cmd_rpm = min(target_rpm, cmd_rpm + accel_step)
            
        elif cmd_rpm > target_rpm:
            cmd_rpm = max(target_rpm, cmd_rpm - 10.0)

        # 이중 제어 루프 (Cascade Control)
        iq_ref = speed_ctrl.compute(cmd_rpm, est_rpm)
        vq_cmd_raw = current_ctrl.compute(iq_ref, sensed_iq)
        vq_cmd = np.clip(vq_cmd_raw, -V_DC_LINK, V_DC_LINK)
        
        # 물리 엔진 업데이트
        true_iq, true_omega_m = motor.update(true_iq, true_omega_m, vq_cmd, load_torque, DT)
        
        # 터미널 실시간 출력 (100ms 간격)
        if step % print_interval == 0 or step == total_steps - 1:
            load_marker = "(!)" if load_torque > 0.005 else "   "
            load_str = f"{load_torque:.3f} {load_marker}"
            # [오류 수정] 12칸 우측 정렬로 테이블 세로줄 완벽 일치화
            print(f"| {t:8.3f} | {cmd_rpm:9.1f} | {act_rpm:9.1f} | {est_rpm:9.1f} | {sensed_iq:10.3f} | {vq_cmd:10.2f} | {load_str:>12} |")
        
        # 그래프 데이터 로깅 (5ms 주기)
        if step % 200 == 0:
            history['t'].append(t)
            history['cmd_rpm'].append(cmd_rpm)
            history['act_rpm'].append(act_rpm)
            history['est_rpm'].append(est_rpm)
            history['true_iq'].append(true_iq)
            history['sensed_iq'].append(sensed_iq)
            history['vq'].append(vq_cmd)
            history['load'].append(load_torque)
            
    print("=" * 96)
    print("Simulation Complete. Rendering Plots...\n")
    return history

# =================================================================
# 7. 통합 시각화
# =================================================================
data = run_master_simulation()

plt.figure(figsize=(14, 12))

plt.subplot(4, 1, 1)
plt.plot(data['t'], data['cmd_rpm'], 'r--', label='Command Profile', linewidth=2)
plt.plot(data['t'], data['act_rpm'], 'b', label='True Actual RPM', linewidth=2, alpha=0.7)
plt.plot(data['t'], data['est_rpm'], 'g', label='Observer Estimated RPM', linewidth=1, alpha=0.8)
plt.axvline(x=1.2, color='k', linestyle=':', label='Load Step Inserted') 
plt.title("Master HIL Simulation: FOC Sensorless Speed Response")
plt.ylabel("Speed [RPM]"); plt.legend(loc='lower right'); plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(data['t'], data['sensed_iq'], 'gray', label='Sensed iq (ADC + Noise)', alpha=0.5)
plt.plot(data['t'], data['true_iq'], 'k', label='True iq (Motor Status)', linewidth=1.5)
plt.axvline(x=1.2, color='k', linestyle=':') 
plt.title("Current Response (Observer sees Gray, Motor is Black)")
plt.ylabel("Current [A]"); plt.legend(loc='upper right'); plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(data['t'], data['vq'], 'm', label='Vq Command (Inverter Output)')
plt.axhline(y=V_DC_LINK, color='r', linestyle='--', label='DC Link Limit (+48V)')
plt.axhline(y=-V_DC_LINK, color='r', linestyle='--')
plt.title("Inverter Voltage Saturation")
plt.ylabel("Voltage [V]"); plt.legend(loc='upper right'); plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(data['t'], data['load'], 'orange', label='External Load Torque', linewidth=2)
plt.title("Disturbance (Load Torque Profile)")
plt.xlabel("Time [s]"); plt.ylabel("Torque [Nm]"); plt.legend(loc='upper left'); plt.grid(True)

plt.tight_layout()
plt.show()