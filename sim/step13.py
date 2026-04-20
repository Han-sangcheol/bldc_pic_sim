import numpy as np
import matplotlib.pyplot as plt

# =================================================================
# 1. 시스템 주기 및 상수
# =================================================================
DT = 25e-6                          # 25us (40kHz 제어 루프)
RPM_TO_RADS = 2 * np.pi / 60        # RPM을 rad/s로 변환하는 상수
RADS_TO_RPM = 60 / (2 * np.pi)      # rad/s를 RPM으로 변환하는 상수
V_DC_LINK = 36.0                    # 36V 전원 시스템

# =================================================================
# 2. 하드웨어 모사 유틸리티 (ADC & LPF)
# =================================================================
class LowPassFilter:
    def __init__(self, cutoff_freq_hz, dt):
        rc = 1.0 / (2.0 * np.pi * cutoff_freq_hz)   # RC 상수 계산
        self.alpha = dt / (rc + dt)                 # LPF 알파 계산 (디지털 필터링 계수)
        self.out_prev = 0.0                         # 이전 출력값 초기화

    def update(self, input_val):
        out = self.alpha * input_val + (1.0 - self.alpha) * self.out_prev   # LPF 공식 적용
        self.out_prev = out                                                 # 현재 출력값을 이전 출력값으로 저장
        return out

class ADCSimulator:
    def __init__(self, resolution_bits=12, max_range=15.0, noise_std=0.03):
        self.levels = 2 ** resolution_bits          # ADC 해상도에 따른 단계 수 계산
        self.max_range = max_range                  # ADC 측정 범위 (예: 최대 15A)  
        self.noise_std = noise_std                  # ADC 측정 노이즈 표준편차

    def read(self, true_value):
        # ADC 센서 노이즈 및 물리적 한계 포화 모사
        noisy_value = true_value + np.random.normal(0, self.noise_std)          # 실제 측정값에 가우시안 노이즈 추가
        noisy_value = np.clip(noisy_value, -self.max_range, self.max_range)     # ADC 최대 범위를 벗어나지 않도록 클램핑
        quantized = np.round((noisy_value / self.max_range) * (self.levels / 2))    # ADC 양수 범위에 맞게 정규화 후 양자화
        return (quantized / (self.levels / 2)) * self.max_range                 # 양자화된 값을 다시 실제 범위로 변환하여 반환

# =================================================================
# 3. PID 제어기 (Conditional Integration Anti-Windup)
# =================================================================
class PIDController:
    def __init__(self, kp, ki, dt, out_min, out_max):
        self.kp, self.ki, self.dt = kp, ki, dt
        self.out_min, self.out_max = out_min, out_max
        self.integral = 0.0

    def compute(self, setpoint, measurement):
        error = setpoint - measurement
        p_out = self.kp * error
        i_step = self.ki * error * self.dt
        out = p_out + self.integral + i_step

        if out > self.out_max:
            out = self.out_max
            if error < 0: self.integral += i_step  
        elif out < self.out_min:
            out = self.out_min
            if error > 0: self.integral += i_step  
        else:
            self.integral += i_step
            
        return out

# =================================================================
# 4. 센서리스 속도 관측기 (BEMF Observer)
# =================================================================
class BEMFObserver:
    def __init__(self, R, L, FluxLinkage, P, dt):
        self.R, self.L = R, L
        self.FluxLinkage = FluxLinkage
        self.P = P
        self.dt = dt
        
        # 전류 미분용 LPF와 역기전력 평활화용 LPF
        self.iq_lpf = LowPassFilter(cutoff_freq_hz=500.0, dt=self.dt)
        self.bemf_lpf = LowPassFilter(cutoff_freq_hz=150.0, dt=self.dt)
        self.prev_iq_filtered = 0.0

    def estimate_rpm(self, prev_vq, iq_measured):
        iq_filt = self.iq_lpf.update(iq_measured)
        diq_dt_approx = (iq_filt - self.prev_iq_filtered) / self.dt
        self.prev_iq_filtered = iq_filt
        
        raw_bemf = prev_vq - (self.R * iq_filt) - (self.L * diq_dt_approx)
        filtered_bemf = self.bemf_lpf.update(raw_bemf)
        
        est_omega_e = filtered_bemf / self.FluxLinkage
        est_omega_m = est_omega_e / self.P 
        return est_omega_m * RADS_TO_RPM

# =================================================================
# 5. 고속 마이크로 모터 물리 엔진 (Plant)
# =================================================================
class HighSpeedMotor:
    def __init__(self):
        self.R = 0.38                   
        self.L = 0.12e-3                
        self.FluxLinkage = 0.0042       
        self.J = 2.0e-6                 # 초소형 로터 관성
        self.B = 4.5e-7                 # 무부하 0.3A 타겟 마찰계수
        self.P = 1                      

    def update(self, iq, omega_m, vq_inverter, load_torque, dt):
        omega_e = omega_m * self.P
        diq_dt = (vq_inverter - self.R * iq - omega_e * self.FluxLinkage) / self.L
        te = 1.5 * self.P * self.FluxLinkage * iq
        dw_dt = (te - load_torque - self.B * omega_m) / self.J
        
        iq_next = iq + diq_dt * dt
        omega_m_next = omega_m + dw_dt * dt
        return iq_next, omega_m_next

# =================================================================
# 6. 마스터 시뮬레이션 통합 실행
# =================================================================
def run_master_simulation():
    motor = HighSpeedMotor()
    observer = BEMFObserver(motor.R, motor.L, motor.FluxLinkage, motor.P, DT)
    adc_iq = ADCSimulator(resolution_bits=12, max_range=15.0, noise_std=0.03)
    
    speed_ctrl = PIDController(kp=0.008, ki=0.1, dt=DT, out_min=-10, out_max=10)
    current_ctrl = PIDController(kp=0.8, ki=150.0, dt=DT, out_min=-V_DC_LINK, out_max=V_DC_LINK)

    true_iq, true_omega_m = 0.0, 0.0
    vq_cmd, prev_vq_cmd, cmd_rpm = 0.0, 0.0, 0.0
    target_rpm = 40000.0
    
    accel_step, max_accel_step, jerk_step = 0.0, 1.0, 0.0005
    history = {'t': [], 'cmd_rpm': [], 'act_rpm': [], 'est_rpm': [], 'true_iq': [], 'sensed_iq': [], 'vq': [], 'load': []}
    
    total_time = 2.0
    total_steps = int(total_time / DT)
    
    print("=" * 96)
    print(f"| {'Time[s]':^8} | {'Cmd RPM':^9} | {'Act RPM':^9} | {'Est RPM':^9} | {'Iq(ADC)[A]':^10} | {'Vq(PWM)[V]':^10} | {'Load[Nm]':^12} |")
    print("-" * 96)
    print_interval = int(0.1 / DT) 

    for step in range(total_steps):
        t = step * DT
        act_rpm = true_omega_m * RADS_TO_RPM
        
        # =============================================================
        # [신규 기능] 가변 불규칙 부하 (Dynamic Variable Load) 모사
        # =============================================================
        load_torque = 0.0
        if t > 1.2:
            base_load = 0.035                                      # 기본 부하 (약 6A 타겟)
            low_freq_var = 0.008 * np.sin(2 * np.pi * 5.0 * t)     # 5Hz 주기 변동 (작업자 손의 압력 변화)
            high_freq_var = 0.003 * np.sin(2 * np.pi * 50.0 * t)   # 50Hz 주기 진동 (공구 떨림 및 마찰 진동)
            white_noise = np.random.normal(0, 0.002)               # 재질 밀도 변화에 따른 랜덤 노이즈
            
            # 모든 변동 요소를 합산하고, 부하가 0 미만(역방향)으로 가지 않도록 클램핑
            load_torque = base_load + low_freq_var + high_freq_var + white_noise
            load_torque = max(0.0, load_torque) 
        # =============================================================
            
        sensed_iq = adc_iq.read(true_iq)
        est_rpm = observer.estimate_rpm(prev_vq_cmd, sensed_iq) 
        prev_vq_cmd = vq_cmd 
        
        # S-Curve 속도 지령
        if cmd_rpm < target_rpm:
            remaining_rpm = target_rpm - cmd_rpm
            stopping_dist = (accel_step**2) / (2 * jerk_step) + (accel_step * 2.0) 
            
            if remaining_rpm <= stopping_dist:
                accel_step = max(0.0001, accel_step - jerk_step)
            else:
                accel_step = min(max_accel_step, accel_step + jerk_step)
            cmd_rpm = min(target_rpm, cmd_rpm + accel_step)
            
        elif cmd_rpm > target_rpm:
            cmd_rpm = max(target_rpm, cmd_rpm - 10.0)

        # FOC 제어 연산 (속도 루프 -> 전류 루프)
        iq_ref = speed_ctrl.compute(cmd_rpm, est_rpm)
        vq_cmd = current_ctrl.compute(iq_ref, sensed_iq)
        
        # 모터 물리 상태 업데이트
        true_iq, true_omega_m = motor.update(true_iq, true_omega_m, vq_cmd, load_torque, DT)
        
        # 터미널 실시간 출력
        if step % print_interval == 0 or step == total_steps - 1:
            load_marker = "(!)" if load_torque > 0.01 else "   "
            load_str = f"{load_torque:.3f} {load_marker}"
            print(f"| {t:8.3f} | {cmd_rpm:9.1f} | {act_rpm:9.1f} | {est_rpm:9.1f} | {sensed_iq:10.3f} | {vq_cmd:10.2f} | {load_str:>12} |")
        
        # 그래프 출력용 데이터 로깅
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
# 7. 통합 시각화 (그래프 출력)
# =================================================================
data = run_master_simulation()

plt.figure(figsize=(14, 12))

# 1. 속도 응답 그래프
plt.subplot(4, 1, 1)
plt.plot(data['t'], data['cmd_rpm'], 'r--', label='Command Profile', linewidth=2)
plt.plot(data['t'], data['act_rpm'], 'b', label='True Actual RPM', linewidth=2, alpha=0.7)
plt.plot(data['t'], data['est_rpm'], 'g', label='Observer Estimated RPM', linewidth=1, alpha=0.8)
plt.axvline(x=1.2, color='k', linestyle=':', label='Variable Load Inserted') 
plt.title("Realistic Medical Motor: Speed Response under Dynamic Load")
plt.ylabel("Speed [RPM]"); plt.legend(loc='lower right'); plt.grid(True)

# 2. 전류 응답 그래프 (가변 부하를 추종하는 모습)
plt.subplot(4, 1, 2)
plt.plot(data['t'], data['sensed_iq'], 'gray', label='Sensed iq (ADC + Noise)', alpha=0.5)
plt.plot(data['t'], data['true_iq'], 'k', label='True iq (Motor Status)', linewidth=1.5)
plt.axvline(x=1.2, color='k', linestyle=':') 
plt.title("Current Response: No-Load ~0.3A -> Dynamic Fluctuations (4A ~ 8A)")
plt.ylabel("Current [A]"); plt.legend(loc='upper right'); plt.grid(True)

# 3. 전압 응답 그래프
plt.subplot(4, 1, 3)
plt.plot(data['t'], data['vq'], 'm', label='Vq Command (Inverter Output)')
plt.axhline(y=V_DC_LINK, color='r', linestyle='--', label='DC Link Limit (+36V)')
plt.axhline(y=-V_DC_LINK, color='r', linestyle='--')
plt.title("Inverter Voltage Output (36V System)")
plt.ylabel("Voltage [V]"); plt.legend(loc='upper right'); plt.grid(True)

# 4. 가변 부하 토크 그래프
plt.subplot(4, 1, 4)
plt.plot(data['t'], data['load'], 'orange', label='Dynamic Load Torque (Base + Sin + Noise)', linewidth=1.5)
plt.title("Disturbance Profile: Fluctuating Mechanical Load starting at 1.2s")
plt.xlabel("Time [s]"); plt.ylabel("Torque [Nm]"); plt.legend(loc='upper left'); plt.grid(True)

plt.tight_layout()
plt.show()