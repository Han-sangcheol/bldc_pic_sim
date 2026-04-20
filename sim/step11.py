import numpy as np
import matplotlib.pyplot as plt

# =================================================================
# 1. 시스템 주기 및 상수
# =================================================================
DT = 25e-6                          # 25us (40kHz Control Loop)
RPM_TO_RADS = 2 * np.pi / 60
RADS_TO_RPM = 60 / (2 * np.pi)
V_DC_LINK = 48.0                    # 인버터 DC 링크 한계 전압

# =================================================================
# 2. 하드웨어 모사 유틸리티
# =================================================================
class LowPassFilter:
    def __init__(self, cutoff_freq_hz, dt):
        rc = 1.0 / (2.0 * np.pi * cutoff_freq_hz)
        self.alpha = dt / (rc + dt)
        self.out_prev = 0.0

    def update(self, input_val):
        out = self.alpha * input_val + (1.0 - self.alpha) * self.out_prev
        self.out_prev = out
        return out

class ADCSimulator:
    def __init__(self, resolution_bits=12, max_range=50.0, noise_std=0.1):
        self.levels = 2 ** resolution_bits
        self.max_range = max_range
        self.noise_std = noise_std

    def read(self, true_value):
        # ADC 센서 노이즈 및 포화 모사
        noisy_value = true_value + np.random.normal(0, self.noise_std)
        noisy_value = np.clip(noisy_value, -self.max_range, self.max_range)
        quantized = np.round((noisy_value / self.max_range) * (self.levels / 2)) 
        return (quantized / (self.levels / 2)) * self.max_range

# =================================================================
# 3. 산업용 표준 PID 제어기 (Conditional Integration Anti-Windup 적용)
# =================================================================
class PIDController:
    def __init__(self, kp, ki, dt, out_min, out_max):
        self.kp, self.ki, self.dt = kp, ki, dt
        self.out_min, self.out_max = out_min, out_max
        self.integral = 0.0

    def compute(self, setpoint, measurement):
        error = setpoint - measurement
        p_out = self.kp * error
        
        # [보완] 완벽한 Anti-Windup (출력이 포화되었을 때 에러 방향을 보고 적분 제한)
        i_step = self.ki * error * self.dt
        out = p_out + self.integral + i_step

        if out > self.out_max:
            out = self.out_max
            if error < 0: self.integral += i_step  # 돌아올 때만 적분 허용
        elif out < self.out_min:
            out = self.out_min
            if error > 0: self.integral += i_step  # 돌아올 때만 적분 허용
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
        
        # [보완] 노이즈 증폭 방지를 위해 LPF 대역폭을 현실적으로 조정
        self.iq_lpf = LowPassFilter(cutoff_freq_hz=500.0, dt=self.dt)
        self.bemf_lpf = LowPassFilter(cutoff_freq_hz=150.0, dt=self.dt)
        self.prev_iq_filtered = 0.0

    def estimate_rpm(self, prev_vq, iq_measured):
        iq_filt = self.iq_lpf.update(iq_measured)
        diq_dt_approx = (iq_filt - self.prev_iq_filtered) / self.dt
        self.prev_iq_filtered = iq_filt
        
        # [보완] 현재 측정된 전류(iq)는 '이전 스텝'의 전압(prev_vq)에 의한 결과임
        raw_bemf = prev_vq - (self.R * iq_filt) - (self.L * diq_dt_approx)
        filtered_bemf = self.bemf_lpf.update(raw_bemf)
        
        est_omega_e = filtered_bemf / self.FluxLinkage
        est_omega_m = est_omega_e / self.P 
        return est_omega_m * RADS_TO_RPM

# =================================================================
# 5. 고속 모터 물리 엔진 (Plant)
# =================================================================
class HighSpeedMotor:
    def __init__(self):
        self.R = 0.38                   
        self.L = 0.12e-3                
        self.FluxLinkage = 0.0042       
        self.J = 0.00003                
        self.B = 0.000005               
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
    adc_iq = ADCSimulator(resolution_bits=12, max_range=20.0, noise_std=0.08)
    
    # [보완] 노이즈 대응을 위해 속도 제어기 게인 하향 안정화 (매우 중요)
    speed_ctrl = PIDController(kp=0.015, ki=0.2, dt=DT, out_min=-15, out_max=15) 
    current_ctrl = PIDController(kp=0.8, ki=150.0, dt=DT, out_min=-V_DC_LINK, out_max=V_DC_LINK)

    true_iq, true_omega_m = 0.0, 0.0
    vq_cmd, prev_vq_cmd, cmd_rpm = 0.0, 0.0, 0.0
    target_rpm = 40000.0
    
    # [보완] 15A 전류 한계로 만들어낼 수 있는 실제 최대 가속도로 S-Curve 조정
    accel_step, max_accel_step, jerk_step = 0.0, 0.5, 0.0002
    
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
        
        load_torque = 0.002
        if t > 1.2:
            load_torque = 0.015 
            
        sensed_iq = adc_iq.read(true_iq)
        # [보완] 위상 동기화 (관측기에는 이전 스텝의 전압을 넣어줌)
        est_rpm = observer.estimate_rpm(prev_vq_cmd, sensed_iq) 
        prev_vq_cmd = vq_cmd 
        
        # S-Curve 가/감속 로직
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

        # 이중 루프 제어 (Cascade)
        iq_ref = speed_ctrl.compute(cmd_rpm, est_rpm)
        vq_cmd = current_ctrl.compute(iq_ref, sensed_iq)
        
        true_iq, true_omega_m = motor.update(true_iq, true_omega_m, vq_cmd, load_torque, DT)
        
        # 터미널 실시간 출력
        if step % print_interval == 0 or step == total_steps - 1:
            load_marker = "(!)" if load_torque > 0.005 else "   "
            load_str = f"{load_torque:.3f} {load_marker}"
            print(f"| {t:8.3f} | {cmd_rpm:9.1f} | {act_rpm:9.1f} | {est_rpm:9.1f} | {sensed_iq:10.3f} | {vq_cmd:10.2f} | {load_str:>12} |")
        
        # 그래프 데이터 로깅
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
plt.title("Master HIL Simulation: Stable Sensorless Speed Response")
plt.ylabel("Speed [RPM]"); plt.legend(loc='lower right'); plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(data['t'], data['sensed_iq'], 'gray', label='Sensed iq (ADC + Noise)', alpha=0.5)
plt.plot(data['t'], data['true_iq'], 'k', label='True iq (Motor Status)', linewidth=1.5)
plt.axvline(x=1.2, color='k', linestyle=':') 
plt.title("Current Response (Robust to ADC Noise)")
plt.ylabel("Current [A]"); plt.legend(loc='upper right'); plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(data['t'], data['vq'], 'm', label='Vq Command (Inverter Output)')
plt.axhline(y=V_DC_LINK, color='r', linestyle='--', label='DC Link Limit (+48V)')
plt.axhline(y=-V_DC_LINK, color='r', linestyle='--')
plt.title("Inverter Voltage Output")
plt.ylabel("Voltage [V]"); plt.legend(loc='upper right'); plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(data['t'], data['load'], 'orange', label='External Load Torque', linewidth=2)
plt.title("Disturbance (Load Torque Profile)")
plt.xlabel("Time [s]"); plt.ylabel("Torque [Nm]"); plt.legend(loc='upper left'); plt.grid(True)

plt.tight_layout()
plt.show()