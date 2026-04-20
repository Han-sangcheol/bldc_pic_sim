import numpy as np
import matplotlib.pyplot as plt

# =================================================================
# 1. 시스템 주기 및 상수
# =================================================================
DT = 25e-6                          # 25us (40kHz 제어 루프)
RPM_TO_RADS = 2 * np.pi / 60
RADS_TO_RPM = 60 / (2 * np.pi)
V_DC_LINK = 36.0                    # [수정] 실제 사용하시는 36V 전원

# =================================================================
# 2. 하드웨어 모사 유틸리티 (ADC & 필터)
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
    def __init__(self, resolution_bits=12, max_range=15.0, noise_std=0.03):
        # [수정] 최대 전류가 6~7A 수준이므로 ADC 측정 범위를 15A로 축소하여 해상도 확보
        self.levels = 2 ** resolution_bits
        self.max_range = max_range
        self.noise_std = noise_std

    def read(self, true_value):
        noisy_value = true_value + np.random.normal(0, self.noise_std)
        noisy_value = np.clip(noisy_value, -self.max_range, self.max_range)
        quantized = np.round((noisy_value / self.max_range) * (self.levels / 2)) 
        return (quantized / (self.levels / 2)) * self.max_range

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
# 5. 고속 마이크로 모터 물리 엔진 (Plant) - 현실 스펙 반영
# =================================================================
class HighSpeedMotor:
    def __init__(self):
        self.R = 0.38                   
        self.L = 0.12e-3                
        self.FluxLinkage = 0.0042       
        # [핵심 수정] 무부하 4만RPM에서 0.3A를 소모하도록 기계 파라미터 대폭 축소
        self.J = 2.0e-6                 # 초소형 로터의 관성 [kg*m^2]
        self.B = 4.5e-7                 # 베어링 및 공기 마찰 계수 (무부하 0.3A 타겟) [Nm*s/rad]
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
    
    # [수정] 모터 관성이 작아졌으므로 속도 제어기 게인을 낮춰 오버슈트 방지
    speed_ctrl = PIDController(kp=0.008, ki=0.1, dt=DT, out_min=-10, out_max=10) # 최대 지령 전류 10A 제한
    current_ctrl = PIDController(kp=0.8, ki=150.0, dt=DT, out_min=-V_DC_LINK, out_max=V_DC_LINK)

    true_iq, true_omega_m = 0.0, 0.0
    vq_cmd, prev_vq_cmd, cmd_rpm = 0.0, 0.0, 0.0
    target_rpm = 40000.0
    
    # 1초 만에 4만 RPM에 부드럽게 도달하도록 S-Curve 파라미터 설정
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
        
        # [수정] 1.2초 지점에서 약 6~7A 전류를 유발하는 실제 부하(0.04 Nm) 인가
        load_torque = 0.0
        if t > 1.2:
            load_torque = 0.04 
            
        sensed_iq = adc_iq.read(true_iq)
        est_rpm = observer.estimate_rpm(prev_vq_cmd, sensed_iq) 
        prev_vq_cmd = vq_cmd 
        
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

        iq_ref = speed_ctrl.compute(cmd_rpm, est_rpm)
        vq_cmd = current_ctrl.compute(iq_ref, sensed_iq)
        
        true_iq, true_omega_m = motor.update(true_iq, true_omega_m, vq_cmd, load_torque, DT)
        
        if step % print_interval == 0 or step == total_steps - 1:
            load_marker = "(!)" if load_torque > 0.01 else "   "
            load_str = f"{load_torque:.3f} {load_marker}"
            print(f"| {t:8.3f} | {cmd_rpm:9.1f} | {act_rpm:9.1f} | {est_rpm:9.1f} | {sensed_iq:10.3f} | {vq_cmd:10.2f} | {load_str:>12} |")
        
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
plt.title("Realistic Medical Motor Simulation: Speed Response")
plt.ylabel("Speed [RPM]"); plt.legend(loc='lower right'); plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(data['t'], data['sensed_iq'], 'gray', label='Sensed iq (ADC + Noise)', alpha=0.5)
plt.plot(data['t'], data['true_iq'], 'k', label='True iq (Motor Status)', linewidth=1.5)
plt.axvline(x=1.2, color='k', linestyle=':') 
plt.title("Current Response: No-Load ~0.3A, Loaded ~6.5A")
plt.ylabel("Current [A]"); plt.legend(loc='upper right'); plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(data['t'], data['vq'], 'm', label='Vq Command (Inverter Output)')
plt.axhline(y=V_DC_LINK, color='r', linestyle='--', label='DC Link Limit (+36V)')
plt.axhline(y=-V_DC_LINK, color='r', linestyle='--')
plt.title("Inverter Voltage Output (36V System)")
plt.ylabel("Voltage [V]"); plt.legend(loc='upper right'); plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(data['t'], data['load'], 'orange', label='External Load Torque', linewidth=2)
plt.title("Disturbance Profile: 0.04 Nm Load applied at 1.2s")
plt.xlabel("Time [s]"); plt.ylabel("Torque [Nm]"); plt.legend(loc='upper left'); plt.grid(True)

plt.tight_layout()
plt.show()