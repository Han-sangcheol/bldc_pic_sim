import numpy as np                  # 수치 계산 라이브러리
import matplotlib.pyplot as plt     # 그래프 시각화 라이브러리

# =================================================================
# 1. 시스템 주기 및 상수
# =================================================================
DT = 25e-6                          # 25us (40kHz Control Loop, 실제 고속 모터 제어 주기)
RPM_TO_RADS = 2 * np.pi / 60        # RPM을 rad/s로 변환하는 상수
RADS_TO_RPM = 60 / (2 * np.pi)      # rad/s를 RPM으로 변환하는 상수
V_DC_LINK = 48.0                    # [추가] 실제 인버터의 DC 링크 전압 한계 (Max 전압)

# =================================================================
# [추가] 2. 실제 하드웨어 모사 유틸리티 (ADC & 필터)
# =================================================================
class LowPassFilter:
    """ 
    실제 MCU 펌웨어에서 가장 많이 사용되는 1차 디지털 저역통과필터 (IIR Filter)
    수식: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
    """
    def __init__(self, cutoff_freq_hz, dt):
        # RC 시정수 계산 및 alpha 값 도출
        rc = 1.0 / (2.0 * np.pi * cutoff_freq_hz)
        self.alpha = dt / (rc + dt)
        self.out_prev = 0.0

    def update(self, input_val):
        out = self.alpha * input_val + (1.0 - self.alpha) * self.out_prev
        self.out_prev = out
        return out

class ADCSimulator:
    """
    실제 전류 센서와 MCU의 ADC(Analog-to-Digital Converter) 과정을 모사
    - 노이즈 삽입 (Gaussian Noise)
    - 양자화 (Quantization, ex: 12-bit ADC)
    """
    def __init__(self, resolution_bits=12, max_range=50.0, noise_std=0.1):
        self.levels = 2 ** resolution_bits           # 12-bit = 4096 단계
        self.max_range = max_range                   # 측정 가능 최대 범위 (예: -50A ~ 50A)
        self.noise_std = noise_std                   # 센서 노이즈 표준편차

    def read(self, true_value):
        # 1. 아날로그 노이즈 추가
        noisy_value = true_value + np.random.normal(0, self.noise_std)
        # 2. ADC 양자화 (이산화)
        quantized = np.round((noisy_value / self.max_range) * (self.levels / 2)) 
        return (quantized / (self.levels / 2)) * self.max_range

# =================================================================
# 3. 고성능 PID 제어기 (Clamping Anti-Windup 적용)
# =================================================================
class PIDController:
    def __init__(self, kp, ki, dt, out_min, out_max):
        self.kp, self.ki = kp, ki
        self.dt = dt
        self.out_min, self.out_max = out_min, out_max
        self.integral = 0.0
        self.prev_error = 0.0 # D제어를 위해 남겨둠 (본 시뮬레이션에서는 PI만 사용)

    def compute(self, setpoint, measurement):
        error = setpoint - measurement
        p_out = self.kp * error
        
        i_candidate = self.integral + (self.ki * error * self.dt)
        full_output = p_out + i_candidate
        
        # [실무 팁] 인버터 전압 포화시 적분기 멈춤 (Anti-Windup)
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
    def __init__(self, R, L, Ke, dt):
        self.R = R
        self.L = L
        self.Ke = Ke
        self.dt = dt
        
        # [추가] 노이즈 환경에서 미분을 위해 LPF 필수 도입
        # 전류 미분용 필터 (Cutoff: 3000Hz) - 위상 지연을 최소화하며 고주파 노이즈 억제
        self.iq_lpf = LowPassFilter(cutoff_freq_hz=3000.0, dt=self.dt)
        # 역기전력 평활화용 필터 (Cutoff: 1500Hz)
        self.bemf_lpf = LowPassFilter(cutoff_freq_hz=1500.0, dt=self.dt)
        
        self.prev_iq_filtered = 0.0

    def estimate_rpm(self, vq, iq_measured):
        """
        Vq = R*iq + L*(diq/dt) + BEMF 에서 BEMF를 역산.
        실제 환경에서는 iq_measured에 노이즈가 가득하므로 반드시 필터링 후 사용해야 함.
        """
        # 1. 전류 노이즈 필터링
        iq_filt = self.iq_lpf.update(iq_measured)
        
        # 2. 필터링된 전류로 변화율(미분) 계산
        diq_dt_approx = (iq_filt - self.prev_iq_filtered) / self.dt
        self.prev_iq_filtered = iq_filt
        
        # 3. 역기전력(BEMF) 도출
        raw_bemf = vq - (self.R * iq_filt) - (self.L * diq_dt_approx)
        
        # 4. 도출된 BEMF 추가 필터링 (미분기로 인해 증폭된 잔여 노이즈 제거)
        filtered_bemf = self.bemf_lpf.update(raw_bemf)
        
        # 5. 각속도 및 RPM 변환
        est_omega = filtered_bemf / self.Ke
        return est_omega * RADS_TO_RPM

# =================================================================
# 5. 고속 모터 물리 엔진 (Plant)
# =================================================================
class HighSpeedMotor:
    def __init__(self):
        # 40,000 RPM 고속 덴탈/의료용 BLDC 모터 사양 기반
        self.R = 0.38           # 저항 [Ohm]
        self.L = 0.12e-3        # 인덕턴스 [H]
        self.Ke = 0.0042        # 역기전력 상수 [V*s/rad]
        self.Kt = 0.0042        # 토크 상수 [Nm/A]
        self.J = 0.00003        # 관성 [kg*m^2]
        self.B = 0.000005       # 마찰 [Nm*s/rad]
        self.P = 1              # 극쌍수

    def update(self, iq, omega, vq_inverter, load_torque, dt):
        # 전기적 모델: Vq = R*iq + L*(diq/dt) + Ke*omega
        diq_dt = (vq_inverter - self.R * iq - self.Ke * omega) / self.L
        
        # 기계적 모델: Te = 1.5 * P * Kt * iq (표면부착형 PMSM 근사)
        te = 1.5 * self.P * self.Kt * iq
        dw_dt = (te - load_torque - self.B * omega) / self.J
        
        # 적분 (Euler Method)
        iq_next = iq + diq_dt * dt
        omega_next = omega + dw_dt * dt
        
        return iq_next, omega_next

# =================================================================
# 6. 마스터 시뮬레이션 통합 실행
# =================================================================
def run_master_simulation():
    motor = HighSpeedMotor()
    observer = BEMFObserver(motor.R, motor.L, motor.Ke, DT)
    adc_iq = ADCSimulator(resolution_bits=12, max_range=20.0, noise_std=0.08) # 12비트 ADC, 80mA 수준의 노이즈 모사
    
    # 이중 루프 제어기 선언 (대역폭 고려하여 게인 재조정)
    speed_ctrl = PIDController(kp=0.08, ki=1.5, dt=DT, out_min=-15, out_max=15) # 전류 지령 한계값 +- 15A
    # 전압 출력 한계는 DC Link 전압(-48V ~ 48V)으로 제한 (실제 인버터 한계 모사)
    current_ctrl = PIDController(kp=1.5, ki=500.0, dt=DT, out_min=-V_DC_LINK, out_max=V_DC_LINK)

    # 상태 변수 (실제 모터 내부 상태 - MCU는 모름)
    true_iq, true_omega = 0.0, 0.0
    
    # 제어 변수
    vq_cmd = 0.0
    cmd_rpm = 0.0
    target_rpm = 40000.0
    
    # S-Curve 가속 변수
    accel_step = 0.0
    max_accel_step = 1.5       # 지침 최대 가속도
    jerk_step = 0.0010         # 슬로우 스타트 감도
    
    # 데이터 로깅용 딕셔너리
    history = {'t': [], 'cmd_rpm': [], 'act_rpm': [], 'est_rpm': [], 'true_iq': [], 'sensed_iq': [], 'vq': [], 'load': []}
    
    total_steps = int(2.0 / DT) # 2.0초 시뮬레이션 (동적 부하 테스트를 위해 시간 늘림)
    
    for step in range(total_steps):
        t = step * DT
        act_rpm = true_omega * RADS_TO_RPM
        
        # [실제 환경 모사 1] 동적 부하 (Dynamic Load) 인가
        # 1.2초 부근에서 치과용 드릴이 뼈/치아에 닿았다고 가정 (부하 급증)
        load_torque = 0.002
        if t > 1.2:
            load_torque = 0.015 
            
        # [실제 환경 모사 2] 센서 샘플링 (노이즈 및 양자화 추가)
        sensed_iq = adc_iq.read(true_iq)
        
        # [기능 1] 센서리스 속도 추정 (Observer) - 센싱된 값 사용
        est_rpm = observer.estimate_rpm(vq_cmd, sensed_iq)
        
        # [기능 2] S-Curve 및 오버슈트 제로 램프 제어
        if cmd_rpm < target_rpm:
            remaining_rpm = target_rpm - cmd_rpm
            stopping_dist = (accel_step**2) / (2 * jerk_step)
            
            if remaining_rpm <= stopping_dist:
                accel_step = max(0.0005, accel_step - jerk_step)
            else:
                accel_step = min(max_accel_step, accel_step + jerk_step)
                
            cmd_rpm = min(target_rpm, cmd_rpm + accel_step)
        elif cmd_rpm > target_rpm:
            cmd_rpm = max(target_rpm, cmd_rpm - 10.0)

        # [기능 3] 제어 루프 (Cascade Control)
        # 3.1 속도 제어기 (추정된 RPM 피드백 사용)
        iq_ref = speed_ctrl.compute(cmd_rpm, est_rpm)
        
        # 3.2 전류 제어기 (센싱된 전류 피드백 사용)
        vq_cmd_raw = current_ctrl.compute(iq_ref, sensed_iq)
        
        # [실제 환경 모사 3] 인버터 포화 (SVPWM Duty Limit)
        vq_cmd = np.clip(vq_cmd_raw, -V_DC_LINK, V_DC_LINK)
        
        # [기능 4] 물리 엔진 업데이트 (실제 모터는 깨끗한 true_iq와 실제 인가된 vq_cmd로 움직임)
        true_iq, true_omega = motor.update(true_iq, true_omega, vq_cmd, load_torque, DT)
        
        # 데이터 샘플링 (메모리 절약을 위해 200 step마다 로깅)
        if step % 200 == 0:
            history['t'].append(t)
            history['cmd_rpm'].append(cmd_rpm)
            history['act_rpm'].append(act_rpm)
            history['est_rpm'].append(est_rpm)
            history['true_iq'].append(true_iq)
            history['sensed_iq'].append(sensed_iq)
            history['vq'].append(vq_cmd)
            history['load'].append(load_torque)
            
    return history

# =================================================================
# 7. 통합 시각화 (그래프 그리기)
# =================================================================
data = run_master_simulation()

plt.figure(figsize=(14, 12))

# 1. 속도 응답 (실제 속도 vs 관측기 추정 속도 vs 명령어)
plt.subplot(4, 1, 1)
plt.plot(data['t'], data['cmd_rpm'], 'r--', label='Command Profile', linewidth=2)
plt.plot(data['t'], data['act_rpm'], 'b', label='True Actual RPM', linewidth=2, alpha=0.7)
plt.plot(data['t'], data['est_rpm'], 'g', label='Observer Estimated RPM', linewidth=1, alpha=0.8)
# 동적 부하 시점 표시
plt.axvline(x=1.2, color='k', linestyle=':', label='Load Step Inserted') 
plt.title("Master HIL Simulation: FOC Sensorless Speed Response")
plt.ylabel("Speed [RPM]"); plt.legend(loc='lower right'); plt.grid(True)

# 2. 전류 응답 (실제 전류 vs 노이즈 낀 센싱 전류)
plt.subplot(4, 1, 2)
plt.plot(data['t'], data['sensed_iq'], 'gray', label='Sensed iq (ADC + Noise)', alpha=0.5)
plt.plot(data['t'], data['true_iq'], 'k', label='True iq (Motor Status)', linewidth=1.5)
plt.axvline(x=1.2, color='k', linestyle=':') 
plt.title("Current Response (Observer sees Gray, Motor is Black)")
plt.ylabel("Current [A]"); plt.legend(loc='upper right'); plt.grid(True)

# 3. 전압 응답 (인버터 포화 상태 확인)
plt.subplot(4, 1, 3)
plt.plot(data['t'], data['vq'], 'm', label='Vq Command (Inverter Output)')
plt.axhline(y=V_DC_LINK, color='r', linestyle='--', label='DC Link Limit (+48V)')
plt.axhline(y=-V_DC_LINK, color='r', linestyle='--')
plt.title("Inverter Voltage Saturation")
plt.ylabel("Voltage [V]"); plt.legend(loc='upper right'); plt.grid(True)

# 4. 외부 부하 토크 시나리오
plt.subplot(4, 1, 4)
plt.plot(data['t'], data['load'], 'orange', label='External Load Torque', linewidth=2)
plt.title("Disturbance (Load Torque Profile)")
plt.xlabel("Time [s]"); plt.ylabel("Torque [Nm]"); plt.legend(loc='upper left'); plt.grid(True)

plt.tight_layout()
plt.show()