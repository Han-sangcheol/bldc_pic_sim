import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from collections import deque

# =================================================================
# 1. 시스템 상수 및 유틸리티 (이전과 동일)
# =================================================================
DT = 25e-6                          
RPM_TO_RADS = 2 * np.pi / 60        
RADS_TO_RPM = 60 / (2 * np.pi)      
V_DC_LINK = 36.0                    

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
        self.levels = 2 ** resolution_bits
        self.max_range = max_range
        self.noise_std = noise_std
    def read(self, true_value):
        noisy_value = true_value + np.random.normal(0, self.noise_std)
        noisy_value = np.clip(noisy_value, -self.max_range, self.max_range)
        quantized = np.round((noisy_value / self.max_range) * (self.levels / 2)) 
        return (quantized / (self.levels / 2)) * self.max_range

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

class BEMFObserver:
    def __init__(self, R, L, FluxLinkage, P, dt):
        self.R, self.L, self.FluxLinkage, self.P, self.dt = R, L, FluxLinkage, P, dt
        self.iq_lpf = LowPassFilter(cutoff_freq_hz=500.0, dt=self.dt)
        self.bemf_lpf = LowPassFilter(cutoff_freq_hz=150.0, dt=self.dt)
        self.prev_iq_filtered = 0.0
    def estimate_rpm(self, prev_vq, iq_measured):
        iq_filt = self.iq_lpf.update(iq_measured)
        diq_dt_approx = (iq_filt - self.prev_iq_filtered) / self.dt
        self.prev_iq_filtered = iq_filt
        raw_bemf = prev_vq - (self.R * iq_filt) - (self.L * diq_dt_approx)
        filtered_bemf = self.bemf_lpf.update(raw_bemf)
        return (filtered_bemf / self.FluxLinkage) / self.P * RADS_TO_RPM

class HighSpeedMotor:
    def __init__(self):
        self.R = 0.38                   
        self.L = 0.12e-3                
        self.FluxLinkage = 0.0042       
        self.J = 2.0e-6                 
        self.B = 4.5e-7                 
        self.P = 1                      
    def update(self, iq, omega_m, vq_inverter, load_torque, dt):
        omega_e = omega_m * self.P
        diq_dt = (vq_inverter - self.R * iq - omega_e * self.FluxLinkage) / self.L
        te = 1.5 * self.P * self.FluxLinkage * iq
        dw_dt = (te - load_torque - self.B * omega_m) / self.J
        return iq + diq_dt * dt, omega_m + dw_dt * dt

# =================================================================
# 2. 실시간 시뮬레이션 엔진 (상태 유지 관리자)
# =================================================================
class RealTimeSimulation:
    def __init__(self):
        self.motor = HighSpeedMotor()
        self.observer = BEMFObserver(self.motor.R, self.motor.L, self.motor.FluxLinkage, self.motor.P, DT)
        self.adc_iq = ADCSimulator()
        self.speed_ctrl = PIDController(kp=0.008, ki=0.1, dt=DT, out_min=-10, out_max=10)
        self.current_ctrl = PIDController(kp=0.8, ki=150.0, dt=DT, out_min=-V_DC_LINK, out_max=V_DC_LINK)

        # 시스템 상태 변수
        self.true_iq, self.true_omega_m = 0.0, 0.0
        self.vq_cmd, self.prev_vq_cmd = 0.0, 0.0
        self.cmd_rpm, self.accel_step = 0.0, 0.0
        self.t_current = 0.0
        
        self.max_accel_step = 1.0
        self.jerk_step = 0.0005

    def step(self, target_rpm, base_load, is_on, steps=1000, log_interval=20):
        """ 한 번의 GUI 프레임 동안 n번의 제어 루프를 고속 연산하여 결과 반환 """
        results = {'t':[], 'cmd_rpm':[], 'act_rpm':[], 'est_rpm':[], 'iq':[], 'vq':[], 'load':[]}
        
        # OFF 상태이면 목표 RPM을 0으로 강제
        actual_target_rpm = target_rpm if is_on else 0.0

        for i in range(steps):
            self.t_current += DT
            act_rpm = self.true_omega_m * RADS_TO_RPM
            
            # 동적 가변 부하 생성 (OFF 상태이거나 회전하지 않을 때는 부하 제거)
            load_torque = 0.0
            if is_on and act_rpm > 1000:
                low_freq = 0.008 * np.sin(2 * np.pi * 5.0 * self.t_current)
                high_freq = 0.003 * np.sin(2 * np.pi * 50.0 * self.t_current)
                noise = np.random.normal(0, 0.002)
                load_torque = max(0.0, base_load + low_freq + high_freq + noise)
                
            sensed_iq = self.adc_iq.read(self.true_iq)
            est_rpm = self.observer.estimate_rpm(self.prev_vq_cmd, sensed_iq) 
            self.prev_vq_cmd = self.vq_cmd 
            
            # S-Curve 램프 제어
            if self.cmd_rpm < actual_target_rpm:
                stopping_dist = (self.accel_step**2) / (2 * self.jerk_step) + (self.accel_step * 2.0) 
                if (actual_target_rpm - self.cmd_rpm) <= stopping_dist:
                    self.accel_step = max(0.0001, self.accel_step - self.jerk_step)
                else:
                    self.accel_step = min(self.max_accel_step, self.accel_step + self.jerk_step)
                self.cmd_rpm = min(actual_target_rpm, self.cmd_rpm + self.accel_step)
            elif self.cmd_rpm > actual_target_rpm:
                self.cmd_rpm = max(actual_target_rpm, self.cmd_rpm - 15.0) # 감속은 조금 더 빠르게

            # 인버터 완전 정지 (Free-wheeling) 조건
            if not is_on and self.cmd_rpm < 100.0 and act_rpm < 100.0:
                self.vq_cmd = 0.0
                self.cmd_rpm = 0.0
            else:
                iq_ref = self.speed_ctrl.compute(self.cmd_rpm, est_rpm)
                self.vq_cmd = self.current_ctrl.compute(iq_ref, sensed_iq)
            
            # 물리 엔진 업데이트
            self.true_iq, self.true_omega_m = self.motor.update(self.true_iq, self.true_omega_m, self.vq_cmd, load_torque, DT)
            
            # 지정된 간격마다 데이터 샘플링 (메모리 및 렌더링 최적화)
            if i % log_interval == 0:
                results['t'].append(self.t_current)
                results['cmd_rpm'].append(self.cmd_rpm)
                results['act_rpm'].append(act_rpm)
                results['est_rpm'].append(est_rpm)
                results['iq'].append(sensed_iq)
                results['vq'].append(self.vq_cmd)
                results['load'].append(load_torque)
                
        return results

# =================================================================
# 3. Matplotlib GUI 대시보드 구축
# =================================================================
sim = RealTimeSimulation()

# 화면에 표시할 시간 길이 (예: 과거 1.5초치 데이터를 화면에 표시)
DISPLAY_TIME = 1.5 
BUFFER_SIZE = int((DISPLAY_TIME / DT) / 20) # 20스텝마다 로깅하므로 크기 조절

# Rolling Buffer (오른쪽으로 데이터가 들어오고 왼쪽으로 빠짐)
history = {
    't': deque(maxlen=BUFFER_SIZE),
    'cmd_rpm': deque(maxlen=BUFFER_SIZE), 'act_rpm': deque(maxlen=BUFFER_SIZE), 'est_rpm': deque(maxlen=BUFFER_SIZE),
    'iq': deque(maxlen=BUFFER_SIZE), 'vq': deque(maxlen=BUFFER_SIZE), 'load': deque(maxlen=BUFFER_SIZE)
}

# GUI 전역 상태 변수
gui_state = {'target_rpm': 40000.0, 'base_load': 0.0, 'is_on': False}

# 레이아웃 설정
fig, axs = plt.subplots(4, 1, figsize=(12, 9))
plt.subplots_adjust(bottom=0.25, hspace=0.4)
fig.canvas.manager.set_window_title("Real-Time Motor FOC Dashboard")

# 그래프 선 객체 미리 생성
line_cmd, = axs[0].plot([], [], 'r--', label='Command RPM', lw=2)
line_act, = axs[0].plot([], [], 'b-', label='Actual RPM', lw=2, alpha=0.7)
line_est, = axs[0].plot([], [], 'g-', label='Estimated RPM', lw=1, alpha=0.8)
line_iq,  = axs[1].plot([], [], 'gray', label='Sensed Iq (ADC)', alpha=0.8)
line_vq,  = axs[2].plot([], [], 'm-', label='Vq Command (PWM)')
line_ld,  = axs[3].plot([], [], 'orange', label='Dynamic Load Torque', lw=2)

# Y축 범위 고정 (그래프가 위아래로 널뛰기 하는 것 방지)
axs[0].set_ylim(-2000, 45000); axs[0].set_ylabel("Speed [RPM]"); axs[0].legend(loc='upper left'); axs[0].grid(True)
axs[1].set_ylim(-15, 15);      axs[1].set_ylabel("Current [A]"); axs[1].legend(loc='upper left'); axs[1].grid(True)
axs[2].set_ylim(-40, 40);      axs[2].set_ylabel("Voltage [V]"); axs[2].legend(loc='upper left'); axs[2].grid(True)
axs[3].set_ylim(-0.01, 0.1);   axs[3].set_ylabel("Torque [Nm]"); axs[3].legend(loc='upper left'); axs[3].grid(True)
axs[3].set_xlabel("Time [s]")

# ---------------- 컨트롤 패널 (Sliders & Buttons) ----------------
ax_rpm  = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor='lightcyan')
ax_load = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_btn  = plt.axes([0.85, 0.10, 0.1, 0.08])

slider_rpm = Slider(ax_rpm, 'Target RPM', 0.0, 45000.0, valinit=40000.0, valstep=1000)
slider_load = Slider(ax_load, 'Base Load [Nm]', 0.0, 0.08, valinit=0.0, valstep=0.005)
btn_onoff = Button(ax_btn, 'TURN ON', color='lightcoral', hovercolor='red')

def update_gui_vars(val):
    gui_state['target_rpm'] = slider_rpm.val
    gui_state['base_load'] = slider_load.val

def toggle_power(event):
    gui_state['is_on'] = not gui_state['is_on']
    if gui_state['is_on']:
        btn_onoff.label.set_text('TURN OFF')
        btn_onoff.color = 'lightgreen'
    else:
        btn_onoff.label.set_text('TURN ON')
        btn_onoff.color = 'lightcoral'
    fig.canvas.draw_idle()

slider_rpm.on_changed(update_gui_vars)
slider_load.on_changed(update_gui_vars)
btn_onoff.on_clicked(toggle_power)

# ---------------- 애니메이션 루프 ----------------
def update_plot(frame):
    # 화면 갱신 주기(예: 30FPS -> 약 33ms) 동안 발생해야 할 시뮬레이션 스텝 수 계산
    # 33ms / 25us = 약 1320 steps. 여유롭게 1500 스텝씩 처리하여 실시간성을 맞춤
    STEPS_PER_FRAME = 1500 
    
    # 물리 엔진 구동 및 데이터 수집
    data = sim.step(gui_state['target_rpm'], gui_state['base_load'], gui_state['is_on'], steps=STEPS_PER_FRAME)
    
    # 원형 버퍼에 데이터 추가
    history['t'].extend(data['t'])
    history['cmd_rpm'].extend(data['cmd_rpm'])
    history['act_rpm'].extend(data['act_rpm'])
    history['est_rpm'].extend(data['est_rpm'])
    history['iq'].extend(data['iq'])
    history['vq'].extend(data['vq'])
    history['load'].extend(data['load'])
    
    # 버퍼에 데이터가 찼을 때만 그래프 렌더링
    if len(history['t']) > 0:
        t_arr = np.array(history['t'])
        
        line_cmd.set_data(t_arr, history['cmd_rpm'])
        line_act.set_data(t_arr, history['act_rpm'])
        line_est.set_data(t_arr, history['est_rpm'])
        line_iq.set_data(t_arr, history['iq'])
        line_vq.set_data(t_arr, history['vq'])
        line_ld.set_data(t_arr, history['load'])
        
        # X축을 가장 최근 시간에 맞춰 오실로스코프처럼 스크롤되도록 설정
        t_max = t_arr[-1]
        t_min = t_max - DISPLAY_TIME
        for ax in axs:
            ax.set_xlim(t_min, t_max)
            
    return line_cmd, line_act, line_est, line_iq, line_vq, line_ld

# 애니메이션 실행 (interval=30은 약 30ms 마다 화면 갱신을 의미)
ani = FuncAnimation(fig, update_plot, interval=30, blit=False, cache_frame_data=False)

plt.show()