import numpy as np
import matplotlib.pyplot as plt

# =================================================================
# 1. PID 제어기 클래스 (MCU 이식용 구조)
# =================================================================
class PIDController:
    def __init__(self, kp, ki, kd, dt, out_min, out_max):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.out_min = out_min
        self.out_max = out_max
        
        self.integral = 0
        self.prev_error = 0

    def compute(self, setpoint, measurement):
        error = setpoint - measurement
        
        # P 항
        p_out = self.kp * error
        
        # I 항 (Anti-windup 적용: 출력이 포화되면 적분 중지)
        self.integral += error * self.dt
        i_out = self.ki * self.integral
        
        # D 항
        d_out = self.kd * (error - self.prev_error) / self.dt
        self.prev_error = error
        
        # 총 출력 및 제한(Saturation)
        output = p_out + i_out + d_out
        if output > self.out_max:
            output = self.out_max
            self.integral -= error * self.dt # Back-calculation 방식의 단순화
        elif output < self.out_min:
            output = self.out_min
            self.integral -= error * self.dt
            
        return output

# =================================================================
# 2. 파라미터 및 하드웨어 설정
# =================================================================
R, L = 1.0, 0.5e-3
Ke, Kt = 0.01, 0.01
J, B = 0.001, 0.0001
P = 1
V_BUS = 24.0
dt = 1e-4 # 10kHz 제어 루프

# 제어기 객체 생성
# 속도 제어기 (Outer Loop): 속도 오차를 보고 필요한 q축 전류(iq)를 결정
speed_pid = PIDController(kp=0.2, ki=1.5, kd=0.001, dt=dt, out_min=-10, out_max=10)

# 전류 제어기 (Inner Loop): 전류 오차를 보고 필요한 전압(Vd, Vq)을 결정
iq_pid = PIDController(kp=5.0, ki=100.0, kd=0, dt=dt, out_min=-V_BUS, out_max=V_BUS)
id_pid = PIDController(kp=5.0, ki=100.0, kd=0, dt=dt, out_min=-V_BUS, out_max=V_BUS)

# =================================================================
# 3. 물리 엔진 및 메인 루프
# =================================================================
def motor_physics_step(x, vd, vq, dt):
    id, iq, w, theta = x
    we = w * P
    did_dt = (vd - R*id + we * L * iq) / L
    diq_dt = (vq - R*iq - we * L * id - we * Ke) / L
    Te = 1.5 * P * Kt * iq
    dw_dt = (Te - B*w) / J
    dtheta_dt = we
    return [id + d * dt for id, d in zip(x, [did_dt, diq_dt, dw_dt, dtheta_dt])]

# 시뮬레이션 변수
t_end = 0.8
steps = int(t_end / dt)
state = [0.0, 0.0, 0.0, 0.0] # [id, iq, w, theta]
history = {'t':[], 'w':[], 'w_ref':[], 'iq':[], 'iq_ref':[]}

# 메인 제어 루프 (Interrupt Service Routine 모사)
for i in range(steps):
    t = i * dt
    curr_id, curr_iq, curr_w, curr_theta = state
    
    # [목표 설정] 0.1초에 Step 입력, 0.5초에 목표 변경
    w_target = 150.0 if t > 0.1 else 0.0
    if t > 0.5: w_target = 80.0
    
    # --- [Step 1] 속도 제어기 (Outer Loop) ---
    # 10kHz로 돌리지만, 실제로는 전류 루프보다 천천히 돌려도 됨
    iq_ref = speed_pid.compute(w_target, curr_w)
    id_ref = 0.0 # 자속 제어 (보통 0)
    
    # --- [Step 2] 전류 제어기 (Inner Loop) ---
    Vq_cmd = iq_pid.compute(iq_ref, curr_iq)
    Vd_cmd = id_pid.compute(id_ref, curr_id)
    
    # --- [Step 3] 물리 엔진 업데이트 ---
    state = motor_physics_step(state, Vd_cmd, Vq_cmd, dt)
    
    # 데이터 저장
    if i % 10 == 0: # 그래프 부하를 줄이기 위해 10스텝마다 저장
        history['t'].append(t)
        history['w'].append(curr_w)
        history['w_ref'].append(w_target)
        history['iq'].append(curr_iq)
        history['iq_ref'].append(iq_ref)

# =================================================================
# 4. 결과 시각화
# =================================================================
plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.plot(history['t'], history['w_ref'], 'r--', label='Target Speed')
plt.plot(history['t'], history['w'], 'b', label='Actual Speed')
plt.title("Cascaded PID: Speed Control")
plt.ylabel("Speed [rad/s]"); plt.legend(); plt.grid(True)

plt.subplot(2,1,2)
plt.plot(history['t'], history['iq_ref'], 'r--', label='iq_ref (from Speed PID)')
plt.plot(history['t'], history['iq'], 'g', label='Actual iq')
plt.title("Current Control Loop")
plt.xlabel("Time [s]"); plt.ylabel("Current [A]"); plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()