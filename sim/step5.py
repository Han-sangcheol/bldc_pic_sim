import numpy as np
import matplotlib.pyplot as plt

# =================================================================
# 1. 하드웨어/모터 파라미터 설정
# =================================================================
R = 1.0          # [Ohm]
L = 0.5e-3       # [H]
Ke = 0.01        # [V*s/rad]
Kt = 0.01        # [Nm/A]
J = 0.001        # [kg*m^2]
B = 0.0001       # [Nm*s/rad]
P = 1            # Pole Pairs
V_BUS = 24.0     # DC Bus Voltage [V]

dt = 1e-4        # 제어 주기 (10kHz = 0.1ms)
T_END = 0.5      # 총 시뮬레이션 시간

# =================================================================
# 2. 센서 데이터 취득 모듈 (Hardware Abstraction Layer)
# 실제 하드웨어 연결 시 이 함수들의 내부를 ADC/Encoder 읽기로 대체
# =================================================================
def read_current_sensors(ia_sim, ib_sim, ic_sim):
    """
    실제 환경: ADC를 통해 상전류를 읽음
    시뮬레이션: 모델에서 계산된 상전류에 노이즈 등을 추가할 수 있음
    """
    return ia_sim, ib_sim, ic_sim

def read_encoder(theta_sim, w_sim):
    """
    실제 환경: Encoder나 Resolver로부터 각도와 속도를 읽음
    """
    return theta_sim, w_sim

# =================================================================
# 3. 제어 알고리즘 모듈 (Control Algorithm)
# 이 부분의 코드는 그대로 MCU firmware에 이식 가능하도록 설계
# =================================================================
def clarke_transform(ia, ib, ic):
    i_alpha = (2/3) * (ia - 0.5*ib - 0.5*ic)
    i_beta = (2/3) * (np.sqrt(3)/2 * ib - np.sqrt(3)/2 * ic)
    return i_alpha, i_beta

def park_transform(i_alpha, i_beta, theta):
    id = i_alpha * np.cos(theta) + i_beta * np.sin(theta)
    iq = -i_alpha * np.sin(theta) + i_beta * np.cos(theta)
    return id, iq

def inv_park_transform(vd, vq, theta):
    v_alpha = vd * np.cos(theta) - vq * np.sin(theta)
    v_beta = vd * np.sin(theta) + vq * np.cos(theta)
    return v_alpha, v_beta

# =================================================================
# 4. 모터 물리 모델 (Motor Physics - 시뮬레이션용)
# =================================================================
def motor_physics_step(x, vd, vq, dt):
    """
    RK4 혹은 오일러 방법을 이용해 미분 방정식을 1스텝 전진시킴
    x = [id, iq, w, theta]
    """
    id, iq, w, theta = x
    we = w * P
    
    # 미분 방정식 (dq 축)
    did_dt = (vd - R*id + we * L * iq) / L
    diq_dt = (vq - R*iq - we * L * id - we * Ke) / L
    
    Te = 1.5 * P * Kt * iq
    dw_dt = (Te - B*w) / J
    dtheta_dt = we
    
    # 오일러 적분 (단순화된 형태)
    id_next = id + did_dt * dt
    iq_next = iq + diq_dt * dt
    w_next = w + dw_dt * dt
    theta_next = theta + dtheta_dt * dt
    
    return [id_next, iq_next, w_next, theta_next]

# =================================================================
# 5. 메인 루프 (Main Control Loop)
# =================================================================
time_steps = int(T_END / dt)
state = [0.0, 0.0, 0.0, 0.0]  # 초기 상태 [id, iq, w, theta]

# 결과 저장을 위한 리스트
history = {'t': [], 'iq': [], 'w': [], 'ia': []}

for i in range(time_steps):
    t = i * dt
    curr_id, curr_iq, curr_w, curr_theta = state
    
    # --- [STEP 1] 센서 데이터 취득 (Simulated Sensors) ---
    # 실제 하드웨어라면 이 단계에서 ADC와 Encoder 값을 가져옴
    # 역 Park/Clarke를 거쳐 ia, ib, ic를 역산(시뮬레이션용)
    va, vb = inv_park_transform(curr_id, curr_iq, curr_theta) # 편의상 전류 대신 전압으로 흐름 예시
    # (실제 시뮬레이션에서는 물리 모델의 id, iq를 그대로 사용)
    
    ia_sens, ib_sens, ic_sens = 0, 0, 0 # 실제라면 여기서 ia, ib 추출
    th_sens, w_sens = read_encoder(curr_theta, curr_w)
    
    # --- [STEP 2] 제어 연산 (Control Logic) ---
    # 목표 속도 100 rad/s
    w_target = 100.0
    
    # 속도 제어 (단순 P 제어기 예시)
    iq_target = 0.5 * (w_target - w_sens) 
    id_target = 0.0  # 효율을 위해 d축은 0으로 제어
    
    # 전압 출력 결정 (여기서는 간단히 전압으로 변환)
    Vd_cmd = id_target * R # 실제로는 PI 제어기가 들어갈 자리
    Vq_cmd = iq_target * R + w_sens * Ke # 역기전력 보상 포함
    
    # --- [STEP 3] 물리 엔진 업데이트 (Simulated Plant) ---
    state = motor_physics_step(state, Vd_cmd, Vq_cmd, dt)
    
    # 데이터 저장
    history['t'].append(t)
    history['iq'].append(curr_iq)
    history['w'].append(curr_w)

# 결과 그래프
plt.figure(figsize=(10, 6))
plt.subplot(2,1,1)
plt.plot(history['t'], history['w'], label='Speed [rad/s]')
plt.axhline(100, color='r', linestyle='--', label='Target')
plt.legend(); plt.grid(True)

plt.subplot(2,1,2)
plt.plot(history['t'], history['iq'], label='q-axis Current [A]')
plt.legend(); plt.grid(True)
plt.show()