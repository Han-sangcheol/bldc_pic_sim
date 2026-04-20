import numpy as np
import matplotlib.pyplot as plt

# 1. Faulhaber 1645L036BHSA 모터 파라미터 (Shared Context)
R = 4.07           # 저항 [Ohm]
L = 150e-6         # 인덕턴스 [H]
Kt = 0.0365        # 토크 상수 [Nm/A]
Ke = 0.0365        # 역기전력 상수 [V*s/rad]
J = 1.2e-7         # 회전자 관성 [kg*m^2] (1.2 g*cm^2)
B = 1e-8           # 마찰 계수 (추정치)

# 2. 제어 및 시뮬레이션 설정
dt = 25e-6         # 제어 주기: 25us (40kHz)
total_time = 0.5   # 총 시뮬레이션 시간 [s]
steps = int(total_time / dt)

# 가감속 제한 (User Command 반영)
accel_limit_rpm = 2    # 25us당 2 RPM 증가
decel_limit_rpm = 10   # 25us당 10 RPM 감소

# 3. 변수 초기화
current_speed_rpm = 0.0
target_speed_rpm = 40000.0  # 목표 속도: 40,000 RPM
v_out = 0.0        # 인버터 출력 전압
i = 0.0            # 모터 전류
omega = 0.0        # 각속도 [rad/s]

history = {"time": [], "speed": [], "current": []}

# 4. 시뮬레이션 루프
for step in range(steps):
    t = step * dt
    
    # [CONTROL LOOP] 25us마다 실행
    # 가속/감속 프로파일 적용
    speed_err = target_speed_rpm - current_speed_rpm
    if speed_err > accel_limit_rpm:
        current_speed_rpm += accel_limit_rpm
    elif speed_err < -decel_limit_rpm:
        current_speed_rpm -= decel_limit_rpm
    else:
        current_speed_rpm = target_speed_rpm
    
    # 단순 PI 제어기 예시 (실제 FOC 알고리즘으로 대체 가능)
    omega_ref = current_speed_rpm * (2 * np.pi / 60)
    error_omega = omega_ref - omega
    v_out = error_omega * 0.1 + 12.0  # 단순 Feed-forward + P제어 예시
    if v_out > 24: v_out = 24  # 전압 제한 (24V)

    # [PHYSICS MODEL] 모터 물리 방정식 계산 (Forward Euler)
    # di/dt = (V - R*i - Ke*omega) / L
    di = (v_out - R * i - Ke * omega) / L * dt
    i += di
    
    # d_omega/dt = (Kt*i - B*omega) / J
    d_omega = (Kt * i - B * omega) / J * dt
    omega += d_omega
    
    # 결과 저장
    history["time"].append(t)
    history["speed"].append(omega * 60 / (2 * np.pi))
    history["current"].append(i)

# 5. 시각화
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history["time"], history["speed"], label='Speed (RPM)')
plt.axhline(40000, color='r', linestyle='--', alpha=0.5)
plt.title("Motor Speed Response (40k RPM)")
plt.ylabel("RPM")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history["time"], history["current"], label='Current (A)', color='orange')
plt.title("Current Response")
plt.ylabel("Amperes")
plt.grid(True)
plt.show()