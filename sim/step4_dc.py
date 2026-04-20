import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 파라미터
R = 1.0     # 저항 [Ohm]
L = 0.5     # 인덕턴스 [H]  
Ke = 0.1    # 역기전력 상수 [V*s/rad]
Kt = 0.1    # 토크 상수 [Nm/A]
J = 0.01    # 회전자 관성 [kg*m^2]
B = 0.001   # 마찰 계수 [Nm*s/rad]

def motor(t, x):
    i, w = x    # 전류와 각속도
    V = 24      # 입력 전압

    di_dt = (V - R*i - Ke*w) / L    # 전류 변화율
    dw_dt = (Kt*i - B*w) / J        # 각속도 변화율

    return [di_dt, dw_dt]           # 반환값은 [di/dt, dw/dt]

sol = solve_ivp(motor, [0, 1], [0, 0], t_eval=np.linspace(0,1,1000)) # 1초 동안 시뮬레이션, 초기 전류와 속도는 0
# 전류 그래프 추가
plt.plot(sol.t, sol.y[0])   # 전류(i) 그래프
plt.plot(sol.t, sol.y[1])   # 각속도(w) 그래프
plt.xlabel("Time")          # 시간 축 레이블
plt.ylabel("Current/Speed")  # 속도 축 레이블       
plt.title("Motor Response")   # 그래프 제목
plt.grid(True)              # 그리드 추가
plt.show()                  # 그래프 표시