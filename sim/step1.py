import numpy as np
import matplotlib
# GUI 충돌 방지를 위해 파일 저장 전용 백엔드 사용
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def run_integrated_sim():
    # --- 1. 가상 모터 스펙 (The Digital Twin) ---
    R = 1.0        # 저항 (Ohm)
    L = 0.1        # 인덕턴스 (H)
    J = 0.02       # 회전자 무게감 (Inertia)
    Kt = 0.5       # 토크 상수
    dt = 0.1       # 0.1초 간격으로 계산
    T_end = 2.0    # 2초 동안 시뮬레이션
    
    # --- 2. 초기 상태 설정 ---
    current = 0.0
    speed = 0.0
    voltage = 10.0
    
    # 결과를 저장할 리스트 (그래프용)
    time_history = []
    current_history = []
    rpm_history = []

    # --- 3. 터미널 출력 헤더 ---
    print(f"{'시간(초)':>6} | {'전류변화(A)':>10} | {'전류(A)':>8} | {'속도(RPM)':>10}")
    print("-" * 50)

    # --- 4. 시뮬레이션 메인 루프 ---
    steps = int(T_end / dt) + 1
    for step in range(steps):
        timestamp = step * dt
        
        # [A] 전기 계산 (전압 -> 전류)
        di = (voltage - current * R) / L * dt
        current += di
        
        # [B] 운동 계산 (전류 -> 속도)
        dw = (current * Kt) / J * dt
        speed += dw
        
        # RPM 변환
        rpm = speed * 60 / (2 * np.pi)
        
        # 데이터 저장
        time_history.append(timestamp)
        current_history.append(current)
        rpm_history.append(rpm)
        
        # 터미널에 한 줄씩 출력
        print(f"{timestamp:8.1f} | {di:10.2f} | {current:10.2f} | {rpm:12.1f}")

    # --- 5. 이미지(그래프) 생성 ---
    plt.figure(figsize=(10, 8))

    # 첫 번째 그래프: 전류 변화
    plt.subplot(2, 1, 1)
    plt.plot(time_history, current_history, 'b-o', label='Current (A)')
    plt.title("Motor Simulation: Electrical & Mechanical Response")
    plt.ylabel("Current [A]")
    plt.grid(True)
    plt.legend()

    # 두 번째 그래프: 속도(RPM) 변화
    plt.subplot(2, 1, 2)
    plt.plot(time_history, rpm_history, 'r-s', label='Speed (RPM)')
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [RPM]")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('basic_motor_sim.png')
    print("\n>>> 시뮬레이션 완료! 'basic_motor_sim.png' 파일이 생성되었습니다.")

if __name__ == "__main__":
    run_integrated_sim()