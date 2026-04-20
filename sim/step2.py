import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def run_firmware_style_sim():
    # --- 1. Faulhaber 1645L036BHSA 물리 파라미터 ---
    R, L, J, Kt, Ke = 4.07, 0.00015, 1.2e-7, 0.0365, 0.0365
    
    # --- 2. 제어 주기 및 가감속 설정 (25us 기준) ---
    dt = 25e-6          # 25us (40kHz)
    acc_step = 2.0      # 가속 시 25us마다 +2 RPM
    dec_step = 10.0     # 감속 시 25us마다 -10 RPM
    
    target_rpm = 40000.0 # 목표 속도
    T_end = 0.2         # 총 시뮬레이션 시간
    
    # 제어 변수 초기화
    curr_speed_rpm = 0.0
    ref_speed_rpm = 0.0
    current = 0.0
    voltage = 0.0
    
    # 기록용 리스트
    history = {'t': [], 'ref': [], 'act': [], 'curr': []}

    # 터미널 헤더 출력
    print(f"{'시간(s)':>8} | {'목표(RPM)':>10} | {'실제(RPM)':>10} | {'전류(A)':>8}")
    print("-" * 50)

    # --- 3. 시뮬레이션 루프 (25us 단위 실행) ---
    steps = int(T_end / dt)
    for step in range(steps):
        t = step * dt
        
        # [Ramp Generator] 가감속 로직
        if ref_speed_rpm < target_rpm:
            ref_speed_rpm += acc_step
            if ref_speed_rpm > target_rpm: ref_speed_rpm = target_rpm
        elif ref_speed_rpm > target_rpm:
            ref_speed_rpm -= dec_step
            if ref_speed_rpm < target_rpm: ref_speed_rpm = target_rpm

        # [Control & Physics] 실제와 거의 흡사한 전압 인가 및 물리 반응
        # (간단한 비례 제어로 목표 속도를 추종한다고 가정)
        v_bemf = Ke * (curr_speed_rpm * 2 * np.pi / 60)
        # 속도 오차에 따른 전압 결정 (P-Control)
        voltage = (ref_speed_rpm - curr_speed_rpm) * 0.01 + v_bemf
        voltage = np.clip(voltage, -36, 36)
        
        # 물리 법칙 계산
        di = (voltage - (current * R) - v_bemf) / L * dt
        current += di
        dw = (current * Kt) / J * dt
        curr_speed_rpm += (dw * 60 / (2 * np.pi)) * dt
        
        # [터미널 출력] 1ms(40 step)마다 화면에 표시
        if step % 40 == 0:
            print(f"{t:8.4f} | {ref_speed_rpm:10.1f} | {curr_speed_rpm:10.1f} | {current:8.3f}")
            # 그래프 기록용 데이터 저장
            history['t'].append(t)
            history['ref'].append(ref_speed_rpm)
            history['act'].append(curr_speed_rpm)
            history['curr'].append(current)

    # --- 4. 그래프 이미지 저장 ---
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(history['t'], history['ref'], 'k--', label='Target Ramp')
    plt.plot(history['t'], history['act'], 'r', label='Actual Speed')
    plt.ylabel("Speed [RPM]"); plt.grid(True); plt.legend()
    plt.title("Faulhaber 1645L Ramp Control Analysis")

    plt.subplot(2, 1, 2)
    plt.plot(history['t'], history['curr'], 'b', label='Current [A]')
    plt.xlabel("Time [s]"); plt.ylabel("Current [A]"); plt.grid(True); plt.legend()

    plt.tight_layout()
    plt.savefig('ramp_control_result.png')
    print(f"\n>>> 완료! 터미널 수치와 'ramp_control_result.png' 파일을 모두 확인하세요.")

if __name__ == "__main__":
    run_firmware_style_sim()