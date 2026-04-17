"""
1. SMO(슬라이딩 모드 관측기)란 무엇인가?
전문가용:
실제 모터의 전기적 수식(Plant)과 동일한 수학 모델을 코드 안에 만들고, 
실제 측정된 전류와 코드 안의 추정 전류 사이의 오차를 
'슬라이딩 모드' 제어 기법으로 강제로 0으로 수렴시켜 역기전력(BEMF)을 
추출하는 장치입니다.
비전문가용:
눈을 감고 빠르게 돌아가는 회전판을 손가락으로 살짝 건드리며 따라가는 것과 같습니다.
회전판과 내 손가락의 위치가 다르면 즉시 손가락을 밀거나 당겨서 위치를 맞춥니다.
이때 내 손가락이 받는 '힘'을 분석하면 회전판이 어디에 있는지 알 수 있습니다.

2. 필요한 입력값과 구하는 법SMO를 돌리려면 모터의 **'신분증'**에 적힌 정보가
필요합니다.
입력 파라미터,의미,구하는 방법
R (Resistance),권선 저항,멀티미터(저항계)로 모터 상간 저항을 측정 후 2로 나눔.
L (Inductance),권선 인덕턴스,LCR 미터로 측정. 고속 모터일수록 매우 작은 값을 가짐.
Ke​ (Back-EMF),역기전력 상수,모터를 외부 힘으로 돌릴 때 발생하는 전압의 크기. 데이터시트 확인.
Ts​ (Sampling),제어 주기,MCU의 타이머 인터럽트 주기 (예: 10kHz = 0.0001초).
Ksmo​ (Gain),관측기 강도,튜닝값. 모터가 만들어내는 역기전력보다 약간 크게 설정.
Glpf​ (Filter),노이즈 필터,"튜닝값. 너무 낮으면 각도가 느려지고, 너무 높으면 노이즈가 심함."
"""

import numpy as np
import matplotlib
matplotlib.use('Agg') # GUI 환경이 없어도 이미지 파일로 저장 가능하게 함
import matplotlib.pyplot as plt

class SensorlessSMO:
    """
    SMO(Sliding Mode Observer) 클래스
    목적: 측정된 전류와 전압을 바탕으로 모터 내부의 역기전력(BEMF)을 추정하고 현재 각도를 계산함
    """
    def __init__(self, R, L, Ke, Ts, K_smo, G_lpf):
        # 모터의 물리적 특성치
        self.R = R  
        self.L = L  
        self.Ke = Ke
        self.Ts = Ts # 시스템이 계산을 수행하는 시간 간격
        
        # SMO의 성능을 결정하는 변수 (Tuning Parameters)
        self.K_smo = K_smo # 오차를 수정하는 강도. 클수록 빠르게 따라가지만 진동이 생김.
        self.G_lpf = G_lpf # 계산된 신호를 얼마나 부드럽게 깎을지 결정 (저역 통과 필터)
        
        # 내부 상태 변수 (이전 값을 기억해야 다음 값을 계산함)
        self.i_alpha_est = 0.0
        self.i_beta_est = 0.0
        self.e_alpha_filt = 0.0
        self.e_beta_filt = 0.0

    def estimate(self, v_alpha, v_beta, i_alpha, i_beta, omega_e):
        """
        [입력값 설명]
        v_alpha, v_beta: MCU가 모터에 준 전압 명령어
        i_alpha, i_beta: 실제 센서로 측정된 모터 전류
        omega_e: 현재 모터의 회전 속도 (각도 보상을 위해 필요)
        """

        # STEP 1: 전류 오차 계산
        # '내가 예상한 전류'와 '실제 측정된 전류'의 차이를 구함
        error_a = self.i_alpha_est - i_alpha
        error_b = self.i_beta_est - i_beta
        
        # STEP 2: 슬라이딩 제어 (오차 수정 신호 생성)
        # 오차가 양수면 (-)를, 음수면 (+)를 곱해 강제로 0으로 보냄 (Signum 함수)
        # 이 v_obs 신호 안에 우리가 원하는 '역기전력' 정보가 숨어 있음
        v_obs_a = self.K_smo * np.sign(error_a)
        v_obs_b = self.K_smo * np.sign(error_b)
        
        # STEP 3: 모델 업데이트 (내부 전류 추정치 갱신)
        # 옴의 법칙(V=IR)과 인덕턴스 수식을 사용하여 다음 주기의 전류를 미리 예측
        # di = (V - R*i - V_obs) * dt / L
        self.i_alpha_est += (self.Ts / self.L) * (v_alpha - self.R * self.i_alpha_est - v_obs_a)
        self.i_beta_est  += (self.Ts / self.L) * (v_beta  - self.R * self.i_beta_est  - v_obs_b)
        
        # STEP 4: 노이즈 제거 (Low Pass Filter)
        # v_obs는 지저분한 신호이므로 필터를 거쳐 깨끗한 역기전력(BEMF) 신호로 바꿈
        self.e_alpha_filt += self.G_lpf * (v_obs_a - self.e_alpha_filt) * self.Ts
        self.e_beta_filt  += self.G_lpf * (v_obs_b - self.e_beta_filt)  * self.Ts
        
        # STEP 5: 각도 추출 (Atan2)
        # 두 축의 역기전력 값을 좌표로 변환하여 로터가 현재 몇 도(Radian)에 있는지 계산
        raw_theta = np.arctan2(-self.e_alpha_filt, self.e_beta_filt)
        
        # STEP 6: 위상 지연 보상 (중요!)
        # 필터를 거치면 신호가 실제보다 뒤처짐. 속도(omega_e)에 맞춰 각도를 앞으로 당겨줌.
        # 보상값 = arctan(현재속도 / 필터계수)
        compensation = np.arctan2(omega_e, self.G_lpf)
        final_theta = np.mod(raw_theta + compensation, 2 * np.pi)
        
        return final_theta

# --- 시뮬레이션 실행 부분 ---
if __name__ == "__main__":
    # 1. 초기 설정 (고속 모터 예시: R=0.1, L=0.5mH)
    Ts = 1e-4 # 10kHz
    smo = SensorlessSMO(R=0.1, L=0.0005, Ke=0.015, Ts=Ts, K_smo=120.0, G_lpf=600.0)
    
    # 2. 가상의 모터 회전 데이터 생성 (3000 RPM)
    t = np.arange(0, 0.04, Ts)
    omega_e = (3000 * 2 * np.pi / 60) * 4 # 4극쌍 가정
    
    actual_theta_history = []
    estimated_theta_history = []
    
    print(">>> SMO 시뮬레이션 계산 중...")
    
    for time in t:
        actual_theta = np.mod(omega_e * time, 2 * np.pi)
        
        # 전압/전류 입력 모사 (실제 환경에선 센서값이 들어옴)
        v_a, v_b = 15 * np.cos(actual_theta), 15 * np.sin(actual_theta)
        i_a, i_b = 8 * np.cos(actual_theta - 0.2), 8 * np.sin(actual_theta - 0.2)
        
        # SMO로 각도 맞추기
        est_theta = smo.estimate(v_a, v_b, i_a, i_b, omega_e)
        
        actual_theta_history.append(actual_theta)
        estimated_theta_history.append(est_theta)

    # 3. 결과 시각화
    plt.figure(figsize=(10, 4))
    plt.plot(t, actual_theta_history, 'k--', label='Real Angle', alpha=0.5)
    plt.plot(t, estimated_theta_history, 'r', label='SMO Estimated')
    plt.title("SMO Sensorless Logic Verification")
    plt.legend()
    plt.grid(True)
    plt.savefig('smo_logic_result.png')
    print(">>> 시뮬레이션 완료. 'smo_logic_result.png' 파일을 확인하세요.")