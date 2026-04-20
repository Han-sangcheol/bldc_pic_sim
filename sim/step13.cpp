#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <algorithm> // std::clamp, std::max, std::min 사용
#include <string>

// =================================================================
// 1. 시스템 주기 및 상수 정의 (constexpr로 컴파일 타임 최적화)
// =================================================================
constexpr float PI = 3.14159265358979323846f;
constexpr float DT = 0.000025f;                  // 25us (40kHz 제어 루프)
constexpr float RPM_TO_RADS = (2.0f * PI / 60.0f);
constexpr float RADS_TO_RPM = (60.0f / (2.0f * PI));
constexpr float V_DC_LINK = 36.0f;               // 36V 구동 시스템

// =================================================================
// 2. 하드웨어 모사 유틸리티 (디지털 필터 및 센서 노이즈)
// =================================================================
class LowPassFilter {
private:
    float alpha;
    float out_prev;

public:
    // 생성자를 통한 초기화
    LowPassFilter(float cutoff_freq_hz, float dt) : out_prev(0.0f) {
        float rc = 1.0f / (2.0f * PI * cutoff_freq_hz);
        alpha = dt / (rc + dt);
    }

    float update(float input_val) {
        out_prev = alpha * input_val + (1.0f - alpha) * out_prev;
        return out_prev;
    }
};

class ADCSimulator {
private:
    int resolution_bits;
    float max_range;
    std::mt19937 gen;                           // 메르센 트위스터 난수 생성기 엔진
    std::normal_distribution<float> noise_dist; // 가우시안 정규 분포 모사기

public:
    ADCSimulator(int bits, float max_r, float noise_std) 
        : resolution_bits(bits), max_range(max_r), 
          gen(std::random_device{}()), noise_dist(0.0f, noise_std) {}

    float read(float true_value) {
        // 1. 센서 백색 잡음(White Noise) 인가
        float noisy_value = true_value + noise_dist(gen);
        
        // 2. 물리적 하드웨어 한계 클램핑 (OP-AMP 포화)
        noisy_value = std::clamp(noisy_value, -max_range, max_range);
        
        // 3. ADC 양자화 (Quantization, 예: 12-bit)
        float levels = static_cast<float>(1 << resolution_bits);
        float half_levels = levels / 2.0f;
        float quantized = std::round((noisy_value / max_range) * half_levels);
        
        return (quantized / half_levels) * max_range;
    }
};

// =================================================================
// 3. 고성능 PID 제어기 (Conditional Integration Anti-Windup)
// =================================================================
class PIDController {
private:
    float kp, ki, dt;
    float out_min, out_max;
    float integral;

public:
    PIDController(float kp, float ki, float dt, float out_min, float out_max)
        : kp(kp), ki(ki), dt(dt), out_min(out_min), out_max(out_max), integral(0.0f) {}

    float compute(float setpoint, float measurement) {
        float error = setpoint - measurement;
        float p_out = kp * error;
        float i_step = ki * error * dt;
        float out = p_out + integral + i_step;

        // 실무형 Anti-Windup: 출력이 포화되었을 때 에러가 반대 방향으로 들어올 때만 적분기 누적 허용
        if (out > out_max) {
            out = out_max;
            if (error < 0.0f) integral += i_step;
        } else if (out < out_min) {
            out = out_min;
            if (error > 0.0f) integral += i_step;
        } else {
            integral += i_step;
        }
        
        return out;
    }
};

// =================================================================
// 4. 센서리스 속도 관측기 (BEMF Observer)
// =================================================================
class BEMFObserver {
private:
    float R, L, FluxLinkage, P, dt;
    LowPassFilter iq_lpf;
    LowPassFilter bemf_lpf;
    float prev_iq_filtered;

public:
    // C++ 초기화 리스트를 통한 필터 객체 동시 초기화
    BEMFObserver(float R, float L, float FluxLinkage, float P, float dt)
        : R(R), L(L), FluxLinkage(FluxLinkage), P(P), dt(dt),
          iq_lpf(500.0f, dt), bemf_lpf(150.0f, dt), prev_iq_filtered(0.0f) {}

    float estimate_rpm(float prev_vq, float iq_measured) {
        float iq_filt = iq_lpf.update(iq_measured);
        float diq_dt_approx = (iq_filt - prev_iq_filtered) / dt;
        prev_iq_filtered = iq_filt;
        
        // 역기전력 = 입력전압 - 저항강하 - 인덕터전압
        float raw_bemf = prev_vq - (R * iq_filt) - (L * diq_dt_approx);
        float filtered_bemf = bemf_lpf.update(raw_bemf);
        
        float est_omega_e = filtered_bemf / FluxLinkage; // 전기적 각속도
        float est_omega_m = est_omega_e / P;             // 기계적 각속도
        
        return est_omega_m * RADS_TO_RPM;
    }
};

// =================================================================
// 5. 물리 엔진 상태 관리자 (Plant - High Speed Micro Motor)
// =================================================================
class HighSpeedMotor {
public: // 관측기 설정을 위해 공개되는 모터 파라미터
    float R = 0.38f;
    float L = 0.12e-3f;
    float FluxLinkage = 0.0042f;
    float P = 1.0f;
    
private:
    // 무부하 4만RPM에서 0.3A를 소모하는 초소형 모터에 맞춘 관성 및 마찰
    float J = 2.0e-6f;                 // 초소형 로터 관성 [kg*m^2]
    float B = 4.5e-7f;                 // 베어링 및 공기 마찰 [Nm*s/rad]
    
    // 모터의 실제 상태 (은닉됨, 센서를 통해서만 밖으로 나감)
    float true_iq = 0.0f;
    float true_omega_m = 0.0f;

public:
    // 데이터 로깅을 위한 Getter 메서드
    float get_true_iq() const { return true_iq; }
    float get_true_omega_m() const { return true_omega_m; }

    // 물리 방정식 업데이트 (1-step Euler Integration)
    void update(float vq_inverter, float load_torque, float dt) {
        float omega_e = true_omega_m * P;
        
        // 전기적 동역학 (dq축 q상 전압 방정식)
        float diq_dt = (vq_inverter - R * true_iq - omega_e * FluxLinkage) / L;
        
        // 기계적 동역학 (전자기 토크 - 부하 - 마찰)
        float te = 1.5f * P * FluxLinkage * true_iq;
        float dw_dt = (te - load_torque - B * true_omega_m) / J;
        
        true_iq += diq_dt * dt;
        true_omega_m += dw_dt * dt;
    }
};

// =================================================================
// 6. 메인 통합 시뮬레이션 루프
// =================================================================
int main() {
    // 1. 객체 인스턴스화
    HighSpeedMotor motor;
    BEMFObserver observer(motor.R, motor.L, motor.FluxLinkage, motor.P, DT);
    ADCSimulator adc_iq(12, 15.0f, 0.03f); // 15A 스케일, 0.03A 노이즈
    
    // 이중 루프 PI 제어기 튜닝
    PIDController speed_ctrl(0.008f, 0.1f, DT, -10.0f, 10.0f);        // 속도 지령 최대 출력: 10A
    PIDController current_ctrl(0.8f, 150.0f, DT, -V_DC_LINK, V_DC_LINK); // 전류 지령 최대 출력: 36V

    // 2. 상태 변수 초기화
    float vq_cmd = 0.0f, prev_vq_cmd = 0.0f;
    float cmd_rpm = 0.0f;
    float target_rpm = 40000.0f;
    
    // S-Curve 프로파일 튜닝 파라미터
    float accel_step = 0.0f;
    float max_accel_step = 1.0f;
    float jerk_step = 0.0005f;

    float total_time = 2.0f;
    int total_steps = static_cast<int>(total_time / DT);
    int print_interval = static_cast<int>(0.1f / DT); // 100ms 간격 출력

    // 터미널 UI 헤더 출력 포맷 세팅
    std::cout << "================================================================================================\n";
    std::cout << "| Time[s]  |  Cmd RPM  |  Act RPM  |  Est RPM  | Iq(ADC)[A] | Vq(PWM)[V] |   Load[Nm]   |\n";
    std::cout << "------------------------------------------------------------------------------------------------\n";
    std::cout << std::fixed; 

    // 외부 부하(외란) 생성을 위한 별도 난수 생성기
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> white_noise_dist(0.0f, 0.002f);

    for (int step = 0; step < total_steps; step++) {
        float t = step * DT;
        float act_rpm = motor.get_true_omega_m() * RADS_TO_RPM;
        
        // -------------------------------------------------------------
        // [기능 1] 동적 가변 부하 (외란) 인가 (1.2초 시점)
        // -------------------------------------------------------------
        float load_torque = 0.0f;
        if (t > 1.2f) {
            float base_load = 0.035f;                                     // 약 6~7A를 유발하는 기본 절삭 부하
            float low_freq = 0.008f * std::sin(2.0f * PI * 5.0f * t);     // 5Hz 작업자 손 압력 변동
            float high_freq = 0.003f * std::sin(2.0f * PI * 50.0f * t);   // 50Hz 드릴 기계적 진동
            float noise = white_noise_dist(gen);                          // 재질 밀도 변화에 따른 잡음
            
            load_torque = base_load + low_freq + high_freq + noise;
            load_torque = std::max(0.0f, load_torque); // 부하가 음수(모터를 밀어주는 방향)가 되지 않도록 방지
        }
        
        // -------------------------------------------------------------
        // [기능 2] 펌웨어 제어 루프 (센서 -> 관측기 -> 제어기)
        // -------------------------------------------------------------
        float sensed_iq = adc_iq.read(motor.get_true_iq());
        
        // 관측기에 이전 스텝의 전압 지령을 넣어주어 위상을 완벽히 맞춤
        float est_rpm = observer.estimate_rpm(prev_vq_cmd, sensed_iq);
        prev_vq_cmd = vq_cmd; 
        
        // S-Curve 오버슈트 방지 알고리즘
        if (cmd_rpm < target_rpm) {
            float remaining_rpm = target_rpm - cmd_rpm;
            float stopping_dist = (accel_step * accel_step) / (2.0f * jerk_step) + (accel_step * 2.0f);
            
            if (remaining_rpm <= stopping_dist) {
                accel_step = std::max(0.0001f, accel_step - jerk_step); // 브레이크
            } else {
                accel_step = std::min(max_accel_step, accel_step + jerk_step); // 가속
            }
            cmd_rpm = std::min(target_rpm, cmd_rpm + accel_step);
        } else if (cmd_rpm > target_rpm) {
            cmd_rpm = std::max(target_rpm, cmd_rpm - 10.0f);
        }

        // PID 연산 (속도 오차로 전류 지령 생성 -> 전류 오차로 전압 지령 생성)
        float iq_ref = speed_ctrl.compute(cmd_rpm, est_rpm);
        vq_cmd = current_ctrl.compute(iq_ref, sensed_iq);
        
        // -------------------------------------------------------------
        // [기능 3] 물리 엔진 업데이트 (현실 세계 시뮬레이션)
        // -------------------------------------------------------------
        motor.update(vq_cmd, load_torque, DT);
        
        // -------------------------------------------------------------
        // [기능 4] C++ 스타일 터미널 모니터링 (100ms 간격 출력)
        // -------------------------------------------------------------
        if (step % print_interval == 0 || step == total_steps - 1) {
            std::string marker = (load_torque > 0.01f) ? "(!)" : "   ";
            
            // iomanip을 이용한 깔끔한 표 정렬 출력
            std::cout << "| " 
                      << std::setw(8) << std::setprecision(3) << t << " | "
                      << std::setw(9) << std::setprecision(1) << cmd_rpm << " | "
                      << std::setw(9) << std::setprecision(1) << act_rpm << " | "
                      << std::setw(9) << std::setprecision(1) << est_rpm << " | "
                      << std::setw(10) << std::setprecision(3) << sensed_iq << " | "
                      << std::setw(10) << std::setprecision(2) << vq_cmd << " | "
                      << std::setw(8) << std::setprecision(3) << load_torque << " " 
                      << marker << " |\n";
        }
    }

    std::cout << "================================================================================================\n";
    std::cout << "Simulation Complete.\n";

    return 0;
}

/*
# -std=c++17 : C++17 표준 문법 사용
# -O2 : 최고 수준의 속도 최적화 적용
g++ -std=c++17 -O2 motor_sim.cpp -o motor_sim

# 윈도우(PowerShell) 환경 실행:
.\motor_sim.exe

# 리눅스/맥 환경 실행:
./motor_sim
*/