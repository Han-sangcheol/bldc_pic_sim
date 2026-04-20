#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// =================================================================
// 1. 시스템 주기 및 상수 정의
// =================================================================
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define DT              0.000025f       // 25us (40kHz 제어 루프)
#define RPM_TO_RADS     (2.0f * M_PI / 60.0f)
#define RADS_TO_RPM     (60.0f / (2.0f * M_PI))
#define V_DC_LINK       36.0f           // 36V 시스템

// 매크로 유틸리티 (실제 펌웨어에서 많이 사용하는 클램핑 함수)
#define LIMIT(val, min, max) ((val) < (min) ? (min) : ((val) > (max) ? (max) : (val)))

// =================================================================
// 2. 수학 및 하드웨어 모사 유틸리티
// =================================================================

// Box-Muller 변환을 이용한 정규분포(가우시안) 노이즈 생성 함수
float randn(float mu, float sigma) {
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;
    if (u1 <= 1e-7f) u1 = 1e-7f; // log(0) 에러 방지
    
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    return mu + sigma * z0;
}

// 저역통과필터(1차 IIR) 구조체
typedef struct {
    float alpha;
    float out_prev;
} LowPassFilter;

void LPF_Init(LowPassFilter* lpf, float cutoff_freq_hz, float dt) {
    float rc = 1.0f / (2.0f * M_PI * cutoff_freq_hz);
    lpf->alpha = dt / (rc + dt);
    lpf->out_prev = 0.0f;
}

float LPF_Update(LowPassFilter* lpf, float input_val) {
    float out = lpf->alpha * input_val + (1.0f - lpf->alpha) * lpf->out_prev;
    lpf->out_prev = out;
    return out;
}

// ADC 양자화 및 노이즈 모사 함수
float ADC_Read(float true_value, int resolution_bits, float max_range, float noise_std) {
    float noisy_value = true_value + randn(0.0f, noise_std);
    noisy_value = LIMIT(noisy_value, -max_range, max_range); // 물리적 포화
    
    float levels = (float)(1 << resolution_bits); // 2^12 = 4096
    float half_levels = levels / 2.0f;
    float quantized = roundf((noisy_value / max_range) * half_levels);
    
    return (quantized / half_levels) * max_range;
}

// =================================================================
// 3. PID 제어기 구조체 (Conditional Integration Anti-Windup)
// =================================================================
typedef struct {
    float kp, ki, dt;
    float out_min, out_max;
    float integral;
} PIDController;

void PID_Init(PIDController* pid, float kp, float ki, float dt, float out_min, float out_max) {
    pid->kp = kp;
    pid->ki = ki;
    pid->dt = dt;
    pid->out_min = out_min;
    pid->out_max = out_max;
    pid->integral = 0.0f;
}

float PID_Compute(PIDController* pid, float setpoint, float measurement) {
    float error = setpoint - measurement;
    float p_out = pid->kp * error;
    float i_step = pid->ki * error * pid->dt;
    float out = p_out + pid->integral + i_step;

    // 실무형 Anti-Windup: 포화 시 에러가 돌아오는 방향일 때만 적분기 업데이트
    if (out > pid->out_max) {
        out = pid->out_max;
        if (error < 0.0f) pid->integral += i_step;
    } else if (out < pid->out_min) {
        out = pid->out_min;
        if (error > 0.0f) pid->integral += i_step;
    } else {
        pid->integral += i_step;
    }
    return out;
}

// =================================================================
// 4. 센서리스 속도 관측기 (BEMF Observer) 구조체
// =================================================================
typedef struct {
    float R, L, FluxLinkage, P, dt;
    LowPassFilter iq_lpf;
    LowPassFilter bemf_lpf;
    float prev_iq_filtered;
} BEMFObserver;

void Observer_Init(BEMFObserver* obs, float R, float L, float FluxLinkage, float P, float dt) {
    obs->R = R;
    obs->L = L;
    obs->FluxLinkage = FluxLinkage;
    obs->P = P;
    obs->dt = dt;
    
    LPF_Init(&obs->iq_lpf, 500.0f, dt);
    LPF_Init(&obs->bemf_lpf, 150.0f, dt);
    obs->prev_iq_filtered = 0.0f;
}

float Observer_Estimate_RPM(BEMFObserver* obs, float prev_vq, float iq_measured) {
    float iq_filt = LPF_Update(&obs->iq_lpf, iq_measured);
    float diq_dt_approx = (iq_filt - obs->prev_iq_filtered) / obs->dt;
    obs->prev_iq_filtered = iq_filt;
    
    // 역기전력 = 전압 - 저항강하 - 인덕터전압
    float raw_bemf = prev_vq - (obs->R * iq_filt) - (obs->L * diq_dt_approx);
    float filtered_bemf = LPF_Update(&obs->bemf_lpf, raw_bemf);
    
    float est_omega_e = filtered_bemf / obs->FluxLinkage;
    float est_omega_m = est_omega_e / obs->P;
    
    return est_omega_m * RADS_TO_RPM;
}

// =================================================================
// 5. 물리 엔진 상태 관리자 (실제 펌웨어엔 없고 시뮬레이션에만 존재)
// =================================================================
typedef struct {
    float R, L, FluxLinkage, J, B, P;
    float true_iq, true_omega_m;
} HighSpeedMotor;

void Motor_Init(HighSpeedMotor* m) {
    m->R = 0.38f;
    m->L = 0.12e-3f;
    m->FluxLinkage = 0.0042f;
    m->J = 2.0e-6f;                 // 초소형 마이크로 모터 관성
    m->B = 4.5e-7f;                 // 무부하 상태 베어링/공기 마찰
    m->P = 1.0f;
    m->true_iq = 0.0f;
    m->true_omega_m = 0.0f;
}

void Motor_Update(HighSpeedMotor* m, float vq_inverter, float load_torque, float dt) {
    float omega_e = m->true_omega_m * m->P;
    
    // 전기 방정식: di/dt = (Vq - R*i - BEMF) / L
    float diq_dt = (vq_inverter - m->R * m->true_iq - omega_e * m->FluxLinkage) / m->L;
    
    // 기계 방정식: 토크 = 1.5 * P * Flux * iq
    float te = 1.5f * m->P * m->FluxLinkage * m->true_iq;
    float dw_dt = (te - load_torque - m->B * m->true_omega_m) / m->J;
    
    // 오일러 적분
    m->true_iq += diq_dt * dt;
    m->true_omega_m += dw_dt * dt;
}

// =================================================================
// 6. 메인 통합 시뮬레이션 루프
// =================================================================
int main() {
    // 랜덤 시드 초기화
    srand((unsigned int)time(NULL));

    // 객체(구조체) 선언 및 초기화
    HighSpeedMotor motor;
    Motor_Init(&motor);
    
    BEMFObserver observer;
    Observer_Init(&observer, motor.R, motor.L, motor.FluxLinkage, motor.P, DT);
    
    PIDController speed_ctrl, current_ctrl;
    PID_Init(&speed_ctrl, 0.008f, 0.1f, DT, -10.0f, 10.0f);        // 전류 지령 최대 +-10A 제한
    PID_Init(&current_ctrl, 0.8f, 150.0f, DT, -V_DC_LINK, V_DC_LINK);

    // 제어 변수 초기화
    float vq_cmd = 0.0f, prev_vq_cmd = 0.0f;
    float cmd_rpm = 0.0f;
    float target_rpm = 40000.0f;
    
    // S-Curve 프로파일 변수
    float accel_step = 0.0f;
    float max_accel_step = 1.0f;
    float jerk_step = 0.0005f;

    float total_time = 2.0f;
    int total_steps = (int)(total_time / DT);
    int print_interval = (int)(0.1f / DT); // 100ms마다 출력

    printf("================================================================================================\n");
    printf("| Time[s]  |  Cmd RPM  |  Act RPM  |  Est RPM  | Iq(ADC)[A] | Vq(PWM)[V] |   Load[Nm]   |\n");
    printf("------------------------------------------------------------------------------------------------\n");

    for (int step = 0; step < total_steps; step++) {
        float t = step * DT;
        float act_rpm = motor.true_omega_m * RADS_TO_RPM;
        
        // -------------------------------------------------------------
        // [동적 가변 부하 생성] 1.2초 이후 외란 인가
        // -------------------------------------------------------------
        float load_torque = 0.0f;
        if (t > 1.2f) {
            float base_load = 0.035f;
            float low_freq_var = 0.008f * sinf(2.0f * M_PI * 5.0f * t);
            float high_freq_var = 0.003f * sinf(2.0f * M_PI * 50.0f * t);
            float white_noise = randn(0.0f, 0.002f);
            
            load_torque = base_load + low_freq_var + high_freq_var + white_noise;
            if (load_torque < 0.0f) load_torque = 0.0f; // 부하 역방향 방지
        }
        
        // -------------------------------------------------------------
        // 제어 루프 (펌웨어 영역)
        // -------------------------------------------------------------
        // 1. 센서 값 읽기 (12-bit ADC, Max 15A, Noise 0.03A 모사)
        float sensed_iq = ADC_Read(motor.true_iq, 12, 15.0f, 0.03f);
        
        // 2. 관측기로 RPM 추정 (위상 동기화를 위해 이전 Vq 사용)
        float est_rpm = Observer_Estimate_RPM(&observer, prev_vq_cmd, sensed_iq);
        prev_vq_cmd = vq_cmd;
        
        // 3. S-Curve 속도 램프 생성
        if (cmd_rpm < target_rpm) {
            float remaining_rpm = target_rpm - cmd_rpm;
            float stopping_dist = (accel_step * accel_step) / (2.0f * jerk_step) + (accel_step * 2.0f);
            
            if (remaining_rpm <= stopping_dist) {
                accel_step = fmaxf(0.0001f, accel_step - jerk_step); // 감속
            } else {
                accel_step = fminf(max_accel_step, accel_step + jerk_step); // 가속
            }
            cmd_rpm = fminf(target_rpm, cmd_rpm + accel_step);
        } else if (cmd_rpm > target_rpm) {
            cmd_rpm = fmaxf(target_rpm, cmd_rpm - 10.0f);
        }

        // 4. 이중 루프 PID 연산
        float iq_ref = PID_Compute(&speed_ctrl, cmd_rpm, est_rpm);
        vq_cmd = PID_Compute(&current_ctrl, iq_ref, sensed_iq);
        
        // -------------------------------------------------------------
        // 모터 물리엔진 업데이트 (실제 세계 영역)
        // -------------------------------------------------------------
        Motor_Update(&motor, vq_cmd, load_torque, DT);
        
        // -------------------------------------------------------------
        // 터미널 시리얼 모니터링 (100ms 간격 출력)
        // -------------------------------------------------------------
        if (step % print_interval == 0 || step == total_steps - 1) {
            const char* load_marker = (load_torque > 0.01f) ? "(!)" : "   ";
            printf("| %8.3f | %9.1f | %9.1f | %9.1f | %10.3f | %10.2f | %8.3f %3s |\n",
                   t, cmd_rpm, act_rpm, est_rpm, sensed_iq, vq_cmd, load_torque, load_marker);
        }
    }

    printf("================================================================================================\n");
    printf("Simulation Complete.\n");

    return 0;
}


/*
# 컴파일 (수학 라이브러리 연동을 위해 -lm 옵션 필수)
gcc -O2 motor_sim.c -o motor_sim -lm

# 실행
./motor_sim
*/