#include "kickback_model.h" // 방금 파이썬에서 생성한 헤더 파일!

// TFLM 필수 헤더들
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// 1. 메모리 아레나 할당 (AI 연산용 RAM 공간, 모델 크기에 따라 조절 필요)
constexpr int kTensorArenaSize = 10 * 1024; // 10KB
uint8_t tensor_arena[kTensorArenaSize];

// 2. TFLite 전역 객체 선언
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

void setup() {
    // 3. 배열에서 모델 읽어오기
    model = tflite::GetModel(kickback_model); // 바로 여기서 배열 이름을 사용합니다!
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        // 모델 버전 에러 처리
        return;
    }

    // 4. 연산자(Op) 리졸버 및 인터프리터 초기화
    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // 5. 메모리 할당
    interpreter->AllocateTensors();

    // 6. 입출력 텐서(데이터 통로) 포인터 연결
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
}

void loop() {
    // 7. IMU 데이터 수집 및 버퍼링 (예: 최근 10개의 샘플)
    // float imu_buffer[10][6];  <-- 실제로는 센서 데이터를 10ms 단위로 모음
    
    // 8. 모인 데이터를 AI 입력 텐서에 복사
    int input_index = 0;
    for(int i = 0; i < 10; i++) {
        input_tensor->data.f[input_index++] = imu_buffer[i][0]; // Accel X
        input_tensor->data.f[input_index++] = imu_buffer[i][1]; // Accel Y
        input_tensor->data.f[input_index++] = imu_buffer[i][2]; // Accel Z
        input_tensor->data.f[input_index++] = imu_buffer[i][3]; // Gyro R
        input_tensor->data.f[input_index++] = imu_buffer[i][4]; // Gyro P
        input_tensor->data.f[input_index++] = imu_buffer[i][5]; // Gyro Y
    }

    // 9. AI 추론 실행 (Inference)
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        // 에러 처리
        return;
    }

    // 10. 결과 확인
    // 출력층의 활성화 함수를 Sigmoid로 했으므로, 0~1 사이의 확률값이 나옵니다.
    float kickback_probability = output_tensor->data.f[0];

    if (kickback_probability > 0.85f) { // 85% 이상 킥백이라 확신하면
        // !! 긴급 상황 !!
        // 하드웨어 모스펫(MOSFET) 차단 신호 발생!
        TriggerEmergencyBrake();
    }
}