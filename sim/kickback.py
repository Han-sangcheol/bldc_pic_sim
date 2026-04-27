import tkinter as tk
import numpy as np
import time
import threading
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split

class KickbackSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("IMU 킥백 시뮬레이터 & AI 학습기")
        
        # 데이터 저장소
        self.normal_data = []
        self.kickback_data = []
        
        # 상태 변수
        self.is_recording = False
        self.current_label = 0 # 0: Normal, 1: Kickback
        self.last_x = 0
        self.last_y = 0
        self.last_time = 0
        self.last_vx = 0
        self.last_vy = 0

        self.setup_ui()

    def setup_ui(self):
        # 컨트롤 패널
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        self.btn_normal = tk.Button(control_frame, text="정상 데이터 기록 (누른 채 마우스 천천히 이동)", width=35, bg="lightgreen")
        self.btn_normal.bind("<ButtonPress-1>", lambda e: self.start_recording(0))
        self.btn_normal.bind("<ButtonRelease-1>", self.stop_recording)
        self.btn_normal.grid(row=0, column=0, padx=5)

        self.btn_kickback = tk.Button(control_frame, text="킥백 데이터 기록 (누른 채 마우스 급격히 꺾기)", width=35, bg="salmon")
        self.btn_kickback.bind("<ButtonPress-1>", lambda e: self.start_recording(1))
        self.btn_kickback.bind("<ButtonRelease-1>", self.stop_recording)
        self.btn_kickback.grid(row=0, column=1, padx=5)

        self.btn_train = tk.Button(control_frame, text="모델 학습 및 C 헤더 추출", command=self.train_and_export, bg="lightblue", font=('Arial', 10, 'bold'))
        self.btn_train.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")

        self.status_var = tk.StringVar()
        self.status_var.set("상태: 대기 중 (데이터를 수집해주세요)")
        tk.Label(self.root, textvariable=self.status_var, font=('Arial', 11)).pack(pady=5)

        # 마우스 입력 캔버스
        self.canvas = tk.Canvas(self.root, width=600, height=400, bg="black")
        self.canvas.pack(pady=10)
        self.canvas.bind("<B1-Motion>", self.record_motion)

    def start_recording(self, label):
        self.is_recording = True
        self.current_label = label
        self.last_time = time.time()
        self.status_var.set(f"상태: {'정상' if label == 0 else '킥백'} 데이터 수집 중...")

    def stop_recording(self, event):
        self.is_recording = False
        n_count = len(self.normal_data)
        k_count = len(self.kickback_data)
        self.status_var.set(f"상태: 수집 완료 (정상: {n_count}샘플, 킥백: {k_count}샘플)")

    def record_motion(self, event):
        if not self.is_recording: return

        current_time = time.time()
        dt = current_time - self.last_time
        if dt < 0.01: return # 약 100Hz 샘플링 제한

        # 마우스 위치 변화량으로 속도(Gyro) 및 가속도(Accel) 시뮬레이션
        dx = event.x - self.last_x
        dy = event.y - self.last_y
        
        vx = dx / dt
        vy = dy / dt
        
        ax = (vx - self.last_vx) / dt
        ay = (vy - self.last_vy) / dt

        # 가상의 6축 데이터 구성 (가속도 X,Y,Z / 자이로 R,P,Y)
        # Z축과 Roll은 노이즈만 추가
        imu_sample = [
            ax * 0.01 + np.random.normal(0, 0.1), # Accel X
            ay * 0.01 + np.random.normal(0, 0.1), # Accel Y
            9.8 + np.random.normal(0, 0.1),       # Accel Z (중력)
            np.random.normal(0, 1.0),             # Gyro Roll
            vx * 0.1 + np.random.normal(0, 1.0),  # Gyro Pitch
            vy * 0.1 + np.random.normal(0, 1.0)   # Gyro Yaw
        ]

        if self.current_label == 0:
            self.normal_data.append(imu_sample)
        else:
            self.kickback_data.append(imu_sample)

        # 캔버스에 시각적 피드백 (선 그리기)
        color = "green" if self.current_label == 0 else "red"
        self.canvas.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, fill=color, outline=color)

        self.last_x, self.last_y = event.x, event.y
        self.last_vx, self.last_vy = vx, vy
        self.last_time = current_time

    def train_and_export(self):
        if len(self.normal_data) < 100 or len(self.kickback_data) < 100:
            self.status_var.set("오류: 정상과 킥백 데이터가 각각 최소 100샘플 이상 필요합니다.")
            return
            
        self.status_var.set("학습 진행 중... 터미널 창을 확인하세요.")
        # UI 멈춤 방지를 위해 스레드로 실행
        threading.Thread(target=self._run_training).start()

    def _run_training(self):
        WINDOW_SIZE = 10 # 10개 샘플(약 100ms) 묶음
        
        def create_windows(data, label):
            X, y = [], []
            for i in range(len(data) - WINDOW_SIZE):
                X.append(data[i:i+WINDOW_SIZE])
                y.append(label)
            return X, y

        X_norm, y_norm = create_windows(self.normal_data, 0)
        X_kick, y_kick = create_windows(self.kickback_data, 1)

        X = np.array(X_norm + X_kick)
        y = np.array(y_norm + y_kick)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

        # 모델 구성
        model = Sequential([
            Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(WINDOW_SIZE, 6)),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # 모델 학습
        model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)
        
        # TFLite 변환
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # C 헤더 배열로 변환
        hex_array = [format(val, '#04x') for val in tflite_model]
        c_str = "#ifndef KICKBACK_MODEL_H\n#define KICKBACK_MODEL_H\n\n"
        c_str += f"// AI Model Size: {len(hex_array)} bytes\n"
        c_str += "const unsigned char kickback_model[] = {\n\t"
        for i, val in enumerate(hex_array):
            c_str += f"{val}, "
            if (i + 1) % 12 == 0: c_str += "\n\t"
        c_str += "\n};\n\n#endif // KICKBACK_MODEL_H"
        
        with open("kickback_model.h", "w") as f:
            f.write(c_str)
            
        self.status_var.set("완료! kickback_model.h 파일이 생성되었습니다.")

if __name__ == "__main__":
    root = tk.Tk()
    app = KickbackSimulatorGUI(root)
    root.mainloop()