import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as patches

# ==========================================
# 1. T-N 곡선 및 LUT(족보) 기본 데이터 생성
# ==========================================
# [왼쪽 그래프용 데이터]
speeds_line = np.linspace(0, 200, 400)
max_torques_line = np.where(speeds_line <= 100, 100.0, 10000.0 / speeds_line)

# [오른쪽 족보(표)용 해상도 데이터]
# 엑셀 표의 행(Row)과 열(Column) 기준값 설정
lut_speeds = np.array([0, 40, 80, 120, 160, 200])   # 열 (Speed)
lut_torques = np.array([100, 80, 60, 40, 20, 0])    # 행 (Torque) - 내림차순 정렬

# ==========================================
# 2. 메인 대시보드 창 분할 (1x2 레이아웃)
# ==========================================
fig = plt.figure(figsize=(14, 7))
fig.canvas.manager.set_window_title('Motor Control Dashboard (T-N & LUT)')
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.25, wspace=0.3)

ax_tn = plt.subplot(1, 2, 1)  # 왼쪽: T-N 곡선
ax_lut = plt.subplot(1, 2, 2) # 오른쪽: 족보(LUT) 표

# ==========================================
# 3. [왼쪽 패널] T-N 곡선 그리기
# ==========================================
ax_tn.plot(speeds_line, max_torques_line, 'k-', linewidth=3, label='Max Torque Limit')
ax_tn.fill_between(speeds_line, 0, max_torques_line, color='lightblue', alpha=0.3)

# 현재 작동 포인트
current_point, = ax_tn.plot([], [], 'o', markersize=15, color='green', markeredgecolor='black', zorder=5)

ax_tn.set_xlim(-10, 210)
ax_tn.set_ylim(-10, 110)
ax_tn.set_xlabel('Motor Speed (%)', fontsize=12)
ax_tn.set_ylabel('Torque Demand (%)', fontsize=12)
ax_tn.set_title('[1] T-N Curve (Capability Map)', fontsize=14, fontweight='bold')
ax_tn.grid(True, linestyle='--', alpha=0.7)

# 상태 텍스트
status_text = ax_tn.text(0.05, 0.92, '', transform=ax_tn.transAxes, fontsize=12, fontweight='bold')

# ==========================================
# 4. [오른쪽 패널] 족보(LUT) 표 그리기
# ==========================================
ax_lut.set_xlim(0, len(lut_speeds))
ax_lut.set_ylim(0, len(lut_torques))
ax_lut.invert_yaxis() # y축을 뒤집어서 0행이 맨 위에 오도록 설정

# 족보 내부의 데이터(각도) 계산 및 텍스트 채우기
for row, t in enumerate(lut_torques):
    for col, s in enumerate(lut_speeds):
        # 해당 칸의 능력 한계 계산
        limit = 100.0 if s <= 100 else 10000.0 / max(s, 1)
        
        if t > limit + 1: # 여유 마진 1%
            val_str = "N/A"
            text_color = 'red'
        elif s <= 100:
            val_str = "0.0°"
            text_color = 'black'
        else:
            # 약계자 각도 계산 로직
            angle = -60.0 * ((s - 100) / 100.0) * (t / limit)
            val_str = f"{angle:.1f}°"
            text_color = 'blue'
            
        # 각 셀의 정중앙에 텍스트 배치
        ax_lut.text(col + 0.5, row + 0.5, val_str, ha='center', va='center', color=text_color, fontsize=11, fontweight='bold')

# 표의 눈금선(Grid) 및 축 설정
ax_lut.set_xticks(np.arange(len(lut_speeds)) + 0.5)
ax_lut.set_xticklabels([f"Spd\n{s}%" for s in lut_speeds])
ax_lut.xaxis.tick_top() # 속도 축을 위로

ax_lut.set_yticks(np.arange(len(lut_torques)) + 0.5)
ax_lut.set_yticklabels([f"Trq\n{t}%" for t in lut_torques])

for x in range(len(lut_speeds) + 1):
    ax_lut.axvline(x, color='gray', lw=1)
for y in range(len(lut_torques) + 1):
    ax_lut.axhline(y, color='gray', lw=1)

ax_lut.set_title('[2] Look-Up Table (Phase Advance Angle)', fontsize=14, fontweight='bold', pad=40)

# 핵심! 제어기가 찾아가는 곳을 표시할 "노란색 하이라이트 박스" 생성
highlight_box = patches.Rectangle((0, 0), 1, 1, fill=True, color='yellow', alpha=0.4, lw=3, edgecolor='orange')
ax_lut.add_patch(highlight_box)

# 하단에 추출된 족보 값 표시
lut_result_text = ax_lut.text(0.5, -0.1, '', transform=ax_lut.transAxes, ha='center', fontsize=14, fontweight='bold', color='purple')

# ==========================================
# 5. 하단 슬라이더 UI (운전자 조작부)
# ==========================================
axcolor = 'lightgoldenrodyellow'
ax_speed = plt.axes([0.15, 0.10, 0.7, 0.04], facecolor=axcolor)
ax_torque = plt.axes([0.15, 0.04, 0.7, 0.04], facecolor=axcolor)

s_speed = Slider(ax_speed, 'Accel (Speed)', 0.0, 200.0, valinit=50.0, valfmt='%0.1f %%')
s_torque = Slider(ax_torque, 'Load (Torque)', 0.0, 100.0, valinit=80.0, valfmt='%0.1f %%')

# ==========================================
# 6. 실시간 제어기 두뇌 로직 (Update)
# ==========================================
def update(val):
    spd = s_speed.val
    trq = s_torque.val
    
    # 1. 왼쪽 T-N 곡선 업데이트
    limit = 100.0 if spd <= 100 else 10000.0 / spd
    current_point.set_data([spd], [trq])
    
    if trq > limit:
        current_point.set_color('red')
        status_text.set_text('Status: LIMIT EXCEEDED')
        status_text.set_color('red')
    else:
        current_point.set_color('green')
        status_text.set_text('Status: NORMAL')
        status_text.set_color('green')
        
    # 2. 오른쪽 LUT(족보) 하이라이트 박스 이동 로직
    # 현재 슬라이더 값과 가장 가까운 족보의 열(속도)과 행(토크)을 찾아냄
    nearest_col = np.abs(lut_speeds - spd).argmin()
    nearest_row = np.abs(lut_torques - trq).argmin()
    
    # 노란색 박스를 해당 위치로 이동
    highlight_box.set_xy((nearest_col, nearest_row))
    
    # 3. 실제 제어기가 계산하는 값 출력
    if trq > limit:
        highlight_box.set_color('red') # 불가능 영역은 빨간색으로
        lut_result_text.set_text('Result: Cannot operate (Out of bounds)')
    else:
        highlight_box.set_color('yellow')
        if spd <= 100:
            angle = 0.0
            lut_result_text.set_text(f'Reading LUT... Applied Angle: {angle:.1f}° (Normal)')
        else:
            angle = -60.0 * ((spd - 100) / 100.0) * (trq / limit)
            lut_result_text.set_text(f'Reading LUT... Applied Angle: {angle:.1f}° (Flux Weakening!)')

    fig.canvas.draw_idle()

# 슬라이더에 업데이트 함수 연결
s_speed.on_changed(update)
s_torque.on_changed(update)

# 초기화
update(None)
plt.show()