import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from bestdori.eventtracker import EventTracker
from bestdori.events import Event

def predict_event_cutoff(input_data, start_ms, end_ms):
    # 转换时间戳为小时单位
    start_h = start_ms / 3600_000
    end_h = end_ms / 3600_000
    total_duration = end_h - start_h  # 总活动时长（小时）

    # 处理输入数据
    times = []
    scores = []
    for ts in sorted(input_data.keys()):
        t = (ts - start_ms) / 3600_000  # 转换为小时单位
        times.append(t)
        scores.append(input_data[ts])
    times = np.array(times)
    scores = np.array(scores)

    # 分阶段处理
    def get_stage(t):
        if t <= 24: return 0
        if t <= 120: return 1
        return 2

    # 计算各阶段速度系数
    stage_coeffs = []
    for stage in range(3):
        stage_mask = [get_stage(t) == stage for t in times]
        stage_times = times[stage_mask]
        stage_scores = scores[stage_mask]

        if len(stage_times) < 1:
            # 默认系数（根据经验值设置）
            if stage == 0:
                coeff = 1.2
            elif stage == 1:
                coeff = 0.9
            else:
                coeff = 1.5
        else:
            # 计算实际速度与平均速度的比值
            elapsed_times = stage_times
            velocities = stage_scores / elapsed_times
            avg_velocity = np.mean(velocities)
            coeff = avg_velocity / (stage_scores[-1] / total_duration)

        stage_coeffs.append(coeff)

    # 使用加权平均调整系数
    stage_durations = [24, 96, 8]
    weighted_sum = sum(c * d for c, d in zip(stage_coeffs, stage_durations))
    adjustment_factor = total_duration / weighted_sum
    stage_coeffs = [c * adjustment_factor for c in stage_coeffs]

    # 预测最终分数
    predicted_total = sum(c * d for c, d in zip(stage_coeffs, stage_durations)) * (scores[-1] / times[-1])

    # 生成预测曲线
    current_t = times[-1]
    prediction_times = np.linspace(current_t, total_duration, 100)
    prediction_scores = []

    for t in prediction_times:
        accumulated = 0
        for i, (coeff, duration) in enumerate(zip(stage_coeffs, stage_durations)):
            start = sum(stage_durations[:i])
            end = start + duration
            if t < start: continue

            effective_t = min(t, end) - start
            if i == 0: effective_t = max(0, effective_t)
            accumulated += coeff * effective_t

        prediction_scores.append(accumulated * (predicted_total / total_duration))

    # 绘制图表
    plt.figure(figsize=(12, 6))
    plt.plot(prediction_times, prediction_scores, label='Predicted Trend', color='orange')
    plt.scatter(times, scores, color='blue', label='Actual Data')

    # 添加阶段标记
    for phase in [24, 120]:
        plt.axvline(phase, color='gray', linestyle='--', alpha=0.5)

    plt.title('Event Score Prediction')
    plt.xlabel('Hours since event start')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    # 转换最终时间为日期格式
    end_datetime = datetime.fromtimestamp(end_ms / 1000)
    print(f"Predicted final cutoff at {end_datetime}: {int(predicted_total):,}")

    plt.show()
    return predicted_total

def event_time(event_id: int, server: int):
    event = Event(event_id)
    info = event.get_info()
    start_time = info["startAt"]
    end_time = info["endAt"]
    startAt = start_time[server]
    endAt = end_time[server]
    return startAt, endAt

def main(Country: int, Activity: int, Rank: int):
    # 实例化 Post 类
    E = EventTracker(server=Country,event=Activity)
    # 调用方法获取信息
    info = E.get_data(tier=Rank)
    r = {}
    result1 = info["cutoffs"]
    for i in result1:
        t = i["time"]
        ep = i["ep"]
        r[t] = ep
    return r


# 示例用法
if __name__ == "__main__":
    Country = int(input("请输入要查询的服务器(0=jp,1=en,2=tw,3=cn,4=kr):"))
    Activity = int(input("请输入要查询的活动id:"))
    Rank = int(input("请输入要查询的数据线:"))
    # 示例数据（时间戳单位为毫秒）
    tm = event_time(Activity,Country)
    event_start = int(tm[0])
    event_end = int(tm[1])

    sample_data = main(Country,Activity,Rank)

    predict_event_cutoff(sample_data, event_start, event_end)