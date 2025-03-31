import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import time
from bestdori.eventtracker import EventTracker
from bestdori.events import Event
import matplotlib

def event_time(event_id: int, server: int):
    event = Event(event_id)
    info = event.get_info()
    start_time = info["startAt"]
    end_time = info["endAt"]
    startAt = start_time[server]
    endAt = end_time[server]
    return startAt, endAt

def time_change(a1): #时间转换

    # 先转换为时间数组
    timeArray = time.strptime(a1, "%Y-%m-%d %H:%M:%S")

    # 转换为时间戳
    timeStamp = int(time.mktime(timeArray))

    return timeStamp

def main(Country: int, Activity: int, Rank: int):
    # 实例化 Post 类
    E = EventTracker(server=Country,event=Activity)
    # 调用方法获取信息
    info = E.get_data(tier=Rank)
    r = {}
    result1 = info["cutoffs"]s
    for i in result1:
        t = i["time"]
        ep = i["ep"]
        r[t] = ep
    return r

def predict_event_score(data_dict, start_time, end_time):
    # 转换时间戳为datetime对象
    start_dt = datetime.datetime.fromtimestamp(start_time / 1000)
    end_dt = datetime.datetime.fromtimestamp(end_time / 1000)

    # 处理输入数据
    timestamps = sorted(data_dict.keys())
    scores = [data_dict[t] for t in timestamps]

    # 计算相对于活动开始时间的小时数
    X_hours = []
    data_dates = []
    for t_ms in timestamps:
        dt = datetime.datetime.fromtimestamp(t_ms / 1000)
        delta = dt - start_dt
        hours = delta.total_seconds() / 3600
        X_hours.append(hours)
        data_dates.append(dt)

    # 转换为numpy数组
    X = np.array(X_hours)
    Y = np.array(scores)

    # 线性回归
    coeffs = np.polyfit(X, Y, 1)
    slope, intercept = coeffs
    regression_line = np.poly1d(coeffs)

    # 计算相关系数
    r, _ = pearsonr(X, Y)

    # 预测结束时间分数
    end_hours = (end_dt - start_dt).total_seconds() / 3600
    predicted_score = regression_line(end_hours)

    # 生成回归线数据
    regression_x = np.linspace(0, end_hours, 100)
    regression_dates = [start_dt + datetime.timedelta(hours=h) for h in regression_x]
    regression_y = regression_line(regression_x)

    # 创建图表
    plt.figure(figsize=(12, 6))

    # 绘制数据点和回归线
    plt.scatter(data_dates, Y, label='观测数据', zorder=3)
    plt.plot(regression_dates, regression_y, 'r-', label='线性回归')
    plt.scatter([end_dt], [predicted_score], c='g', marker='*', s=200,
                label=f'预测值: {predicted_score:.2f}', zorder=4)

    # 设置图表格式
    plt.title('活动分数线预测', fontsize=14)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('分数', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 设置日期格式
    date_form = mdates.DateFormatter('%m/%d %H:%M')
    plt.gca().xaxis.set_major_formatter(date_form)
    plt.gcf().autofmt_xdate()

    # 显示相关系数
    plt.text(0.05, 0.9, f'Pearson r = {r:.3f}',
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

    return predicted_score, r


# 示例用法
if __name__ == "__main__":
    matplotlib.rc("font", family='Microsoft YaHei')
    # 示例数据（时间戳单位：毫秒）
    Country = int(input("请输入要查询的服务器(0=jp,1=en,2=tw,3=cn,4=kr):"))
    Activity = int(input("请输入要查询的活动id:"))
    Rank = int(input("请输入要查询的数据线:"))
    data = main(Country,Activity,Rank)

    # 活动时间（示例）
    tm = event_time(Activity,Country)
    start_time = int(tm[0])
    end_time = int(tm[1])

    # 运行预测
    score, r_value = predict_event_score(data, start_time, end_time)

    print(f"预测结束分数线: {score:.2f}")
    print(f"相关系数: {r_value:.3f}")