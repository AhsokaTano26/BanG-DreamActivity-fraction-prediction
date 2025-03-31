import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from bestdori.eventtracker import EventTracker
import time
from bestdori.events import Event

def nonlinear_predict(data_dict, start_time, end_time, degree=2):
    # 时间转换
    start_dt = datetime.fromtimestamp(start_time / 1000)
    end_dt = datetime.fromtimestamp(end_time / 1000)

    # 处理数据
    timestamps = sorted(data_dict.keys())
    scores = [data_dict[t] for t in timestamps]

    # 计算时间差（小时）
    X_hours = np.array([(datetime.fromtimestamp(t / 1000) - start_dt).total_seconds() / 3600
                        for t in timestamps]).reshape(-1, 1)
    Y = np.array(scores)

    # 生成完整时间轴
    total_hours = (end_dt - start_dt).total_seconds() / 3600
    full_X = np.linspace(0, total_hours, 1000).reshape(-1, 1)

    # 多项式回归模型
    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )
    model.fit(X_hours, Y)

    # 预测值
    full_pred = model.predict(full_X)
    end_pred = model.predict([[total_hours]])[0]

    # 计算R²
    r2 = r2_score(Y, model.predict(X_hours))

    # 时间转换绘图数据
    plot_dates = [start_dt + timedelta(hours=h) for h in full_X.ravel()]
    obs_dates = [datetime.fromtimestamp(t / 1000) for t in timestamps]

    # 绘图
    plt.figure(figsize=(14, 7))

    # 绘制观测点
    plt.scatter(obs_dates, Y, c='#1f77b4', s=80, label='观测数据', zorder=5)

    # 绘制预测曲线
    plt.plot(plot_dates, full_pred, 'r-', lw=2,
             label=f'{degree}次多项式拟合 (R²={r2:.3f})')

    # 绘制预测终点
    plt.scatter([end_dt], [end_pred], c='g', marker='*', s=300,
                edgecolor='black', label=f'预测终值: {end_pred:.2f}', zorder=6)

    # 图表装饰
    plt.title(f'非线性回归预测 (多项式次数={degree})', fontsize=14)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('分数', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')

    # 日期格式化
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.gcf().autofmt_xdate()

    # 显示关键信息
    plt.legend(loc='upper left', fontsize=10)
    plt.text(0.02, 0.88, f'活动时长: {total_hours:.1f}小时',
             transform=ax.transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    return end_pred, r2


# 模型比较函数
def compare_models(data_dict, start_time, end_time, max_degree=4):
    timestamps = sorted(data_dict.keys())
    X = np.array([(datetime.fromtimestamp(t / 1000) -
                   datetime.fromtimestamp(start_time / 1000)).total_seconds() / 3600
                  for t in timestamps])
    Y = np.array([data_dict[t] for t in timestamps])

    plt.figure(figsize=(14, 8))

    # 绘制原始数据
    plt.scatter(X, Y, s=80, label='观测数据', zorder=5)

    # 生成预测范围
    x_pred = np.linspace(0, (datetime.fromtimestamp(end_time / 1000) -
                             datetime.fromtimestamp(start_time / 1000)).total_seconds() / 3600,
                         100)

    # 测试不同次数多项式
    colors = plt.cm.viridis(np.linspace(0, 1, max_degree))
    for degree in range(1, max_degree + 1):
        model = make_pipeline(
            PolynomialFeatures(degree=degree),
            LinearRegression()
        )
        model.fit(X.reshape(-1, 1), Y)
        y_pred = model.predict(x_pred.reshape(-1, 1))
        r2 = r2_score(Y, model.predict(X.reshape(-1, 1)))

        plt.plot(x_pred, y_pred, lw=2,
                 label=f'Degree {degree} (R²={r2:.3f})',
                 color=colors[degree - 1])

    plt.title('不同多项式次数回归比较', fontsize=14)
    plt.xlabel('时间 (小时)', fontsize=12)
    plt.ylabel('分数', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

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
    # 打印信息
    #print(info)
    #print(type(info))
    r = {}
    result1 = info["cutoffs"]
    #print(result1)
    #print(type(result1))
    for i in result1:
        t = i["time"]
        ep = i["ep"]
        r[t] = ep
    return r


# 示例使用
if __name__ == "__main__":
    Country = int(input("请输入要查询的服务器(0=jp,1=en,2=tw,3=cn,4=kr):"))
    Activity = int(input("请输入要查询的活动id:"))
    Rank = int(input("请输入要查询的数据线:"))
    #活动时间
    tm = event_time(Activity,Country)
    start_time = int(tm[0])
    end_time = int(tm[1])

    # 模拟非线性增长数据
    test_data = main(Country,Activity,Rank)

    # 绘制模型比较
    compare_models(test_data, start_time, end_time, max_degree=4)

    # 执行二次多项式预测
    final_score, r2 = nonlinear_predict(test_data, start_time, end_time, degree=2)
    print(f"预测最终分数: {final_score:.2f}")
    print(f"模型R²值: {r2:.3f}")