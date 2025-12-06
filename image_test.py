# !/user/bin/env python
# -*- coding: UTF-8

"""
PROJECT_NAME: BanG-Dream-Activity-fraction-prediction
PRODUCT_NAME: PyCharm
FILE_NAME: image_test
CREAT_USER: Tano26
CREAT_DATE: 2025/9/24
CREAT_TIME: 00:27
"""

from API.GetEventTracker import main
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def image(Activity):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 您的数据
    data = main(Country = 3,Activity = Activity ,Rank = 2000)

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 将时间戳转换为datetime对象
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')

    # 创建图表
    plt.figure(figsize=(15, 8))

    # 绘制折线图
    plt.plot(df['datetime'], df['ep'], linewidth=2, color='blue', alpha=0.7)

    # 添加标题和标签
    plt.title(f'活动{Activity} 时间与EP值关系图', fontsize=16, fontweight='bold')
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('EP值', fontsize=12)

    # 格式化x轴日期显示
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()  # 自动旋转日期标签

    # 添加网格
    plt.grid(True, alpha=0.3)

    # 显示图表
    plt.tight_layout()
    plt.show()


    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    # 第一个子图：完整趋势
    ax1.plot(df['datetime'], df['ep'], linewidth=2, color='red')
    ax1.set_title(f'活动{Activity} EP值随时间变化趋势', fontsize=14, fontweight='bold')
    ax1.set_ylabel('EP值', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # 第二个子图：EP值变化率（每日增量）
    df['date'] = df['datetime'].dt.date
    daily_ep = df.groupby('date')['ep'].max().diff().fillna(0)

    ax2.bar(daily_ep.index, daily_ep.values, alpha=0.7, color='green')
    ax2.set_title(f'活动{Activity} 每日EP值增量', fontsize=14, fontweight='bold')
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_ylabel('日增量', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 创建散点图查看分布
    plt.figure(figsize=(15, 8))
    plt.scatter(df['datetime'], df['ep'], alpha=0.5, color='purple', s=10)
    plt.title(f'活动{Activity} EP值分布散点图', fontsize=16, fontweight='bold')
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('EP值', fontsize=12)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 打印统计信息
    print("数据统计信息:")
    print(f"总数据点数: {len(df)}")
    print(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    print(f"EP值范围: {df['ep'].min():,} 到 {df['ep'].max():,}")
    print(f"EP值平均值: {df['ep'].mean():,.2f}")
    print(f"EP值中位数: {df['ep'].median():,}")

    # 计算总增长
    total_growth = df['ep'].max() - df['ep'].min()
    print(f"总增长: {total_growth:,}")

if __name__ == '__main__':
    for i in range(270,271):
        image(i)