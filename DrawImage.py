import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from models_method import DatabaseManager, EventManager
from encrypt import encrypt
import ast

def image(Country: int, Activity: int, Rank: int):
    ID = encrypt(f"{Activity}" + f"{Country}" + f"{Rank}")
    info = EventManager().get_info_by_event_name(ID)
    data_list = info.PointRank
    data = ast.literal_eval(data_list)
    # 2. 数据处理
    # 注意：你的时间戳是13位（毫秒级），需要除以1000转为秒级
    x_times = [datetime.fromtimestamp(item['time'] / 1000) for item in data]
    y_eps = [item['ep'] for item in data]

    # 3. 设置图表大小和风格
    plt.figure(figsize=(12, 6))  # 设置画布宽12，高6
    plt.style.use('seaborn-v0_8-darkgrid') # 使用网格风格，让图表更清晰 (可选)

    # 4. 绘制连线图
    # marker='o' 表示在数据点位置画圆点，linestyle='-' 表示实线连接
    plt.plot(x_times, y_eps, marker='o', linestyle='-', color='#1f77b4', label='EP Trend')

    # 5. 格式化 X 轴（时间轴）
    # 设置时间显示格式为 "月-日 时:分" (例如: 12-26 18:00)
    # 如果需要包含年份，可以改为 '%Y-%m-%d %H:%M'
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

    # 自动旋转日期标签，防止重叠
    plt.gcf().autofmt_xdate()

    # 6. 添加标题和标签
    plt.title(f"{Activity} - {Country} - {Rank}", fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('EP Value', fontsize=12)

    # 显示具体数值（可选：在每个点上方显示数字）
    # for a, b in zip(x_times, y_eps):
    #     plt.text(a, b, str(b), ha='center', va='bottom', fontsize=8)

    # 7. 显示图表
    plt.tight_layout() # 自动调整布局，防止标签被切掉
    plt.savefig(f'image/{Activity}-{Country}-{Rank}.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # for i in range(1,293):
    #     image(Country=3, Activity=i, Rank=2000)
    image(Country=3, Activity=298, Rank=1000)