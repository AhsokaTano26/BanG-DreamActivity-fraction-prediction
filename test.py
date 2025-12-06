# !/user/bin/env python
# -*- coding: UTF-8

"""
PROJECT_NAME: BanG-Dream-Activity-fraction-prediction
PRODUCT_NAME: PyCharm
FILE_NAME: test
CREAT_USER: Tano26
CREAT_DATE: 2025/9/24
CREAT_TIME: 00:24
"""

import pandas as pd
from API.GetEventTracker import main

Activity = 273
Rank = 2000
# 您的数据
data = main(Country = 3,Activity = Activity,Rank = Rank) # 这里应该是您完整的数据

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 将时间戳转换为可读的日期时间格式（毫秒级时间戳）
df['datetime'] = pd.to_datetime(df['time'], unit='ms')

# 重新排列列的顺序
df = df[['datetime', 'time', 'ep']]

# 设置显示选项，确保所有行都能显示
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# 打印表格
print("数据表格：")
print(df)

# 也可以保存到CSV文件
df.to_csv(f'time_ep_data({Activity}-{Rank}).csv', index=False)
print("\n数据已保存到 time_ep_data.csv 文件")

# 显示基本统计信息
print("\n基本统计信息：")
print(f"数据总行数: {len(df)}")
print(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
print(f"EP值范围: {df['ep'].min()} 到 {df['ep'].max()}")