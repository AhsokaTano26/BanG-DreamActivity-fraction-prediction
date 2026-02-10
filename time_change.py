import time


def time_change(a1): #时间转换

    # 先转换为时间数组
    timeArray = time.strptime(a1, "%Y-%m-%d %H:%M:%S")

    # 转换为时间戳
    timeStamp = int(time.mktime(timeArray))

    return timeStamp
