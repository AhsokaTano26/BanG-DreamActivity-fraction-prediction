import time


def time_change(a1): #时间转换

    # 先转换为时间数组
    timeArray = time.strptime(a1, "%Y-%m-%d %H:%M:%S")

    # 转换为时间戳
    timeStamp = int(time.mktime(timeArray))

    return timeStamp

if __name__ == '__main__':
    a = time_change("2025-03-26 10:00:00")
    print(a)
    print(type(a))
