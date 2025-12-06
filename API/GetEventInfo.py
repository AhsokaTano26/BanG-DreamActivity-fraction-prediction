# !/user/bin/env python
# -*- coding: UTF-8

"""
PROJECT_NAME: BanG-Dream-Activity-fraction-prediction
PRODUCT_NAME: PyCharm
FILE_NAME: GetEventInfo
CREAT_USER: Tano26
CREAT_DATE: 2025/9/24
CREAT_TIME: 01:50
"""

from bestdori.events import Event

def get_event_info(Activity: int, Country: int):
    event = Event(Activity)
    info = event.get_info()
    # print(info)
    start_time = info["startAt"]
    end_time = info["endAt"]
    startAt = start_time[Country]
    endAt = end_time[Country]
    return info,startAt, endAt

def get_event_info_async(Activity: int, Country: int):
    event = Event(Activity)
    info = event.get_info()
    # print(info)
    start_time = info["startAt"]
    end_time = info["endAt"]
    startAt = start_time[Country]
    endAt = end_time[Country]
    return info,startAt, endAt

if __name__ == '__main__':
    #main(270,3)
    #print(type(main(270,3)))
    a = get_event_info(270,3)
    print(a)
    print(type(a))