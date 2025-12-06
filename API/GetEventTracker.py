from bestdori.eventtracker import EventTracker
import asyncio

async def event_tracker_async(Country: int, Activity: int, Rank: int) -> list:
    """
    异步从Bestdori获取活动时间戳-分数信息
    时间戳（time）：ms
    分数（ep）：ep
    """
    # 实例化 Post 类
    E = EventTracker(server=Country,event=Activity)
    # 调用方法获取信息
    info = E.get_data(tier=Rank)
    result = info["cutoffs"]
    return result

def event_tracker(Country: int, Activity: int, Rank: int) -> list:
    """
    同步从Bestdori获取活动时间戳-分数信息
    时间戳（time）：ms
    分数（ep）：ep
    """
    # 实例化 Post 类
    E = EventTracker(server=Country,event=Activity)
    # 调用方法获取信息
    info = E.get_data(tier=Rank)
    result = info["cutoffs"]
    return result


# a = asyncio.run(event_tracker_async(Country = 3,Activity = 270,Rank = 2000))
# print(a)
# print(type(a))

