from models_method import DatabaseManager, EventManager
from model import Event
from API.GetEventTracker import event_tracker
from API.GetEventInfo import get_event_info
from encrypt import encrypt

Country_list = ["日本","国际","中国台湾","中国大陆","韩国"]

def get_info(Country: int, Activity: int, Rank: int):
    Point = str(event_tracker(Country = Country,Activity = Activity,Rank = Rank))
    info, startAt, endAt = get_event_info(Activity = Activity,Country = Country)
    ID = encrypt(f"{Activity}" + f"{Country}" + f"{Rank}")
    if EventManager().get_info_by_event_name(ID):
        print(f"活动{Activity} - {Country} - {Rank}已存在，进行下一个")
    else:
        EventManager().create_new_event(
            ID=ID,
            EventID=Activity,
            EventName=info['eventName'][Country],
            EventType=info['eventType'],
            StartAt=startAt,
            EndAt=endAt,
            PointRank=Point,
            Rank=Rank,
            Country=Country_list[Country]
        )
        print(f"活动{Activity} - {Country} - {Rank}已成功写入数据库")

def main(event_id: int, Country_id: int ,Rank: int):
    for eid in range(1,event_id):
        print(f"正在处理活动{eid} - {Country_id} - {Rank}")
        try:
            get_info(Country = Country_id, Activity = eid, Rank = Rank)
        except Exception as e:
            print(f"处理活动{eid} - {Country_id} - {Rank} 发生异常：{e}")

if __name__ == '__main__':
    main(event_id = 293, Country_id = 3, Rank = 1000)
    # get_info(Country=3, Activity=1, Rank=2000)