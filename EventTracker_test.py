from bestdori.eventtracker import EventTracker

def main() -> None:
    Country = str(input("请输入要查询的服务器(0=jp;1=en;2=tw;3=cn;4=kr):"))
    Rank = int(input("请输入要查询的数据线:"))
    Activity = int(input("请输入要查询的活动id:"))
    # 实例化 Post 类
    E = EventTracker(server=Country,event=Activity)
    # 调用方法获取信息
    info = E.get_data(tier=Rank)
    # 打印信息
    print(info)
    print(type(info))

main()