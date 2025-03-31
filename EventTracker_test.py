from bestdori.eventtracker import EventTracker

def main(Country: int, Activity: int, Rank: int):
    #Country,Activity,Rank = int(input("请输入要查询的服务器(0=jp,1=en,2=tw,3=cn,4=kr):")),int(input("请输入要查询的活动id:")),int(input("请输入要查询的数据线:"))
    # 实例化 Post 类
    E = EventTracker(server=Country,event=Activity)
    # 调用方法获取信息
    info = E.get_data(tier=Rank)
    # 打印信息
    #print(info)
    #print(type(info))
    r = {}
    result1 = info["cutoffs"]
    print(result1)
    #print(type(result1))
    for i in result1:
        t = i["time"]
        ep = i["ep"]
        r[t] = ep
    return r



a = main(Country = 3,Activity = 270,Rank = 2000)
#a = main(Country = int(input("请输入要查询的服务器(0=jp,1=en,2=tw,3=cn,4=kr):")),Activity = int(input("请输入要查询的活动id:")),Rank = int(input("请输入要查询的数据线:")))
print(a)
print(type(a))
