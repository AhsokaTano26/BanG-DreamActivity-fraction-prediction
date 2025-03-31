from bestdori.events import Event

def main(event_id: int, server: int):
    event = Event(event_id)
    info = event.get_info()
    start_time = info["startAt"]
    end_time = info["endAt"]
    startAt = start_time[server]
    endAt = end_time[server]
    return startAt, endAt

if __name__ == '__main__':
    #main(270,3)
    #print(type(main(270,3)))
    a = main(270,3)
    print(a)
    print(type(a))
    b = a[1]
    print(b)
    print(type(b))