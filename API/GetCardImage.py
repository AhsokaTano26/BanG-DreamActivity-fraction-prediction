from bestdori.cards import Card
from PIL import Image
from io import BytesIO

def main(event_id: int,type,flag):
    event = Card(event_id)
    if flag == "Y":
        info = event.get_trim(type)
    else:
        info = event.get_card(type)
    return info

def card_info(event_id: int):
    event = Card(event_id)
    info = event.get_info()
    return info

if __name__ == '__main__':
    flag = input("是否为无背景图片？Y/N:")
    id = int(input("请输入卡片ID："))
    c = input("是否特训卡？Y/N:")
    if c == "Y":
        a = main(id,"after_training",flag)
        image = Image.open(BytesIO(a))
        image.show()
        image.save(f'{id}_特训后.png')
    else:
        a = main(id,"normal",flag)
        image = Image.open(BytesIO(a))
        image.show()
        image.save(f'{id}_特训前.png')

