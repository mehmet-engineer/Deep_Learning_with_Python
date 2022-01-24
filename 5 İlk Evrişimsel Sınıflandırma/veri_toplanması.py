#pip install mss
#pip install keyboard

import time
import keyboard
import uuid
from PIL import Image  #python image library
from mss import mss    #multiple screen shots

"""
http://www.trex-game.skipser.com/
"""

# ekran görüntüsü alacağımız bölge;
# siteyi açınca F11 ile tam ekran yaptıktan sonra ss al
# sol üst köşenin koordinatları x=722,y=294
bolge = {"top":294, "left":722, "width":250, "height":140}

screen = mss()

i = 0  #toplam basılan tuş sayacı

def record_screen(id,key):
    global i
    i = i + 1
    print("{}: {}".format(key,i))
    img = screen.grab(bolge)
    im = Image.frombytes("RGB",img.size,img.rgb)
    if key == "up":
        im.save("veriseti\\up\\{}_{}_{}.png".format(key,id,i))
    if key == "down":
        im.save("veriseti\\down\\{}_{}_{}.png".format(key,id,i))
    if key == "right":
        im.save("veriseti\\right\\{}_{}_{}.png".format(key,id,i))

is_exit = False

def exit():
    global is_exit
    is_exit = True

keyboard.add_hotkey("esc",exit)

record_id = uuid.uuid4()

while True:
    if is_exit == True:
        print("çıkış yapıldı.")
        break

    try:
        if keyboard.is_pressed(keyboard.KEY_UP) == True:
            record_screen(record_id,"up")
            time.sleep(0.1)
        elif keyboard.is_pressed(keyboard.KEY_DOWN) == True:
            record_screen(record_id,"down")
            time.sleep(0.1)
        elif keyboard.is_pressed("right") == True:
            record_screen(record_id,"right")
            time.sleep(0.1)
        else:
            pass

    except RuntimeError:
        continue