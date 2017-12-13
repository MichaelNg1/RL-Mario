import pyscreeze
import time
from PIL import Image
'''
raw_input()input("Press Enter to continue...")
'''
if __name__ == "__main__":
    while True:
        location = pyscreeze.locateOnScreen('locate_window.png')
        print location
        if location:
            print location
            break

    x = location[0]
    y = location[1]
    width = location[2]
    height = location[3]
    height_compensation = 90
    y -= height_compensation
    height += height_compensation
    location_region = (x, y, width, height)
    time.sleep(1)

    for i in range(100):
        # screenshot every 0.5 second
        time.sleep(0.045)
        t1 = time.time()
        im = pyscreeze.screenshot(region=location_region)
        im_resize = im.resize((84, 84), Image.ANTIALIAS)
        im_gray = im_resize.convert('L')
        # im_gray.save(str(i)+'gray.jpg')
        t2 = time.time()
        print t2-t1
