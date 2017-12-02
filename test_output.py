import pykeyboard
import time
k = pykeyboard.PyKeyboard()
#some sample practice
'''
#time.sleep(2)
#k.press_keys(['a',k.left_key,'b'])
time.sleep(2)
k.press_key('a')
time.sleep(0.1)
k.release_key('a')
print 1
time.sleep(0.1)
print 2
k.press_key(k.right_key)
time.sleep(1)
k.release_key(k.right_key)
print 3
time.sleep(0.5)
print 4
k.press_key('a')
k.press_key(k.left_key)

#time.sleep(1)
time.sleep(0.1)
k.release_key('a')
time.sleep(0.1)
#k.release_key(k.left_key)

'''
'''
time.sleep(2)
a = ['a',k.left_key,'a',k.right_key,'a',k.left_key]
for i in a:
    #time.sleep(0.01)
    k.press_key(i)
    time.sleep(1)
    k.release_key(i)
'''
'''
ke = pykeyboard.PyKeyboardEvent
print ke.run()'''


#Demo for level 1
time.sleep(5)
print 1
k.press_key(k.right_key)
time.sleep(2.097)
k.press_key('a')
print 2
time.sleep(0.161)
k.release_key('a')
time.sleep(1.423)
k.press_key('a')
print 3
time.sleep(0.291)
k.release_key('a')
time.sleep(0.781)
k.press_key('a')
print 4
time.sleep(0.573)
k.release_key('a')
time.sleep(0.685)
k.press_key('a')
print 5
time.sleep(0.532)
k.release_key('a')
time.sleep(0.354)
k.press_key('a')
print 6
time.sleep(0.19)
k.release_key('a')
time.sleep(1.095)
k.press_key('a')
print 7
time.sleep(0.602)
k.release_key('a')
time.sleep(1.413)
k.press_key('a')
print 8
time.sleep(0.222)
k.release_key('a')
time.sleep(1.895)
k.press_key('a')
print 9
time.sleep(0.29)
k.release_key('a')
time.sleep(0.66)
k.press_key('a')
print 10
time.sleep(0.114)
k.release_key('a')
time.sleep(0.914)
k.press_key('a')
print 11
time.sleep(0.15)
k.release_key('a')
time.sleep(0.778)
k.press_key('a')
print 12
time.sleep(0.131)
k.release_key('a')
time.sleep(1.736)
k.press_key('a')
print 13
time.sleep(0.92)
k.release_key('a')
time.sleep(0.462)
k.press_key('a')
print 14
time.sleep(0.562)
k.release_key('a')
time.sleep(0.1)
k.press_key('a')
print 15
time.sleep(0.255)
k.release_key('a')
time.sleep(0.325)
k.press_key('a')
print 16
time.sleep(0.813)
k.release_key('a')
time.sleep(0.711)
k.press_key('a')
print 17
time.sleep(0.486)
k.release_key('a')
time.sleep(0.22)
k.press_key('a')
print 18
time.sleep(0.12)
k.release_key('a')
time.sleep(0.84)
k.press_key('a')
print 19
time.sleep(0.322)
k.release_key('a')
time.sleep(1.466)
k.press_key('a')
print 20
time.sleep(0.58)
k.release_key('a')
time.sleep(0.493)
k.press_key('a')
print 21
time.sleep(0.861)
k.release_key('a')
time.sleep(0.124)
k.press_key('a')
print 22
time.sleep(0.744)
k.release_key('a')
time.sleep(0.836)
k.press_key('a')
print 23
time.sleep(0.421)
k.release_key('a')
