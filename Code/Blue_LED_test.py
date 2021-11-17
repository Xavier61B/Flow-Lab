from board import LED
from machine import Pin, PWM, Timer
from time import sleep

#led = Pin(LED, mode=Pin.OUT)
led_1 = Pin(27, mode=Pin.OUT)
led_2 = Pin(33, mode=Pin.OUT)

b1 = 0
b2 = 100

L1 = PWM(led_1,freq=200,duty=b1,timer=0)
L2 = PWM(led_2,freq=200,duty=b2,timer=1)

def tcb(timer):
    global b1
    global b2
    
    if b1 < 100:
        b1 += 1
    else:
        b1 = 0

    if b2 > 0:
        b2 -= 1
    else:
        b2 = 100

    L1.duty(b1)
    L2.duty(b2)

t1 = Timer(2)
t1.init(period=50, mode=t1.PERIODIC, callback=tcb)