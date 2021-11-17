from board import LED
from machine import Pin, PWM, Timer
from time import sleep

#led = Pin(LED, mode=Pin.OUT)
led_1 = Pin(27, mode=Pin.OUT)
led_2 = Pin(33, mode=Pin.OUT)

led_1(1)
led_2(2)