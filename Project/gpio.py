from periphery import GPIO
from time import sleep

led_rood = GPIO("/dev/gpiochip2", 13, "out")  # pin 37
led_groen = GPIO("/dev/gpiochip2", 13, "out")  # pin 37
led_geel = GPIO("/dev/gpiochip2", 13, "out")  # pin 37

def toggle_led(prediction):
    if prediction == 'aardbei':
        led_rood.write(True) 
    if prediction == 'kers':
        led_rood.write(False)     

    if prediction == 'boom':
        led_groen.write(True)     
    if prediction == 'gras':
        led_groen.write(False)

    if prediction == 'kaas':
        led_geel.write(True)     
    if prediction == 'zon':
        led_geel.write(False)
    
    sleep(1)
