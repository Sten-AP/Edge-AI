from periphery import GPIO
import time

led_rood = GPIO("/dev/gpiochip2", 13, "out")  # pin 37
# led_groen = GPIO("/dev/gpiochip2", 13, "out")  # pin 37
# led_geel = GPIO("/dev/gpiochip2", 13, "out")  # pin 37

try:
  while True:
    led_rood.write(True)
    time.sleep(1)
    led_rood.write(False)
    time.sleep(1)
finally:
  led_rood.write(False)
  led_rood.close()
