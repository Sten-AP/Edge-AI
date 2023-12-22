from periphery import GPIO

led_groen = GPIO("/dev/gpiochip0", 8, "out")  # pin 31
led_geel = GPIO("/dev/gpiochip4", 13, "out")  # pin 36
led_rood = GPIO("/dev/gpiochip2", 13, "out")  # pin 37

def toggle_led(prediction):
    # Groene LED aan-/uitzetten
    if prediction == 'boom':
        led_groen.write(True)     
    if prediction == 'gras':
        led_groen.write(False)

    # Gele LED aan-/uitzetten
    if prediction == 'kaas':
        led_geel.write(True)     
    if prediction == 'zon':
        led_geel.write(False)
            
    # Rode LED aan-/uitzetten
    if prediction == 'aardbei':
        led_rood.write(True) 
    if prediction == 'kers':
        led_rood.write(False)     
        