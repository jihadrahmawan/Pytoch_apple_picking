from asyncore import read
from socket import timeout
import serial
import time
import pydobot
import socket
import cv2
import keyboard

def motor_triger (device, state, var):
    device.motor(0,0,0)

pause = False
def onkeypress(event):
    global pause
    if event.name == 'n' or event.name == 'N':
        pause = True
    
start = False
def onkeypress_strat(event):
    global start
    if event.name == 'm' or event.name == 'M':
        start = True

if __name__ == '__main__':
    arduino = serial.Serial('COM4', 9600, timeout=1)
    device = pydobot.Dobot(port='COM8', verbose=False)
    wait_ready = 0
    motor_active = 0
    motor_disactive = 0
    pick_ready = True
    vision_message = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) 
    vision_message.bind(("127.0.0.1", 1111))
    vision_message.setblocking(False)
    proxy = 3
    detect = 0
    type_quality = 0
    data_vision=None
    #keyboard.on_press(onkeypress)
    #keyboard.on_press(onkeypress_strat)
    cnt = 0
    cnt1 = 0
    ready = 0
    entry = 0
    last_detect = 0
    array_entry = []
    data_qual = 0
    new_array = []
    conveyor = 0
    visiononline = False
    start_motor = 0
    start_motor1 = 0
    start_motor2 = 0 
    last_conveyor = 0
    conveyor_active = False
    try :
        #device.set_home(1)
        while(1):
            data = arduino.readline()
            try:
                data_vision, addr = vision_message.recvfrom(1024)
            except socket.error:
                pass
            
            try:
                proxy = int (data)
            except ValueError:
                #print ("Waiting Proxymity sensor ready")
                pass
            
            
            if data_vision is not None:
                visiononline = True
                data_vision.decode("utf-8")
                try:
                    detect = int (data_vision[0:4])
                    quality = int (data_vision[4:5])
                    conveyor = int (data_vision[5:6])
                except ValueError:
                    #print ("Vision Error")
                    pass
            else:

                print ("Pytorch is Offline, Waiting...")

            if detect != last_detect:
                array_entry.append(quality)

            if conveyor != last_conveyor:
                if conveyor == 1:
                    conveyor_active =True
                if conveyor == 0:
                    conveyor_active = False
            
            ready += 1
            if  visiononline:
                if conveyor_active:
                    start_motor1+=1
                    if start_motor1 == 2:
                        start_motor2=0
                        device.motor(0, 1, 2000)
                    if start_motor1 > 5:
                        start_motor1 = 6
                else:
                    start_motor2+=1
                    if start_motor2 == 2:
                        start_motor1=0
                        device.motor(0, 0, 2000)
                    if start_motor2 > 5:
                        start_motor2 = 6


                if (proxy == 0 and pick_ready == True):
                    motor_disactive = 0
                    motor_active+=1
                    if (motor_active == 2):
                        if len(array_entry) != 0 :
                            type_quality = array_entry[0]
                            del array_entry[0]
                        if type_quality == 1 :
                            device.grip(False)
                            device.motor(0, 0, 2000)
                            device.speed(50,80)
                            device.move_to(243, -5, 135, 7.6, 1, False)
                                    
                            device.move_to(243, -5, 125, 7.6, 1, False)
                            device.grip(True)
                            device.wait(500)
                            device.move_to(243, -5, 135, 7.6, 1, False)
                            device.speed(200,180)

                                    #p1
                            device.move_to(-10, -240, 138,-83 , 0, False)
                            device.grip(False)
                            device.wait(500)
                            device.move_to(243, -5, 135, 7.6, 1, False)
                            motor_active = 3
                            pick_ready = False

                else:
                    motor_active = 0 
                    motor_disactive += 1
                    if motor_disactive == 1700:
                        pick_ready = True
                        device.suck(False)
                        device.motor(0, 1, 2000)
                        motor_disactive = 1800

                print ("Detect Entry = ", array_entry)
                print ("PICKING      = ", new_array)
                print ("conveyor     = ", conveyor)

           
            last_detect = detect
            last_conveyor = conveyor

    
    finally:
        device.motor(0, 0, 0)
        device.suck(False)
        arduino.close()
        device.close()
        vision_message.close()
