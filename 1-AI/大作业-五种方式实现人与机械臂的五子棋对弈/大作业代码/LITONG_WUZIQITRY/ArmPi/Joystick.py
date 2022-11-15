#!/usr/bin/env python3
# encoding: utf-8
import os
import time
import json
import pygame
import requests

url = "http://127.0.0.1:9030/jsonrpc"
cmd = {
    "method":"SetBusServoPulse",
    "params": [],
    "jsonrpc": "2.0",
    "id": 0,
    }

step_width = 10
key_map = {"PSB_CROSS":2, "PSB_CIRCLE":1, "PSB_SQUARE":3, "PSB_TRIANGLE":0,
        "PSB_L1": 4, "PSB_R1":5, "PSB_L2":6, "PSB_R2":7,
        "PSB_SELECT":8, "PSB_START":9, "PSB_L3":10, "PSB_R3":11};
action_map = ["CROSS", "CIRCLE", "", "SQUARE", "TRIANGLE", "L1", "R1", "L2", "R2", "SELECT", "START", "", "L3", "R3"]

os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.display.init()
pygame.joystick.init()
if pygame.joystick.get_count() > 0:
    js=pygame.joystick.Joystick(0)
    js.init()
    jsName = js.get_name()
    print("Name of the joystick:", jsName)
    jsAxes=js.get_numaxes()
    print("Number of axif:",jsAxes)
    jsButtons=js.get_numbuttons()
    print("Number of buttons:", jsButtons);
    jsBall=js.get_numballs()
    print("Numbe of balls:", jsBall)
    jsHat= js.get_numhats()
    print("Number of hats:", jsHat)

connected = False
change = [500,500,136,931,795,500]
while True:
    if os.path.exists("/dev/input/js0") is True:
        if connected is False:
            jscount =  pygame.joystick.get_count()
            if jscount > 0:
                try:
                    js=pygame.joystick.Joystick(0)
                    js.init()
                    connected = True
                except Exception as e:
                    print(e)
            else:
                pygame.joystick.quit()
    else:
        if connected is True:
            connected = False
            js.quit()
            pygame.joystick.quit()
    if connected is True:
        pygame.event.pump()      
        try:
            if js.get_button(key_map["PSB_R1"]) :
                change[0] -= step_width
                change[0] = 0 if change[0] < 0 else change[0]
                cmd["params"] = [20, 1, 1, change[0]]
                r = requests.post(url, json = cmd).json()
            if js.get_button(key_map["PSB_L1"])  :
                change[0] += step_width
                change[0] = 1000 if change[0] > 1000 else change[0]
                cmd["params"] = [20, 1, 1, change[0]]
                r = requests.post(url, json = cmd).json()
            if js.get_button(key_map["PSB_SQUARE"]) :
                change[1] -= step_width
                change[1] = 0 if change[1] < 0 else change[1]
                cmd["params"] = [20, 1, 2, change[1]]
                r = requests.post(url, json = cmd).json()
            if js.get_button(key_map["PSB_CIRCLE"]) :
                change[1] += step_width
                change[1] = 1000 if change[1] > 1000 else change[1]
                cmd["params"] = [20, 1, 2, change[1]]
                r = requests.post(url, json = cmd).json()
            if js.get_button(key_map["PSB_R2"]) :
                change[2] += step_width
                change[2] = 1000 if change[2] > 1000 else change[2]
                cmd["params"] = [20, 1, 3, change[2]]
                r = requests.post(url, json = cmd).json() 
            if js.get_button(key_map["PSB_L2"]) :
                change[2] -= step_width
                change[2] = 0 if change[2] < 0 else change[2]
                cmd["params"] = [20, 1, 3, change[2]]
                r = requests.post(url, json = cmd).json() 
            if js.get_button(key_map["PSB_TRIANGLE"]) :
                change[3] += step_width
                change[3] = 1000 if change[3] > 1000 else change[3]
                cmd["params"] = [20, 1, 4, change[3]]
                r = requests.post(url, json = cmd).json()
            if js.get_button(key_map["PSB_CROSS"]) :
                change[3] -= step_width
                change[3] = 0 if change[3] < 0 else change[3]
                cmd["params"] = [20, 1, 4, change[3]]
                r = requests.post(url, json = cmd).json()
            hat = js.get_hat(0)
            if hat[0] > 0 :
                change[5] -= step_width
                change[5] = 0 if change[5] < 0 else change[5]
                cmd["params"] = [20, 1, 6, change[5]]
                r = requests.post(url, json = cmd).json()
            elif hat[0] < 0:
                change[5] += step_width
                change[5] = 1000 if change[5] > 1000 else change[5]
                cmd["params"] = [20, 1, 6, change[5]]
                r = requests.post(url, json = cmd).json()
            if hat[1] > 0 :
                change[4] -= step_width
                change[4] = 0 if change[4] < 0 else change[4]
                cmd["params"] = [20, 1, 5, change[4]]
                r = requests.post(url, json = cmd).json()
            elif hat[1] < 0:
                change[4] += step_width
                change[4] = 0 if change[4] > 1000 else change[4]
                cmd["params"] = [20, 1, 5, change[4]]
                r = requests.post(url, json = cmd).json()

            lx = js.get_axis(0)
            ly = js.get_axis(1)
            rx = js.get_axis(2)
            ry = js.get_axis(3)
            if lx < -0.5 :
                change[5] += step_width
                change[5] = 1000 if change[5] > 1000 else change[5]
                cmd["params"] = [20, 1, 6, change[5]]
                r = requests.post(url, json = cmd).json()
            elif lx > 0.5:             
                change[5] -= step_width
                change[5] = 0 if change[5] < 0 else change[5]
                cmd["params"] = [20, 1, 6, change[5]]
                r = requests.post(url, json = cmd).json()

            l3_state = js.get_button(key_map["PSB_L3"])
            if ly < -0.5 :
                if not l3_state:
                    change[4] -= step_width
                    change[4] = 0 if change[4] < 0 else change[4]
                    cmd["params"] = [20, 1, 5, change[4]]
                    r = requests.post(url, json = cmd).json()
                else:
                    change[3] += step_width
                    change[3] = 1000 if change[3] > 1000 else change[3]
                    cmd["params"] = [20, 1, 4, change[3]]
                    r = requests.post(url, json = cmd).json()
            elif ly > 0.5:
                if not l3_state:
                    change[4] += step_width
                    change[4] = 1000 if change[4] > 1000 else change[4]
                    cmd["params"] = [20, 1, 5, change[4]]
                    r = requests.post(url, json = cmd).json()
                else:
                    change[3] -= step_width
                    change[3] = 0 if change[3] < 0 else change[3]
                    cmd["params"] = [20, 1, 4, change[3]]
                    r = requests.post(url, json = cmd).json()
            if rx > 0.5 :
                change[1] += step_width
                change[1] = 1000 if change[1] > 1000 else change[1]
                cmd["params"] = [20, 1, 2, change[1]]
                r = requests.post(url, json = cmd).json()
            elif rx < -0.5:
                change[1] -= step_width
                change[1] = 0 if change[1] < 0 else change[1]
                cmd["params"] = [20, 1, 2, change[1]]
                r = requests.post(url, json = cmd).json()
            if ry > 0.5 :
                change[3] -= step_width
                change[3] = 0 if change[3] < 0 else change[3]
                cmd["params"] = [20, 1, 4, change[3]]
                r = requests.post(url, json = cmd).json() 
            elif ry < -0.5:
                change[3] += step_width
                change[3] = 1000 if change[3] > 1000 else change[3]
                cmd["params"] = [20, 1, 4, change[3]]
                r = requests.post(url, json = cmd).json()                  
            if js.get_button(key_map["PSB_START"]):
                change = [500,500,136,931,795,500]
                cmd["params"] =  [1000, 6, 1, 500, 2, 500, 3, 136, 4, 931, 5, 795, 6, 500]
                r = requests.post(url, json = cmd).json()                
        except Exception as e:
            print(e)
            connected = False          
    time.sleep(0.06)
