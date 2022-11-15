#!/usr/bin/python3
# coding=utf8
import os
import sys
sys.path.append('/home/pi/ArmPi/')
import time
import logging
import threading
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple
from jsonrpc import JSONRPCResponseManager, dispatcher
from ArmIK.ArmMoveIK import *
import HiwonderSDK as hwsdk
import HiwonderSDK.Board as Board
import HiwonderSDK.ActionGroupControl as AGC
import Functions.Running as Running
import Functions.ColorTracking as ColorTrack
import Functions.ColorSorting as ColorSort
import Functions.ColorPalletizing as ColorPalletiz
from Functions.ASRControl import *

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)

__RPC_E01 = "E01 - Invalid number of parameter!"
__RPC_E02 = "E02 - Invalid parameter!"
__RPC_E03 = "E03 - Operation failed!"
__RPC_E04 = "E04 - Operation timeout!"
__RPC_E05 = "E05 - Not callable"

HWSONAR = None
QUEUE = None

initMove()

@dispatcher.add_method
def SetPWMServo(*args, **kwargs):
    ret = (True, ())
    arglen = len(args)
    if 0 != (arglen % 3):
        return (False, __RPC_E01)
    try:
        servos = args[0:arglen:3]
        pulses = args[1:arglen:3]
        use_times = args[2:arglen:3]
        for s in servos:
            if s < 1 or s > 6:
                return (False, __RPC_E02)
        dat = zip(servos, pulses, use_times)
        for (s, p, t) in dat:
            Board.setPWMServoPulse(s, p, t)
    except Exception as e:
        print(e)
        ret = (False, __RPC_E03)
    return ret

@dispatcher.add_method
def SetBusServoPulse(*args, **kwargs):
    ret = (True, ())
    arglen = len(args)
    if (args[1] * 2 + 2) != arglen or arglen < 4:
        return (False, __RPC_E01)
    try:
        servos = args[2:arglen:2]
        pulses = args[3:arglen:2]
        use_times = args[0]
        for s in servos:
           if s < 1 or s > 6:
                return (False, __RPC_E02)
        dat = zip(servos, pulses)
        for (s, p) in dat:
            Board.setBusServoPulse(s, p, use_times)
    except Exception as e:
        print(e)
        ret = (False, __RPC_E03)
    #return ret

@dispatcher.add_method
def SetBusServoDeviation(*args):
    ret = (True, ())
    arglen = len(args)
    if arglen != 2:
        return (False, __RPC_E01)
    try:
        servo = args[0]
        deviation = args[1]
        Board.setBusServoDeviation(servo, deviation)
    except Exception as e:
        print(e)
        ret = (False, __RPC_E03)

@dispatcher.add_method
def GetBusServosDeviation(args):
    ret = (True, ())
    data = []
    if args != "readDeviation":
        return (False, __RPC_E01)
    try:
        for i in range(1, 7):
            dev = Board.getBusServoDeviation(i)
            if dev is None:
                dev = 999
            data.append(dev)
        ret = (True, data)
    except Exception as e:
        print(e)
        ret = (False, __RPC_E03)
    return ret 

@dispatcher.add_method
def SaveBusServosDeviation(args):
    ret = (True, ())
    if args != "downloadDeviation":
        return (False, __RPC_E01)
    try:
        for i in range(1, 7):
            dev = Board.saveBusServoDeviation(i)
    except Exception as e:
        print(e)
        ret = (False, __RPC_E03)
    return ret 

@dispatcher.add_method
def UnloadBusServo(args):
    ret = (True, ())
    if args != 'servoPowerDown':
        return (False, __RPC_E01)
    try:
        for i in range(1, 7):
            Board.unloadBusServo(i)
    except Exception as e:
        print(e)
        ret = (False, __RPC_E03)

@dispatcher.add_method
def GetBusServosPulse(args):
    ret = (True, ())
    data = []
    if args != 'angularReadback':
        return (False, __RPC_E01)
    try:
        for i in range(1, 7):
            pulse = Board.getBusServoPulse(i)
            if pulse is None:
                ret = (False, __RPC_E04)
                return ret
            else:
                data.append(pulse)
        ret = (True, data)
    except Exception as e:
        print(e)
        ret = (False, __RPC_E03)
    return ret 

@dispatcher.add_method
def StopBusServo(args):
    ret = (True, ())
    if args != 'stopAction':
        return (False, __RPC_E01)
    try:     
        AGC.stop_action_group()
    except Exception as e:
        print(e)
        ret = (False, __RPC_E03)

@dispatcher.add_method
def RunAction(args):
    ret = (True, ())
    if len(args) == 0:
        return (False, __RPC_E01)
    try:
        threading.Thread(target=AGC.runAction, args=(args, )).start()
    except Exception as e:
        print(e)
        ret = (False, __RPC_E03)
        
@dispatcher.add_method
def ArmMoveIk(*args):   
    ret = (True, ())
    if len(args) != 7:
        return (False, __RPC_E01)
    try:
        result = setPitchRangeMoving((args[0], args[1], args[2]), args[3], args[4], args[5], args[6])
        ret = (True, result)
    except Exception as e:
        print(e)
        ret = (False, __RPC_E03)
    return ret
        
@dispatcher.add_method
def SetBrushMotor(*args, **kwargs):
    ret = (True, ())
    arglen = len(args)
    if 0 != (arglen % 2):
        return (False, __RPC_E01)
    try:
        motors = args[0:arglen:2]
        speeds = args[1:arglen:2]
        for m in motors:
            if m < 1 or m > 4:
                return (False, __RPC_E02)
        dat = zip(motors, speeds)

        for m, s in dat:
            Board.setMotor(m, s)
    except:
        ret = (False, __RPC_E03)
    return ret

@dispatcher.add_method
def GetSonarDistance():
    global HWSONAR
    ret = (True, 0)
    try:
        ret = (True, HWSONAR.getDistance())
    except:
        ret = (False, __RPC_E03)
    return ret

@dispatcher.add_method
def GetBatteryVoltage():
    ret = (True, 0)
    try:
        ret = (True, Board.getBattery())
    except Exception as e:
        print(e)
        ret = (False, __RPC_E03)
    return ret

@dispatcher.add_method
def SetSonarRGBMode(mode = 0):
    global HWSONAR
    HWSONAR.setRGBMode(mode)
    return (True, (mode,))

@dispatcher.add_method
def SetSonarRGB(index, r, g, b):
    global HWSONAR
    if index == 0:
        HWSONAR.setRGB(1, (r, g, b))
        HWSONAR.setRGB(2, (r, g, b))
    else:
        HWSONAR.setRGB(index, (r, g, b))
    return (True, (r, g, b))

@dispatcher.add_method
def SetSonarRGBBreathCycle(index, color, cycle):
    global HWSONAR
    HWSONAR.setBreathCycle(index, color, cycle)
    return (True, (index, color, cycle))

@dispatcher.add_method
def SetSonarRGBStartSymphony():
    global HWSONAR
    HWSONAR.startSymphony()
    return (True, ())

def runbymainth(req, pas):
    if callable(req):
        event = threading.Event()
        ret = [event, pas, None]
        QUEUE.put((req, ret))
        count = 0
        #ret[2] =  req(pas)
        #print('ret', ret)
        while ret[2] is None:
            time.sleep(0.01)
            count += 1
            if count > 200:
                break
        if ret[2] is not None:
            if ret[2][0]:
                return ret[2]
            else:
                return (False, __RPC_E03 + " " + ret[2][1])
        else:
            return (False, __RPC_E04)
    else:
        return (False, __RPC_E05)

@dispatcher.add_method
def SetSonarDistanceThreshold(new_threshold = 30): 
    return runbymainth(Avoidance.setThreshold, (new_threshold,))

@dispatcher.add_method
def GetSonarDistanceThreshold():
    return runbymainth(Avoidance.getThreshold, ())

@dispatcher.add_method
def LoadFunc(new_func = 0):
    return runbymainth(Running.loadFunc, (new_func, ))

@dispatcher.add_method
def UnloadFunc():
    return runbymainth(Running.unloadFunc, ())

@dispatcher.add_method
def StartFunc():
    return runbymainth(Running.startFunc, ())

@dispatcher.add_method
def StopFunc():
    return runbymainth(Running.stopFunc, ())

@dispatcher.add_method
def FinishFunc():
    return runbymainth(Running.finishFunc, ())

@dispatcher.add_method
def Heartbeat():
    return runbymainth(Running.doHeartbeat, ())

@dispatcher.add_method
def GetRunningFunc():
    #return runbymainth("GetRunningFunc", ())
    return (True, (0,))

@dispatcher.add_method
def ColorTracking(*target_color):
    return runbymainth(ColorTrack.setTargetColor, target_color)

@dispatcher.add_method
def ColorSorting(*target_color):
    return runbymainth(ColorSort.setTargetColor, target_color)

@dispatcher.add_method
def ColorPalletizing(*target_color):
    return runbymainth(ColorPalletiz.setTargetColor, target_color)

@Request.application
def application(request):
    dispatcher["echo"] = lambda s: s
    dispatcher["add"] = lambda a, b: a + b
    #print(request.data)
    response = JSONRPCResponseManager.handle(request.data, dispatcher)
    return Response(response.json, mimetype='application/json')

def startRPCServer():
#    log = logging.getLogger('werkzeug')
#    log.setLevel(logging.ERROR)
    run_simple('', 9030, application)

if __name__ == '__main__':
    startRPCServer()
