import numpy as np
import serial
import time
from scipy.optimize import fsolve
import math

ser = serial.Serial('COM8', 9600)

time1 = 0
time2 = 0
time3 = 0
As = 0
Bs = 0
Cs = 0
counter = 0
sound1i = []
sound1t = []
sound2i = []
sound2t = []
sound3i = []
sound3t = []

def findtimes():
    global time1
    global time2
    global time3
    time1 = sound1t[sound1i.index(max(sound1i))]
    time2 = sound2t[sound2i.index(max(sound2i))]
    time3 = sound3t[sound3i.index(max(sound3i))]
        
def first(time1,time2,time3):
    global As
    global Bs
    global Cs
    if(time1 < time2 and time1 < time3):
        As = 0.0
        Bs = time2 - time1
        Cs = time3 - time1
    elif(time2 < time1 and time2 < time3):
        As = time1 - time2
        Bs = 0.0
        Cs = time3 - time2
    else:
        As = time1 - time3
        Bs = time2 - time3
        Cs = 0.0
    return As,Bs,Cs
    
def equations(p):
    global As
    global Bs
    global Cs
    x,y,r = p
    return((0.1525-x)**2+((math.sqrt(3)*0.1525)-y)**2-(math.fabs(r)+340.29*(Cs/1000000.0))**2,
	   x**2+y**2-(math.fabs(r)+340.29*(As/1000000.0))**2,
           (0.305-x)**2+y**2-(math.fabs(r)+340.29*(Bs/1000000.0))**2)

def record(st):
    global counter
    counter+=1
    tandI = st.split()
    mic = int(tandI[1])
    inten = int(tandI[0])
    if(mic%4 == 1):
        sound1i.append(inten)
        sound1t.append(mic-1)
    if(mic%4 == 2):
        sound2i.append(inten)
        sound2t.append(mic-2)
    if(mic%4 == 3):
        sound3i.append(inten)
        sound3t.append(mic-3)
   
while True:
    line = ser.readline().strip()
    if(line is not ''):
        record(line)
    if(len(sound1i) >= 1 and len(sound2i)>=1 and len(sound3i)>= 1):
        findtimes()
        As,Bs,Cs = first(time1,time2,time3)
        print As
        print Bs
        print Cs
        sound1i = []
        sound1t = []
        sound2i = []
        sound2t = []
        sound3i = []
        sound3t = []
        if(As < Bs and As < Cs and Bs < Cs):
            x,y,r = fsolve(equations, (-5,-5,1))
        elif(As < Bs and As < Cs and Bs > Cs):
            x,y,r = fsolve(equations, (-5,5,1))
        elif(As > Bs and As < Cs and Cs > Bs):
            x,y,r = fsolve(equations, (5,-5,1))
        else:
            x,y,r = fsolve(equations, (5,5,1))
        print x,y
        counter = 0
