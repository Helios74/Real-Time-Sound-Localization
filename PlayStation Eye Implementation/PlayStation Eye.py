import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy import signal, fftpack
import struct
import cv2
import funcs

#Instantiation of Variables
a = 0.0
b = 0.0
c = 0.0
d = 0.0

#Object Delaration for audio recording device
CHUNK = 8192
FORMAT = pyaudio.paInt16
CHANNELS = 4
RATE = 96000
RECORD_SECONDS = 30

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index = 12)

#Speed of Sound
sound = 343.21

#Physical Offset of Microphones
x1 = 0
x2 = 20.67/1000
x3 = 41.33/1000
x4 = 62/1000

#Establish equations for solving of sound source location
def equations(p):
     global a
     global b
     global c
     global d
     r = abs((x2**2+x3**2-x4**2+(sound*a)**2+(sound * d)**2-(sound*b)**2-(sound*c)**2)/(2*sound*(b-a-d+c)))
     x,y = p
     return ((x1-x)**2+y**2-(r+sound*(a))**2,(x4-x)**2+y**2-(r+sound*(d))**2)

#Allocates parts of the raw input to each microphone's independent array
def allocate(array):
     m1 = array[0::4]
     m3 = array[1::4]
     m2 = array[2::4]
     m4 = array[3::4]
     return m1,m2,m3,m4
 
#Finds where microphones exceed a certain value
def find(threshold,a1,a2,a3,a4):
     h1 = np.where(a1 > threshold)
     h2 = np.where(a2 > threshold)
     h3 = np.where(a3 > threshold)
     h4 = np.where(a4 > threshold)
     return h1,h2,h3,h4
 
#Changes difference in ticks to differences in seconds
def quantify(points):
     copy = points*1
     minn = min(points)
     copy -= minn
     copy *=(1/48000.0)
     return copy

#Uses matplotlib to display values taken in by the microphone array
def display(plot,typ,valarray):
     if(typ == "lin"):
          x = np.linspace(0, CHUNK, CHUNK)
          y = valarray*1
          plot.plot(x, y)
     else:
          freq_list = np.fft.fft(valarray)
          plot.plot(freq_list)

#Finds the x and y using simply the first detection of any sound
def method1():
     global a
     global b
     global c
     global d
     for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
         data = stream.read(CHUNK)
         count = len(data)/2
         format = "<%dh"%(count)
         x = np.array(struct.unpack(format,data))
         mic1,mic2,mic3,mic4 = allocate(x)
         hits1, hits2, hits3, hits4 = find(15000,mic1,mic2,mic3,mic4)
         if(len(hits1[0])>0 and len(hits2[0])>0 and len(hits3[0])>0 and len(hits4[0])>0):
             points = np.array([hits1[0][0],hits2[0][0],hits3[0][0],hits4[0][0]]).astype(np.float32)
             print points
             points = quantify(points)
             a = points[0]
             b = points[1]
             c = points[2]
             d = points[3]
             if(a < d):
                 x,y = fsolve(equations, (-1,1))
             else:
                 x,y = fsolve(equations, (1,1))
             if(x < 10 and y < 10):
                 print x,abs(y)
             else:
                  pass
               
#Finds the source of sound using a matching technique over several samples
def method2():
     global a
     global b
     global c
     global d
     for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
         data = stream.read(CHUNK)
         count = len(data)/2
         format = "<%dh"%(count)
         x = np.array(struct.unpack(format,data))
         mic1,mic2,mic3,mic4 = allocate(x)
         hits = []
         hits1,hits2,hits3,hits4 = find(15000,mic1,mic2,mic3,mic4)
         if(len(hits1[0])>0 and len(hits2[0])>0 and len(hits3[0])>0 and len(hits4[0])>0):
              '''
              fig,ax = plt.subplots()
              fig,bx = plt.subplots()
              display(mic1,ax,"lin")
              display(mic2,ax,"lin")
              display(mic3,ax, "lin")
              display(mic4,ax, "lin")
              display(mic1,bx,"fft")
              display(mic2,bx,"fft")
              display(mic3,bx, "fft")
              display(mic4,bx, "fft")
              plt.show()
              '''            
              mics = np.array([mic1,mic2,mic3,mic4])
              hits.append(hits1[0][0])
              hits.append(hits2[0][0])
              hits.append(hits3[0][0])
              hits.append(hits4[0][0])
              print hits
              seq = hits.index(min(hits))
              sequence = fftpack.fft(mics[seq])
              m1 = fftpack.fft(mic1)
              m2 = fftpack.fft(mic2)
              m3 = fftpack.fft(mic3)
              m4 = fftpack.fft(mic4)
              modseq = -sequence.conjugate()
              points = [np.float64(np.argmax(np.abs(fftpack.ifft(modseq*m1)))),
                        np.float64(np.argmax(np.abs(fftpack.ifft(modseq*m2)))),
                        np.float64(np.argmax(np.abs(fftpack.ifft(modseq*m3)))),
                        np.float64(np.argmax(np.abs(fftpack.ifft(modseq*m4))))]
              print points
              points = quantify(points)
              a = points[0]
              b = points[1]
              c = points[2]
              d = points[3]
              if(a < d):
                  x,y = fsolve(equations, (-0.5,0.5))
              else:
                  x,y = fsolve(equations, (0.5,0.5))
              if(x < 10 and y < 10):
                   print x,abs(y)
              else:
                   pass

def method3():
     global a
     global b
     global c
     global d
     for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
         data = stream.read(CHUNK)
         count = len(data)/2
         format = "<%dh"%(count)
         x = np.array(struct.unpack(format,data))
         mic1,mic2,mic3,mic4 = allocate(x)
         hits = []
         hits1,hits2,hits3,hits4 = find(15000,mic1,mic2,mic3,mic4)
         if(len(hits1[0])>0 and len(hits2[0])>0 and len(hits3[0])>0 and len(hits4[0])>0):
              
              fig,ax = plt.subplots()
              fig,bx = plt.subplots()
              display(ax,"lin",mic1)
              display(ax,"lin",mic2)
              display(ax, "lin",mic3)
              display(ax, "lin",mic4)
              display(bx,"fft",mic1)
              display(bx,"fft",mic2)
              display(bx, "fft",mic3)
              display(bx, "fft",mic4)
              plt.show()
              mics = np.array([mic1,mic2,mic3,mic4])
              hits.append(hits1[0][0])
              hits.append(hits2[0][0])
              hits.append(hits3[0][0])
              hits.append(hits4[0][0])
              seq = hits.index(min(hits))
              points = [np.float64(np.argmax(signal.correlate(mic1,mics[seq]))-8191),
                        np.float64(np.argmax(signal.correlate(mic2,mics[seq]))-8191),
                        np.float64(np.argmax(signal.correlate(mic3,mics[seq]))-8191),
                        np.float64(np.argmax(signal.correlate(mic4,mics[seq]))-8191)]
              print points
              points = quantify(points)
              a = points[0]
              b = points[1]
              c = points[2]
              d = points[3]
              plt.show()
              if(a < d):
                  x,y = fsolve(equations, (-0.5,0.5))
              else:
                  x,y = fsolve(equations, (0.5,0.5))
              if(x < 10 and y < 10):
                   print x,abs(y)
              else:
                   pass

#Transforms normal x and y values to be viable for the graph structure
def plottransform(val):
     cop = val*1
     cop*=50
     cop+=500
     return cop

#Plots location of sound sources in real time
def plotter(array,x,y,lx,ly):
     display = array*1
     y *=-1
     ly*=-1
     x = plottransform(x)
     y = plottransform(y)
     lx = plottransform(lx)
     ly = plottransform(ly)
     display[y-5:y+5,x-5:x+5] = (0,0,255)
     display[ly-5:ly+5,lx-5:lx+5] = (0,0,255)
     cv2.line(display,(x,y),(lx,ly),(0,0,255),5)
     funcs.show(display)
     return display

displayl = np.zeros((1000,1000,3))
L = np.float32([[1,0,500],[0,1,500],[0,0,1]])
displayl[495:505,:] = (0,0,0)
displayl[:,495:505]= (0,0,0)
displayl= cv2.warpPerspective(displayl,L,(1000,1000))
displayl+=255

method3()
