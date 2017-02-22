import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fftpack
import struct
import cv2
import funcs
import numpy.random
import math

#Object Delaration for audio recording device
CHUNK = 8192
FORMAT = pyaudio.paInt16
CHANNELS = 4
RATE = 48000
RECORD_SECONDS = 30

p = pyaudio.PyAudio()
#Creation of stream object for capturing of sound bytes
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index = 13)

#Creation of second stream object in order to attain accurate y values

aux = p.open(format = FORMAT,
             channels = CHANNELS,
             rate = RATE,
             input = True,
             frames_per_buffer = CHUNK,
             input_device_index = 14)


#Speed of Sound
sound = 343.21

#Physical Offset of Microphones
x1 = 0
x2 = 20.67/1000
x3 = 41.33/1000
x4 = 62.0/1000

#Resamples particle filter
def resample(weights):
    cum_weights=weights.cumsum()
    cum_weights/=np.max(cum_weights)
    samples=np.random.sample(weights.size)
    indices=np.searchsorted(cum_weights,samples)
    return indices
 
#Establishes a random x,y-filled array for prticle filter manipulation
def initial_samples(n):
    samples=np.zeros((n,2))
    samples[:,0] = np.random.uniform(-2,2,n)
    samples[:,1] = np.random.uniform(0,2,n)
    return samples

#Creates a step within a linear filter
def step(samples,pe1,pe2,pe3,ype1,ype2,ype3,dx=.1):
    x=samples[:,0]
    y=samples[:,1]
    timediff1 = np.sqrt(x**2+y**2)*(1/sound)
    timediff2 = np.sqrt((x2-x)**2+y**2)*(1/sound)
    timediff3 = np.sqrt((x3-x)**2+y**2)*(1/sound)
    timediff4 = np.sqrt((x4-x)**2+y**2)*(1/sound)
    p1 = timediff2-timediff1
    p2 = timediff3-timediff2
    p3 = timediff4-timediff3
    td1 = np.sqrt((-2-x)**2+y**2)*(1/sound)
    td2 = np.sqrt((-2-x)**2+(x2-y)**2)*(1/sound)
    td3 = np.sqrt((-2-x)**2+(x3-y)**2)*(1/sound)
    td4 = np.sqrt((-2-x)**2+(x4-y)**2)*(1/sound)
    p1y = td2-td1
    p2y = td3-td2
    p3y = td4-td3
    wx=gaussian(p1,p2,p3,pe1,pe2,pe3,90000000000)
    wy=gaussian(p1y,p2y,p3y,ype1,ype2,ype3,90000000000)
    w = 5*wy+wx
    samples=samples[resample(w),:]
    h,w=samples.shape[:2]
    samples+=dx*(np.random.sample((h,w))*2-1)
    return samples

#Performs a weighting using the three diffs of expected area and actual
def gaussian(p1,p2,p3,p1e,p2e,p3e,s):
    return 1/(1+s*((p1-p1e)**2+(p2-p2e)**2+(p3-p3e)**2))


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

#Transforms normal x and y values to be viable for the graph structure
def plottransform(val,dim):
     cop = val*1
     cop /=2.0
     if(dim == 1):
         cop*=(1920/2)
         math.floor(cop)
         cop+=1920/2
     else:
         cop*=(1080/2)
         math.floor(cop)
         cop*=-1
         cop+=1080/2
     return int(cop)

#Plot the first point localized by the array
def initialplot(array,x,y):
    display = array*1
    x = plottransform(x,1)
    y = plottransform(y,2)
    display[y-5:y+5,x-5:x+5] = (0,0,0)
    return display

#Plots location of sound sources in real time
def plotcont(array,x,y,lx,ly):
     display = array*1
     x = plottransform(x,1)
     y = plottransform(y,2)
     lx = plottransform(lx,1)
     ly = plottransform(ly,2)
     display[y-5:y+5,x-5:x+5] = (0,0,0)
     cv2.line(display,(x,y),(lx,ly),(0,0,0),5)
     return display

#Generates the main array for display purposes
def initialarray():
     displayl = np.zeros((1920,1080,3))
     L = np.float32([[1,0,500],[0,1,500],[0,0,1]])
     displayl= cv2.warpPerspective(displayl,L,(1920,1080))
     displayl+=255
     displayl[535:545,:] = (0,0,0)
     displayl[:,955:965]= (0,0,0)
     cv2.putText(displayl, "-2m", (960,1050), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
     cv2.putText(displayl, "2m", (960,30), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
     cv2.putText(displayl, "1m", (960,270), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
     cv2.putText(displayl, "-1m", (960,810), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
     cv2.putText(displayl, "-2m", (30,570), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
     cv2.putText(displayl, "2m", (1870,570), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
     cv2.putText(displayl, "1m", (1440,570), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
     cv2.putText(displayl, "-1m", (480,570), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
     cv2.putText(displayl, "This is an Animation Displaying an Example of", (1100,30), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
     cv2.putText(displayl, "Tracking Output", (1100,60), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
     return displayl

#Changes difference in ticks to differences in seconds
def quantify(points):
     global RATE
     copy = points*1
     minn = np.min(points)
     copy -= minn
     copy *=(1/(RATE*1.0))
     return copy
     
#Uses matplotlib to display values taken in by the microphone array
def display(valarray,plot):
    x = np.linspace(0, CHUNK, CHUNK)
    y = valarray*1
    plot.plot(x, y)

samples = initial_samples(5000)

#Displays the result of a Fourier Transform in the Waveform utilizing matplotlib
def displayfft(mic1,mic2,mic3,mic4):
     fig,ax = plt.subplots()
     fig,bx = plt.subplots()
     display(mic1,ax)
     display(mic2,ax)
     display(mic3,ax)
     display(mic4,ax)
     freq_list = fftpack.fft(mic1)
     seq = -freq_list.conjugate()
     m2 = fftpack.fft(mic2)
     m3 = fftpack.fft(mic3)
     m4 = fftpack.fft(mic4)
     shiftone = np.abs(fftpack.ifft(m2*seq))
     shifttwo = np.abs(fftpack.ifft(m3*seq))
     shiftthree = np.abs(fftpack.ifft(m4*seq))
     bx.plot(shiftone)
     bx.plot(shifttwo)
     bx.plot(shiftthree)
     plt.show()

#Finds the x and y using simply the first detection of any sound
def method1():
     global samples
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
             p1e = b-a
             p2e = c-b
             p3e = d-c
             samples = step(samples,p1e,p2e,p3e)
             return np.average(samples,axis = 0)
               
#Finds the source of sound using a matching technique over several samples
def method2():
     global samples
     for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
         data = stream.read(CHUNK)
         data2 = aux.read(CHUNK)
         count = len(data)/2
         format1 = "<%dh"%(count)
         count2 = len(data2)/2
         format2 = "<%dh"%(count2)
         x = np.array(struct.unpack(format1,data))
         y = np.array(struct.unpack(format2,data2))
         mic1,mic2,mic3,mic4 = allocate(x)
         ymic1,ymic2,ymic3,ymic4 = allocate(y)
         hits = []
         yhits = []
         hits1,hits2,hits3,hits4 = find(3000,mic1,mic2,mic3,mic4)
         yhits1,yhits2,yhits3,yhits4 = find(3000,ymic1,ymic2,ymic3,ymic4)
         if(len(hits1[0])>0 and len(hits2[0])>0 and len(hits3[0])>0 and len(hits4[0])>0
            and len(yhits1[0])>0 and len(yhits2[0])>0 and len(yhits3[0])>0 and len(yhits4[0])>0):            
              mics = np.array([mic1,mic2,mic3,mic4])
              ymics = np.array([ymic1,ymic2,ymic3,ymic4])
              hits.append(hits1[0][0])
              hits.append(hits2[0][0])
              hits.append(hits3[0][0])
              hits.append(hits4[0][0])
              yhits.append(yhits1[0][0])
              yhits.append(yhits2[0][0])
              yhits.append(yhits3[0][0])
              yhits.append(yhits4[0][0])
              xseq = np.argmin(hits)
              yseq = np.argmin(yhits)
              sequence = fftpack.fft(mics[xseq])
              ysequence = fftpack.fft(ymics[yseq])
              m1 = fftpack.fft(mic1)
              m2 = fftpack.fft(mic2)
              m3 = fftpack.fft(mic3)
              m4 = fftpack.fft(mic4)
              ym1 = fftpack.fft(ymic1)
              ym2 = fftpack.fft(ymic2)
              ym3 = fftpack.fft(ymic3)
              ym4 = fftpack.fft(ymic4)
              modseq = -sequence.conjugate()
              ymodseq = -ysequence.conjugate()
              points = [np.float64(np.argmax(np.abs(fftpack.ifft(modseq*m1)))),
                        np.float64(np.argmax(np.abs(fftpack.ifft(modseq*m2)))),
                        np.float64(np.argmax(np.abs(fftpack.ifft(modseq*m3)))),
                       np.float64(np.argmax(np.abs(fftpack.ifft(modseq*m4))))]
              ypoints = [np.float64(np.argmax(np.abs(fftpack.ifft(ymodseq*ym1)))),
                         np.float64(np.argmax(np.abs(fftpack.ifft(ymodseq*ym2)))),
                         np.float64(np.argmax(np.abs(fftpack.ifft(ymodseq*ym3)))),
                         np.float64(np.argmax(np.abs(fftpack.ifft(ymodseq*ym4))))]
              points = quantify(points)
              ypoints = quantify(ypoints)
              xa = points[0]
              xb = points[1]
              xc = points[2]
              xd = points[3]
              xp1e = xb-xa
              xp2e = xc-xb
              xp3e = xd-xc
              ya = ypoints[0]
              yb = ypoints[1]
              yc = ypoints[2]
              yd = ypoints[3]
              yp1e = yb-ya
              yp2e = yc-yb
              yp3e = yd-yc
              samples = step(samples,xp1e,xp2e,xp3e,yp1e,yp2e,yp3e)
              x = samples[:,0]
              y = samples[:,1]
              x = x[y>0]
              y = y[y>0]
              newdone = np.zeros((len(y),2))
              newdone[:,0] = x
              newdone[:,1] = y
              x,y = np.average(newdone,axis = 0)
              x*=-1
              print x,y
              return x,y


#Cross-Correlation Based localization method
def method3():
     global samples
     for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
         data = stream.read(CHUNK)
         count = len(data)/2
         format = "<%dh"%(count)
         x = np.array(struct.unpack(format,data))
         mic1,mic2,mic3,mic4 = allocate(x)
         hits = []
         hits1,hits2,hits3,hits4 = find(4000,mic1,mic2,mic3,mic4)
         if(len(hits1[0])>0 and len(hits2[0])>0 and len(hits3[0])>0 and len(hits4[0])>0):
              mics = np.array([mic1,mic2,mic3,mic4])
              displayfft(mic1,mic2,mic3,mic4)
              hits.append(hits1[0][0])
              hits.append(hits2[0][0])
              hits.append(hits3[0][0])
              hits.append(hits4[0][0])
              seq = np.argmin(hits)
              points = [np.float64(np.argmax(signal.correlate(mic1,mics[seq]))-8191),
                        np.float64(np.argmax(signal.correlate(mic2,mics[seq]))-8191),
                        np.float64(np.argmax(signal.correlate(mic3,mics[seq]))-8191),
                        np.float64(np.argmax(signal.correlate(mic4,mics[seq]))-8191)]
              points = quantify(points)
              a = points[0]
              b = points[1]
              c = points[2]
              d = points[3]
              p1e = b-a
              p2e = c-b
              p3e = d-c
              samples = step(samples,p1e,p2e,p3e)
              return np.average(samples, axis = 0)


def update(displayl):
    displayl[535:545,:] = (0,0,0)
    displayl[:,955:965]= (0,0,0)
    cv2.putText(displayl, "-2m", (960,1050), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.putText(displayl, "2m", (960,30), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.putText(displayl, "1m", (960,270), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.putText(displayl, "-1m", (960,810), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.putText(displayl, "-2m", (30,570), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.putText(displayl, "2m", (1870,570), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.putText(displayl, "1m", (1440,570), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.putText(displayl, "-1m", (480,570), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.putText(displayl, "This is an Animation Displaying an Example of", (1100,30), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.putText(displayl, "Tracking Output", (1100,60), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    
k = initialarray()
x = 0
y = 0
lx = 0
ly = 0
i =  0

while(True):
    x,y = method2()
    i+=1
    if(i>10):
        if(lx == 0 and ly == 0):
            k = initialplot(k,x,y)
        else:
            k = plotcont(k,x,y,lx,ly)
        k+=25
        update(k)
        lx = x
        ly = y
        funcs.saveimage(str(i)+"pic.png",k)
        
