import os
import spidev
import time

# Function to convert digital data to Volts
def Volts(data, places, Vref):
  return round((data * Vref) / float(4096), places)

# Function to read Digital Data from a MCP3208 channel
# Channel 0-7
def ReadADCChannel(channel):
  adc = spi.xfer2([6 + ((channel&4) >> 2),(channel&3) << 6, 0])
  data = ((adc[1] & 15) << 8) + adc[2]
  return data

# Reference Voltage, Jumper selected 5.0 (default), 3.3, 1.0, or 0.3 Volts
Vref = 5.0

# (jumper CE0 on) chip = 0 (default), (jumper CE1 on) chip = 1
chip = 0

# Open SPI bus
spi = spidev.SpiDev()
spi.open(0, chip)

while True:

  c0 = ReadADCChannel(0)
  c1 = ReadADCChannel(1)
  c2 = ReadADCChannel(2)

  # results
  print "--------------------------------------------"
  print("Channel 0 : {} ({}V)".format(c0, Volts(c0, 2, Vref)))
  print("Channel 1 : {} ({}V)".format(c1, Volts(c1, 2, Vref)))
  print("Channel 2 : {} ({}V)".format(c2, Volts(c2, 2, Vref)))

  # Wait
  time.sleep(5)
