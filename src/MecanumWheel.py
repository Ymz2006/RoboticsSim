import numpy
import numpy as np
import tensorflow as tf
import pandas as pd
import csv

r = 0.08
lx = 0.5
ly = 0.7\

inv = numpy.array( [[1, -1, -(lx+ly)], [1, 1, lx+ly], [1, 1, -(lx+ly)], [1, -1, lx+ly]])


data_array = []

Xcount = 25
Ycount = 25
Wcount = 25

Xmax = 0.2
Ymax = 0.2
Wmax = 0.1


Xmin = 0
Ymin = 0
Wmin = 0

for vx in range(0, Xcount+1):
    for vy in range(0, Ycount+1):
        for wz in range(0, Wcount+1):
            vx_vel = Xmin +vx * (Xmax - Xmin)/Xcount
            vy_vel = Ymin + vy * (Ymax- Ymin)/Ycount
            wz_vel = Wmin + wz * (Wmax - Wmin)/Wcount
            angular_vel = numpy.array([vx_vel,vy_vel,wz_vel])
            output_angular_vel = numpy.dot(  (1/r * inv), angular_vel)

            data_array.append([vx_vel, vy_vel, wz_vel, output_angular_vel[0], output_angular_vel[1], output_angular_vel[2], output_angular_vel[3]])

with open('mecanum_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['vx', 'vy', 'wz', 'w1', 'w2', 'w3', 'w4'])
    writer.writerows(data_array)
