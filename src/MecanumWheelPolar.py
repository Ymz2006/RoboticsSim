import random
import numpy
import numpy as np
import tensorflow as tf
import pandas as pd
import csv

r = 0.08
lx = 0.3
ly = 0.6

inv = numpy.array( [[1, -1, -(lx+ly)], [1, 1, lx+ly], [1, 1, -(lx+ly)], [1, -1, lx+ly]])
fwd = numpy.array( [[1, 1, 1, 1], [-1, 1, 1, -1], [-1/(lx + ly), 1/(lx + ly), -1/(lx+ly), 1/(lx+ly)]])

data_array = []
data_array_speed = []

MaxRot = 5
MinRot = -5


for i in range (1000):
    inp_angular_vel = np.array([random.uniform(-5, 5) for _ in range(4)])
    output_vel = np.array( r/4 * np.dot(fwd, inp_angular_vel))


    curr_row = np.append(inp_angular_vel, output_vel[2])
    curr_row_speed = np.append(inp_angular_vel, np.sqrt(output_vel[1] * output_vel[1] + output_vel[2] * output_vel[2]))
    data_array.append(curr_row)
    data_array_speed.append(curr_row_speed)

with open('mecanum_polar_rot.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['w1', 'w2', 'w3', 'w4', 'angle'])
    writer.writerows(data_array)

with open('mecanum_polar_speed.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['w1', 'w2', 'w3', 'w4', 'speed'])
    writer.writerows(data_array_speed)