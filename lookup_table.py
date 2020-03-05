# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree


number_wavelengths=16

#def find_pandas(x_array): 
#    table = pd.read_csv('C:/Users/Teresa/Desktop/test_lookup.csv')
#    print(table)
#    matrix_intensities = table.iloc[:,2:]
#    print(matrix_intensities)
#    matching_array = matrix_intensities[cKDTree(matrix_intensities).query(x_array, k=1)[1]]
#    print(matching_array)
#    return(matching_array)
    
def find_numpy(x_array):
    table = np.array(list(csv.reader(open('C:/Users/Teresa/Desktop/test_lookup.csv', "r"), delimiter=",")))
#    table = np.genfromtxt('C:/Users/Teresa/Desktop/test_lookup.csv', delimiter=',', names=True, case_sensitive=True)
    print(table)
    table = table[1:].astype(float)
    matrix_intensities = table[:,2:]
    print(matrix_intensities)
    matching_array = table[cKDTree(matrix_intensities).query(x_array, k=1)[1]]
#    print("measured Intensities = " + str(x_array))
#    print("simulated intensities = " + str(matching_array[2:]))
#    print("density = " + matching_array[0])
#    print("diameter = " + matching_array[1])
    return(matching_array)
    
def create_lookup(start_density, end_density, start_dia, end_dia, start_wl, end_wl, diffraction_index):
    return(0)
     
# row: [density, size, I(400nm), I(500nm), I(600nm), ...]
nms = [[0.1, 1, 10, 19, 18, 13, 9, 13, 15],\
       [0.2, 1, 12, 18, 17, 13, 14, 9, 11],\
       [0.3, 1, 15, 18, 16, 14, 16, 16, 17],\
       [0.4, 1, 13, 17, 15, 14, 14, 16, 14],\
       [0.5, 1, 12, 11, 14, 13, 11, 12, 13],\
       [0.1, 2, 16, 13, 13, 16, 16, 17, 18],\
       [0.2, 2, 17, 12, 12, 16, 11, 15, 14],\
       [0.3, 2, 16, 17, 11, 16, 9, 14, 14],\
       [0.4, 2, 18, 11, 18, 13, 14, 13, 14],\
       [0.5, 2, 13, 15, 17, 12, 9, 8, 7],\
       [0.1, 3, 12, 11, 16, 13, 9, 12, 15],\
       [0.2, 3, 11, 17, 15, 12, 11, 15, 14],\
       [0.2, 3, 18, 13, 14, 13, 13, 14, 14],\
       [0.2, 3, 16, 15, 13, 12, 13, 14, 15]]
#nms_2 = [[500, 0.4, 1, 5]]
#
nms_t = np.array(nms)
#nms_t = np.append(nms_t,nms_2, axis=0)

#print(nms_t)
#print(nms_t[0][1])
#print(nms_t[0][2])

f = open('C:/Users/Teresa/Desktop/test_lookup.csv', 'w', newline='')

with f:

    writer = csv.writer(f)
    writer.writerow(['density', 'diameter', 'I(400)', 'I(450)', 'I(500)', 'I(550)', 'I(600)', 'I(650)', 'I(700)'])
    for row in nms_t:
        writer.writerow(row)

test_array = [11, 12, 16.6, 11, 9, 15, 14]

matching_array = find_numpy(test_array)

print("measured Intensities = " + str(test_array))
print("simulated intensities = " + str(matching_array[2:]))
print("density = " + str(matching_array[0]))
print("diameter = " + str(matching_array[1]))

wavelengths = [[400, 450, 500, 550, 600, 650, 700]]

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(wavelengths, test_array, s=10, c='b', marker="s", label='measured')
ax1.scatter(wavelengths, matching_array[2:], s=10, c='r', marker="o", label='simulated')
plt.legend(loc='upper left');
plt.show()
