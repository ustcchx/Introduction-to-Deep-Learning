# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 22:42:13 2024

@author: 28577
"""
import matplotlib.pyplot as plt 
x = range(60)
y = []
with open("1.txt", "r") as file:
    for line in file:
        number = float(line.strip())
        y.append(number)
y = y + [129.99072, 129.7604, 129.53802, 129.32359, 129.1171, 128.91855, 128.72794, 128.54527, 128.37055, 128.20377, 128.04493, 127.89403,
         127.75108, 127.61606, 127.48899, 127.36986, 127.25867, 127.15543, 127.06012, 126.97276, 126.89334, 126.82186, 126.75833, 126.70273, 
         126.65508, 126.61537, 126.5836, 126.55978, 126.54389, 126.53595]
plt.ylim(min(y), max(y)) 
plt.plot(x, y, color="red")
plt.savefig("../loss/runing-loss-100.png")