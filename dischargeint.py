
import numpy as np
import pandas as pd
import _pickle as pickle
from datetime import datetime
import csv
import math

survey = {}
iter = -1

#generating date to integer
def to_integer(dt_time):
    dt_time= datetime.strptime(dt_time, '%Y-%m-%d')
    result = 10000*dt_time.year + 100*dt_time.month + dt_time.day
    return result

outer_list = []
iter = -1
previous_pid = 0
current_pid = 1000002
middle_list = []
number_patients = 0
miss_data = 0


d = []
temp = []
number = 1
## input data
#for line in open("discharge2.csv"):
for line in open("blooddischarge.csv"):
    x = line.rstrip().split(',')
    iter += 1
    temp = x
    if iter == 0:
        d.append(x)
        continue
    #print(x[0])
    temp[0] = int(x[0])
    #temp[1] = to_integer(x[1])
    temp[1] = int(x[1])
    for i in range(2,len(x)):
        if temp[i] == "":
            temp[i] = 0
        try:
            temp[i] = float(temp[i])
            temp[i] = int(temp[i])
        except:
            #print(temp[i])
            if temp[i] == "True":
                #print('*')
                temp[i] = 1
            else:
                temp[i] = 0
    d.append(temp)


## output data: csv version of patient data which all the values are integer, except date
#with open("intdischarge.csv", "w") as file:
with open("intblooddis.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(d)
