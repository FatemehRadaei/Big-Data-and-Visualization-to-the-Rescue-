

import numpy as np
import pandas as pd
import _pickle as pickle
from datetime import datetime
import re
import csv
import math


d = []
tresh = []
iter = -1
for line in open("int2discharge.csv"):
    x = line.rstrip().split(',')
    iter += 1
    if iter == 0:
        d.append(x)
        continue
    if iter == 1 or iter == 2:
        continue
    if iter == 3:
        for i in range(0,len(x)):
            tresh.append(x[i])
    if iter > 4:
        for i in range(1,len(x)):
            if x[i] >= tresh[i]:
                x[i] = 1
            else:
                x[i] = 0
        d.append(x)

with open("bindischarge.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(d)
