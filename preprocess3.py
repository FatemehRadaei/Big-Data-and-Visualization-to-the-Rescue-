import numpy as np
import pandas as pd
import _pickle as pickle
from datetime import datetime
import json


#generating date_float
def to_integer(dt_time):
#    dt_time= datetime.strptime(dt_time, '%m/%d/%y')
    dt_time= datetime.strptime(dt_time, '%Y-%m-%d')
    result = 10000*dt_time.year + 100*dt_time.month + dt_time.day
    return result

# def get_labels_dict(filename):
#     survey = {}
#     iter = -1
#     for line in open(filename):
#         x = line.split(',')
#         iter += 1
#         if iter > 0:
#             survey[int(x[0])] = int(x[1])
#     return survey
##### @Ferreshteh

def get_labels_dict(filename):
    survey = {}
    iter = -1
    for line in open(filename):
        x = line.strip().split(',')
        iter += 1
        if iter > 0:
            summ = 0
            numm = 0
            avg = 0.0
            for i in range(1,8):
                #print(x[i])
                if x[i] == 'TRUE' or x[i] == '1':
                    summ += 1
                    numm += 1
                    #print("*")
                elif x[i] == 'FALSE' or x[i] == '0':
                    numm += 1

            if numm < 0.01:
                avg = 0
            else:
                avg = (1.0*summ) / (1.0*numm)
            #print('*', avg)
            if avg > 0.2:
                avg = 1
            else:
                avg = 0
            #print(int(x[0]), '+', avg)
            pid = int(x[0])
            if pid in survey:
                survey[pid] = max(avg,survey[pid])
            else:
                survey[pid] = avg
    return survey


####
def make_binary(col_ind, val):
    i = col_ind
    t = val
    if i==1:
        #age
        if t > 65:
            t = 1
        else:
            t = 0
    if i> 30 and i <67:
        #heart rate
        if t > 85:
            t = 1
        else:
            t = 0
    if i> 66 and i < 103:
        #systolic blood presure
        if t > 140:
            t = 1
        else:
            t = 0
    if i > 102 and i < 141:
        #diastolic blood pressure
        if t < 60:
            t = 1
        else:
            t = 0
    if i > 140 and i < 178:
        #weight, hoever we need bmi instead
        if t > 100:
            t = 1
        else:
            t = 0
    if i > 177 and i < 183:
        #health status score
        if t > 5:
            t = 1
        else:
            t = 0
    if i > 182 and i < 186:
        #health maintance score?
        if t > 70:
            t = 1
        else:
            t = 0
    if i > 185 and i < 189:
        #self management score
        if t > 70:
            t = 1
        else:
            t = 0
    if i > 188 and i < 192:
        #self confidence
        if t > 70:
            t = 1
        else:
            t = 0
    if i > 191 and i < 202:
        #living heart failure score
        if t > 2:
            t = 1
        else:
            t = 0
    if i > 201 and i < 207:
        #self isolation scores
        if t > 15:
            t = 1
        else:
            t = 0
    if i > 206 and i < 212:
        #gds score
        if t > 5:
            t = 1
        else:
            t = 0
    return t

def store_in_batches(file_name, per_person_data_all, start_ind, end_ind, batch_size):
    # we need to sort based on the number of visits.
    # Each batch has to contain patiens with the same number of visits
    per_person_data = per_person_data_all[start_ind:end_ind]
    per_person_data.sort(key=lambda x: len(x))
    final_list = []
    batch_data_list = []
    batch_label_list = []
    last_person_num_visitis = -1
    for i in range(len(per_person_data)):
        if len(batch_data_list) >= 1 and last_person_num_visitis != len(per_person_data[i]):
            final_list.append((batch_data_list, batch_label_list))
            batch_data_list = []
            batch_label_list = []
            last_person_num_visitis = -1
        batch_data_list.append(per_person_data[i])
        batch_label_list.append(per_person_data[i][0][3])
        last_person_num_visitis = len(per_person_data[i])
        if len(batch_data_list) >= batch_size or i == end_ind - 1:
            final_list.append((batch_data_list, batch_label_list))
            batch_data_list = []
            batch_label_list = []
            last_person_num_visitis = -1

    with open(file_name,'wb') as f:
        pickle.dump(final_list,f)

def store_in_json(file_name, per_person_data_all, start_ind, end_ind):
    # we follow the bach method for consistency
    # so we need to sort based on the number of visits
    per_person_data = per_person_data_all[start_ind:end_ind]
    per_person_data.sort(key=lambda x: len(x))
    final_list = []
    for i in range(len(per_person_data)):
        person_data = per_person_data[i]
        pid = person_data[0][0]
        label = person_data[0][3]
        visits_features = [visit_data[2] for visit_data in person_data]
        final_list.append({
            'pid': pid,
            'visits_features': visits_features,
            'label': label,
        })

    with open(file_name,'w') as f:
        json.dump(final_list,f)



## main code

survey = get_labels_dict("EHR/survey3.csv")
ones =0
total=0
for key, val in survey.items():
    ones += val
    total += 1
print("map_labels", ones * 100.0 / total)
#final := list[batch_data]
#batch_data := (list[per_person_visits], list[per_person_label]),
#per_person_label := 0 or 1
#per_person_visits := list[per_visit]
#per_visit := (date_gap, date_float_formatted, list[features], label_int)


# outer_list will be all per_person_data without batching or anything
outer_list = []

iter = -1
previous_pid = 0
current_pid = 0
middle_list = []
number_patients = 0
miss_data = 0
ones = 0
total =0
for line in open("EHR/finalmerged2.csv"):
    x = line.split(',')
    iter += 1
    if iter == 0:
        continue
    #print(x[0])
    current_pid = int(x[0])
    if iter == 1:
        previous_pid = current_pid
    if current_pid != previous_pid:
        if len(middle_list) >= 1:
            number_patients += 1
            outer_list.append(middle_list)
        middle_list = []

    if current_pid not in survey:
        miss_data += 1
        continue

    newx = []
    for i in range(1,len(x)):
        t = x[i]

        if int(float(t)) == 1:
            newx.append(i)

        x[i] = t
    previous_pid = current_pid
    ones +=survey[current_pid]
    total += 1
    newtup = (current_pid, x[212], newx, survey[current_pid])
    middle_list.append(newtup)
print("data_labels", ones * 100.0 / total)
ones =0
total=0
for key, val in survey.items():
    ones += val
    total += 1
print("map_labels", ones * 100.0 / total)

# adding the last patient
if len(middle_list) >= 1:
    number_patients += 1
    outer_list.append(middle_list)

print("size of outer_list=", len(outer_list))

training_cut = int(60 * number_patients / 100)
validation_cut = int(70 * number_patients / 100)
sume = 0
for i in range(validation_cut,len(outer_list)):
    #print(outer_list[i][0][3])
    sume += outer_list[i][0][3]
#print(sume)
batch_size = 1000
store_in_batches(
    'EHR/train.pckl',
    outer_list,
    0,
    training_cut,
    batch_size,
)
store_in_batches(
    'EHR/val.pckl',
    outer_list,
    training_cut,
    validation_cut,
    batch_size,
)
store_in_batches(
    'EHR/test.pckl',
    outer_list,
    validation_cut,
    len(outer_list),
    batch_size,
)

store_in_json(
    'EHR/all_data1.json',
    outer_list,
    0,
    len(outer_list),
)
