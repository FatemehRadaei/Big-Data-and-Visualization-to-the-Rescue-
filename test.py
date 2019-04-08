import torch
from torch import optim,nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import time
import os
import pickle
import argparse
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from matplotlib import pyplot as plt

from functions import date_converter, decay_fn, get_dates, get_inverted_dates

parser = argparse.ArgumentParser()
parser.add_argument("--ver", help="which model to use", default='retain',
    type=str)
parser.add_argument("--emb", help="embedding size of model", default=128,
    type=int)
parser.add_argument("--hid", help="hidden size of model", default=128,
    type=int)
parser.add_argument("--epoch", help="number of epochs to load", default=20,
    type=int)
parser.add_argument("--pnum", help="max number of patients to use", default=1000000, type=int)
parser.add_argument("--decay", help="which decay function to use", default=1,
    type=int)
parser.add_argument("--cuda", help="whether to use cuda",
    action="store_true")
parser.add_argument("--task", help="which data to test on", type=str, default='I50')
parser.add_argument("--lr", help="learning rate size", type=float, default=0.001)
parser.add_argument("--nfeatures", help="number of features", default=1400, type=int)

args = parser.parse_args()
task = args.task
lr = args.lr
pnum = args.pnum
nfeatures = args.nfeatures
args.ver = args.ver.lower().strip()
if args.ver=='retain':
    from models.retain_bidirectional import RETAIN
elif args.ver=='time':
    from models.retain_time import RETAIN
elif args.ver=='gru':
    from models.gru_bidirectional import GRU
else:
    print("Error! --ver must be either 'retain', 'time', or 'gru'")
    import sys
    sys.exit()

hid = args.hid
emb = args.emb
print(args)

# @V
# load data
with open('data/%s/test.pckl'%task,'rb') as f:
    test_data = pickle.load(f)

if args.ver=='gru':
    model = GRU(emb, hid, 1, args.cuda)
else:
    model = RETAIN(emb, hid, 1, nfeatures, args.cuda)
if args.cuda:
    model.cuda()

# for epoch in [args.epoch]:
# for epoch in range(9,15):
recal = []
prec = []
for epoch in range(1, args.epoch+1):
    # if args.cuda:
    #     model.load_state_dict(torch.load('experiments/I50/saved_weights/%s_epochs_%d_cuda.pckl'%(args.ver,args.epoch)))
    # else:
    #     model.load_state_dict(torch.load('experiments/I50/saved_weights/%s_epochs_%d_cpu.pckl'%(args.ver,args.epoch)))
    # @V
    # if args.ver=='time':
    #     name = args.ver + '_' + str(args.decay)
    # else:
    #     name = args.ver
    # if args.cuda:
    #     file_name = 'experiments/I50/saved_weights/%s_epochs_%d_cuda.pckl'%(name,epoch)
    # else:
    #     file_name = 'experiments/I50/saved_weights/%s_epochs_%d_cpu.pckl'%(name,epoch)
    if args.ver=='ex':
        # e.g. experiments/H26/ex-1_128_0.01/
        save_dir = 'experiments/%s/%s-%d_%d_%s'%(task,args.ver,time_ver,hid,str(lr))
    else:
        # e.g. experiments/H26/gru_128_0.01/
        save_dir = 'experiments/%s/%s_%d_%s'%(task,args.ver,hid,str(lr))
    weight_dir = os.path.join(save_dir,'saved_weights')
    if args.cuda:
        file_name = os.path.join(weight_dir,'%d_cuda.pckl'%(epoch))
    else:
        file_name = os.path.join(weight_dir,'%d_cpu.pckl'%(epoch))
    # file_name = 'experiments/I50/saved_weights/time_epochs_best_cuda.pckl'
    model.load_state_dict(torch.load(file_name))
    model.eval()
    print("Loading from %s" %file_name)

    loss_list = []
    correct_list = []
    predict_list = []
    score_list = []
    # @V
    max_i = len(test_data)
    if max_i > pnum:
        max_i = pnum
    for i in range(max_i):
        X,y = test_data[i]
        #print(y)
        if (len(X) <= 1):
            continue
        date_list = []
        input_list = []
        for sample in X:
            _,dates_,inputs_,_ = zip(*sample)
            #date_list.append(get_dates(dates_))
            input_list.append(list(inputs_))
        inputs = model.list_to_tensor(input_list)
        dates = Variable(torch.Tensor(date_list), requires_grad=False)
        # targets = Variable(torch.LongTensor(np.array(y,dtype=int)))
        if args.cuda:
            dates = dates.cuda()
            targets = targets.cuda()
        if args.ver=='time':
            outputs = model(inputs,dates)
        else:
            outputs = model(inputs)
        outputs = F.sigmoid(outputs.squeeze())

        # append to lists
        correct_list.extend(y)
        score_list.extend(outputs.data.cpu().tolist())
        predict_list.extend((outputs>0.5).data.tolist())
        # loss_list.append(loss.data[0])
    print('Epoch %d' %(epoch))
    print(sum(predict_list)/len(predict_list))
    print(predict_list)
    print(len(predict_list))
    str_acc = "Avg. ACC: %1.3f" %(accuracy_score(correct_list,predict_list))
    str_macro_auc = "Avg. AUC: %1.3f" %(roc_auc_score(correct_list,score_list,'macro'))
    str_prec = "Avg. prec: %1.3f" %(precision_score(correct_list,predict_list))
    str_recall = "Avg. rec: %1.3f" %(recall_score(correct_list,predict_list))
    log_list=[str_acc,str_macro_auc,str_prec,str_recall]
    recal.append(recall_score(correct_list,predict_list))
    prec.append(precision_score(correct_list,predict_list))
    for log in log_list:
        print(log)
plt.plot(recal,prec)
plt.show()
