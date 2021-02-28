import numpy as np
import pandas as pd
import math
import random as rand

file_path_tr = './data/train.txt'
file_path_ts = './data/test.txt'
#read training and test data
traicom = pd.read_csv(file_path_tr, names="x", dtype={'x': object})
tra_r = list()

testcom = pd.read_csv(file_path_ts, names="x", dtype={'x': object})
test_r = list()

#obtain labels and attributes(16x16) for training and test sets
for i in range(len(traicom)):
    if len(traicom['x'].iloc[i])==1:
        tra_r.append(traicom['x'].iloc[i])

tra_x = traicom[~traicom['x'].isin(list(map(str, range(10))))].reset_index(drop=True)

for i in range(len(testcom)):
    if len(testcom['x'].iloc[i])==1:
        test_r.append(testcom['x'].iloc[i])

test_x = testcom[~testcom['x'].isin(list(map(str, range(10))))].reset_index(drop=True)


#convert 16x16 attribute matrix to 256 dimensional row matrix for each instance
n = len(tra_r)
x=[[]]*n
for i in range(n):
    lst = []
    for j in range(16*i, 16*(i+1)):
        for k in tra_x['x'].iloc[j]:
            lst.append(k)
    x[i] = lst

x_test=[[]]*n
for i in range(n):
    lst = []
    for j in range(16*i, 16*(i+1)):
        for k in test_x['x'].iloc[j]:
            lst.append(k)
    x_test[i] = lst


#initialize weights and record true labels
k=10; d=256; k_idx=list(range(k)); d_idx=list(range(d+1));
w = np.zeros((k,d+1))
for i in k_idx:
    for j in d_idx:
        w[i][j] = rand.uniform(-.0001, .0001)

true = list(map(int, tra_r))

r = np.zeros((len(tra_r), k))
for i in range(len(tra_r)):
    for j in range(k):
        r[i][j] = 1 if int(tra_r[i])==j else 0

true_test = list(map(int, test_r))

r_test = np.zeros((len(test_r), k))
for i in range(len(test_r)):
    for j in range(k):
        r_test[i][j] = 1 if int(test_r[i])==j else 0

#define a function for confusion matrix calculation
def conf_mat(true, predicted, k):
    cm = [[0] * k for i in range(k)]
    for t, p in zip(true, predicted):
        cm[t][p] += 1
    return cm

#define a function for misclassification error calculation
def calc_error(confusion_matrix):
    total = sum(sum(x) for x in confusion_matrix)
    return 1-(sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))/total)
    #error=1-accuracy

#logistic regression of m epochs
o = np.zeros((k,1))
o_test = np.zeros((k,1))
l=.99 #decrement factor of eta
eta = .001
n_idx = list(range(n))
m = list(range(50)) # m: epochs, n: instances

err_list = []
err_list_test = []
for eph in m:
    pred = np.zeros((600,1))
    pred_test = np.zeros((600,1))
    n_idx = list(range(n))
    rand.shuffle(n_idx) #shuffle instance indices
    for ins in range(n):
        t = n_idx[0] #take the first element of shuffled instance list as current instance
        x_t = [1] + list(map(int, x[t])) #add bias unit to x matrix
        y_t = [0]*k #y_t: estimation for training set
        o = np.zeros((10,1))
        x_te = [1] + list(map(int, x_test[t]))
        y_te = [0]*k #y_te: estimation for test set
        o_te = np.zeros((10,1))

        o = np.dot(w,x_t)
        o_te = np.dot(w,x_te)
        y_t = np.dot(np.exp(o),(1/sum(np.exp(o)))) #softmax function for training estimations
        y_te = np.dot(np.exp(o_te),(1/sum(np.exp(o_te)))) #softmax function for test estimations
        for i in k_idx:
            for j in d_idx:
                w[i][j] += eta*(r[t][i]-y_t[i])*x_t[j] #update weights
        #choose predicted class as the max y_t value (probability of classes)
        pred[t] = np.argmax(y_t)
        pred_test[t] = np.argmax(y_te)
        n_idx.remove(t) #remove instance t from set
    eta *= l #decrease learning rate
    cm = conf_mat(true, list(map(int, pred)), k) #confusion matrix for training
    err_list.append(calc_error(cm)) #record errors for training
    cm_test = conf_mat(true_test, list(map(int, pred_test)), k) #confusion matrix for test
    err_list_test.append(calc_error(cm_test)) #record errors for training

names = [x for x in list(map(str,range(10)))]
df_cm = pd.DataFrame(cm, index=names, columns=names) #convert matrix to indexed dataframe
print(df_cm)

df_cm_test = pd.DataFrame(cm_test, index=names, columns=names) #convert matrix to indexed dataframe
print(df_cm_test)

#plotting errors for training and test sets
import matplotlib.pyplot as plt
plt.plot(range(50), err_list, 'r-')
plt.plot(range(50), err_list_test, 'b-')
plt.gca().grid(color='gray', linestyle='--', linewidth=0.3)
plt.title('Training & Test Errors (%)')
plt.xlabel('Epochs')
plt.ylabel('Error (%)')
plt.legend(['Training', 'Test'])
for spine in plt.gca().spines.values():
    spine.set_visible(False)
