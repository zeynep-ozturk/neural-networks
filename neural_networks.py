import numpy as np
import matplotlib.pyplot as plt
import itertools

#read the copy pasted data
fname = './data/data.txt'
with open(fname) as f:
    content = f.readlines()
content = [x.strip().split() for x in content]
x = [float(x[0]) for x in content]
r = [float(x[1]) for x in content]
#40-30-30% percent split of train, test and validation sets
tra_x = x[:40]; tra_r = r[:40]; tra_y = [[]]*len(tra_r) #train set
val_x = x[40:70]; val_r = r[40:70]; val_y = [[]]*len(val_r) #validation set
test_x = x[70:]; test_r = r[70:]; test_y = [[]]*len(test_r) #test set

#define a function for calculating both sigmoid and derivative of the sigmoid function
def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))

#plot function
def hiddenfigures(H, wx_list, zh_list, zhT_list, T_zero, e):
    for cv, h in enumerate(H):
        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
        axs = axs.ravel()
        for h_unit in range(H[cv]):
            #print lines that enter hidden units
            wx_x=np.vstack((tra_x,wx_list[sum(H[:cv]):sum(H[:cv+1]), :][h_unit] )).T
            wx_x_sorted=wx_x[wx_x[:,0].argsort()]
            axs[0].plot(wx_x_sorted[:,0], wx_x_sorted[:,1], '--', alpha=.6)
            #print hidden unit outputs
            zh_x=np.vstack((tra_x, zh_list[sum(H[:cv]):sum(H[:cv+1]), :][h_unit])).T
            zh_x_sorted=zh_x[zh_x[:,0].argsort()]
            axs[1].plot(zh_x_sorted[:,0], zh_x_sorted[:,1], '--', alpha=.6)
            axs[1].set_title('H={} Epoch={}'.format(H[cv], e))
            #print hidden unit weighted outputs
            zhT_x=np.vstack((tra_x,zhT_list[sum(H[:cv]):sum(H[:cv+1]),:][h_unit])).T
            zhT_x_sorted=zhT_x[zhT_x[:,0].argsort()]
            axs[2].plot(zhT_x_sorted[:,0], zhT_x_sorted[:,1], '--', alpha=.6)
        for i in range(3):
            axs[i].plot(tra_x, tra_r, 'b+', alpha=.6) #training data
            pred=np.vstack((tra_x, tra_y_list[cv])).T
            pred_sorted=pred[pred[:,0].argsort()]
            axs[i].plot(pred_sorted[:,0],pred_sorted[:,1],'b-',alpha=.8)#fittedvalues
            axs[i].set_ylim([-6, 6])
            axs[i].axhline(T_zero[cv], linestyle='-', color='blue', alpha=.3) #T0 line

d=[40, 30, 30] #dimensions of training, validation and test sets
H = [2, 4, 30] #number of hidden units
Hplus1 = [x+1 for x in H]
eph=500 #number of epochs
l=.999; eta = .1 # l: decrement factor, eta: learning rate
alpha=0.5 #momentum factor
e_list=[9, 199, 499] #epoch numbers for training data figures with best H
best_H=[H[1]]

#sizes of training, validation and test sets are different, thus MSE is used instead of SSE
err_tra_list = [[] for i in range(len(H))] #list for training MSE's
err_val_list = [[] for i in range(len(H))] #list for validation MSE's
tra_y_list = np.zeros((len(H),len(tra_x))) #list for training predictions
T_zero = [[] for i in range(len(H))] #list for bias hidden unit weight to output
T_zero2 = [] #list for bias hidden unit weight to output

w_list = np.zeros((sum(H), 2)) #list for final weights on 3 different H values
T_list = np.zeros((sum(H)+3, 1)) #list for final weights on 3 different H values
wx_list = np.zeros((sum(H), len(tra_x))) #list for hidden unit lines
zh_list = np.zeros((sum(H), len(tra_x))) #list for hidden unit outputs
zhT_list = np.zeros((sum(H), len(tra_x))) #list for hidden unit weighted outputs
wx_list2 = np.zeros((sum(H), len(tra_x))) #list for hidden unit lines
zh_list2 = np.zeros((sum(H), len(tra_x))) #list for hidden unit outputs
zhT_list2 = np.zeros((sum(H), len(tra_x))) #list for hidden unit weighted outputs
for cv, h in enumerate(H):
    #initialize first and second layer weights
    w=np.random.uniform(-.01, .01, (h, 2))
    T=np.random.uniform(-.01, .01, h+1)
    for e in range(eph):
        #print("epoch: ", e)
        err_tra=0
        err_val=0
        t1_idx = list(range(d[0]))
        np.random.shuffle(t1_idx)
        prev_dw=0 #initialize weight change of previous instance
        for t1, t2 in itertools.zip_longest(t1_idx, range(d[1])): # d[1] is the dimension of the validation set
            #training set
            tra_x_t1=np.append(1, tra_x[t1])
            zh = sigmoid(np.dot(w,tra_x_t1))
            zh_all = np.append(1, zh) #set bias unit to 1
            tra_y = np.sum(T*zh_all)
            dT = eta*(tra_r[t1]-tra_y)*zh_all
            dw = np.outer((eta*(tra_r[t1]-tra_y)*T[1:])*sigmoid(zh, derivative=True), tra_x_t1)-alpha*prev_dw
            prev_dw=dw # weight change of previous instance for momentum calculations
            w+=dw
            T+=dT
            err_tra+= pow((tra_r[t1]-tra_y), 2)/len(tra_x) # MSE for training instances
            #store predictions for the last epoch
            if e==eph-1:
                tra_y_list[cv][t1]=tra_y
            #validation set
            try:
                val_x_t2=np.append(1, val_x[t2])
                zh_val = sigmoid(np.dot(w,val_x_t2))
                zh_val_all = np.append(1, zh_val) #set bias unit to 1
                val_y = np.sum(T*zh_val_all)
                err_val+= pow((val_r[t2]-val_y), 2)/len(val_x) # MSE for validation #instances
            except:
                pass
        eta*=l
        err_tra_list[cv].append(err_tra)
        try:
            err_val_list[cv].append(err_val)
        except:
            pass
        if e in e_list and h==H[1]:
            wx2=np.dot(w,np.vstack((np.ones((40,)), tra_x)))
            wx_list2[sum(H[:cv]):sum(H[:cv+1]),:]=wx2
            zhs2=sigmoid(np.dot(w,np.vstack((np.ones((40,)), tra_x))))
            zh_list2[sum(H[:cv]):sum(H[:cv+1]),:]=zhs2
            zhT2=(((sigmoid(np.dot(w,np.vstack((np.ones((40,)), tra_x))))).T)*T[1:]).T
            zhT_list2[sum(H[:cv]):sum(H[:cv+1]),:]=zhT2
            T_zero2.append(T[0])
            hiddenfigures(best_H, wx_list2, zh_list2, zhT_list2, T_zero2, e+1)

    T_zero[cv]=T[0]
    w_list[sum(H[:cv]):sum(H[:cv+1]),:]=w
    T_list[sum(Hplus1[:cv]):sum(Hplus1[:cv+1])]=T.reshape(H[cv]+1, 1)
    wx=np.dot(w,np.vstack((np.ones((40,)), tra_x)))
    wx_list[sum(H[:cv]):sum(H[:cv+1]),:]=wx
    zhs=sigmoid(np.dot(w,np.vstack((np.ones((40,)), tra_x))))
    zh_list[sum(H[:cv]):sum(H[:cv+1]),:]=zhs
    zhT=(((sigmoid(np.dot(w,np.vstack((np.ones((40,)), tra_x))))).T)*T[1:]).T
    zhT_list[sum(H[:cv]):sum(H[:cv+1]),:]=zhT
    print("MSE for training with H={} is {}".format(H[cv],err_tra_list[cv][-1]))
    print("MSE for validation with H={} is {}".format(H[cv],err_val_list[cv][-1]))

#plot errors
fig,ax = plt.subplots()
ax.plot(range(eph), err_tra_list[1], '-', alpha=.8)
ax.plot(range(eph), err_val_list[1], '--', dashes=(1,2), alpha=.8 )
ax.legend(['Training', 'Validation'])
plt.title('MSE for Training&Validation Sets with H={}'.format(H[1]))
plt.xlabel('Epochs')
plt.ylabel('MSE')
ax.set_ylim([0,1])

#test set
err_test=0 #MSE for test set
for t in range(d[2]): # d[2] is the dimension of the test set
    test_x_t=np.append(1, test_x[t])
    zh = sigmoid(np.dot(w_list[2:6],test_x_t))
    zh_all = np.append(1, zh) #set bias unit to 1
    test_y = np.sum(T_list[3:8]*zh_all.reshape(5,1))
    err_test+= pow((test_r[t]-test_y), 2)/len(test_x) # MSE for test set instances
MSE_test = err_test
print(MSE_test)
hiddenfigures(H, wx_list, zh_list, zhT_list, T_zero, e+1)
