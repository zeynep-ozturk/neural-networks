#read the copy pasted data
fname = r'./data/data.txt'
with open(fname) as f:
    content = f.readlines()
content = [x.strip().split() for x in content]
x = [float(x[0]) for x in content]
r = [float(x[1]) for x in content]

#40-30-30% percent split of train, test and validation sets
tra_x = x[:40]; tra_r = r[:40]; tra_y = [[]]*len(tra_r) #train set
val_x = x[40:70]; val_r = r[40:70]; val_y = [[]]*len(val_r) #validation set
test_x = x[70:]; test_r = r[70:]; test_y = [[]]*len(test_r) #test set

#initialize centers
H = [2, 4, 15] #number of hidden units
l=.999; eta = .1 # l: decrement factor of learning rate, eta: learning rate
eph=20 #number of epochs
mh_list = np.zeros((sum(H), ))
sh_list = np.zeros((sum(H), ))
b_list = np.zeros((len(tra_x), sum(H)))
for cv in range(len(H)):
    init = np.random.randint(0, 40, H[cv])
    mh_list[sum(H[:cv]):sum(H[:cv+1])] = [tra_x[j] for j in init]
    dist_mat = np.zeros((len(tra_x), H[cv]))
    for h in range(H[cv]):
        dist_mat[:,h]=abs(tra_x-np.repeat(mh_list[sum(H[:cv]):sum(H[:cv+1])][h], len(tra_x)))
    t1_idx = list(range(len(tra_x)))
    np.random.shuffle(t1_idx) #shuffle the training data
    for e in range(eph):
        for t1 in t1_idx:
            min_idx = np.argmin(dist_mat[t1,:])
            mh_list[sum(H[:cv]):sum(H[:cv+1])][min_idx]  += eta*(tra_x[t1]-mh_list[sum(H[:cv]):sum(H[:cv+1])][min_idx])
        eta *= l
    dist_mat_fin = np.zeros((len(tra_x), H[cv]))
    #find std devs by taking the half of the max intra-cluster distance
    for h in range(H[cv]):
        dist_mat_fin[:,h] = abs(tra_x-np.repeat(mh_list[sum(H[:cv]):sum(H[:cv+1])][h], len(tra_x)))
        for t1 in t1_idx:
            b_list[t1, sum(H[:cv]):sum(H[:cv+1])]=[1 if i==np.min(dist_mat_fin[t1,:]) else 0 for i in dist_mat_fin[t1,:]]
    sh_list[sum(H[:cv]):sum(H[:cv+1])] = np.max(b_list[:,sum(H[:cv]):sum(H[:cv+1])]*dist_mat_fin, axis=0)/2
sh_list[sh_list==0]=0.01
#plot centers to check the output is reasonable
plt.figure()
plt.plot(tra_x, np.zeros(len(tra_x)), 'bo')
plt.plot(mh_list[sum(H[:1]):sum(H[:1+1])], np.zeros(len(mh_list[sum(H[:1]):sum(H[:1+1])])), 'ro', alpha=.5 )

# ## NORMALIZED RBF

l1=.999; eta1 = .15 # l: decrement factor of learning rate, eta: learning rate
l2=l1; eta2 = eta1/1000 # l: decrement factor of learning rate, eta: learning rate
alpha=0.75 #momentum factor
eph_rbf=5 #number of epochs

wh_list = np.zeros((sum(H), )) #list for final weights on 3 different H values
ph_list = np.zeros((sum(H), )) #list for hidden unit outputs
gh_list = np.zeros((sum(H), )) #list for hidden unit normalized outputs
whgh_list = np.zeros((sum(H), )) #list for hidden unit normalized and weighted outputs
mh_list_fin = np.zeros((sum(H), )) #final list of centers
sh_list_fin = np.zeros((sum(H), )) #final list of std devs

err_tra_list = [[] for i in range(len(H))] #list for training MSE's
err_val_list = [[] for i in range(len(H))] #list for validation MSE's
tra_y_list = np.zeros((len(H),len(tra_x))) #list for training predictions

for cv in range(len(H)):
    wh=np.random.uniform(-.1, .1, (H[cv], ))
    mh=mh_list[sum(H[:cv]):sum(H[:cv+1])]
    sh=sh_list[sum(H[:cv]):sum(H[:cv+1])]
    for e in range(eph_rbf):
        err_tra=0
        err_val=0
        t1_idx = list(range(len(tra_x)))
        np.random.shuffle(t1_idx)
        prev_dw=0 #initialize weight change of previous instance
# training and validation set are processed together
        for t1, t2 in itertools.zip_longest(t1_idx, range(len(val_x))):
     #training set
            a=np.repeat(tra_x[t1], len(mh_list[sum(H[:cv]):sum(H[:cv+1])]))-mh_list[sum(H[:cv]):sum(H[:cv+1])]
            b=sh_list[sum(H[:cv]):sum(H[:cv+1])]
            ph = np.exp(-np.divide(a*a.T, 2*b*b.T))
            gh = ph/sum(ph)
            tra_y = np.dot(wh,gh)
            dw = eta1*(tra_r[t1]-tra_y)*gh-alpha*prev_dw
            prev_dw=dw
            dm = (eta2)*(tra_r[t1]-tra_y)*(wh-tra_y)*gh*np.divide((tra_x[t1]-mh),np.power(b, 2))
            ds = (eta2)*(tra_r[t1]-tra_y)*(wh-tra_y)*gh*np.divide(np.dot((tra_x[t1]-mh), (tra_x[t1]-mh).T),np.power(b, 3))
            wh+=dw
            #check whether centers go outside the input range
            if np.min(mh+dm)>=min(tra_x) and np.max(mh+dm)<=max(tra_x):
                mh+=dm
            #check whether std devs becomes too small or too big
            if np.min(sh+ds)>=0.01 and np.max(sh+ds)<=4/(2*H[cv]):
                sh+=ds
            err_tra+= pow((tra_r[t1]-tra_y), 2)/len(tra_x) # MSE for training instances
            #store predictions for the last epoch
            if e==eph_rbf-1:
                tra_y_list[cv][t1]=tra_y
            #validation set
            try:
                a=np.repeat(val_x[t2], len(mh_list[sum(H[:cv]):sum(H[:cv+1])]))-mh_list[sum(H[:cv]):sum(H[:cv+1])]
                ph_val = np.exp(-np.divide(a*a.T, 2*b*b.T))
                val_y = np.dot(wh,gh)
                err_val+= pow((val_r[t2]-val_y), 2)/len(val_x) # MSE for validation
            except:
                pass
        eta1*=l1
        eta2*=l2
        err_tra_list[cv].append(err_tra)
        try:
            err_val_list[cv].append(err_val)
        except:
            pass
    mh_list_fin[sum(H[:cv]):sum(H[:cv+1])]=mh
    sh_list_fin[sum(H[:cv]):sum(H[:cv+1])]=sh
    wh_list[sum(H[:cv]):sum(H[:cv+1])]=wh
    ph_list[sum(H[:cv]):sum(H[:cv+1])]=ph
    gh_list[sum(H[:cv]):sum(H[:cv+1])]=gh
    whgh_list[sum(H[:cv]):sum(H[:cv+1])]=np.dot(wh,gh)
    print("MSE for training with H={} is {}".format(H[cv],err_tra_list[cv][-1]))
    print("MSE for validation with H={} is {}".format(H[cv],err_val_list[cv][-1]))
#plot function
def plot_rbf(H, ph_list, gh_list, whgh_list):
    for cv in range(len(H)):
        fig, axs = plt.subplots(1,3, sharex=True, sharey=True)
        axs = axs.ravel()
        for i in range(3):
            axs[i].plot(tra_x, tra_r, 'b.', alpha=.7)
            #plot cluster centers
            axs[i].plot(mh_list[sum(H[:cv]):sum(H[:cv+1])], np.zeros(len(mh_list[sum(H[:cv]):sum(H[:cv+1])])), 'r.', alpha=.7)
            pred=np.vstack((tra_x, tra_y_list[cv])).T
            pred_sorted=pred[pred[:,0].argsort()]
            axs[i].plot(pred_sorted[:,0], pred_sorted[:,1], '-') #fitted values
            axs[i].set_ylim([-2, 2])
        #plot ph
        gaus = np.zeros((sum(H), len(tra_x)))
        for h_unit in range(H[cv]):
            mhh=mh_list_fin[sum(H[:cv]):sum(H[:cv+1])]
            m=[np.repeat(m, len(tra_x)) for m in mhh]
            s=sh_list_fin[sum(H[:cv]):sum(H[:cv+1])]
            gaus[h_unit,:]=np.exp(-np.power(tra_x-m[h_unit], 2)/(2*np.power(s[h_unit],2)))
            x_gaus=np.vstack((tra_x, gaus[h_unit,:])).T
            x_gaus_sorted=x_gaus[x_gaus[:,0].argsort()]
            axs[0].plot(x_gaus_sorted[:,0], x_gaus_sorted[:,1], '--')
        #plot gh
        denom=np.sum(gaus, axis=0)
        for h_unit in range(H[cv]):
            gh_h=gaus[h_unit,:]/denom
            x_gh=np.vstack((tra_x, gh_h)).T
            x_gh_sorted=x_gh[x_gh[:,0].argsort()]
            axs[1].plot(x_gh_sorted[:,0], x_gh_sorted[:,1], '--')
            axs[1].set_title('Normalized RBF H={}, ph, gh and wh*gh values'.format(H[cv]))
            #plot wh*gh
            whgh = wh_list[sum(H[:cv]):sum(H[:cv+1])][h_unit]*gh_h
            x_whgh=np.vstack((tra_x, whgh)).T
            x_whgh_sorted=x_whgh[x_whgh[:,0].argsort()]
            axs[2].plot(x_whgh_sorted[:,0], x_whgh_sorted[:,1], '--')
plot_rbf(H, ph_list, gh_list, whgh_list)

#plot for training and validation error
fig,ax = plt.subplots()
ax.plot(range(eph_rbf), err_tra_list[1], '-', alpha=.8)
ax.plot(range(eph_rbf), err_val_list[1], '--', dashes=(1,2), alpha=.8 )
ax.legend(['Training', 'Validation'])
plt.title('RBF MSE for Training&Validation Sets with H={}'.format(H[1]))
plt.xlabel('Epochs')
plt.ylabel('MSE')
ax.set_ylim([0,1])
plt.xticks(np.arange(min(range(eph_rbf)), max(range(eph_rbf))+2, 1.0))

#MSE calculation for test set
cv=1 #second experiment is the best one
err_test=0 #MSE for test set
for t in range(len(test_x)):
    a=np.repeat(test_x[t], len(mh_list[sum(H[:cv]):sum(H[:cv+1])]))-mh_list[sum(H[:cv]):sum(H[:cv+1])]
    b=sh_list[sum(H[:cv]):sum(H[:cv+1])]
    ph_test = np.exp(-np.divide(a*a.T, 2*b*b.T))
    test_y = np.dot(wh,gh)
    err_test+= pow((test_r[t]-test_y), 2)/len(test_x) # MSE for validation instances
MSE_test = err_test
print(MSE_test)


### MIXTURE of EXPERTS
eph_rbf=30 #number of epochs

wh_list = np.zeros((sum(H), )) #list for final weights on 3 different H values
vh_list = np.zeros((sum(H), )) #list for gating values on 3 different H values
ph_list = np.zeros((sum(H), )) #list for hidden unit outputs
gh_list = np.zeros((sum(H), )) #list for hidden unit normalized outputs
whgh_list = np.zeros((sum(H), )) #list for hidden unit normalized and weighted outputs
mh_list_fin = np.zeros((sum(H), )) #final list of centers
sh_list_fin = np.zeros((sum(H), )) #final list of std devs

err_tra_list = [[] for i in range(len(H))] #list for training MSE's
err_val_list = [[] for i in range(len(H))] #list for validation MSE's
tra_y_list = np.zeros((len(H),len(tra_x))) #list for training predictions

for cv in range(len(H)):
    vh=np.random.uniform(-.1, .1, (H[cv], ))
    v0=np.random.uniform(-.1, .1, (H[cv], ))
    mh=mh_list[sum(H[:cv]):sum(H[:cv+1])]
    sh=sh_list[sum(H[:cv]):sum(H[:cv+1])]
    for e in range(eph_rbf):
        err_tra=0
        err_val=0
        t1_idx = list(range(len(tra_x)))
        np.random.shuffle(t1_idx)
        prev_dv=0 #initialize weight change of previous instance
        prev_dv0=0
	    # training and validation set are processed together
        for t1, t2 in itertools.zip_longest(t1_idx, range(len(val_x))):
            #training set
            wh=np.dot(vh,tra_x[t1])+v0
            #print(v0, vh, wh)
            a=np.repeat(tra_x[t1], len(mh_list[sum(H[:cv]):sum(H[:cv+1])]))-mh_list[sum(H[:cv]):sum(H[:cv+1])]
            b=sh_list[sum(H[:cv]):sum(H[:cv+1])]
            ph = np.exp(-np.divide(a*a.T, 2*b*b.T))
            gh = ph/sum(ph) #radial gating
            tra_y = np.dot(wh,gh)
            dv = eta1*(tra_r[t1]-tra_y)*gh*tra_x[t1]-alpha*prev_dv
            prev_dv=dv
            dm = (eta2)*(tra_r[t1]-tra_y)*(wh-tra_y)*gh*np.divide((tra_x[t1]-mh),np.power(b, 2))
            ds = (eta2)*(tra_r[t1]-tra_y)*(wh-tra_y)*gh*np.divide(np.dot((tra_x[t1]-mh), (tra_x[t1]-mh).T),np.power(b, 3))
            vh+=dv
            dv0=eta1*(tra_r[t1]-tra_y)*gh-alpha*prev_dv0
            prev_dv0=dv0
            v0+=dv0
            #check whether centers go outside the input range
            if np.min(mh+dm)>=min(tra_x) and np.max(mh+dm)<=max(tra_x):
                mh+=dm
            #check whether std devs becomes too small or too big
            if np.min(sh+ds)>=0.01 and np.max(sh+ds)<=4/(2*H[cv]):
                sh+=ds
            err_tra+= pow((tra_r[t1]-tra_y), 2)/len(tra_x) # MSE for training instances
            #store predictions for the last epoch
            if e==eph_rbf-1:
                tra_y_list[cv][t1]=tra_y
            #validation set
            try:
                a=np.repeat(val_x[t2], len(mh_list[sum(H[:cv]):sum(H[:cv+1])]))-mh_list[sum(H[:cv]):sum(H[:cv+1])]
                ph_val = np.exp(-np.divide(a*a.T, 2*b*b.T))
                val_y = np.dot(wh,gh)
                err_val+= pow((val_r[t2]-val_y), 2)/len(val_x) # MSE for validation
            except:
                pass
        eta1*=l1
        eta2*=l2
        err_tra_list[cv].append(err_tra)
        try:
            err_val_list[cv].append(err_val)
        except:
            pass
    mh_list_fin[sum(H[:cv]):sum(H[:cv+1])]=mh
    sh_list_fin[sum(H[:cv]):sum(H[:cv+1])]=sh
    wh_list[sum(H[:cv]):sum(H[:cv+1])]=wh
    vh_list[sum(H[:cv]):sum(H[:cv+1])]=vh
    ph_list[sum(H[:cv]):sum(H[:cv+1])]=ph
    gh_list[sum(H[:cv]):sum(H[:cv+1])]=gh
    whgh_list[sum(H[:cv]):sum(H[:cv+1])]=np.dot(wh,gh)
    print("MSE for training with H={} is {}".format(H[cv],err_tra_list[cv][-1]))
    print("MSE for validation with H={} is {}".format(H[cv],err_val_list[cv][-1]))

#plot function
def plot_moe(H, ph_list, gh_list, wh_list, whgh_list):
    for cv in range(len(H)):
        fig, axs = plt.subplots(1,4, sharex=True, sharey=True)
        axs = axs.ravel()
        for i in range(4):
            axs[i].plot(tra_x, tra_r, 'b.', alpha=.7)
            #plot cluster centers
            axs[i].plot(mh_list[sum(H[:cv]):sum(H[:cv+1])], np.zeros(len(mh_list[sum(H[:cv]):sum(H[:cv+1])])), 'r.', alpha=.7)
            pred=np.vstack((tra_x, tra_y_list[cv])).T
            pred_sorted=pred[pred[:,0].argsort()]
            axs[i].plot(pred_sorted[:,0], pred_sorted[:,1], '-') #fitted values
            axs[i].set_ylim([-2, 2])
        #plot ph
        gaus = np.zeros((sum(H), len(tra_x)))
        for h_unit in range(H[cv]):
            mhh=mh_list_fin[sum(H[:cv]):sum(H[:cv+1])]
            m=[np.repeat(m, len(tra_x)) for m in mhh]
            s=sh_list_fin[sum(H[:cv]):sum(H[:cv+1])]
            gaus[h_unit,:]=np.exp(-np.power(tra_x-m[h_unit], 2)/(2*np.power(s[h_unit],2)))
            x_gaus=np.vstack((tra_x, gaus[h_unit,:])).T
            x_gaus_sorted=x_gaus[x_gaus[:,0].argsort()]
            axs[0].plot(x_gaus_sorted[:,0], x_gaus_sorted[:,1], '--')
        #plot gh
        denom=np.sum(gaus, axis=0)
        for h_unit in range(H[cv]):
            gh_h=gaus[h_unit,:]/denom
            x_gh=np.vstack((tra_x, gh_h)).T
            x_gh_sorted=x_gh[x_gh[:,0].argsort()]
            axs[1].plot(x_gh_sorted[:,0], x_gh_sorted[:,1], '--')
            axs[1].set_title('Cooperative MoE H={}, ph, gh, wh and wh*gh values'.format(H[cv]), x=1.08)
            #plot wh
            wh_h = np.dot(vh_list[sum(H[:cv]):sum(H[:cv+1])][h_unit],tra_x)+np.dot(v0[h_unit],np.repeat(1,len(tra_x)))
            x_wh=np.vstack((tra_x, wh_h)).T
            x_wh_sorted=x_wh[x_wh[:,0].argsort()]
            axs[2].plot(x_wh_sorted[:,0], x_wh_sorted[:,1], '--')
            #plot wh*gh
            whgh = wh_h*gh_h
            x_whgh=np.vstack((tra_x, whgh)).T
            x_whgh_sorted=x_whgh[x_whgh[:,0].argsort()]
            axs[3].plot(x_whgh_sorted[:,0], x_whgh_sorted[:,1], '--')
plot_moe(H, ph_list, gh_list, wh_list, whgh_list)

#plot for training and validation error
fig,ax = plt.subplots()
ax.plot(range(eph_rbf), err_tra_list[1], '-', alpha=.8)
ax.plot(range(eph_rbf), err_val_list[1], '--', dashes=(1,2), alpha=.8 )
ax.legend(['Training', 'Validation'], loc=2)
plt.title('MoE MSE for Training&Validation Sets with H={}'.format(H[1]))
plt.xlabel('Epochs')
plt.ylabel('MSE')
ax.set_ylim([0,1])
plt.xticks(np.arange(min(range(eph_rbf)), max(range(eph_rbf))+2, 5.0))

#MSE calculation for test set
cv=1 #second experiment is the best one
err_test=0 #MSE for test set
for t in range(len(test_x)):
    a=np.repeat(test_x[t], len(mh_list[sum(H[:cv]):sum(H[:cv+1])]))-mh_list[sum(H[:cv]):sum(H[:cv+1])]
    b=sh_list[sum(H[:cv]):sum(H[:cv+1])]
    ph_test = np.exp(-np.divide(a*a.T, 2*b*b.T))
    test_y = np.dot(wh,gh)
    err_test+= pow((test_r[t]-test_y), 2)/len(test_x) # MSE for validation instances
MSE_test = err_test
print(MSE_test)
