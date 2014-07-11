#!/bin/python
from movielens import *
from numpy import *
from numpy import array
from scipy.cluster.vq import *
import math

# Store data in arrays
user = []
item = []
rating = []
rating_test = []

# Load the movie lens dataset into arrays
d = Dataset()
d.load_users("C:/Users/BenBen/Documents/Google Drive/UD741 unsupervised learning/data/u.user", user)
d.load_items("C:/Users/BenBen/Documents/Google Drive/UD741 unsupervised learning/data/u.item", item)
d.load_ratings("C:/Users/BenBen/Documents/Google Drive/UD741 unsupervised learning/data/u.base", rating)
d.load_ratings("C:/Users/BenBen/Documents/Google Drive/UD741 unsupervised learning/data/u.test", rating_test)

n_users = len(user)
n_items = len(item)

# The utility matrix stores the rating for each user-item pair in the matrix form.
utility = zeros((n_users, n_items))
for r in rating:
    utility[r.user_id-1][r.item_id-1] = r.rating

# Finds the average rating for each user and stores it in the user's object
for i in range(0, n_users):
    user[i].avg_r = mean(utility[i])
print utility

##################################################
### Perform clustering on users and items
f = open("C:/Users/BenBen/Documents/Google Drive/UD741 unsupervised learning/data/u.item", "r")
text = f.read()
entries = re.split("\n+", text)
i=[]
for entry in entries:
    e = entry.split('|', 24)
    if len(e) == 24:
        i.append([int(e[5]), int(e[6]), int(e[7]), int(e[8]), int(e[9]), int(e[10]), int(e[11]), int(e[12]), int(e[13]), int(e[14]), int(e[15]), int(e[16]), int(e[17]), int(e[18]), int(e[19]), int(e[20]), int(e[21]),int(e[22]), int(e[23])])
f.close()
iarray = numpy.array(i)
k=20
res, idx = kmeans2(iarray,k)
 
util_clus=[]
for i in range(0,n_users): 
    tmp=[]
    avg=zeros(k)
    for m in range(0,k):
        tmp.append([])
    for j in range(0,n_items):
        if utility[i][j]!=0:
            tmp[idx[j]].append(utility[i][j])    
    for m in range(0,k):
        avg[m]=mean(tmp[m]) 
    util_clus.append(avg)

##################################################
# Predit the ratings of the user-item pairs in rating_test
pearson=[]
for i in range(0,n_users):
    for j in range(0,n_users):
        if i!=j:
            x=util_clus[i]
            y=util_clus[j]
            x_bar=sum(a for a in x if a>0)/sum(a>0 for a in x)
            y_bar=sum(a for a in y if a>0)/sum(a>0 for a in y)          
            sm=0
            sqsm=0
            for k in range(0,len(x)):
                if math.isnan(x[k])==False and math.isnan(y[k])==False:
                    sm=sm+(x[k]-x_bar)*(y[k]-y_bar)
                    sqsm=sqsm+(x[k]-x_bar)**2*(y[k]-y_bar)**2
            if sqsm!=0:
                pcs=sm/sqsm
            else:
                pcs=0
            pearson.append([i,j,pcs])
        
#check=[]
#for t in range(0,len(pearson)):
#    if math.isnan(pearson[i][2])==True:
#        check.append(pearson[i])

f = open("C:/Users/BenBen/Documents/Google Drive/UD741 unsupervised learning/data/u.test", "r")
text = f.read()
entries = re.split("\n+", text)
test=[]
for entry in entries:
    e = entry.split('\t', 4)
    if len(e) == 4:
        test.append([int(e[0]),int(e[1]),int(e[2]),int(e[3])])
f.close()
testarray = numpy.array(test)

#testuser=set()
#for t in range(0,len(test)):
#    testuser.add(test[t][0])

testitem={}
for u in range(1,n_users+1):
    testitem[u]=[]
    for t in range(0,len(test)):
        if test[t][0]==u:
            testitem[u].append(test[t][1])
    
top_n=3
guess=[]
for u in range(1,n_users+1):
    tmp1=util_clus[u-1]
    tmp2=tmp1[~numpy.isnan(tmp1)]
    copy=sorted(tmp2, reverse=True)
    idx_n=[]
    for i in range(0,len(copy)):
        idx_n.append(util_clus[u-1].tolist().index(copy[i]))
    for item_id in testitem[u]:
        clus_id=idx[item_id-1]
        value_n=[]
        for peer_id in idx_n:
            if math.isnan(util_clus[peer_id][clus_id])==False:
                value_n.append(util_clus[peer_id][clus_id])
        guess.append([u,item_id,round(mean(value_n))])
        
##################################################
# Find mean-squared error

##################################################
# Find % Accuracy
accuracy=[]
for k in range(0,len(test)):
    accuracy.append(guess[k][2]-test[k][2])
pctg0=sum(a==0 for a in accuracy)/float(len(test))
pctg1=sum(a==1 or a==-1 for a in accuracy)/float(len(test))
pctg2=sum(a==2 or a==-2 for a in accuracy)/float(len(test))
pctg3=sum(a==3 or a==-3 for a in accuracy)/float(len(test))
pctg4=sum(a==4 or a==-4 for a in accuracy)/float(len(test))
pctgs=[pctg0,pctg1,pctg2,pctg3,pctg4]

##################################################
# Guesses the ratings that user with id, user_id, might give to item with id, i_id.
# We will consider the top_n similar users to do this.
def guess(user_id, i_id, top_n):
    user_idx=user_id-1
    item_idx=item_id-1
    tmp=zeros(n_users)
    x=utility[user_idx]
    for j in range(0,n_users):
        if j!=user_idx:
            y=utility[j]
            tmp[j]=pcs(x,y) 
    tmp=tmp.tolist()
    copy=sorted(tmp, reverse=True)
    idx_n=[]
    for i in range(0,top_n):
        if copy[i]!=0:
            idx_n.append(tmp.index(copy[i]))
    print idx_n 
    value_n=[]
    for idx in idx_n:
        if utility[idx][item_id]!=0:
            value_n.append(utility[i][item_idx])
    avg=mean(value_n)
    return avg   