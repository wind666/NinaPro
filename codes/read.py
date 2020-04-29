import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from scipy import signal
import matplotlib.pyplot as plt
%matplotlib inline
from window_slider import Slider
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.svm import LinearSVC, SVC

#from sklearn.cross_validation import train_test_split

from sklearn.datasets import make_moons, make_circles, make_classification

#read EMG signals from 10 subjects (DB1,Exercise B) 10 channes and 17+1 classes
EMG=[]
subject=[]
lab=[]
for j in range(1,11):
    dire=("C:/Users/HP/Desktop/git_clone/NinaPro/unzip_data/"+'S' + str(j) + '_' + 'A1_E2.mat')
    ninapro_DB1=loadmat(dire)
    
    lab.extend(ninapro_DB1['restimulus'])

    sub=(ninapro_DB1['subject'])
    su=[sub[0][0]]*len(ninapro_DB1['restimulus'])
    subject.extend(su)
    
    EMG.extend(ninapro_DB1['emg'])

#2-D array to 2-D list
#sub = [[row.flat[0] for row in line] for line in subject]
label = [[row.flat[0] for row in line] for line in lab]
data = [[row.flat[0] for row in line] for line in EMG]
#change label format to (n_samples,) . Also data format is (n_samples,n_features)
label=(np.ravel(label)).tolist()
np.shape(label),np.shape(data),np.shape(subject)

# down sampling to 1 KHz
subject=subject[::2]
label=label[::2]
data=data[::2]
# Define a dictionary containing EMG data 
dataemg = {'subject': subject,
           'label':label,
           'data': data}
#dataemg
#make a Dataframes
columns_sub = ['subject']
columns_label = ['label']
columns_data = [ 'ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8','ch9','ch10']

df_subject = pd.DataFrame(subject, columns=columns_sub)
df_label= pd.DataFrame(label, columns=columns_label)
df_data = pd.DataFrame(data, columns=columns_data)

df_subject['label']=df_label
df_subject[columns_data]=df_data[columns_data]

#df_subject
"""Define window function that bucket_size & overlap of windows are as inputs"""
#pip install window-slider
from window_slider import Slider
def window (list,bucket_size,overlap_count):
    wind=[]
    #list = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    slider = Slider(bucket_size,overlap_count)
    slider.fit(list)       
    while True:
        window_data = (slider.slide()).tolist()
        # do your stuff
        #print(window_data)
        wind.append(window_data)
        if slider.reached_end_of_list(): break
    return(wind)
"""#example:
list1 = np.array([0, 1, 2, 3, 4, 5, 6, 7,8,9])
window(list1,4,2)"""
"""found all index in list"""
def all_indices(value, qlist):
    indices = []
    idx = -1
    while True:
        try:
            idx = qlist.index(value, idx+1)
            indices.append(idx)
        except ValueError:
            break
    return indices
#class0=all_indices(0,dataemg['label'])
#class1=all_indices(1,dataemg['label'])
#class2=all_indices(2,dataemg['label'])
#make a copy from dataset
import copy

#emg_17class=dataemg.copy()
emg_17class = copy.deepcopy(dataemg)
#delete class 0 from dataset
for index in sorted(class0, reverse=True):
    del emg_17class['subject'][index]
    del emg_17class['label'][index]
    del emg_17class['data'][index]
#windowing data base on their label (200 samples with 50 samples overlap 'f=1KhZ') 
numner_of_subject=10 #len(np.unique(emg_17class['subject']))
number_of_label=17  #len(np.unique(emg_17class['label']))
wind_data=[]
wind_label=[]
wind_subject=[]
for su in range(1,numner_of_subject+1):
    
    m=emg_17class['subject'].index(su)
    for la in range(1,number_of_label+1):
        #wind_label.append(la)
        if (la==number_of_label and su!=numner_of_subject):
            b=emg_17class['subject'].index(su+1)
        elif(la==number_of_label and su==numner_of_subject):
            b=len(emg_17class['label']) 
        else:
            b=emg_17class['label'].index(la+1,m)
                
        a=emg_17class['label'].index(la,m)
        
        wind=window(np.array(emg_17class["data"][a:b]).T,200,50)
        wind_data.extend(wind)
        wind_label.extend([la]*len(wind))
        wind_subject.extend([su]*len(wind))
#np.shape(wind_data),np.shape(wind_label),np.shape(wind_subject)        

np.shape(wind_data),np.shape(wind_data[0]) # wind_data[0] is number of channel*number of samples in each sample 
# & wind_data is number of windows*number of channels (that every channel contain a window (200 sample))
RMS=[]

for i in range(len(wind_data)):
    for j in range(len(wind_data[i])):
        RMS.append(np.sqrt(np.mean(list(map(lambda x:pow(x,2),wind_data[i][j])))))
RMS=(np.reshape(RMS,(len(wind_data),len(wind_data[0])))).tolist()
print(np.shape(emg_17class['data']),np.shape(emg_17class['label']))
np.shape(RMS),np.shape(wind_label)

from sklearn.model_selection import train_test_split
x_train , x_test,y_train,y_test=train_test_split(RMS , wind_label,test_size=0.25,random_state=42,stratify=wind_label) 
np.shape(x_train),np.shape(y_train),np.shape(x_test),np.shape(y_test)

#MLP
from sklearn.neural_network import MLPClassifier

clf=MLPClassifier(hidden_layer_sizes=(25,20),alpha=0.002,verbose=True ,  max_iter=300)
#clf=MLPClassifier(hidden_layer_sizes=(100),alpha=0.002,verbose=True ,  max_iter=300)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
from sklearn.metrics import accuracy_score
print("accuracy = %.2f"%accuracy_score(y_test,y_pred))
accuracy=np.mean(y_pred==y_test)
print("accuracy of test data=%.2f" %accuracy)
y_pred_train=clf.predict(x_train)
accuracy=np.mean(y_pred_train==y_train)
print("accuracy of train data=%.2f" %accuracy)
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred, labels=np.unique(y_test))
plt.figure(figsize=(8,8))
plt.imshow(cm,cmap=plt.cm.Blues,interpolation='nearest')
plt.xticks(range(17))
plt.yticks(range(17))
plt.colorbar()
plt.show()
#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
clf=LDA()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
accuracy=np.mean(y_pred==y_test)
print("accuracy of test data=%.2f" %accuracy)
y_pred_train=clf.predict(x_train)
accuracy=np.mean(y_pred_train==y_train)
print("accuracy of train data=%.2f" %accuracy)
#QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
clf=QDA()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
accuracy=np.mean(y_pred==y_test)
print("accuracy of test data=%.2f" %accuracy)
y_pred_train=clf.predict(x_train)
accuracy=np.mean(y_pred_train==y_train)
print("accuracy of train data=%.2f" %accuracy)
#RF
from sklearn.ensemble import RandomForestClassifier as RF
clf = RF(max_depth=200, random_state=0)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
accuracy=np.mean(y_pred==y_test)
print("accuracy of test data=%.2f" %accuracy)
y_pred_train=clf.predict(x_train)
accuracy=np.mean(y_pred_train==y_train)
print("accuracy of train data=%.2f" %accuracy)

#KNN
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
accuracy=np.mean(y_pred==y_test)
print("accuracy of test data=%.2f" %accuracy)
y_pred_train=clf.predict(x_train)
accuracy=np.mean(y_pred_train==y_train)
print("accuracy of train data=%.2f" %accuracy)
#SVM
from sklearn import svm
clf = svm.SVC(decision_function_shape='ovo',kernel='rbf',C=1,gamma=2)
clf.fit(x_train,y_train)
#clf.fit(x_train,y_train,kernel='rbf')
y_pred=clf.predict(x_test)
accuracy=np.mean(y_pred==y_test)
print("accuracy of test data=%.2f" %accuracy)
y_pred_train=clf.predict(x_train)
accuracy=np.mean(y_pred_train==y_train)
print("accuracy of train data=%.2f" %accuracy)
# All classifiers


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=200),
    RandomForestClassifier(max_depth=200, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    QDA()]

X = StandardScaler().fit_transform(RMS)

x_train , x_test,y_train,y_test=train_test_split(RMS , wind_label,test_size=0.25,random_state=42,stratify=wind_label) 



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
# iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(x_train, y_train)
    y_pred=clf.predict(x_test)
    print(name)
    print("accuracy = %.2f"%accuracy_score(y_test,y_pred))

# All classifiers


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    QDA()]

X = StandardScaler().fit_transform(emg_17class['data'])

x_train , x_test,y_train,y_test=train_test_split(emg_17class['data'] , emg_17class['label'],test_size=0.25,random_state=42,stratify=emg_17class['label']) 



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
# iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(x_train, y_train)
    y_pred=clf.predict(x_test)
    print(name)
    print("accuracy = %.2f"%accuracy_score(y_test,y_pred))
