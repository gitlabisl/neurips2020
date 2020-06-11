import torch
import torch.nn as nn
import numpy as np
import scipy
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from keras.layers import Dropout

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#  load the features from rnn or lstm or hor and normalize
X = (X - X.max())/(X.max() - X.min())

# whatever channel u wud like to or conjoint channels or concatenated channels
channel = 1
#### Train and Test For Channel 1 
#### Probe wise train and probe wise test
X = X[:,:,:,channel]

y = y.reshape((30,24))

X_d = X.reshape((X.shape[0]*X.shape[1],X.shape[2]))
y_d = y.reshape(y.shape[0]*y.shape[1])
X_train, X_test, y_train, y_test = train_test_split(X_d, y_d,stratify=y_d, test_size=0.3)

Accs = []
Test_accs = []
kfold = StratifiedKFold(4, True, 1)
val_sensitivity =[]
val_specificity=[]
# model for cross-validation
for train, val in kfold.split(X_train,y_train):
        X_tra, X_val, y_tra, y_val = X_train[train], X_train[val], y_train[train], y_train[val]
        print(X_tra.shape,X_val.shape,y_tra.shape,y_val.shape)
        model = Sequential()
        model.add(Dropout(0.1, input_shape=(50,)))
        model.add(Dense(100, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(1,activation='sigmoid'))
# compile the keras model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit the keras model on the dataset
        model.fit(X_tra, y_tra, epochs=300, batch_size=10,shuffle=True)
        # evaluate the keras model
        _, accuracy = model.evaluate(X_val, y_val)
        Accs.append(accuracy*100)
        print('Validation_Accuracy: %.2f' % (accuracy*100))
        val_tn, val_fp, val_fn, val_tp = confusion_matrix(y_val,model.predict_classes(X_val)).ravel()
        val_sensitivity.append(val_tp/(val_tp+val_fn))
        val_specificity.append(val_tn/(val_tn+val_fp))
Accs = np.array(Accs)
val_sensitivity = np.array(val_sensitivity)
val_specificity = np.array(val_specificity)
Avg_Validation_accuracy= np.mean(Accs)
Std_Validation_accuracy = np.std(Accs)
Avg_Validation_sensitivity = np.mean(val_sensitivity)
Std_Validation_sensitivity = np.std(val_sensitivity)
Avg_Validation_specificity = np.mean(val_specificity)
Std_validation_specificity = np.std(val_specificity)
# final model

model = Sequential()
model.add(Dropout(0.1, input_shape=(50,)))
model.add(Dense(100, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dense(1,activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=300, batch_size=10,shuffle=True)
# evaluate the keras model
_, test_accuracy = model.evaluate(X_test,y_test)
print('Test_Accuracy: %.2f' % (test_accuracy*100))
test_tn, test_fp, test_fn, test_tp = confusion_matrix(y_test,model.predict_classes(X_test)).ravel()
test_sensitivity = test_tp/(test_tp+test_fn)
test_specificity= test_tn/(test_tn+test_fp)

# #### Subject wise test for Channel 1
subj_predictions = []
for subj in range(X.shape[0]):
    n = np.count_nonzero(model.predict_classes(X[subj,:,:]) == 1)
    if n> 18:
        subj_predictions.append(1)
    else :
        subj_predictions.append(0)

true_labels = y[:,0]

subj_tn, subj_fp, subj_fn, subj_tp = confusion_matrix(true_labels,subj_predictions).ravel()
subj_accuracy = (subj_tp+subj_tn)/(subj_tn+subj_fp+subj_fn+subj_tp)
print('subject accuracy ', subj_accuracy)
subj_sensitivity = subj_tp/(subj_tp+subj_fn)
subj_specificity= subj_tn/(subj_tn+subj_fp)





