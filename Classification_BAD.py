import torch
import torch.nn as nn
import numpy as np
import scipy
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

# fix random seed for reproducibility

#  Load the BAD features for perp(Perpetrator) and wit(Witness)
X = annots['perp'][:,:,1]
y = np.ones(annots['perp'].shape[0])
X = np.concatenate((X,annots['wit'][:,:,1]),axis = 0)
y = np.concatenate((y,np.zeros(annots['wit'].shape[0])))


X = (X-np.min(X))/(np.max(X)-np.min(X))

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.3)

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
        model.add(Dropout(0.1, input_shape=(750,)))
        model.add(Dense(1500, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(750, activation='relu'))        
        model.add(Dropout(0.2))
        model.add(Dense(100, activation='relu'))
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
model.add(Dropout(0.1, input_shape=(750,)))
model.add(Dense(1500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(750, activation='relu'))        
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
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
scipy.io.savemat('/home/xx/Documents/NIPS_20/results_bp/amplitude_200.mat', mdict={'mean_Val': Avg_Validation_accuracy,'std_Val':Std_Validation_accuracy,
                                                                                            'mean_Val_Sensi': Avg_Validation_sensitivity,'mean_Val_Speci':Avg_Validation_specificity,
                                                                                            'std_Val_Sensi':Std_Validation_sensitivity,'std_Val_Speci':Std_validation_specificity,'Test_acc': test_accuracy*100,'test_sensi':test_sensitivity,'test_speci':test_specificity})
