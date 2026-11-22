# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:58:50 2019

@author: Abhishek
Input Features : pupil metrics of both eyes and medianSI
Output : 0 (No-Task)/1 (Task)
"""
from sklearn import preprocessing
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
from keras import regularizers
from numpy.random import seed
seed(1)
#%%
#Exploring and Processing the Data
df = pd.read_csv('dataset12.csv')
## Basic correlogram
#sns.pairplot(df,hue='Class')
#%%
df_Notask = df [df['Class']== 0] #splitting no-task rows
df_task = df [df['Class']== 1]  #splitting Task rows

##print the shape of the dataframe
print("Task matrix dimension",df_Notask.shape) #(41, 7)
print("Task matrix dimension",df_task.shape) #(80, 7)

random_Notask = df_Notask.sample(n=10) #randomly selecting no task related columns
random_task = df_task.sample(n=13) #randomly selecting task related rows
frames = [random_Notask,random_task] #storing randomly selcted task and no-task in a directory
df_keys=pd.concat(frames,keys=['NoTask','Task'])
dataset=df_keys.values
#%%
#Spliting dataset into feature vector and predicting vector
X=dataset[:,0:8]
Y=dataset[:,8]
#%%
#Scaling of feature vectors
#min-max scaler, which scales the dataset so that all the input features lie
#between 0 and 1 inclusive
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
#%%
#Train-test splitting
#X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.0)
#%%
# define the keras model
model = Sequential()
model.add(Dense(160, kernel_regularizer=regularizers.l2(0.001), input_dim=8, activation='relu')) #previous:160
model.add(Dense(80, kernel_regularizer=regularizers.l2(0.001), activation='relu')) #previous:80
#model.add(Dense(160, input_dim=6, activation='relu')) #previous:160
#model.add(Dense(80, activation='relu')) #previous:80
model.add(Dense(1, activation='sigmoid'))
model.summary()
#%%
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
poly_NN=model.fit(X_scale, Y, epochs=100, batch_size=1)
model.save('testtttt.h5')
#%%
# save the model to disk
#filename = 'model_poly_NN.sav'
#pickle.dump(poly_NN, open(filename, 'wb'))

#%%
# make class predictions with the model

from sklearn import preprocessing
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
from keras import regularizers
from keras.models import load_model
from numpy.random import seed
seed(1)

model = load_model('testtttt.h5')

df1=pd.read_csv('testdata12.csv')
df1.describe(include='all')
#Visualizing pairplot with two classes
#sns.pairplot(df,hue='Task')
df_task1 = df1 [df1['Class']== 1]  #splitting Task rows
df_Notask1 = df1 [df1['Class']== 0] #splitting no-task rows
random_task1 = df_task1.sample(n=86) #randomly selecting task related columns
random_Notask1 = df_Notask1.sample(n=42) #randomly selecting no-task related rows
frames1 = [random_task1,random_Notask1] #storing randomly selcted task and no-task in a directory
df_keys1=pd.concat(frames1,keys=['Task','NoTask'])
dataset1=df_keys1.values
X_test=dataset1[:,0:8]
Y_test=dataset1[:,8]
min_max_scaler = preprocessing.MinMaxScaler()
X_scale1 = min_max_scaler.fit_transform(X_test)

#predictions = model.predict_classes(X_scale1)
### summarize the first 10 cases
#for i in range(128):
#	print('%s => %d (expected %d)' % (X_scale1[i].tolist(), predictions[i], Y_test[i]))

predictions = model.predict_classes(X_scale1)
## summarize the first 10 cases


TP=0
TN=0
FP=0
FN=0

for i in range(128):
    val1=predictions[i]
    val2=Y_test[i]
    #print('%s => %d (expected %d)' % (X_scale1[i].tolist(), predictions[i], Y_test[i]))
    if(val1==1 and val2==1):
        TP+=1
    elif(val1==1 and val2==0):
        FP+=1
    elif(val1==0 and val2==0):
        TN+=1
    else:
        FN+=1
print(TP,FN,TN,FP)
Num=(TP+TN)
Dem=(TP+TN+FP+FN)
print()
Accuracy=Num/float(Dem)
print("Accuracy of the Model : %f"%(Accuracy*100)) 

