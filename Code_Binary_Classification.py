#Objective 1
#Train a Multi Layer Perceptron network model for Binary classification
#Objective 2
#Find few parameters of Multi Layer Perceptron network by GridSerchCrossValidation method from train data
###################################################
##BINARY CLASSIFICATION using a predefined model architecture

##############################################

#Binary classification using Multilayer perceptron

#If you are using spyder you need to install scikeras-
#by typing pip install scikeras at anaconda prompt


#Import required libraries
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import models
from scikeras.wrappers import KerasClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import tensorflow as tf

#PART 1: BASELINE MODEL
#Observation1: Task description
#Refer to provided paper and write in brief about classification task


#Read data from csv file 

df= pd.read_csv("Acoustic_Extinguisher_Fire_Dataset.csv")

#Observation2:Note type of data variable df, its shape-number of rows and columns

print("The variable, df is of type:", type(df))
df.shape

#observe first few data instances to get rough idea of type of  variables present in df
df.head()

#Observation3: Note column names and identify predictors and response variables
df.columns

#Observation4:Check id data has any missing values
df.isna().any()

#Observation5:Note data types of predictors and response variable
df.info()

#Observation6:Find number of classes and number of instances of each class
#Note if dataset is balanced or imbalanced
df['FUEL'].value_counts()

#Observation7:if labels are categorical encode them into numeric form, otherwise skip this step
df['FUEL'] = df.FUEL.astype('category').cat.codes

df['FUEL']  #verify that labels are encoded

#Convert dataframe into array since GridSearchCV works with array data
dataset=df.to_numpy()

# Split into predictor\input(X) and response\output(Y) variables
X = dataset[:,0:6]
Y = dataset[:,6]

#Split data into train and test sets(80-20), use random_state for reproducible results
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=123)
#Observation8:Note shape of train and test data
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape
#Observation9: Note class distributiion in train and test data
unique, counts = np.unique(Y_train, return_counts=True)
print(np.asarray((unique, counts)).T)
print("The class distribution in train data is:",np.asarray((unique, counts)).T)

unique, counts = np.unique(Y_test, return_counts=True)
print(np.asarray((unique, counts)).T)
print("The class distribution in test data is:",np.asarray((unique, counts)).T)


#Choose a baseline model with a single hidden layer and number of neurons equal to a number between number of neurons in input and output layers

#For binary classification number of neuron in output layer =1, activation function in output layer=sigmoid
#For multiclass classification number of neurons in output layer=number of classes
#and activation function in output layer=softmax
#fOR BINARY CLASSIFICATION: LOSS FUNCTION = BINARY_CROSSENTROPY
#FOR MULTICLASS CLASSIFICATION:LOSS FUNCTION=CATEGORICAL CROSS ENTROPY

#Observation 10: Choose and report network architecture
mlp_classifier = models.Sequential(
    [
        keras.layers.Dense(4, activation="relu", input_shape=(X_train.shape[1],)),
        #keras.layers.Dense(4, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ]
)


#Observation 11: Report number oftrainable network parameters
mlp_classifier.summary()

#Observation 12:Initiate model, report model parameters chosen
#NOTE THAT HERE ALL MODEL PARAMETERS ARE CHOSEN BEFORE HAND AND RE NOT FOUND FROM DATA
scikeras_classifier = KerasClassifier(model=mlp_classifier,
                                      optimizer="adam",
                                      loss=keras.losses.binary_crossentropy,
                                      batch_size=8,
                                      epochs=100,
                                      verbose=0,
                                      validation_split=0.1
                                      )

#Observation 13:Train model using train data by calling train() function

scikeras_classifier.fit(X_train, Y_train)

#Observation 14: Plot loss versus epoch curve for train and validation data to study learning process

scikeras_classifier.history_.keys()
epochs=len(scikeras_classifier.history_["loss"])
plt.figure(1)
plt.plot(scikeras_classifier.history_["loss"],'-b', label='Train loss')
plt.plot(scikeras_classifier.history_["val_loss"],'--r', label='Val loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Observation 15: Obtain accuracy on train and test data using trained model
#Determine if the model is overfit, good fit or underfit

print("Train Accuracy : {:.2f}".format(scikeras_classifier.score(X_train, Y_train)))
print("Test  Accuracy : {:.2f}".format(scikeras_classifier.score(X_test, Y_test)))


#Observation 16: Compute and plot Confusion matrix
#Comment on classification performance of chosen model 

Y_preds = scikeras_classifier.predict(X_test)

cm = confusion_matrix(Y_test, Y_preds, labels=scikeras_classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=scikeras_classifier.classes_)
disp.plot()

plt.show()



################################################
#PART 2:Hyperparameter tuning using KerasClassifier from scikeras and GridSearchCV from sklearn


## Function to create model, required for KerasClassifier
#Observation 17:Choose network architecture, number of hidden layer, number of neurons in hidden layer,
#activation function in hidden layer and in output layer

def create_model():
 # create model
 model = keras.models.Sequential()
 model.add(keras.layers.Dense(4, input_shape=(X_train.shape[1],), activation='relu'))  #First hidden layer, 4=num of neuron
 model.add(keras.layers.Dense(1, activation='sigmoid'))     #output layer
 # Compile model
 model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 return model

#Observation18: Choose loss function
#create a  keras model for the Classification task and  wrap it inside of scikeras model
#loss function for binary classification= 'binary_crossentropy
model = KerasClassifier(model=create_model,loss='binary_crossentropy', verbose=1)

#In this work we are tuning hyperparameters batch_size and epochs
#Observation19: Note range of hyperparameters used for gridsearch
# define the grid search parameters
batch_size = [5,10]
epochs = [5,5]

#Create dictionary of hyperparameters and their values
param_grid = dict(batch_size=batch_size, epochs=epochs)

#Observation20:Create  an instance of GridSearchCV by giving it scikeras model and hyperparameters dictionary; choose k=5 for cross validation
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)


#Apply fit() on an instance of GridSearchCV which will perform 
#grid searchby trying different combinations of the  hyperparameters
#to find the combination which gives the best result.
grid_result = grid.fit(X_train,Y_train)

#Observation21:Note hyperparameters setting that gave the best 
#result along with  the best accuracy score

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']


#Compute predictions on Test data 

Y_preds = grid.predict(X_test)
#Observation 22:Evaluate Accuracy metric on both train and test datasets
#to check the performance of the model with the best hyperparameters setting.
print("Test  Accuracy : {:.2f}".format(grid.score(X_test, Y_test)))
print("Train Accuracy : {:.2f}".format(grid.score(X_train, Y_train)))

#Observation 23:Plot confusion matrix for Test data
cm = confusion_matrix(Y_test, Y_preds, labels=grid.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=grid.classes_)
disp.plot()

plt.show()

#############################################
#Part 3: Visit the site https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/ 
#learn at least 1 model parameter by grid search method
#The parameter can be:
    #number of neurons in hidden layer
    #type of optimizer algorithm
    #optimizer learning rate
    #activation function
    
#Observation 24: Note down the model parameter found by gridsearchcv
#and the accuracy of model on train and test data
# Use scikit-learn to grid search the batch size and epochs
def create_model():
 # create model
 model = keras.models.Sequential()
 model.add(keras.layers.Dense(4, input_shape=(X_train.shape[1],), activation='relu'))  #First hidden layer, 4=num of neuron
 model.add(keras.layers.Dense(1, activation='sigmoid'))     #output layer
 # Compile model
 model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 return model
# split into input (X) and output (Y) variables
X = dataset[:,0:6]
Y = dataset[:,6]
# create model
model = KerasClassifier(model=create_model, loss="binary_crossentropy", epochs=5, batch_size=5, verbose=0)
# define the grid search parameters
optimizer = ['SGD', 'RMSprop', 'Adam']
param_grid = dict(optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
    
print("Hello World")
print("ABC")
print("CDE")