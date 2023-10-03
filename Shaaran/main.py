import tensorflow as tf
import pickle
import math
import os
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from skimage.transform import resize
from sklearn.metrics import f1_score, precision_score,recall_score,confusion_matrix
from sklearn.model_selection import train_test_split
from skimage.io import imread
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import time

datadir = "C:\\Users\\shaar\\OneDrive\\Documents\\NvidiaJetson_SignDetection\\Shaaran\\dataset"
pickle_filepath = "C:\\Users\\shaar\\OneDrive\\Documents\\NvidiaJetson_SignDetection\\Shaaran\\pickled.pickle"
i=0

# Load or initialize the processed images data
flat_data_arr = []
target_arr = []
model_filename = 'SVM.joblib'
categories = ['road','stop']

if os.path.exists(pickle_filepath):
    with open(pickle_filepath, "rb") as f:
        df = pickle.load(f)
else:

    for category in categories:
        print(f'loading... category: {category}')
        path = os.path.join(datadir,str(category))
        for img in os.listdir(path):
        # Read and load the image into an array.
            img_array=imread(os.path.join(path,img))
        # Resize the image to size of 32x32 pixels with 3 slots for the RGB values
            img_resized=resize(img_array,(112,112,3))
        # Flatten the resized image into a 1D array and append it to the flat_data_arr list.
            flat_data_arr.append(img_resized.flatten())
        # Appending the 'target' values (what the image actually is)
            target_arr.append(category)
        print(f'loaded category:{category} successfully')
    flat_data=np.array(flat_data_arr)
    target=np.array(target_arr)



    df = pd.DataFrame(flat_data)
    df['Target'] = target
    with open(pickle_filepath, 'wb') as file:
        pickle.dump(df, file)

    print(f'Data saved to {pickle_filepath}')



# df contains the dataframe for the inputs
x=df.iloc[:,:-1] # Contains the rgb vals for each pixel
y=df.iloc[:,-1] # Contains the targets (what it is)

# Train:Test split of 85:15, I believe this is a substantial enough split such that we are unlikely to overtrain and we have enough data to test with.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=3,stratify=y)


svcModel_filename = 'SVCModel.joblib'
if os.path.exists(svcModel_filename):
    print("Found Existing SVC")
    svcModel = joblib.load(svcModel_filename)
else:
    print("Training SVC")
    svcModel = svm.SVC(kernel='rbf',gamma=0.001,C=10,max_iter=100000000)
    ## Using original 85:15 train-test split
    svcModel.fit(x_train,y_train)
    joblib.dump(svcModel, svcModel_filename)

print(f"SVC Model score is: {svcModel.score(x_test,y_test)}")
svc_pred = svcModel.predict(x_test)

# Calculate the Precision, Recall and F1-score for the SVM model
prec = precision_score(y_test,svc_pred,average='weighted')
recall = recall_score(y_test,svc_pred,average='weighted')
f1 = f1_score(y_test,svc_pred,average='weighted')

print("Precision Score for SVM:",prec)
print("Recall Score for SVM:",recall)
print("F1 Score for SVM:",f1)


# num_samples = 1  # Change this to the number of random elements you want to select

# # Use sample method to select random rows
# random_rows = x_test.sample(n=num_samples, random_state=48)

# # Use loc to get the corresponding rows from y_test
# random_labels = y_test.loc[random_rows.index]


# ## For testing ##
# print(random_rows.shape)
# print(random_labels.shape)
# print(random_labels)
images = ["testStop.jpg","testRoad.jpg","hardTestRoad.jpg"]

for image in images:

    print("PREDICTING SVC")
    start_time = time.time()
    svcPred = svcModel.predict(resize(imread(image),(112,112,3)).reshape(1,-1))
    end_time = time.time()


    print(svcPred)

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"Execution time: {elapsed_time} seconds")

