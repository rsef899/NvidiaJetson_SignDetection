
import tensorflow as tf
import pickle
import math
import os
import numpy as np
import joblib
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from skimage.transform import resize
from sklearn.metrics import f1_score, precision_score,recall_score,confusion_matrix
from sklearn.model_selection import train_test_split
from skimage.io import imread
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import time

print(sklearn.__version__)
print(joblib.__version__)
datadir = "/Users/shaaranelango/Downloads/NvidiaJetson_SignDetection/Shaaran/dataset"
pickle_filepath = "/Users/shaaranelango/Downloads/NvidiaJetson_SignDetection/Shaaran/pickled.pickle"
i=0
dataGen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=False,vertical_flip=False,fill_mode="nearest")
# Load or initialize the processed images data
flat_data_arr = []
target_arr = []
model_filename = 'SVM.joblib'
categories = ['road','stop','speed','yellow','red','green']
# categories = ['yellow','speed','stop','red','green']

if os.path.exists(pickle_filepath):
    with open(pickle_filepath, "rb") as f:
        df = pickle.load(f)
        print(df)
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


print(df.shape)
# df contains the dataframe for the inputs
x=df.iloc[:,:-1] # Contains the rgb vals for each pixel
y=df.iloc[:,-1] # Contains the targets (what it is)

#x = MinMaxScaler().fit_transform(x) 
x_scaled = StandardScaler().fit_transform(x)

x = pd.DataFrame(x_scaled, columns=x.columns)

# Train:Test split of 85:15, I believe this is a substantial enough split such that we are unlikely to overtrain and we have enough data to test with.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5,stratify=y)
print(x_train)
# augmented_images = []
# augmented_labels = []
# num_augmentations = 2 # Number of augmented versions of this image
# for image, label in zip(x_train, y_train):
#     image = np.array(image)
#     image = image.reshape((1,) + image.shape)  # Add a batch dimension
#     for _ in range(num_augmentations):
#         augmented_image = dataGen.flow(image, batch_size=1).next()[0]
#         augmented_images.append(augmented_image)
#         augmented_labels.append(label)

# # Convert augmented data back to NumPy arrays
# x_augmented = np.array(augmented_images)
# y_augmented = np.array(augmented_labels)

# # Combine the original data and augmented data
# x_train_augmented = np.concatenate((x_train, x_augmented))
# y_train_augmented = np.concatenate((y_train, y_augmented))
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.70,random_state=3,stratify=y) # Using a very small training set (0.1) to get good hyperparams before doing the big training
# param_grid={'C':[0.1,1,10],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
# svc = svm.SVC(probability = True)
# print("The training of the SVC model is started, please wait for while as it may take few minutes to complete")
# svcModel=GridSearchCV(svc,param_grid)
# # After running for about 12 hours, the best params were given as: [C = 0.1, gamma = 0.0001, kernel = poly]
# svcModel.fit(x_train,y_train)
# print('The Model is trained well with the given parameters')
# print(svcModel.best_params_)

svcModel_filename = 'SVCModel.joblib'
if os.path.exists(svcModel_filename):
    print("Found Existing SVC")
    svcModel = joblib.load(svcModel_filename)
else:
    print("Training SVC")
    svcModel = svm.SVC(kernel='poly',gamma=0.0001,C=0.1,max_iter=1000,probability=False)
    ## Using original 85:15 train-test split
    svcModel.fit(x_train,y_train)
    joblib.dump(svcModel, svcModel_filename)

print(f"SVC Model score is: {svcModel.score(x_test,y_test)}")
svc_pred = svcModel.predict(x_test)

# Calculate the Precision, Recall and F1-score for the SVM model
prec = precision_score(y_test,svc_pred,average='weighted')
recall = recall_score(y_test,svc_pred,average='weighted')
f1 = f1_score(y_test,svc_pred,average='weighted')


#y_pred_labels = np.argmax(y_test, axis=1) # Grabbing the labels from the predictions
## Confusion Matrix (just for some testing)
confusion = confusion_matrix(y_test, svc_pred,labels=categories)
print(confusion)

print("Precision Score for SVM:",prec)
print("Recall Score for SVM:",recall)
print("F1 Score for SVM:",f1)


num_samples = 5  # Change this to the number of random elements you want to select

# Use sample method to select random rows
random_rows = x_test.sample(n=num_samples, random_state=48)

# Use loc to get the corresponding rows from y_test
random_labels = y_test.loc[random_rows.index]


## For testing ##
print(random_rows.shape)
print(random_labels.shape)
print(random_labels)

svcPred = svcModel.predict(random_rows)
print(svcPred)
images = ["blue.png","purple.png","speed.png","stop.png","yellow.png","hardTestRoad.jpg","testRoad.jpg","testStop.jpg"]
for image in images:

    print("PREDICTING SVC")
    start_time = time.time()
    svcPred = svcModel.predict(resize(imread(image),(112,112,3)).reshape(1,-1))
    end_time = time.time()


    print(svcPred)

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"Execution time: {elapsed_time} seconds")

