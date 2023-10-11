import joblib
from sklearn import svm
import numpy as np
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import cv2

import show_time

ORIENTATIONS = 8
PIXELS_PER_CELL = 8
CELLS_PER_BLOCK = 2

# don't change either of these
IMAGE_SIZE = 32
COLOR_PARTITIONS = 8
PARTITION_SIZE = IMAGE_SIZE // COLOR_PARTITIONS

def save_svm_model(model):
    joblib.dump(model, 'svm_model.joblib')


def load_svm_model():
    return joblib.load('svm_model.joblib')


def plot_testing_image(image, color_features):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Image 1')

    plt.subplot(1, 2, 2)
    plt.imshow(color_features, cmap='gray')
    plt.title('Image 2')

    plt.show()


def fit_and_train_svm_model(x_training, x_valid, y_training, y_valid, save_model=False):
    # we only use hog cos some colors are really off, might not be useful
    hog_features_training = []
    hog_features_valid = []
    color_features_training = []
    color_features_valid = []

    # apply hog on the data to get features
    x_training_not_flat = x_training.reshape(786, 32, 32, 3)
    for image in x_training_not_flat:
        hog_features = hog(image, orientations=ORIENTATIONS, pixels_per_cell=(PIXELS_PER_CELL, PIXELS_PER_CELL),
                           cells_per_block=(CELLS_PER_BLOCK, CELLS_PER_BLOCK), channel_axis=-1)
        hog_features_training.append(hog_features)

        # note that we are removing the borders of the pic (so its the center 24x24)
        # add the color_feature
        color_features = np.zeros((COLOR_PARTITIONS-2,COLOR_PARTITIONS-2,3))

        # try add those color params
        for i in range(0+1,COLOR_PARTITIONS-1):
            for j in range(0+1,COLOR_PARTITIONS-1):
                
                # find the mean color over this 4x4 pixel region
                color_features[i-1,j-1,0] = np.mean(image[i*PARTITION_SIZE:(i+1)*PARTITION_SIZE, j*PARTITION_SIZE:(j+1)*PARTITION_SIZE, 0])
                color_features[i-1,j-1,1] = np.mean(image[i*PARTITION_SIZE:(i+1)*PARTITION_SIZE, j*PARTITION_SIZE:(j+1)*PARTITION_SIZE, 1])
                color_features[i-1,j-1,2] = np.mean(image[i*PARTITION_SIZE:(i+1)*PARTITION_SIZE, j*PARTITION_SIZE:(j+1)*PARTITION_SIZE, 2])

        # make sure to normalize it
        color_features_norm = (color_features - color_features.min()) / (color_features.max() - color_features.min())
        color_features_training.append(color_features_norm)

        # only uncomment if we want to:
        # plot_testing_image(image, color_features)

    color_features_training_flat = np.array(color_features_training).reshape(786, 108)

    x_valid_not_flat = x_valid.reshape(197, 32, 32, 3)
    for image in x_valid_not_flat:
        hog_features = hog(image, orientations=ORIENTATIONS, pixels_per_cell=(PIXELS_PER_CELL, PIXELS_PER_CELL),
                           cells_per_block=(CELLS_PER_BLOCK, CELLS_PER_BLOCK), channel_axis=-1)
        hog_features_valid.append(hog_features)

        # note that we are removing the borders of the pic (so its the center 24x24)
        # add the color_feature
        color_features = np.zeros((COLOR_PARTITIONS-2,COLOR_PARTITIONS-2,3))

        # try add those color params
        for i in range(0+1,COLOR_PARTITIONS-1):
            for j in range(0+1,COLOR_PARTITIONS-1):
                
                # find the mean color over this 4x4 pixel region
                color_features[i-1,j-1,0] = np.mean(image[i*PARTITION_SIZE:(i+1)*PARTITION_SIZE, j*PARTITION_SIZE:(j+1)*PARTITION_SIZE, 0])
                color_features[i-1,j-1,1] = np.mean(image[i*PARTITION_SIZE:(i+1)*PARTITION_SIZE, j*PARTITION_SIZE:(j+1)*PARTITION_SIZE, 1])
                color_features[i-1,j-1,2] = np.mean(image[i*PARTITION_SIZE:(i+1)*PARTITION_SIZE, j*PARTITION_SIZE:(j+1)*PARTITION_SIZE, 2])

        # make sure to normalize it
        color_features_norm = (color_features - color_features.min()) / (color_features.max() - color_features.min())
        color_features_valid.append(color_features_norm)

        # only uncomment if we want to:
        # plot_testing_image(image, color_features)

    color_features_valid_flat = np.array(color_features_valid).reshape(197, 108)

    show_time.print_time(False, True)
    print("(HOG finished)")
    show_time.print_time(True, True)

    param_grid = {
    'C': [0.1, 0.5, 1, 3, 5, 7, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': [0.0001, 0.01, 0.1, 0.7, 1, 10],
    'max_iter': [1400, 2000, 10000, 100000000]
    }

    # from the param grid, my best hyperparams are C:0.1, gamma:1, kernel:poly, max_iter:1400

    #my_svc = svm.SVC()

    #gridsearch = GridSearchCV(estimator=my_svc, param_grid=param_grid, cv=5)
    #gridsearch.fit(np.concatenate((hog_features_training, color_features_training_flat), axis=1), y_training)
    #print(gridsearch.best_params_)
    #model = gridsearch.best_estimator_

    # train the model
    model = svm.SVC(kernel='rbf', gamma=0.1, C=1, max_iter=1400, probability=True)
    model.fit(np.concatenate((hog_features_training, color_features_training_flat), axis=1), y_training)

    # do validation on the current params
    y_pred = model.predict(np.concatenate((hog_features_valid, color_features_valid_flat), axis=1))
    accuracy = accuracy_score(y_valid, y_pred)
    precision = precision_score(y_valid, y_pred, average='micro')
    score = f1_score(y_valid, y_pred, average="micro")
    print(f"Accuracy Score: {accuracy * 100}%")
    print(f"Precision Score: {precision * 100}%")
    print(f"F1 Score: {score * 100}%")
    print(confusion_matrix(y_valid, y_pred))

    if save_model:
        save_svm_model(model)


def validation(x_testing, y_testing):
    print("\nSVM testing info:\n")

    # check model based off of testing params
    model = load_svm_model()

    # hog the model
    hog_features_testing = []
    color_features_testing = []

    x_testing_not_flat = x_testing.reshape(246, 32, 32, 3)
    for image in x_testing_not_flat:
        hog_features = hog(image, orientations=ORIENTATIONS, pixels_per_cell=(PIXELS_PER_CELL, PIXELS_PER_CELL),
                           cells_per_block=(CELLS_PER_BLOCK, CELLS_PER_BLOCK), channel_axis=-1)
        hog_features_testing.append(hog_features)

        # note that we are removing the borders of the pic (so its the center 24x24)
        # add the color_feature
        color_features = np.zeros((COLOR_PARTITIONS-2,COLOR_PARTITIONS-2,3))

        # try add those color params
        for i in range(0+1,COLOR_PARTITIONS-1):
            for j in range(0+1,COLOR_PARTITIONS-1):
                
                # find the mean color over this 4x4 pixel region
                color_features[i-1,j-1,0] = np.mean(image[i*PARTITION_SIZE:(i+1)*PARTITION_SIZE, j*PARTITION_SIZE:(j+1)*PARTITION_SIZE, 0])
                color_features[i-1,j-1,1] = np.mean(image[i*PARTITION_SIZE:(i+1)*PARTITION_SIZE, j*PARTITION_SIZE:(j+1)*PARTITION_SIZE, 1])
                color_features[i-1,j-1,2] = np.mean(image[i*PARTITION_SIZE:(i+1)*PARTITION_SIZE, j*PARTITION_SIZE:(j+1)*PARTITION_SIZE, 2])

        # make sure to normalize it
        color_features_norm = (color_features - color_features.min()) / (color_features.max() - color_features.min())
        color_features_testing.append(color_features_norm)

    color_features_testing_flat = np.array(color_features_testing).reshape(246, 108)

    show_time.print_time(False, True)
    print("(HOG finished)")
    show_time.print_time(True, True)

    # test it by predicting
    y_pred = model.predict(np.concatenate((hog_features_testing, color_features_testing_flat), axis=1))
    accuracy = accuracy_score(y_testing, y_pred)
    precision = precision_score(y_testing, y_pred, average='macro')
    recall = recall_score(y_testing, y_pred, average='macro')
    score = f1_score(y_true=y_testing, y_pred=y_pred, average="macro")
    print(f"Accuracy Score: {accuracy * 100}%")
    print(f"Precision Score: {precision * 100}%")
    print(f"Recall Score: {recall * 100}%")
    print(f"F1 Score: {score * 100}%")
    print(classification_report(y_true=y_testing, y_pred=y_pred))
    print(confusion_matrix(y_testing, y_pred))


def individual_test(x_testing, y_testing):
    model = load_svm_model()

    # show the guess and actual for an image, to check if we are guessing correctly

    for img_num in range(0,240):
        image_flat = x_testing[img_num, :]
        image = np.array(image_flat).reshape(32, 32, 3)
        hog_features = hog(image, orientations=ORIENTATIONS, pixels_per_cell=(PIXELS_PER_CELL, PIXELS_PER_CELL),
                           cells_per_block=(CELLS_PER_BLOCK, CELLS_PER_BLOCK), channel_axis=-1)
        
        # note that we are removing the borders of the pic (so its the center 24x24)
        # add the color_feature
        color_features = np.zeros((COLOR_PARTITIONS-2,COLOR_PARTITIONS-2,3))

        # try add those color params
        for i in range(0+1,COLOR_PARTITIONS-1):
            for j in range(0+1,COLOR_PARTITIONS-1):
                
                # find the mean color over this 4x4 pixel region
                color_features[i-1,j-1,0] = np.mean(image[i*PARTITION_SIZE:(i+1)*PARTITION_SIZE, j*PARTITION_SIZE:(j+1)*PARTITION_SIZE, 0])
                color_features[i-1,j-1,1] = np.mean(image[i*PARTITION_SIZE:(i+1)*PARTITION_SIZE, j*PARTITION_SIZE:(j+1)*PARTITION_SIZE, 1])
                color_features[i-1,j-1,2] = np.mean(image[i*PARTITION_SIZE:(i+1)*PARTITION_SIZE, j*PARTITION_SIZE:(j+1)*PARTITION_SIZE, 2])

        # make sure to normalize it
        color_features_norm = (color_features - color_features.min()) / (color_features.max() - color_features.min())

        feature_vector = np.concatenate((hog_features, color_features_norm.reshape(-1)), axis=0).reshape(1, -1)

        print(f"prediction: {model.predict(feature_vector)[0]}")
        print(f"actual: {y_testing[img_num]}")

        plt.imshow(image)
        plt.show()


def visual_all_test(x_testing, y_testing):
    model = load_svm_model()

    images = np.array(x_testing).reshape(246,32,32,3)

    # Define the number of rows and columns for subplots
    num_rows, num_cols = 7, 8  # Assuming you want 7 rows and 8 columns of images

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    # Iterate through the images and display them in subplots
    for i in range(num_rows):
        for j in range(num_cols):
            img_index = i * num_cols + j
            if img_index < images.shape[0]:

                image = images[i*8 + j, :]
                hog_features = hog(image, orientations=ORIENTATIONS, pixels_per_cell=(PIXELS_PER_CELL, PIXELS_PER_CELL),
                                   cells_per_block=(CELLS_PER_BLOCK, CELLS_PER_BLOCK), channel_axis=-1)

                title = f"prediction: {model.predict(np.array(hog_features).reshape(1, -1))[0]} " \
                        f"actual: {y_testing[i*8 + j]}"

                # Display the image in the current subplot
                axes[i, j].imshow(images[img_index])
                axes[i, j].axis('off')  # Turn off axis for better visualization
                axes[i, j].set_title(title)

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the subplots
    plt.show()