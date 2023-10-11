import pandas as pd
import os
import pickle
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
import joblib
import matplotlib.pyplot as plt

import mlp_model
import svm_model
import show_time


def plot_category_sizes(category_sizes):
    # plots the size of each category for data analysis
    bar_x_values = range(0, 43)
    plt.bar(bar_x_values, category_sizes)
    plt.xlabel("category #")
    plt.ylabel("category frequency")
    plt.show()


def read_categories():
    # reads all the categories to an array for analysis
    labelsFile = "Assignment-Dataset/labels.csv"
    df = pd.read_csv(labelsFile)
    return df.to_numpy()


def read_in_data(save_to_file=True, read_labels=True, num_categories=43):
    # first have a look at the data
    # and put it into a dataframe

    # initialize array that hold flattened, normalized image data
    # as well as array that holds target info to append to the above one
    flat_data_arr = []
    target_arr = []

    if read_labels:
        # this is for data analysis of the dataset
        category_sizes = []
        categoryArray = read_categories()

    # data will be found in this directory (is the renamed unzipped folder from kaggle - must be added in)
    dataDirectory = 'Assignment-Dataset/myData'

    # add in all files
    Category_numbers = range(0, num_categories)

    for i in Category_numbers:

        if read_labels:
            # change it to category name
            cat_name = categoryArray[i][1]
        else:
            cat_name = str(i);

        if read_labels:
            # initialize count of images in a category
            category_count = 0

        # display what category is being loaded for debugging
        print(f'loading... category ({i}/{num_categories-1}) :\t{cat_name}')
        path = os.path.join(dataDirectory, str(i))

        for img in os.listdir(path):
            img_array = imread(os.path.join(path, img))
            # resizing is important to ensure all images are 32 by 32 px
            # also, this normalizes the data for us
            img_resized = resize(img_array, (32, 32, 3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(i)
            if read_labels:
                category_count += 1

        # create the category count list as we load in the dataset
        if read_labels:
            category_sizes.append(category_count)
        print(f'finished loaded category : {cat_name}')

    # finish up the dataframe, adding the labels to the data as well
    flat_data = np.array(flat_data_arr)
    target = np.array(target_arr)
    df = pd.DataFrame(flat_data)
    df['Target'] = target
    print(df.shape)

    # only if we want to save to file, do this step
    if save_to_file:
        pickle.dump(df, open('data.pickle', 'wb'))
        if read_labels:
            pickle.dump(category_sizes, open('category_sizes.pickle', 'wb'))


def open_data(read_labels=True):
    if read_labels:
        return pickle.load(open('data.pickle', 'rb')), pickle.load(open('category_sizes.pickle', 'rb'))
    else:
        return pickle.load(open('data.pickle', 'rb'))


def quick_analysis(dataframe):
    # give a quick analysis of the data, to make sure we understand how it looks, and how big each image array is
    print("\nData Analysis:")
    print(f'shape: {dataframe.shape}')
    print(f"first 10 :\n{dataframe.head(10)}")
    offset = 10
    first_10_with_offset = dataframe.iloc[offset:offset + 10]
    print(f"random 10 :\n{first_10_with_offset}")
    print(f"")


def split_dataset(df_split, valid_size, test_size, do_print=False, do_save_file=False):
    # separate into values and labels
    x = df_split.drop(['Target'], axis=1).values
    y = df_split['Target'].values

    # split into training, testing and validation
    x_train_val, x_testing_split, y_train_val, y_testing_split = train_test_split(x, y,
                                                                                  test_size=test_size,
                                                                                  random_state=1)
    x_training_split, x_valid_split, y_training_split, y_valid_split = train_test_split(x_train_val, y_train_val,
                                                                                        test_size=valid_size,
                                                                                        random_state=1)

    # print out if we want to see the shape of our datasets for training / testing / validation
    if do_print:
        print(np.shape(x_training_split))
        print(np.shape(x_testing_split))
        print(np.shape(x_valid_split))
        print(np.shape(y_training_split))
        print(np.shape(y_testing_split))
        print(np.shape(y_valid_split))

    # save to file if we want
    if do_save_file:
        joblib.dump(x_training_split, "x_training.joblib")
        joblib.dump(x_testing_split, "x_testing.joblib")
        joblib.dump(x_valid_split, "x_valid.joblib")
        joblib.dump(y_training_split, "y_training.joblib")
        joblib.dump(y_testing_split, "y_testing.joblib")
        joblib.dump(y_valid_split, "y_valid.joblib")

    return x_training_split, x_testing_split, x_valid_split, y_training_split, y_testing_split, y_valid_split


def load_split_dataset():
    x_training_split = joblib.load("x_training.joblib")
    x_testing_split = joblib.load("x_testing.joblib")
    x_valid_split = joblib.load("x_valid.joblib")
    y_training_split = joblib.load("y_training.joblib")
    y_testing_split = joblib.load("y_testing.joblib")
    y_valid_split = joblib.load("y_valid.joblib")

    return x_training_split, x_testing_split, x_valid_split, y_training_split, y_testing_split, y_valid_split


# show the time taken for debugging purposes
show_time.print_time(True, True)

# get full dataset from file
# read_in_data(read_labels=False, num_categories=6)
df = open_data(read_labels=False)
# quick_analysis(df)

# plot bar graph of size of each category
# plot_category_sizes(category_sizes)

# split into training and testing and validation datasets, saving to file as well
# split_dataset(df, 0.2, 0.2, True, True)
x_training, x_testing, x_valid, y_training, y_testing, y_valid = load_split_dataset()

# try train the models
# mlp_model.fit_and_train_mlp_model(x_training, x_valid, y_training, y_valid, 0.01, 2500, True)
# svm_model.fit_and_train_svm_model(x_training, x_valid, y_training, y_valid, True)
# show_time.print_time(True, True)

# now that we have correct hyper-parameters, use testing dataset to check model
svm_model.validation(x_testing, y_testing)
svm_model.individual_test(x_testing, y_testing)
# svm_model.visual_all_test(x_testing, y_testing)
# mlp_model.validation(x_testing, y_testing)

show_time.print_time(False, True)
