import joblib
from skimage.feature import hog
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

import show_time


def save_mlp_model(model):
    joblib.dump(model, 'mlp_model.joblib')


def load_mlp_model():
    return joblib.load('mlp_model.joblib')


def fit_and_train_mlp_model(x_training, x_valid, y_training, y_valid, learning_rate, iterations, save_model=False):
    # we only use hog cos some colors are really off, might not be useful
    hog_features_training = []
    hog_features_valid = []

    # note the first value here are dependent on the splits
    # also, this is just so we can get HOG from the training set and validation set
    x_training_not_flat = x_training.reshape(52660, 32, 32, 3)
    for image in x_training_not_flat:
        hog_features = hog(image, orientations=8, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), channel_axis=-1)
        hog_features_training.append(hog_features)

    x_valid_not_flat = x_valid.reshape(13165, 32, 32, 3)
    for image in x_valid_not_flat:
        hog_features = hog(image, orientations=8, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), channel_axis=-1)
        hog_features_valid.append(hog_features)

    show_time.print_time(False, True)
    print("(HOG finished)")
    show_time.print_time(True, True)

    # train the model on hog values
    model = MLPClassifier(hidden_layer_sizes=(144, 77), random_state=1, learning_rate_init=learning_rate,
                          max_iter=iterations, early_stopping=True, learning_rate='adaptive')
    model.fit(hog_features_training, y_training)

    # now validate it
    y_pred = model.predict(hog_features_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    precision = precision_score(y_valid, y_pred, average='micro')
    print(f"Accuracy Score: {accuracy * 100}%")
    print(f"Precision Score: {precision * 100}%")
    print(classification_report(y_true=y_valid, y_pred=y_pred))

    # save the model to joblib file if we want to
    if save_model:
        save_mlp_model(model)


def validation(x_testing, y_testing):

    print("\nMLP testing info:\n")

    # check model based off of testing params
    model = load_mlp_model()

    # hog the model
    hog_features_testing = []

    x_testing_not_flat = x_testing.reshape(7314, 32, 32, 3)
    for image in x_testing_not_flat:
        hog_features = hog(image, orientations=8, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), channel_axis=-1)
        hog_features_testing.append(hog_features)

    show_time.print_time(False, True)
    print("(HOG finished)")
    show_time.print_time(True, True)

    # show validation results
    y_pred = model.predict(hog_features_testing)
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
