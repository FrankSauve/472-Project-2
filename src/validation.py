import os
import pickle
from LogColors import LogColors
from sklearn.metrics import accuracy_score
from processing import get_dataset


def load_classifier(filename):
    """
    Load a pickle file
    :param filename: Name of the pick file
    :return: The classifier saved in the pickle file
    """
    with open(os.path.abspath(os.path.join(os.getcwd(), "../pickles/", filename)), "rb") as pickle_file:
        classifier = pickle.load(pickle_file)
    return classifier


def DT_classify(features, labels):
    """
    Classify using the trained DT classifier
    :param features: List of features from validation set
    :param labels: List of labels from validation set
    :return:
    """
    DT_Classifier = load_classifier("DT_Classifier.pkl")

    ds1_validation_predicted = DT_Classifier.predict(features)

    accuracy = accuracy_score(labels, ds1_validation_predicted)

    print("Decision Tree accuracy: ", LogColors.OKBLUE, accuracy, LogColors.ENDC)


def MNB_classify(features, labels):
    """
    Classify using the trained MNB classifier
    :param features: List of features from validation set
    :param labels: List of labels from validation set
    :return:
    """
    MNB_Classifier = load_classifier("MNB_Classifier.pkl")

    ds1_validation_predicted = MNB_Classifier.predict(features)

    accuracy = accuracy_score(labels, ds1_validation_predicted)

    print("Multinomial Naive Bayes accuracy: ", LogColors.OKBLUE, accuracy, LogColors.ENDC)
