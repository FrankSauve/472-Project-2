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

def classify(features, labels, type):
    """
    Classify using the trained classifier
    :param features: List of features from validation set
    :param labels: List of labels from validation set
    :return:
    """
    classifier = load_classifier(type+"_Classifier.pkl")

    ds_validation_predicted = classifier.predict(features)

    accuracy = accuracy_score(labels, ds_validation_predicted)

    print(type+" accuracy: ", LogColors.OKBLUE, accuracy, LogColors.ENDC)



# TODO: Remove obsolete methods
def DT_classify(features, labels):
    """
    Classify using the trained DT classifier
    :param features: List of features from validation set
    :param labels: List of labels from validation set
    :return:
    """
    DT_Classifier = load_classifier("DT_Classifier.pkl")

    ds_validation_predicted = DT_Classifier.predict(features)

    accuracy = accuracy_score(labels, ds_validation_predicted)

    print("Decision Tree accuracy: ", LogColors.OKBLUE, accuracy, LogColors.ENDC)


def MNB_classify(features, labels):
    """
    Classify using the trained MNB classifier
    :param features: List of features from validation set
    :param labels: List of labels from validation set
    :return:
    """
    MNB_Classifier = load_classifier("MNB_Classifier.pkl")

    ds_validation_predicted = MNB_Classifier.predict(features)

    accuracy = accuracy_score(labels, ds_validation_predicted)

    print("Multinomial Naive Bayes accuracy: ", LogColors.OKBLUE, accuracy, LogColors.ENDC)


def BNB_classify(features, labels):
    """
    Classify using the trained BNB classifier
    :param features: List of features from validation set
    :param labels: List of labels from validation set
    :return:
    """
    BNB_Classifier = load_classifier("BNB_Classifier.pkl")

    ds_validation_predicted = BNB_Classifier.predict(features)

    accuracy = accuracy_score(labels, ds_validation_predicted)

    print("Bernoulli Naive Bayes accuracy: ", LogColors.OKBLUE, accuracy, LogColors.ENDC)


def GNB_classify(features, labels):
    """
    Classify using the trained GNB classifier
    :param features: List of features from validation set
    :param labels: List of labels from validation set
    :return:
    """
    GNB_Classifier = load_classifier("GNB_Classifier.pkl")

    ds_validation_predicted = GNB_Classifier.predict(features)

    accuracy = accuracy_score(labels, ds_validation_predicted)

    print("Gaussian Naive Bayes accuracy: ", LogColors.OKBLUE, accuracy, LogColors.ENDC)
