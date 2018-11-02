import os
import pickle
from LogColors import LogColors
from sklearn.metrics import accuracy_score
from processing import get_dataset


def load_classifier(filename):
    with open(os.path.abspath(os.path.join(os.getcwd(), "../pickles/", filename)), "rb") as pickle_file:
        classifier = pickle.load(pickle_file)
    return classifier


def MNB_classify():
    MNB_Classifier = load_classifier("MNB_Classifier.pkl")
    ds1_validation_features, ds1_validation_labels = get_dataset("ds1/ds1Val.csv")

    ds1_validation_predicted = MNB_Classifier.predict(ds1_validation_features)

    accuracy = accuracy_score(ds1_validation_labels, ds1_validation_predicted)

    print("Multinomial Naive Bayes accuracy: ", LogColors.OKBLUE, accuracy, LogColors.ENDC)
