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

def classify(features, labels, ds, type):
    """
    Classify using the trained classifier
    :param features: List of features from validation set
    :param labels: List of labels from validation set
    :return:
    """
    classifier = load_classifier(type+"_Classifier.pkl")

    ds_validation_predicted = classifier.predict(features)

    accuracy = accuracy_score(labels, ds_validation_predicted)

    try:
        output = open(os.path.dirname(__file__) + "/../output/ds" + ds + "Test-" + type + ".txt", "w+")

        i = 0
        output_text = ""
        while i < len(ds_validation_predicted):
            output_text += str(i + 1) + ", " + str(ds_validation_predicted[i]) + "\n"
            i += 1

        output.write(output_text)
    except Exception:
        print(LogColors.FAIL + "🛑 File IO exception" + LogColors.ENDC)
        exit(0)

    print(type+" accuracy: ", LogColors.OKBLUE, accuracy, LogColors.ENDC)
    return accuracy
