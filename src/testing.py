import os
from LogColors import LogColors
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib


def load_classifier(filename, ds):
    """
    Load a pickle file
    :param filename: Name of the pick file
    :param ds: Number of the dataset
    :return: The classifier saved in the pickle file
    """
    with open(os.path.abspath(os.path.join(os.getcwd(), "../joblibs/ds" + ds + filename)), "rb") as joblib_file:
        classifier = joblib.load(joblib_file)
    return classifier


def classify(features, ds, type):
    """
    Classify using the trained classifier
    :param features: List of features from validation set
    :param labels: List of labels from validation set
    :param ds: Number of the dataset
    :param type: Name of the classifier
    """
    classifier = load_classifier(type+"_Classifier.pkl", ds)

    ds_test_predicted = classifier.predict(features)

    try:
        output = open(os.path.dirname(__file__) + "/../output/ds" + ds + "Test-" + type + ".txt", "w+")

        i = 0
        output_text = ""
        while i < len(ds_test_predicted):
            output_text += str(i + 1) + ", " + str(ds_test_predicted[i]) + "\n"
            i += 1

        output.write(output_text)
    except Exception:
        print(LogColors.FAIL + "🛑 File IO exception" + LogColors.ENDC)
        exit(0)

