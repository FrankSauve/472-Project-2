import os
import pickle
from sklearn.naive_bayes import MultinomialNB
from processing import get_dataset


def save_classifier(classifier, filename):
    with open(os.path.abspath(os.path.join(os.getcwd(), "../pickles/", filename)), "wb") as pickle_file:
        pickle.dump(classifier, pickle_file)


ds1_training_features, ds1_training_labels = get_dataset("ds1/ds1Train.csv")


def MNB_train():
    MNB_Classifier = MultinomialNB()
    MNB_Classifier.fit(ds1_training_features, ds1_training_labels)
    save_classifier(MNB_Classifier, "MNB_Classifier.pkl")
