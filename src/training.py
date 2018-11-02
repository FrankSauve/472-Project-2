import os
import pickle
from sklearn.naive_bayes import MultinomialNB


def save_classifier(classifier, filename):
    """
    Save the a classifier as a pickle file
    :param classifier: Classifier object
    :param filename: Name of the file to save
    """
    with open(os.path.abspath(os.path.join(os.getcwd(), "../pickles/", filename)), "wb") as pickle_file:
        pickle.dump(classifier, pickle_file)


def MNB_train(features, labels):
    """
    Use the Multinomial Naive Bayes classifier to train
    and saves the classifier as pickle file
    :param features: List of features from training set
    :param labels: List of labels from training set
    """
    MNB_Classifier = MultinomialNB()
    MNB_Classifier.fit(features, labels)
    save_classifier(MNB_Classifier, "MNB_Classifier.pkl")
