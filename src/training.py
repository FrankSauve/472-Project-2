import os
import pickle
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, ComplementNB
from sklearn.neural_network import MLPClassifier
import warnings


def warn(*args, **kwargs):
    pass
warnings.warn = warn

def save_classifier(classifier, filename):
    """
    Save the a classifier as a pickle file
    :param classifier: Classifier object
    :param filename: Name of the file to save
    """
    with open(os.path.abspath(os.path.join(os.getcwd(), "../pickles/", filename)), "wb") as pickle_file:
        pickle.dump(classifier, pickle_file)


def DT_train(features, labels):
    """
    Use the Decision tree classifier to train
    and saves the classifier as pickle file
    :param features: List of features from training set
    :param labels: List of labels from training set
    """
    DT_Classifier = tree.DecisionTreeClassifier(criterion='entropy')
    DT_Classifier.fit(features, labels)
    save_classifier(DT_Classifier, "DT_Classifier.pkl")


def MNB_train(features, labels):
    """
    Use the Multinomial Naive Bayes classifier to train
    and saves the classifier as pickle file
    :param features: List of features from training set
    :param labels: List of labels from training set
    """
    MNB_Classifier = MultinomialNB(alpha=10.0)
    MNB_Classifier.fit(features, labels)
    save_classifier(MNB_Classifier, "MNB_Classifier.pkl")

def BNB_train(features, labels):
    """
    Use the Bernoulli Naive Bayes classifier to train
    and saves the classifier as pickle file
    :param features: List of features from training set
    :param labels: List of labels from training set
    """
    BNB_Classifier = BernoulliNB(alpha=0.5)
    BNB_Classifier.fit(features, labels)
    save_classifier(BNB_Classifier, "BNB_Classifier.pkl")

def GNB_train(features, labels):
    """
    Use the Gaussian Naive Bayes classifier to train
    and saves the classifier as pickle file
    :param features: List of features from training set
    :param labels: List of labels from training set
    """
    GNB_Classifier = GaussianNB(var_smoothing=1e-01)
    GNB_Classifier.fit(features, labels)
    save_classifier(GNB_Classifier, "GNB_Classifier.pkl")

def CNB_train(features, labels):
    """
    Use the Complement Naive Bayes classifier to train
    and saves the classifier as pickle file
    :param features: List of features from training set
    :param labels: List of labels from training set
    """
    CNB_Classifier = ComplementNB(alpha=10.0)
    CNB_Classifier.fit(features, labels)
    save_classifier(CNB_Classifier, "CNB_Classifier.pkl")

def MLP_NN_train(features, labels):
    """
    Use the Multi-layer Perceptron Neural Network classifier to train
    and saves the classifier as pickle file
    :param features: List of features from training set
    :param labels: List of labels from training set
    """

    MLP_NN_Classifier = MLPClassifier(batch_size=500)
    MLP_NN_Classifier.fit(features, labels)
    save_classifier(MLP_NN_Classifier, "MLP_NN_Classifier.pkl")

