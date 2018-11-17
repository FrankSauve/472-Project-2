import os
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
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
    with open(os.path.abspath(os.path.join(os.getcwd(), "../joblibs/", filename)), "wb") as joblib_file:
        joblib.dump(classifier, joblib_file)


def DT_train(features, labels, ds):
    """
    Use the Decision tree classifier to train
    and saves the classifier as pickle file
    :param features: List of features from training set
    :param labels: List of labels from training set
    :param ds: Number of the dataset
    """
    DT_Classifier = tree.DecisionTreeClassifier(criterion='entropy')
    DT_Classifier.fit(features, labels)
    save_classifier(DT_Classifier, "ds" + ds + "DT_Classifier.pkl")


def MNB_train(features, labels, ds):
    """
    Use the Multinomial Naive Bayes classifier to train
    and saves the classifier as pickle file
    :param features: List of features from training set
    :param labels: List of labels from training set
    :param ds: Number of the dataset
    """
    MNB_Classifier = MultinomialNB(alpha=10.0)
    MNB_Classifier.fit(features, labels)
    save_classifier(MNB_Classifier, "ds" + ds + "MNB_Classifier.pkl")


def BNB_train(features, labels, ds):
    """
    Use the Bernoulli Naive Bayes classifier to train
    and saves the classifier as pickle file
    :param features: List of features from training set
    :param labels: List of labels from training set
    :param ds: Number of the dataset
    """
    BNB_Classifier = BernoulliNB(alpha=0.5)
    BNB_Classifier.fit(features, labels)
    save_classifier(BNB_Classifier, "ds" + ds + "BNB_Classifier.pkl")


def GNB_train(features, labels, ds):
    """
    Use the Gaussian Naive Bayes classifier to train
    and saves the classifier as pickle file
    :param features: List of features from training set
    :param labels: List of labels from training set
    :param ds: Number of the dataset
    """
    GNB_Classifier = GaussianNB(var_smoothing=1e-01)
    GNB_Classifier.fit(features, labels)
    save_classifier(GNB_Classifier, "ds" + ds + "GNB_Classifier.pkl")


def CNB_train(features, labels, ds):
    """
    Use the Complement Naive Bayes classifier to train
    and saves the classifier as pickle file
    :param features: List of features from training set
    :param labels: List of labels from training set
    :param ds: Number of the dataset
    """
    CNB_Classifier = ComplementNB(alpha=10.0)
    CNB_Classifier.fit(features, labels)
    save_classifier(CNB_Classifier, "ds" + ds + "CNB_Classifier.pkl")


def MLP_NN_train(features, labels, ds):
    """
    Use the Multi-layer Perceptron Neural Network classifier to train
    and saves the classifier as pickle file
    :param features: List of features from training set
    :param labels: List of labels from training set
    :param ds: Number of the dataset
    """

    MLP_NN_Classifier = MLPClassifier(batch_size=500)
    MLP_NN_Classifier.fit(features, labels)
    save_classifier(MLP_NN_Classifier, "ds" + ds + "MLP_NN_Classifier.pkl")

