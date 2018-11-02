import training
import validation
from LogColors import LogColors
from processing import get_dataset

while True:
    # TODO: Ask which dataset to use.
    option = input(LogColors.HEADER + "Which action do you want to execute?\n" + LogColors.ENDC +
                                      "1. Train MNB\n"
                                      "2. Validate MNB\n"
                                      + LogColors.FAIL + "0. Exit\n" + LogColors.ENDC)

    if option == "1":
        # Get the training features and labels of dataset 1
        ds1_training_features, ds1_training_labels = get_dataset("ds1/ds1Train.csv")
        training.MNB_train(ds1_training_features, ds1_training_labels)

    elif option == "2":
        ds1_validation_features, ds1_validation_labels = get_dataset("ds1/ds1Val.csv")
        validation.MNB_classify(ds1_validation_features, ds1_validation_labels)

    elif option == "0":
        exit(0)

