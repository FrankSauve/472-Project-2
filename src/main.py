import training
import validation
from LogColors import LogColors
from processing import get_dataset

while True:
    # Choose which dataset
    option1 = 1
    while option1 >= 0 and option1 <= 2:
        option1 = input(LogColors.HEADER + "Which dataset do you wish to use?\n" + LogColors.ENDC +
        "1. Dataset 1\n"
        "2. Dataset 2\n"
        + LogColors.FAIL + "0. Exit\n" + LogColors.ENDC)

    if option1 == "0":
        exit(0)

# TODO: ADD MORE NUMBERS TO OPTION2

    # Choose which ML alg to use
    option2 =1
    while option2 >=0 and option2 <=6:
        option2 = input(LogColors.HEADER + "Which action do you want to execute?\n" + LogColors.ENDC +
                                          "1. Train DT\n"
                                          "2. Validate DT\n"
                                          "3. Train MNB\n"
                                          "4. Validate MNB\n"
                                          + LogColors.FAIL + "0. Exit\n" + LogColors.ENDC)

    if option2 == "0":
        exit(0)

    ds_training_features = None
    ds_training_labels = None
    ds_validation_features = None
    ds_validation_labels = None

    if option2 % 2 == 1:
        ds_training_features, ds_training_labels = get_dataset("ds" + option1 + "/ds" + option1 + "Train.csv")
    else
        ds_training_features, ds_training_labels = get_dataset("ds" + option1 + "/ds" + option1 + "Val.csv")

    if option2 == "1":
        training.DT_train(ds_training_features, ds_training_labels)
    elif option2 == "3":
        training.MNB_train(ds_training_features, ds_training_labels)
    elif option2 == "5":
        training.BNB_train(ds_training_features, ds_training_labels)
    elif option2 == "7":
        training.GNB_train(ds_training_features, ds_training_labels)
    elif option2 == "9":
        training.CNB_train(ds_training_features, ds_training_labels)

    elif option2 == "2":
        validation.classify(ds_validation_features, ds_validation_labels, "DT")
    elif option2 == "4":
        validation.classify(ds_validation_features, ds_validation_labels, "MNB")
    elif option2 == "6":
        validation.classify(ds_validation_features, ds_validation_labels, "BNB")
    elif option2 == "8":
        validation.classify(ds_validation_features, ds_validation_labels, "GNB")
    elif option2 == "10":
        validation.classify(ds_validation_features, ds_validation_labels, "CNB")

