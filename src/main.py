import training
import validation
from LogColors import LogColors
from processing import get_dataset

while True:
    # TODO: Ask which dataset to use.

    option1 = 1
    while option1 >= 0 and option1 <= 2:
        option1 = input(LogColors.HEADER + "Which dataset do you wish to use?\n" + LogColors.ENDC +
        "1. Dataset 1\n"
        "2. Dataset 2\n"
        + LogColors.FAIL + "0. Exit\n" + LogColors.ENDC)

    if option1 == "0":
        exit(0)

    option2 =1
    while option2 >=0 and option2 <=6:
        option2 = input(LogColors.HEADER + "Which action do you want to execute?\n" + LogColors.ENDC +
                                          "1. Train DT\n"
                                          "2. Validate DT\n"
                                          "3. Train MNB\n"
                                          "4. Validate MNB\n"
                                          + LogColors.FAIL + "0. Exit\n" + LogColors.ENDC)

    if option2 == "1":
        # Get the training features and labels of the dataset
        ds_training_features, ds_training_labels = get_dataset("ds"+option1+"/ds"+option1+"Train.csv")
        training.DT_train(ds_training_features, ds_training_labels)

    elif option2 == "2":
        ds_validation_features, ds_validation_labels = get_dataset("ds"+option1+"/ds"+option1+"Val.csv")
        validation.DT_classify(ds_validation_features, ds_validation_labels)

    elif option2 == "3":
        # Get the training features and labels of the dataset
        ds_training_features, ds_training_labels = get_dataset("ds"+option1+"/ds"+option1+"Train.csv")
        training.MNB_train(ds_training_features, ds_training_labels)

    elif option2 == "4":
        ds_validation_features, ds_validation_labels = get_dataset("ds"+option1+"/ds"+option1+"Val.csv")
        validation.MNB_classify(ds_validation_features, ds_validation_labels)
    elif option2 == "0":
        exit(0)

