import training
import validation
from LogColors import LogColors
from processing import get_dataset

def exitOnZero(check):
    if check == 0:
        exit(check)

while True:
    # Choose which dataset
    option1 = ""
    validOption = False
    while not validOption:
        option1 = input(LogColors.HEADER + "Which dataset do you wish to use?\n" + LogColors.ENDC +
        "1. Dataset 1\n"
        "2. Dataset 2\n"
        + LogColors.FAIL + "0. Exit\n" + LogColors.ENDC)

        if (int(option1) >= 0) and (int(option1) <= 2):
            validOption = True

    exitOnZero(int(option1))

    # Choose which ML alg to use
    option2 = ""
    validOption = False
    while not validOption:
        option2 = input(LogColors.HEADER + "Which action do you want to execute?\n" + LogColors.ENDC +
                                          "1. Train DT\n"
                                          "2. Validate DT\n"
                                          "3. Train NB\n"
                                          "4. Validate NB\n"
                                          "5. Train MLP NN\n"
                                          "6. Validate MLP NN\n"
                                          + LogColors.FAIL + "0. Exit\n" + LogColors.ENDC)
        if (int(option2) >=0) and (int(option2) <= 6):
            validOption = True

    exitOnZero(int(option2))

    # Choose which NB alg to use if nb was chosen
    if(option2 == "3") or (option2 == "4"):
        option3 = ""
        validOption = False
        while not validOption:
            option3 = input(LogColors.HEADER + "Which type of Naïve-Bayes do you want to " +
                            ("train" if (option2 == "3") else "validate") + "?\n" + LogColors.ENDC +
                            "1. Multinomial NB\n"
                            "2. Bernoulli NB\n"
                            "3. Gaussian NB\n"
                            "4. Complement NB\n"
                            + LogColors.FAIL + "0. Exit\n" + LogColors.ENDC)
            if (int(option3) >= 0) and (int(option3) <= 3):
                validOption = True

        exitOnZero(int(option3))

    # Training
    if int(option2) % 2 == 1:
        print(LogColors.OKGREEN + "Training..." + LogColors.ENDC)

        ds_training_features, ds_training_labels = get_dataset("ds" + option1 + "/ds" + option1 + "Train.csv")

        if option2 == "1":
            training.DT_train(ds_training_features, ds_training_labels)
        elif option2 == "3":
            if option3 == "1":
                training.MNB_train(ds_training_features, ds_training_labels)
            elif option3 == "2":
                training.BNB_train(ds_training_features, ds_training_labels)
            elif option3 == "3":
                training.GNB_train(ds_training_features, ds_training_labels)
            elif option3 == "4":
                training.CNB_train(ds_training_features, ds_training_labels)
        elif option2 == "5":
            training.MLP_NN_train(ds_training_features, ds_training_labels)

        print(LogColors.OKGREEN + "Trained Successfully ✔" + LogColors.ENDC)
    # Validating
    else:
        print(LogColors.OKGREEN + "Validating..." + LogColors.ENDC)

        ds_validation_features, ds_validation_labels = get_dataset("ds" + option1 + "/ds" + option1 + "Val.csv")

        if option2 == "2":
            validation.classify(ds_validation_features, ds_validation_labels, option1, "DT")
        elif option2 == "4":
            if option3 == "1":
                validation.classify(ds_validation_features, ds_validation_labels, option1, "MNB")
            elif option3 == "2":
                validation.classify(ds_validation_features, ds_validation_labels, option1, "BNB")
            elif option3 == "3":
                validation.classify(ds_validation_features, ds_validation_labels, option1, "GNB")
            elif option3 == "4":
                validation.classify(ds_validation_features, ds_validation_labels, option1, "CNB")
        elif option2 == "6":
            validation.classify(ds_validation_features, ds_validation_labels, option1, "MLP_NN")

        print(LogColors.OKGREEN + "Validated Successfully ✔" + LogColors.ENDC)

