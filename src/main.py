import training
import validation
from LogColors import LogColors
from processing import get_dataset


def exitOnZero(check):
    if check == 0:
        exit(check)


test = ""
validOption = False
while not validOption:
    test = input(LogColors.HEADER + "How do you wish to test the learning algorithms?\n" + LogColors.ENDC +
                    "1. Custom Test\n"
                    "2. Partial Test\n"
                    "3. Full Test\n"
                    + LogColors.FAIL + "0. Exit\n" + LogColors.ENDC)

    if (int(test) >= 0) and (int(test) <= 3):
        validOption = True

exitOnZero(int(test))

if test == "1":
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
            if (int(option2) >= 0) and (int(option2) <= 6):
                validOption = True

        exitOnZero(int(option2))

        # Choose which NB alg to use if nb was chosen
        if (option2 == "3") or (option2 == "4"):
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
# Run partial testing program
if test == "2":

    # Choose which ML alg to use
    option1 = ""
    validOption = False
    while not validOption:
        option1 = input(LogColors.HEADER + "Which algorithm do you wish to test?\n" + LogColors.ENDC +
                        "1. DT\n"
                        "2. MNB\n"
                        "3. BNB\n"
                        "4. GNB\n"
                        "5. CNB\n"
                        "6. MLP NN\n"
                        + LogColors.FAIL + "0. Exit\n" + LogColors.ENDC)
        if (int(option1) >= 0) and (int(option1) <= 6):
            validOption = True

    exitOnZero(int(option1))

    print(LogColors.OKGREEN + "Running Partial testing Program...\n" + LogColors.ENDC)

    dataset = 1
    while dataset <= 2:

        print(LogColors.OKBLUE + "Dataset " + str(dataset) + "\n" + LogColors.ENDC)

        average = 0
        k = ""

        if int(option1) == 1:
            k = "DT"
        elif int(option1) == 2:
            k = "MNB"
        elif int(option1) == 3:
            k = "BNB"
        elif int(option1) == 4:
            k = "GNB"
        elif int(option1) == 5:
            k = "CNB"
        elif int(option1) == 6:
            k = "MLP_NN"

        print(LogColors.BOLD + k + " Validation values and Average" + LogColors.ENDC)

        j = 0
        maxRuns = 5
        while j < maxRuns:

            ds_training_features, ds_training_labels = get_dataset(
                "ds" + str(dataset) + "/ds" + str(dataset) + "Train.csv")

            if int(option1) == 1:
                training.DT_train(ds_training_features, ds_training_labels)
            elif int(option1) == 2:
                training.MNB_train(ds_training_features, ds_training_labels)
                j = maxRuns
            elif int(option1) == 3:
                training.BNB_train(ds_training_features, ds_training_labels)
                j = maxRuns
            elif int(option1) == 4:
                training.GNB_train(ds_training_features, ds_training_labels)
                j = maxRuns
            elif int(option1) == 5:
                training.CNB_train(ds_training_features, ds_training_labels)
                j = maxRuns
            elif int(option1) == 6:
                training.MLP_NN_train(ds_training_features, ds_training_labels)

            ds_validation_features, ds_validation_labels = get_dataset(
                "ds" + str(dataset) + "/ds" + str(dataset) + "Val.csv")

            average += validation.classify(ds_validation_features, ds_validation_labels, str(dataset), k)
            j += 1
        if (int(option1) == 1) or (int(option1) == 6):
            average /= maxRuns
            print("Total Average: " + str(average) + "\n\n")
        else:
            print("\n")
        dataset += 1

# Run Full testing program
else:
    print(LogColors.OKGREEN + "Running Testing Program...\n" + LogColors.ENDC)

    dataset = 1
    while dataset <= 2:

        print(LogColors.OKBLUE + "Dataset " + str(dataset) + "\n" + LogColors.ENDC)

        average = 0

        i = 1
        while i <= 6:
            k = ""
            if i == 1:
                k = "DT"
            elif i == 2:
                k = "MNB"
            elif i == 3:
                k = "BNB"
            elif i == 4:
                k = "GNB"
            elif i == 5:
                k = "CNB"
            elif i == 6:
                k = "MLP_NN"

            print(LogColors.BOLD + k + " Validation values and Average" + LogColors.ENDC)

            average = 0
            j = 0
            maxRuns = 5
            while j < maxRuns:

                ds_training_features, ds_training_labels = get_dataset(
                    "ds" + str(dataset) + "/ds" + str(dataset) + "Train.csv")

                if i == 1:
                    training.DT_train(ds_training_features, ds_training_labels)
                elif i == 2:
                    training.MNB_train(ds_training_features, ds_training_labels)
                    j = maxRuns
                elif i == 3:
                    training.BNB_train(ds_training_features, ds_training_labels)
                    j = maxRuns
                elif i == 4:
                    training.GNB_train(ds_training_features, ds_training_labels)
                    j = maxRuns
                elif i == 5:
                    training.CNB_train(ds_training_features, ds_training_labels)
                    j = maxRuns
                elif i == 6:
                    training.MLP_NN_train(ds_training_features, ds_training_labels)

                ds_validation_features, ds_validation_labels = get_dataset(
                    "ds" + str(dataset) + "/ds" + str(dataset) + "Val.csv")

                average += validation.classify(ds_validation_features, ds_validation_labels, str(dataset), k)
                j += 1
            if (i == 1) or (i == 6):
                average /= maxRuns
                print("Total Average: " + str(average) + "\n\n")
            else:
                print("\n")
            i += 1
        dataset += 1
