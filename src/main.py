import training
import validation
from LogColors import LogColors

while True:
    option = input(LogColors.HEADER + "Which action do you want to execute?\n" + LogColors.ENDC +
                                      "1. Train MNB\n"
                                      "2. Validate MNB\n"
                                      + LogColors.FAIL + "0. Exit\n" + LogColors.ENDC)

    if option == "1":
        training.MNB_train()
    elif option == "2":
        validation.MNB_classify()
    elif option == "0":
        exit(0)

