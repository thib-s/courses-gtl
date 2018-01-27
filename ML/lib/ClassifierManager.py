import copy
import logging
import pickle
import timeit

import os
import pandas as pd
from sklearn.metrics import classification_report

STORE_PATH = "trained_classifiers/ClassifierManagerDict"


class ClassifierManager:
    """
    This class acts as a DB for classifier. You can add a new classifier, and it will train it, test it,
    and store the results on disk

    """

    def __init__(self):
        """
        Called when initializing the classifier
        """
        self.dict = {}
        if os.path.isfile(STORE_PATH + ".pkl"):
            self.load_backup()
        self.logger = logging.getLogger()

    def add_trained_classifier(self,
                               key_name,
                               classifier_instance,
                               training_duration,
                               prediction_duration,
                               predicted_data,
                               report):
        """
        add a trained classifier to the dictionnary
        :param key_name: the name to get this element back this will also be used for plotting
        :param classifier_instance: the instance to add
        :param training_duration: the duration of the training phase
        :param prediction_duration: the duration of the prediction phase
        :param predicted_data: the classifier results on the test set
        :param report: the report of the prediction phase
        :return: the element added (no deep copy for performance purposes)
        """
        if self.dict.keys().__contains__(key_name):
            self.logger.warning('the classifier ' + key_name + ' already exist in this manager')
        element = {'instance': classifier_instance,
                   'training_duration': training_duration,
                   'prediction_duration': prediction_duration,
                   'predicted_data': predicted_data.copy(),
                   'report': report}
        self.dict[key_name] = element
        return element

    def add_classifier(self, key_name, classifer_instance, train_df: pd.DataFrame, test_df: pd.DataFrame,
                       get_xy_function):
        """
        train and evaluate the classifier and store the results
        :param key_name: the key in the dict, also used for plotting
        :param classifer_instance:
        :param train_df: the pandas dataframe corresponding to the train dataset
        :param test_df: the pandas dataframe corresponding to the test dataset
        :param get_xy_function: the function used to extract the x and y vector from dataframes
        :return: the element added (no deep copy for performance purposes)
        """
        if self.dict.keys().__contains__(key_name):
            self.logger.info("already trained model, skipping model "+key_name)
            return self.dict[key_name]
        # get the vectors
        (X_test, y_test) = get_xy_function(test_df)
        (X_train, y_train) = get_xy_function(train_df)
        # train the instance
        start_time = timeit.default_timer()
        classifer_instance.fit(X_train, y_train)
        fit_duration = timeit.default_timer() - start_time
        # make predictions on test set
        start_time = timeit.default_timer()
        predicted = classifer_instance.predict(X_test)
        predict_duration = timeit.default_timer() - start_time
        # generate report
        report = classification_report(y_test, predicted)
        # now we have everything to add data to dict
        result =  self.add_trained_classifier(
            key_name,
            classifer_instance,
            fit_duration,
            predict_duration,
            predicted,
            report
        )
        self.do_backup()
        return result

    def pretty_print(self, key_name):
        """
        display the results of a classifier
        :param key_name:
        :return:
        """
        print("===================================")
        print("* " + key_name + " *")
        print("===================================")
        print("training time:" + str(self.dict[key_name]['training_duration']))
        print("prediction time:" + str(self.dict[key_name]['prediction_duration']))
        print("report" + self.dict[key_name]['report'])

    def do_backup(self):
        if os.path.isfile(STORE_PATH + ".pkl"):
            os.rename(STORE_PATH + ".pkl", STORE_PATH + ".bck")
        pickle.dump(self.dict, open(STORE_PATH + ".pkl", "wb"))

    def load_backup(self):
        assert os.path.isfile(STORE_PATH + ".pkl"), "no backup file found at " + STORE_PATH + ".pkl"
        self.dict = pickle.load(open(STORE_PATH + ".pkl", "rb"))
