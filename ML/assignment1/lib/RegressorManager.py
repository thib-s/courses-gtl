import copy
import logging
import pickle
import timeit

import os
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

STORE_PATH = "trained_regressors/RegressorManagerDict"


class RegressorManager:
    """
    This class acts as a DB for regressor. You can add a new regressor, and it will train it, test it,
    and store the results on disk

    """

    def __init__(self):
        """
        Called when initializing the regressor
        """
        self.dict = {}
        if os.path.isfile(STORE_PATH + ".pkl"):
            self.load_backup()
        self.logger = logging.getLogger()
        self.train_override = False

    def add_trained_regressor(self,
                               key_name,
                               regressor_instance,
                               training_duration,
                               prediction_duration,
                               predicted_data,
                               report):
        """
        add a trained regressor to the dictionnary
        :param key_name: the name to get this element back this will also be used for plotting
        :param regressor_instance: the instance to add
        :param training_duration: the duration of the training phase
        :param prediction_duration: the duration of the prediction phase
        :param predicted_data: the regressor results on the test set
        :param report: the report of the prediction phase
        :return: the element added (no deep copy for performance purposes)
        """
        if self.dict.keys().__contains__(key_name):
            self.logger.warning('the regressor ' + key_name + ' already exist in this manager')
        element = {'instance': regressor_instance,
                   'training_duration': training_duration,
                   'prediction_duration': prediction_duration,
                   'predicted_data': predicted_data.copy(),
                   'report': report}
        self.dict[key_name] = element
        return element

    def add_regressor(self, key_name, classifer_instance, X_train, y_train, X_test, y_test):
        """
        train and evaluate the regressor and store the results
        :param key_name: the key in the dict, also used for plotting
        :param classifer_instance:
        :param train_df: the pandas dataframe corresponding to the train dataset
        :param test_df: the pandas dataframe corresponding to the test dataset
        :return: the element added (no deep copy for performance purposes)
        """
        if self.dict.keys().__contains__(key_name) and (not self.train_override):
            self.logger.info("already trained model, skipping model " + key_name)
            return self.dict[key_name]
        (
            key_name,
            classifer_instance,
            fit_duration,
            predict_duration,
            predicted,
            report
        ) = RegressorManager.train_regressor(
            key_name,
            classifer_instance,
            X_train, y_train, X_test, y_test)
        # now we have everything to add data to dict
        result = self.add_trained_regressor(
            key_name,
            classifer_instance,
            fit_duration,
            predict_duration,
            predicted,
            report
        )
        self.do_backup()
        return result

    @staticmethod
    def train_regressor(key_name, classifer_instance, X_train, y_train, X_test, y_test):
        """
        train and evaluate the regressor and return the results
        :param key_name: the key in the dict, also used for plotting
        :param classifer_instance:
        :param train_df: the pandas dataframe corresponding to the train dataset
        :param test_df: the pandas dataframe corresponding to the test dataset
        :return: the element added (no deep copy for performance purposes)
        """
        # train the instance
        start_time = timeit.default_timer()
        classifer_instance.fit(X_train, y_train)
        fit_duration = timeit.default_timer() - start_time
        # make predictions on test set
        start_time = timeit.default_timer()
        predicted = classifer_instance.predict(X_test)
        predict_duration = timeit.default_timer() - start_time
        # generate report
        report = mean_absolute_error(y_test, predicted)
        return (
            key_name,
            classifer_instance,
            fit_duration,
            predict_duration,
            predicted,
            report
        )

    def pretty_print(self, key_name):
        """
        display the results of a regressor
        :param key_name:
        :return:
        """
        print("===================================")
        print("* " + key_name + " *")
        print("===================================")
        print("training time:" + str(self.dict[key_name]['training_duration']))
        print("prediction time:" + str(self.dict[key_name]['prediction_duration']))
        print("report" + str(self.dict[key_name]['report']))

    def do_backup(self):
        if os.path.isfile(STORE_PATH + ".pkl"):
            os.rename(STORE_PATH + ".pkl", STORE_PATH + ".bck")
        pickle.dump(self.dict, open(STORE_PATH + ".pkl", "wb"))

    def load_backup(self):
        assert os.path.isfile(STORE_PATH + ".pkl"), "no backup file found at " + STORE_PATH + ".pkl"
        self.dict = pickle.load(open(STORE_PATH + ".pkl", "rb"))
