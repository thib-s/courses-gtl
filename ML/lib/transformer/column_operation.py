#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class ColumnOperation(BaseEstimator, TransformerMixin):

    def __init__(self, operation_function, output_column: str):
        """
        Apply the operation_function to each row of the dataset and store it into the output_column
        :param operation_function: the function to apply ( row => value )
        :param output_column: the name of the output column
        """
        self.operation_function = operation_function
        self.output_column = output_column

    def fit(self, X, y=None):
        """nothing to do in fit"""
        return self

    def transform(self, df: pd.DataFrame):
        """
        Parameters
        ----------

        df : pandas dataframe 

        Returns
        -------

        Transformed pandas dataframe
        """
        # perform the query for each element
        df[self.output_column] = df.apply(lambda row: self.operation_function(row), axis=1)
        return df