#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import pandas as pd
from pandas.io.json import json_normalize
import json
import requests
from sklearn.base import BaseEstimator, TransformerMixin


logger = logging.getLogger(__name__)

def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

def default_result_format(jsonArray):
    data = []
    for js_item in jsonArray:
        data.append(json_normalize(flatten_json(js_item)))
    return pd.DataFrame(data)


class WebRequestEnrich(BaseEstimator, TransformerMixin):

    def __init__(self, url_build, format_result_function=default_result_format):
        """
        enrich a dataset using web queries
        :param url_build: the function used to get the api url for a row (function: row => url)
        :param format_result_function: the function used to convert the text result to a dataframe row (function: list(json) => dataFrame)
        """
        self.url_build = url_build
        self.format_result = format_result_function

    def fit(self, X, y=None):
        """nothing to do in fit"""
        return self

    @staticmethod
    def _run_request(row, url_build):
        url = url_build(row)
        logging.info("getting infos from: " + url)
        response = requests.get(url=url)
        return json.loads(response.text, strict=False)

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
        # see https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future for async requests
        jsonArray = df.apply(lambda row: WebRequestEnrich._run_request(row, self.url_build), axis=1)
        # format the results
        enrichment_df = self.format_result(jsonArray)
        # appends the result in a new column of the dataframe
        df = pd.concat([df, enrichment_df], axis=1)
        return df