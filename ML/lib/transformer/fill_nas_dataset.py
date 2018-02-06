#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 23:03:22 2017

@author: marvin
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
import logging
logger = logging.getLogger(__name__)


class FillNAsDataset(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None):
        self.max_datetime = max(pd.to_datetime(X['content.removalDate']))
        self.agreg_roomCount = np.mean(X['content.roomCount'])
        self.agreg_lat = np.mean(X['content.geoPoint.lat'])
        self.agreg_lon = np.mean(X['content.geoPoint.lon'])
        #self.agreg_roomCount = X.groupby('dep')['content.roomCount']
        #self.agreg_lat = X.groupby('dep')['content.geoPoint.lat']
        #self.agreg_lon = X.groupby('dep')['content.geoPoint.lon']
        return self

    def transform(self,df):
        """
        transform a dataframe to 

        Parameters
        ----------

        df : pandas dataframe 

        Returns
        -------
        
        Transformed pandas dataframe
        """
        #fill removalDate with the youngest date
        df['content.removalDate'] = df['content.removalDate'].fillna(self.max_datetime)
        #fill roomCount lat lon with mean in the department
        #df['content.roomCount'] = self.agreg_roomCount.transform(lambda x: x.fillna(x.mean()))
        #df['content.geoPoint.lat'] = self.agreg_lat.transform(lambda x: x.fillna(x.mean()))  
        #df['content.geoPoint.lon'] = self.agreg_lon.transform(lambda x: x.fillna(x.mean()))
        df['content.roomCount'] = df['content.roomCount'].fillna(self.agreg_roomCount)
        df['content.geoPoint.lat'] = df['content.geoPoint.lat'].fillna(self.agreg_lat)  
        df['content.geoPoint.lon'] = df['content.geoPoint.lon'].fillna(self.agreg_lon)
        
        #fill property.land with False
        df['content.property.land'] = df['content.property.land'].fillna(False)#Checking for missing data

        return df