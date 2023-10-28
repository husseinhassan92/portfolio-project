#!/usr/bin/python3
"""module for creating graphs"""

import pandas as pd
import requests
from config import settings
from models.data import AlphaVantageAPI
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


av = AlphaVantageAPI()


class GraphBuilder:
    def __init__(self):
        pass

    def close_graph(self, ticker, n_observations=None):
        """time series graph about closing prices"""
        df = av.get_daily(ticker)
        if n_observations <= len(df):
            new_df = df.iloc[:n_observations, :]
        else:
            new_df = df

        fig = px.line(new_df, x=new_df.index, y=new_df['close'],
                      title=f"Closing Price of {ticker}")

        return fig

    def volume_graph(self, ticker, n_observations=None):
        """time series graph about Sales Volume"""
        df = av.get_daily(ticker)
        if n_observations <= len(df):
            new_df = df.iloc[:n_observations, :]
        else:
            new_df = df

        fig = px.line(new_df, x=new_df.index, y=new_df['volume'],
                      title=f"Sales Volume for {ticker}")

        return fig