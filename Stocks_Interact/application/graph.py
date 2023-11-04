#!/usr/bin/python3
"""module for creating graphs"""

import pandas as pd
import requests
from config import settings
from application.data import AlphaVantageAPI
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


av = AlphaVantageAPI()


class GraphBuilder:
    def __init__(self,ticker="", n_observations=255):
        self.ticker = ticker
        self.n_observations = n_observations

    def get_data(self, ticker="", n_observations=255):
        df = av.get_daily(ticker)
        if self.n_observations <= len(df):
            self.new_df = df.iloc[:n_observations, :]
        else:
            self.new_df = df

    def close_graph(self):
        """time series graph about closing prices"""
        fig = px.line(self.new_df, x=self.new_df.index, y=self.new_df['close'],
                      title=f"Closing Price of {self.ticker}")
        return fig

    def volume_graph(self):
        """time series graph about Sales Volume"""
        fig = px.line(self.new_df, x=self.new_df.index, y=self.new_df['volume'],
                      title=f"Sales Volume for {self.ticker}")
        return fig

    def moving_average(self, n_days=10):
        """graph about moving average"""
        self.new_df[f"MA for {n_days} days"] = self.new_df['close'].rolling(n_days).mean()
        fig = px.line(self.new_df, x=self.new_df.index,
                      y=['close',f"MA for {n_days} days"],
                      title=f"Moving Average of {n_days} days for {self.ticker}")
        return fig

    def daily_return(self):
        """graph about daily return"""
        self.new_df['Daily Return'] = self.new_df['close'].pct_change()
        fig = make_subplots(rows=2, cols=1,
                    subplot_titles=(f"Daily Return for {self.ticker}", "Most Frequent Daily Return"))
        fig.append_trace(go.Line(x=self.new_df.index,
                                 y=self.new_df['Daily Return']), row=1, col=1)
        fig.append_trace(go.Histogram(x=self.new_df['Daily Return']), row=2, col=1)
        fig.update_layout(height=1000, width=1000, title_text="Daily Return")
        
        return fig

    def risk(self):
        """graph about risk invest based on returns"""
        returns = self.new_df['close'].pct_change()
        rets = returns.dropna()
        fig = px.scatter(x=[rets.mean(),0], y=[rets.std(),0],
                labels=dict(x='Expected return', y='Risk'),
                title = f'risk by investing in {self.ticker}')
        return fig