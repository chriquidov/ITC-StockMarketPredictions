import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


class StockPrediction():
    def __init__(self, stock_symbol="SPY", interval='1d', period='max'):
        """Initializes the class, creates the following arguments:
        stock_symbol: Ticker
        interval: period between any sample in the data
        df: downloaded data

        """
        self.period = period
        self.stock_symbol = stock_symbol
        self.interval = interval
        self.df = self.download_ticker(self.stock_symbol, self.interval)
        self.train = None  # train set from train_test_split
        self.test = None  # test set from train_test_split
        self.d = 0

        # baseline model
        self.baseline_history = []
        self.baseline_pred = []
        self.baseline_RMSE = None

        # model
        self.model = None
        self.pred = []
        self.history = []

        # different predictions
        self.avg_preds=None
        self.exp_preds=None
        self.ravg_preds=None
        self.drift_preds=None
        self.naive_preds=None
    @st.cache(suppress_st_warning=True,allow_output_mutation=True)
    def download_ticker(self, ticker, interval):
        """params:
            ticker: default is SPY
            returned_cols: (list) default is Adj Close
            interval: one of 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo, default is 1d
            returns: df with ticker data index is timeseries"""
        return yf.download(ticker, interval=interval, period=self.period)

    def train_test_split(self, date_split,col="Adj Close", diff=0, pct_change=False, train_size=.99):
        """Splits dataframe into train and test according to train_size"""
        self.d = diff
        self.original_price = col
        if date_split:
            ind_split=self.df.index.get_loc(date_split)
        else:
            ind_split = int(np.floor(len(self.df.index) * train_size))
        self.ind_split = ind_split
        # Train
        self.train = self.df.iloc[:ind_split][col] if self.d == 0 else \
            (self.df.iloc[:ind_split][col].diff(self.d).dropna() if not pct_change else \
                 self.df.iloc[:ind_split][col].pct_change(self.d).dropna() * 100)
        # test
        self.test = self.df.iloc[ind_split:][col] if self.d == 0 else \
            (self.df.iloc[ind_split:][col].diff(self.d).dropna() if not pct_change else \
                 self.df.iloc[ind_split:][col].pct_change(self.d).dropna() * 100)
        print("train, test split success")

    def baseline_predict_all(self, p=1, q=1, diff=False):
        self.baseline_pred = []
        self.baseline_history = self.train.to_list()

        arima_bar=st.progress(0)
        for t in range(len(self.test)):
            bsl_model = ARIMA(self.baseline_history, order=(p, q if not diff else 0, q))
            bsl_fit = bsl_model.fit()
            self.baseline_pred.append(bsl_fit.forecast()[0])
            self.baseline_history.append(self.test[t])
            arima_bar.progress(t+1)
        arima_bar.empty()
        self.baseline_RMSE = np.sqrt(
            ((self.test.to_numpy() - np.array(self.baseline_pred)) ** 2).sum() / len(self.test.index))


    def baseline_predict(self, scope=1, p=4, q=1):
        baseline_model = ARIMA(pd.concat([self.train, self.test]), order=(p, q if not diff else 0, q))
        bsl_fit = baseline_model.fit()
        return pd.Series(bsl_fit.forecast(steps=scope))

    def plot_baseline_performance(self):
        plt.figure(figsize=(12, 8))
        ax = self.test.plot(label="y_true")
        pd.Series(self.baseline_pred, index=self.test.index).plot(label="y_pred_baseline", ax=ax)
        ax.grid(True, which="both")
        ax.legend()
        ax.set_title(f"RMSE_Baseline: {self.baseline_RMSE:.4f}")

    def exp_predict_all(self):
        """Makes exponantial smoothing predictions for all test set, adds it as a new attribute: self.exp_preds"""
        exp_pred = pd.Series()
        exp_history = self.train.to_list()
        exp_bar = st.progress(0)
        for t in range(len(self.test)):
            exp_model = SimpleExpSmoothing(exp_history, initialization_method='estimated')
            exp_fit = exp_model.fit()
            exp_pred[self.test.index[t]] = exp_fit.forecast()[0]
            exp_history.append(self.test[t])
            exp_bar.progress(t + 1)
        exp_bar.empty()
        self.exp_preds = exp_pred
        self.exp_RMSE = mean_squared_error(self.test, self.exp_preds, squared=False)

    def exp_predict(self):
        """Makes Unique prediction for the next period"""
        exp_model = SimpleExpSmoothing(self.df['Adj Close'], initialization_method='estmiated').fit()
        return pd.Series(exp_model.forecast())

    def ravg_predict_all(self, window=5):
        """Makes predictions based on the average method, parameter window is the number of sample used for the prediction"""
        self.ravg_preds = self.df['Adj Close'].iloc[self.ind_split - window:].rolling(window=window,
                                                                                      closed='left').mean().dropna()

    def naive_predict_all(self):
        self.naive_preds = (self.df['Adj Close'].iloc[self.ind_split - 2:].shift(1) + self.df['Adj Close'].iloc[
                                                                                      self.ind_split - 2:].diff().shift(
            1)).dropna()

    def drift_predict_all(self):
        self.df['my_index'] = range(len(self.df))
        self.drift_preds = self.df['Adj Close'].shift(1) + (
                    self.df['Adj Close'].shift(1) - self.df['Adj Close'].iloc[0]) / self.df['my_index']
        self.drift_preds = self.drift_preds.iloc[self.ind_split:]
        self.df.drop(columns=['my_index'])

    def avg_predict_all(self):
        self.avg_preds = self.df['Adj Close'].shift(1).expanding().mean().iloc[self.ind_split:]

    def all_model_predict_all(self):
        """If called, will make predictions using all the models"""
        self.baseline_predict_all()
        self.exp_predict_all()
        self.ravg_predict_all()
        self.naive_predict_all()
        self.drift_predict_all()
        self.avg_predict_all()

    def profiling(self, cols=["Adj Close"]):
        """Automatic EDA"""
        # head
        print("The head of the DataFrame:")
        display(self.df.head())
        print("\n")

        # shape
        print("The shape of the DataFrame:")
        display(self.df.shape)
        print("\n")

        # index
        print("The time stamp for each record:")
        display(self.df.index)
        print("\n")

        # plot the distribution of the day
        print("The distribution of the weekday recorded:")
        sns.countplot(x=self.df.index.strftime('%A'))
        plt.show()
        print("\n")

        # gap
        print("The start date and the current end date for the data:")
        display(self.df.index[0], self.df.index[-1])
        print("\n")

        # nan check
        print("If there is missing values in the data:")
        display(self.df.isna().sum())
        print("\n")

        # describe
        print("Describe the time series:")
        display(self.df.describe())
        print("\n")

        for col in cols:
            # plot time series values
            print("Plot the time series:")
            plt.figure(figsize=(12, 8))
            self.plot(col)
            plt.title = f"{col} Price"
            plt.show()

            # plot diff one day
            print("Plot the change in the time series:")
            plt.figure(figsize=(12, 8))
            self.df[col].diff().plot()
            plt.title = f"{col} Price Change"
            plt.show()
            # plot diff acf
            print(f"Plot the Auto-Correlation Function of {col} Price Change:")
            self.plot_acf(col)
            plt.show()
            # plot diff pacf
            print(f"Plot the Partial Auto-Correlation Function of {col} Price Change:")
            self.plot_pacf(col)
            plt.show()

    def plot(self, col="Adj Close"):
        self.df[col].plot()

    def plot_acf(self, col="Adj Close"):
        plot_acf(self.df[col].diff().dropna())

    def plot_pacf(self, col="Adj Close"):
        plot_pacf(self.df[col].diff().dropna())


class Invest_shorten():
    def __init__(self, model_name, true_adj_close, preds=None, pred_recs=None):
        """The class receives the true proces of the stock and the predictions of prices"""
        self.model_name = model_name
        self.daily_table = pd.DataFrame()  # Initializes the table
        self.daily_table['Adj_close'] = true_adj_close  # Adds the true prices
        if preds is not None:
            self.daily_table['preds'] = preds  # Adds predictions
            self.daily_table['pred_rec'] = (self.daily_table['Adj_close'].shift(1) < self.daily_table.preds).astype(
                int)  # Makes recommandation based on predictions
        else:
            self.daily_table['pred_rec'] = pred_recs  # Makes recommandation based on imput pred_recs
        self.daily_table['profit_day'] = self.daily_table[
            'Adj_close'].pct_change()  # Checks for stocks profit on this day
        self.daily_table['profit_pred'] = np.select([self.daily_table.pred_rec == 1, self.daily_table.pred_rec == 0],
                                                    [self.daily_table.profit_day,
                                                     0]) + 1  # Constructs system profit based on stock profit and recommandation
        self.daily_table.iloc[0, -1] = self.daily_table['Adj_close'].iloc[
            0]  # Initialises the initial money as initial stock price
        self.daily_table['money_end_day'] = np.cumprod(self.daily_table.profit_pred)

    def plot_money_end_day_evolution(self,all_strat=False):
        """Plots the total amount of money of stock vs prediction based"""
        if not all_strat:
            f=plt.figure(figsize=(13, 10))
            self.daily_table['Adj_close'].plot( label='True prices')
            self.daily_table.money_end_day.plot( label='Preds')
            plt.title('Investing in this strategy this is the evolution of your Portfolio', fontsize=15)
            plt.legend()
            return f
        else:
            self.daily_table.money_end_day.plot(label=self.model_name)

    def plot_daily_pct_change(self):
        plt.figure(figsize=(13, 10))
        self.daily_table.profit_day.plot(label='true changes')
        plt.plot(self.daily_table.profit_pred[1:] - 1, label='Preds')
        plt.legend();

    def make_summary_table(self):
        summary = pd.DataFrame()
        summary['RMSE'] = [mean_squared_error(self.daily_table.Adj_close, self.daily_table.preds,
                                              squared=False) if 'preds' in self.daily_table.columns else None]
        summary['Stock_return'] = [(self.daily_table.Adj_close.iloc[-1] / self.daily_table.Adj_close.iloc[0]) - 1]
        summary['System_return'] = [(self.daily_table.money_end_day.iloc[-1] / self.daily_table.money_end_day[0]) - 1]
        self.true_recommandations = (self.daily_table['Adj_close'].shift(1) < self.daily_table['Adj_close']).astype(int)
        summary['Recommandation_accuracy'] = [accuracy_score(self.true_recommandations, self.daily_table.pred_rec)]
        summary['std_system'] = [self.daily_table.money_end_day.std()]
        summary['std_stock'] = [self.daily_table.Adj_close.std()]
        summary['model'] = [self.model_name]
        summary.set_index('model', inplace=True)
        self.summary_table = summary
        return summary


