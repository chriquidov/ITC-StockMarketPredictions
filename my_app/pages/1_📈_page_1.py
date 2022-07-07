import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score
import datetime
from pred_classes import StockPrediction, Invest_shorten
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(page_title="Plotting Demo", page_icon="📈")
st.sidebar.header("Plotting Demo")

ticker = st.text_input('Please input one ticker', 'SPY')


date_start=st.date_input('Date to test the predictions on',max_value=datetime.datetime.now() - datetime.timedelta(1),
                         value=datetime.datetime.now() - datetime.timedelta(100))

spy=StockPrediction(ticker)

try:
    spy.train_test_split(date_start.isoformat())
except:
    st.write('Please choose a valid ticker')
    st.stop()
st.write("Want to know more about the models we're using?")
st.write("Check out the [article](https://medium.com/@pengzhang_15103/ccccc0837c2f) of the project!")

options = st.selectbox(
    'Please select the models you would like to test:',
    ( 'Exponantial Smoothing','All of them', 'Moving average', 'Average', 'Drift Method',
     'Naive', 'Arima'))

strategy_dict = {'All of them': spy.all_model_predict_all, 'Exponantial Smoothing': spy.exp_predict_all,
                 'Moving average': spy.ravg_predict_all,
                 'Average': spy.avg_predict_all, 'Drift Method': spy.drift_predict_all, 'Naive': spy.naive_predict_all,
                 'Arima': spy.baseline_predict_all}
strategy_dict[options]()

preds_dict = {'Exponantial Smoothing': spy.exp_preds, 'Moving average': spy.ravg_preds, 'Average': spy.avg_preds,
              'Drift Method': spy.drift_preds, 'Naive': spy.naive_preds, 'Arima': spy.baseline_pred}

if options != 'All of them':
    preds = preds_dict[options]

    money_ravg = Invest_shorten(options, spy.test, preds)
    summary=money_ravg.make_summary_table()

    pred_count=money_ravg.daily_table.pred_rec.loc[money_ravg.daily_table.pred_rec==1].size /money_ravg.daily_table.size
    fig=money_ravg.plot_money_end_day_evolution()
    st.pyplot(fig=fig)

    st.write(f'The model predicts than the stock will go up {pred_count:.0%} of the time')


else:

    summary = pd.DataFrame()
    f = plt.figure(figsize=(13, 10))
    plt.title('Portfolio value evolution using all the strategies ')
    for k,v in preds_dict.items():

        money = Invest_shorten(k,spy.test,v)
        summary=pd.concat([summary,money.make_summary_table()])
        money.daily_table['Adj_close'].plot(label='True prices')
        money.plot_money_end_day_evolution(all_strat=True)

    plt.legend()
    st.pyplot(f)

pd.set_option('max_colwidth', 60)
summary=summary.style.highlight_max(axis=0,subset=['System_return','Recommandation_accuracy'],
                            color='lightgreen').highlight_min(color='lightgreen',axis=0
                                                              ,subset=['RMSE','std_system'])
summary=summary.format({"System_return": "{:20,.2f}%",
                          "Recommandation_accuracy": "{:20,.2f}%",
                          "RMSE": "{:20,.2f}",
                        "std_stock": "{:20,.2f}","Stock_return": "{:20,.2f}%",
                          "std_system":"{:20,.2f}"})
st.dataframe(summary)

