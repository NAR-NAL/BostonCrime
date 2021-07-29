import pandas as pd
import matplotlib.pyplot as plt
import itertools as it
from pylab import rcParams
import statsmodels.api as sm


#https://towardsdatascience.com/time-series-forecasting-for-road-accidents-in-uk-f940e5970988
def read_data():
    crime_2021 = pd.read_csv('./crime_2021.csv')
    crime_2021 = crime_2021[(crime_2021['OFFENSE_DESCRIPTION']=="VANDALISM") | (crime_2021['OFFENSE_DESCRIPTION']
                                                                                =="VERBAL DISPUTE")]

    crime_2020 = pd.read_csv('./crime_2020.csv')
    crime_2020 = crime_2020[(crime_2020['OFFENSE_DESCRIPTION']=="VANDALISM") | (crime_2020['OFFENSE_DESCRIPTION']
                                                                                =="VERBAL DISPUTE")]
    print(crime_2021.head())
    print(crime_2020.head())

    df = pd.concat([crime_2020,crime_2021], ignore_index=True)
    df.reset_index(inplace=True)
    print(df)
    df['OCCURRED_ON_DATE'] = pd.to_datetime(df['OCCURRED_ON_DATE'])
    df['incident_count'] = 1

    df = df.sort_values(by=['OCCURRED_ON_DATE'])
    print(df.head())
    df.set_index('OCCURRED_ON_DATE',inplace=True)
    df_vandalism = df[(df['OFFENSE_DESCRIPTION']=="VANDALISM")]
    df_verbaldispute = df[(df['OFFENSE_DESCRIPTION']=="VERBAL DISPUTE")]
    return(df_vandalism,df_verbaldispute)

def try_model(df_incident):

    y = df_incident['incident_count'].resample('D').sum()
    y.head()
    y.to_excel('output.xlsx')
    y.plot(figsize=(15, 6))
    plt.show()

    rcParams['figure.figsize'] = 16, 10
    decomposition = sm.tsa.seasonal_decompose(y, model='additive')
    fig = decomposition.plot()
    plt.show()

    p = d = q = range(0, 2)
    pdq = list(it.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(it.product(p, d, q))]
    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

    list_param = []
    list_param_seasonal = []
    list_aic = []
    list_bic = []

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                print(param,param_seasonal,results.aic)
                list_param_seasonal.append(param_seasonal)
                list_param.append(param)
                list_aic.append(results.aic)
                list_bic.append(results.bic)

                #print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    df_Params = pd.DataFrame(
        {'Param': list_param,
         'Seasonal': list_param_seasonal,
         'AICScore': list_aic,
         'BICScore': list_bic
         })

    df_Params.to_excel('params.xlsx')
    return(df_Params,y)

#print(df_minimizeAIC)
# Who scored more points ?
def FineTuneAndPredict(df_Params,y):
    df_minimizeAIC = df_Params[['BICScore', 'AICscore']].min(axis=1)
    #df_minimizeAIC = df_Params[df_Params.AICScore == df_Params.AICScore.min()]
    #print(df_minimizeAIC)
    print(df_Params.tail())
    df_minimizeAIC.reset_index(inplace=True)
    print(df_minimizeAIC.Param.values[0])
    mod = sm.tsa.statespace.SARIMAX(y,
                                    order=df_minimizeAIC.Param.values[0],
                                    seasonal_order=df_minimizeAIC.Seasonal.values[0],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)



    results = mod.fit()
    print(results.summary().tables[1])




    results.plot_diagnostics(figsize=(16, 8))
    plt.show()




    pred = results.get_prediction(start=pd.to_datetime('2021-07-09'),end=pd.to_datetime('2021-12-31'), dynamic=False)
    pred_ci = pred.conf_int()
    ax = y['2020':].plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    #print(pred_ci.iloc[:, 0])
    #print(pred_ci.iloc[:, 1])
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    #ax.set_xlabel('Date')
    #ax.set_ylabel('Furniture Sales')
    #plt.legend()
    plt.show()


if __name__ == "__main__":
    [df_vandalism,df_verbaldispute] = read_data()
    print(df_vandalism.head())
    [df_Params, y] = try_model(df_vandalism)
    FineTuneAndPredict(df_Params, y)

