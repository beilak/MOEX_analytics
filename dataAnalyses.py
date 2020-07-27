import mysql.connector
import pandas        as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy        as np

import statsmodels.api as sm
from statsmodels.iolib.table import SimpleTable
from sklearn.metrics import r2_score
import ml_metrics as metrics
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import math


def saveDataFromDBtoExcel():
    print("Start==>>Load from MySQL")
    
    # !!!!!!!!!!!!!!!!!!!!!!!
    mySqlHost = "";
    mySqlUser = "";
    mySqlPass = "";
    mySqlDataBase = "";
    
    mydb = mysql.connector.connect(
        host=mySqlHost,
        user=mySqlUser,
        passwd=mySqlPass,
        database=mySqlDataBase,
        autocommit=True
    )
    mycursor = mydb.cursor()

    sSql = r" SELECT "
    sSql += r" `TRADEDATE`, `SECID`, `OPEN`, `CLOSE`, `LOW`, `HIGH` "
    sSql += r" FROM `MOEX` "
    sSql += r" WHERE `SECID` = 'SBER' "
    sSql += r" and `TRADEDATE` > '2018-01-01' "

    mycursor.execute(sSql)
    myresult = mycursor.fetchall()

    df = pd.DataFrame(columns=['TRADEDATE', 'SECID', 'OPEN', 'CLOSE', 'LOW', 'HIGH'])
    i = 0
    for raw in myresult:
        df.loc[i] = raw
        i += 1
    writer = ExcelWriter('moexData.xlsx')
    df.to_excel(writer, 'moexData', index=False)
    writer.save()
    print("End ==>> Loaded from MySQL")


def loadData():
    # loadDataFromExcel()
    print("Start Load from file (raw data)")
    file = 'moexData.xlsx'
    xl = pd.ExcelFile(file)
    df1 = xl.parse("moexData")
    dataIn = df1
    data = [np]
    data[0] = dataIn.values
    print("End read")
    return dataIn.values  # data


def get_data():
    dataSet = loadData()
    data = []
    line = []
    for dataLine in dataSet:
        line = dataLine
        data.append(line)
    return data


# Function that calls ARIMA model to fit and forecast the data
def StartARIMAForecasting(Actual, P, D, Q):
    try:
        model = ARIMA(Actual, order=(P, D, Q))
        model_fit = model.fit(disp=0)
        prediction = model_fit.forecast()[0]
        return prediction
    except Exception:
        return 0


def predict_arima(mass):
    masslog = []
    currindex = 0
    for item in mass:
        res = math.log(item[0]) - math.log(mass[currindex - 1][0])
        if res < -1:
            res = -1
        elif res > 1:
            res = 1
        if currindex > 0:
            masslog.append([res])
        currindex += 1
    predicted = StartARIMAForecasting(masslog, 1, 1, 0)
    return float(predicted)

def get_next_mass_for_arima(index, dataset, columnindex):
    data_out = []
    first = index - 8
    if first < 0:
        return []
    data = dataset[first:index]
    for item in data:
        data_out.append([item[columnindex]])
    return data_out

def get_arima(dataset):
    data = []
    datalow = []
    datahigh = []
    line = []
    currindex = 0
    for dataline in dataset:
        line = dataline
        mass = []
        arrima_pred = 0
        mass = get_next_mass_for_arima(currindex, dataset, 4)
        if len(mass) > 0:
            arrima_pred = dataline[4] + dataline[4] * predict_arima(mass)
        arrima_pred = round(arrima_pred, 2)
        line = np.append(line, arrima_pred)

        mass = []
        mass = get_next_mass_for_arima(currindex, dataset, 5)
        if len(mass) > 0:
            arrima_pred = dataline[5] + dataline[5] * predict_arima(mass)
        arrima_pred = round(arrima_pred, 2)
        line = np.append(line, arrima_pred)

        currindex += 1
        data.append(line)

    return data


def __get_de_mark(dataSet):
    # dataSet = loadData()
    data = []
    line = []
    iIndex = 0
    for dataLine in dataSet:
        pOpen = dataLine[2]
        pClose = dataLine[3]
        pLow = dataLine[4]
        pHigh = dataLine[5]

        line = dataLine
        try:
            nextOpen = dataSet[iIndex + 1][2]
            nextMin = dataSet[iIndex + 1][4]
            nextMax = dataSet[iIndex + 1][5]
        except Exception as e:
            nextOpen = 0

        # calc Demark range
        # # Close < Open
        if (pClose < pOpen):
            predictMax = ((pHigh + pClose + 2 * pLow) / 2) - pLow
            predictMin = ((pHigh + pClose + 2 * pLow) / 2) - pHigh
        elif (pClose > pOpen):
            predictMax = ((2 * pHigh + pLow + pClose) / 2) - pLow
            predictMin = ((2 * pHigh + pLow + pClose) / 2) - pHigh
        elif (pClose == pOpen):
            predictMax = ((pHigh + pLow + 2 * pClose) / 2) - pLow
            predictMin = ((pHigh + pLow + 2 * pClose) / 2) - pHigh

        ooMin = (nextMin - predictMin) / nextMin * 100
        ooMax = (nextMax - predictMax) / nextMax * 100

        line = np.append(line, nextOpen)
        line = np.append(line, nextMin)
        line = np.append(line, nextMax)

        line = np.append(line, predictMin)
        line = np.append(line, predictMax)

        line = np.append(line, ooMin)
        line = np.append(line, ooMax)

        data.append(line)
        iIndex += 1
    return data


def get_woody(dataset):
    data = []
    line = []
    for item in dataset:
        line = item
        C = item[3]
        L = item[4]
        H = item[5]
        P = (H + L + 2 * C) / 4
        R1 = 2 * P - L
        S1 = 2 * P - H
        line = np.append(line, S1)
        line = np.append(line, R1)
        data.append(line)
    return data

def get_camarilla(dataset):
    data = []
    line = []
    for item in dataset:
        line = item
        C = item[3]
        L = item[4]
        H = item[5]
        R1 = (H - L) * 1.1 / 12 + C
        S1 = C - (H - L) * 1.1 / 12
        line = np.append(line, S1)
        line = np.append(line, R1)
        data.append(line)
    return data

def get_Fibonacci(dataset):
    data = []
    line = []
    for item in dataset:
        line = item
        C = item[3]
        L = item[4]
        H = item[5]
        P = (H + L + C) / 3
        R1 = (2 * P) - L
        S1 = (2 * P) - H
        line = np.append(line, S1)
        line = np.append(line, R1)
        data.append(line)
    return data

def save_data(data, name):
    df = pd.DataFrame(data, columns=['DATE', 'SECID', 'OPEN', 'CLOSE', 'LOW', 'HIGH',
                                     'NEXT_OPEN', 'NEXT_LOW', 'NEXT_HIGH', 'predictMIN',
                                     'predictMAX', 'deviationMIN', 'deviationMAX', 'arimaLOW',
                                     'arimaHIGH', 'woodyLow', 'woodyHIGH', 'CamarillaLOW', 'CamarillaHIGH',
                                     'FimonacciLOW', 'FibonacciHIGH'])
    writer = ExcelWriter(name)
    df.to_excel(writer, 'dataAn', index=False)
    writer.save()
    print("Save to Excel -> Success")



def loadDataAfterSave(filein):
    # saveDataFromDBtoExcel()
    print("Start Load from file (after save data)")
    file = filein
    xl = pd.ExcelFile(file)
    df1 = xl.parse("dataAn")
    dataIn = df1
    data = [np]
    data[0] = dataIn.values
    print("End read")
    return dataIn.values  # data


def main():
    #dataset = get_data()
    #dataset = get_de_mark(dataset)
    #dataset = get_arima(dataset)
    dataset = loadDataAfterSave('dataAn3.xlsx')
    #dataset = get_woody(dataset)
    #dataset = get_camarilla(dataset)
    dataset = get_Fibonacci(dataset)
    save_data(dataset, 'dataAn4.xlsx')


main()
