import os
import random
import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import deque

from dotenv import load_dotenv
load_dotenv()

DATA_DIR = os.getenv("dataset_dir")

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3

markets = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]

def classify(current, future):
    return 1 if float(future) > float(current) else 0

def preprocess_df(df):
    df = df.drop('future', 1)

    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])

        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target:
            buys.append([seq, target])
        else:
            sells.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y

def get_training_data(MARKET_TO_PREDICT):
    main_df = pd.DataFrame()

    for r in markets:
        dataset = f"{DATA_DIR}/{r}.csv"
        df = pd.read_csv(dataset, names=[
                         "time", "open", "high", "low", "close", "volume"])

        df.rename(columns={
                "close": f"{r}_close",
                "volume": f"{r}_volume"
            }, inplace=True)

        df.set_index("time", inplace=True)
        df = df[[f"{r}_close", f"{r}_volume"]]

        if len(main_df):
            main_df = main_df.join(df)
        else:
            main_df = df

    main_df["future"] = main_df[f"{MARKET_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)

    main_df["target"] = list(map(classify, main_df[f"{MARKET_TO_PREDICT}_close"], main_df["future"]))

    times = sorted(main_df.index.values)

    last_5perc = times[-int(0.05*len(times))]

    validation_main_df = main_df[(main_df.index >= last_5perc)]
    main_df = main_df[(main_df.index < last_5perc)]

    train_x, train_y = preprocess_df(main_df)
    validation_x, validation_y = preprocess_df(validation_main_df)

    print(f"train data: {len(train_x)} validation: {len(validation_x)}")
    print(f"Dont buy: {train_y.count(0)}, buys: {train_y.count(1)}")
    print(f"VALIDATION dont buy: {validation_y.count(0)}, buy: {validation_y.count(1)}")

    # print(main_df[[f"{MARKET_TO_PREDICT}_close", "future", "target"]].head(10))
    return train_x, train_y, validation_x, validation_y
