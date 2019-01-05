import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

DATA_DIR = os.getenv("dataset_dir")

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
MARKET_TO_PREDICT = "LTC-USD"


main_df = pd.DataFrame()

markets = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]

def classify(current, future):
    return 1 if float(future) > float(current) else 0

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

# print(main_df[[f"{MARKET_TO_PREDICT}_close", "future", "target"]].head(10))
