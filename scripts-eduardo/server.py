import flwr as fl
import tensorflow as tf
from tensorflow import keras    
from tensorflow.keras.models import Sequential   # to flatten the input data
from tensorflow.keras.layers import Dense,Dropout,LSTM ,Conv1D,MaxPooling1D,Flatten,TimeDistributed, Input,RepeatVector,ConvLSTM2D
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from matplotlib import pyplot as plt
from typing import Dict
import sys
import numpy as np


def fit_round(rnd: int) -> Dict:
    return {"rnd": rnd}

if __name__ == "__main__":
    np.random.seed(0)
    tf.random.set_seed(0)
    strategy = fl.server.strategy.FedAvg(
            fraction_fit=0.9,
            min_fit_clients=2,
            min_available_clients=2,
            on_fit_config_fn=fit_round,
        )
    config = fl.server.ServerConfig(num_rounds=25)
    fl.server.start_server(server_address="127.0.0.1:8080", strategy=strategy, config=config)
