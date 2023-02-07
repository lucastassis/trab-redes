import sys
import pandas as pd
import flwr as fl
from collections import OrderedDict
import tensorflow as tf
from tensorflow import keras    
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Dropout,LSTM ,Conv1D,MaxPooling1D,Flatten,TimeDistributed, Input,RepeatVector,ConvLSTM2D, Activation
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score, mean_absolute_percentage_error
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import csv


def scale_dataset(x_train, x_test, y_train, y_test):
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    scalers={}
    for column in x_train.columns:
      scaler = MinMaxScaler(feature_range=(-1,1))
      s_s = scaler.fit_transform(x_train[column].values.reshape(-1, 1))
      s_s=np.reshape(s_s,len(s_s))
      scalers['scaler_'+ column] = scaler
      x_train[column]=s_s

    for column in x_test.columns:
      scaler = scalers['scaler_'+column] 
      s_s = scaler.transform(x_test[column].values.reshape(-1, 1))
      s_s=np.reshape(s_s,len(s_s))
      x_test[column]=s_s
    
    y_train = target_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1))
    y_test = target_scaler.transform(y_test.to_numpy().reshape(-1, 1))
    return x_train, x_test, y_train, y_test, target_scaler, scalers

def CNN_LSTM_compile(x_train, y_train, horizon):
    #adicionando uma dimensão extra a entrada pois para as camadas conv1d e lstm a entrada precisa ter 3 dimensoes
    x_train = np.array(x_train)[...,None]
    model = tf.keras.models.Sequential([
        Conv1D(16, kernel_size=1, padding='causal',strides=1,input_shape=(x_train.shape[1], x_train.shape[2])),
        Activation('relu'),
        MaxPooling1D(strides=1),
        Conv1D(16, kernel_size=1,padding='causal', strides=1),
        Activation('relu'),
        MaxPooling1D(strides=1),
        LSTM(16, return_sequences=False),
        Activation('tanh'),
        Dense(8),
        Dense(horizon)
    ], name="lstm_cnn")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    #tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    return model



class FLClient(fl.client.NumPyClient):
    def __init__(self, model,x_train, x_test, y_train, y_test, host):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.host = host


    def get_parameters(self,config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        es = EarlyStopping(monitor='val_loss', patience=20)
        checkpoint = ModelCheckpoint('Modelos/best_model'+ self.host +'_Federado.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        self.model.fit(self.x_train, self.y_train, epochs=100,validation_split=0.1,batch_size=32, verbose=0,callbacks=[es,checkpoint])
        self.model = load_model('Modelos/best_model'+ self.host +'_Federado.h5')
        #print(f"Training finished for round {config['rnd']}")
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"mae": accuracy}

def main() -> None:
    np.random.seed(0)
    tf.random.set_seed(0)
    path = sys.argv[1]
    print(path)
    aux = path.split('\\')
    print(aux)
    processed = aux[-1][10]
    print(processed)
    horizon = 1 
    df = pd.read_csv(path)
    df.set_index('timestamp', inplace=True)
    

    target = 'bufferLevel'
    #target = 'calculatedBitrate'
    X = df.loc[:, 'CPU_use':]  
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=False, random_state=0)
    X, x_test, y, y_test, target_scaler, scalers = scale_dataset(X, x_test, y, y_test)
    model = CNN_LSTM_compile(X, y, horizon)
    client = FLClient(model,X, x_test, y, y_test,processed)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
    test_path = "E:\\Mestrado\\Laboratório-de-redes\\trab-redes-master\\trab-redes-master\\results\\last-results\\results-final-test4\\processed_1.csv"
    df_test = pd.read_csv(test_path)
    df_test.set_index('timestamp', inplace=True)
    X = df_test.loc[:, 'CPU_use':]  
    y = df_test[target]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=False, random_state=0)
    X, x_test, y, y_test, target_scaler, scalers = scale_dataset(X, x_test, y, y_test)
    pred = client.model.predict(X)

    pred_rescaled = target_scaler.inverse_transform(pred.reshape(-1, 1))

    y_test_rescaled =  target_scaler.inverse_transform(y)
    mape = mean_absolute_percentage_error(y_test_rescaled, pred_rescaled)
    rmse = mean_squared_error(y_test_rescaled, pred_rescaled, squared=False)
    score = r2_score(y_test_rescaled, pred_rescaled)
    print('R-squared score CNN-LSTM: ', round(score,4))  
    print('MAPE CNN-LSTM: ', mape)
    print('RMSE CNN-LSTM: ', rmse)

    row = [score,mape,rmse,processed]
    f = open('E:\\Mestrado\\Laboratório-de-redes\\trab-redes-master\\trab-redes-master\\scripts-eduardo\\csvs\\test1.csv', 'a')
    writer = csv.writer(f)
    writer.writerow(row)
    f.close()


    plt.figure()
    plt.plot(pred_rescaled, label="Predito")
    plt.plot(y_test_rescaled, label="Real")
    plt.legend(loc="lower center")
    plt.xlabel('Tempo')
    plt.ylabel('bufferLevel')
    #plt.ylabel('calculatedBitrate')
    #plt.title('Predição CalculatedBitrate no Teste4/h1 utilizando Aprendizado Federado')
    #plt.savefig('Imagens/Preditito vs. Bitrate Real CNN-LSTM combination para o host 1.png')  
    plt.title('Predição BufferLevel no Teste4/h1 utilizando Aprendizado Federado')
    plt.savefig('Imagens/Preditito vs. Buffer Level Real CNN-LSTM combination para o host 1.png')  


if __name__ == "__main__":
    main()
