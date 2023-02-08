from tabnanny import verbose
import pandas as pd
import sys
import tensorflow as tf
from tensorflow.keras import backend as K    
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense,Dropout,LSTM ,Conv1D,MaxPooling1D,Flatten,TimeDistributed, Input,RepeatVector,GlobalAveragePooling1D,BatchNormalization,AlphaDropout,Activation
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from matplotlib import pyplot as plt
from typing import Dict
import numpy as np
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_percentage_error
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

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
        Conv1D(32, kernel_size=1, padding='causal',strides=1,input_shape=(x_train.shape[1], x_train.shape[2])),
        Activation('relu'),
        MaxPooling1D(strides=1),
        Conv1D(32, kernel_size=1,padding='causal', strides=1),
        Activation('relu'),
        MaxPooling1D(strides=1),
        LSTM(32, return_sequences=False),
        Activation('tanh'),
        Dense(32),
        Dense(horizon)
    ], name="lstm_cnn")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    #tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    return model


if __name__ == "__main__":
    np.random.seed(0)
    tf.random.set_seed(0)
    path = sys.argv[1]
    aux = path.split('/')
    processed = aux[-1][10]
   
    diff = False
    n_steps = 3
    horizon = 1 
    df = pd.read_csv(path)
    df.set_index('timestamp', inplace=True)
    
    #adiciona_lag(df, n_steps, horizon,diff)
    #columns = ['frameRate']+dataframe.columns[dataframe.columns.str.contains('horizon_frameRate_')].to_list()
    X = df.loc[:, 'CPU_use':]  
    print(X.head())
    y = df['calculatedBitrate']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=False, random_state=0)
    x_train, x_test, y_train, y_test, target_scaler, scalers = scale_dataset(x_train, x_test, y_train, y_test)
    es = EarlyStopping(monitor='val_loss', patience=500)
    model = CNN_LSTM_compile(x_train, y_train, horizon)
    checkpoint = ModelCheckpoint('Modelos/best_model '+ processed + '.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    history = model.fit(x_train, y_train, epochs=500, validation_split=0.10, batch_size=32,verbose=0,callbacks=[es,checkpoint])
    model = load_model('Modelos/best_model '+ processed + '.h5')
    pred = model.predict(x_test)
    pred_rescaled = target_scaler.inverse_transform(pred)
    y_test_rescaled =  target_scaler.inverse_transform(y_test)
    mape = mean_absolute_percentage_error(y_test_rescaled, pred_rescaled)
    rmse = mean_squared_error(y_test_rescaled, pred_rescaled, squared=False)
    score = r2_score(y_test_rescaled, pred_rescaled)
    print('R-squared score CNN-LSTM: ', round(score,4))  
    print('MAPE CNN-LSTM: ', mape)
    print('RMSE CNN-LSTM: ', rmse)
    plt.figure()
    plt.plot(pred_rescaled, label="Predito")
    plt.plot(y_test_rescaled, label="Real")
    plt.legend(loc="upper left")
    plt.xlabel('Time')
    plt.ylabel('Bitrate')
    plt.title('Preditito vs. Real Bitrate CNN-LSTM para o host ' + str(processed))
    plt.savefig('Imagens/Preditito vs. Real Bitrate CNN-LSTM para o host ' + str(processed) +'.png')  

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Learning curve para host ' + str(processed))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('Imagens/Learning curve para host ' + str(processed) +'.png') 
    plt.figure()
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('MAE curve para host ' + str(processed))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('Imagens/MAE curve para host ' + str(processed))  
    
    
