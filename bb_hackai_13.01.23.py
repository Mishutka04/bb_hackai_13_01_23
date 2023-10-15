import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.src.layers import GRU, Dropout

clip = np.load('dataset/clip.npy')
w = np.load('dataset/weight.npy')
x_train = np.load("dataset/y_smp_test.npy")
y_train = np.load("dataset/pars_smp_train.npy")


model = Sequential()
model.add(GRU(units=128,
              return_sequences=True,
              input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(GRU(units=64))
model.add(Dense(32))
# model.add(Dropout(0.5))
model.add(Dense(15))
model.summary()
model.compile(optimizer='adam',
              loss='mse', metrics=["mae"])

model.compile(optimizer='adam',
              loss='mse', metrics=["mae"])

model.fit(x_train[:100000,:,], y_train[:100000,:,], epochs=5, batch_size=128)

pred = model.predict(x_train[:100001:200001])

def create_models(x_train, y_train, s):
    model = Sequential()
    model.add(GRU(units=128,
                  return_sequences=True,
                  input_shape=(15, 1)))
    # model.add(GRU(units=64))
    model.add(Dense(32))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer='adam',
                  loss='mse', metrics=["mae"])
    model.fit(x_train,
              y_train[:, :, s],
              epochs=5, batch_size=128, validation_split=0.1)
    return model

def merge_arrays(arr1, arr2, arr3, arr4, arr5, arr6):
    fin_arr = np.empty((arr1.shape[0],arr1.shape[1],6))
    for i in range(len(arr1)):
        for j in range(len(arr1[0])):
            semi_arr = np.hstack([arr1[i][j], arr2[i][j],
                                      arr3[i][j], arr4[i][j],
                                      arr5[i][j], arr6[i][j]])
            fin_arr[i][j] = semi_arr
    return fin_arr

arr = []

x_train = np.load('dataset/pars_smp_test.npy')
y_train = np.load('dataset/random_submit.npy')

for i in range(6):
        model = create_models(x_train, y_train, i)
        pred = model.predict(pred)
        print(pred.shape)
        arr.append(pred[:100000])
        
fin_arr = merge_arrays(arr[0], arr[1],arr[2],arr[3],arr[4],arr[5])

def calculate_metrics(forecast, true_values, weights = 1/15*np.ones(shape=(15,11)), clip = 10**6*np.ones(shape=(15,6)), quantiles = [0.1,0.25,0.5,0.75,0.9]):
    # Define diff between forecast
    diff = true_values - forecast
    # Mean loss
    RMSE = np.mean(np.minimum(diff[:,:,0]**2, clip[None,:,0]), axis=0)**0.5
    L_mean = np.exp(np.sum(-weights[:,0]*RMSE))
    # Quantile loss
    quantile_loss = np.mean(np.minimum(((diff[:,:,1:]>0)*(np.array(quantiles)[None,None,:]) + (diff[:,:,1:]<=0)*(1 - np.array(quantiles)[None,None,:]))*abs(diff[:,:,1:]), clip[None,:,1:]), axis=0)
    L_quantile = np.exp(np.sum(-weights[:,1:6]*quantile_loss))
    # Calibration loss
    L_calibration = np.exp(np.sum(-weights[:,6:]*np.abs(np.array(quantiles)-np.mean(diff[:,:,1:]<0, axis=0))))
    return 0.45*L_mean + 0.45*L_quantile + 0.1*L_calibration


metrics = calculate_metrics(fin_arr, x_train, weights= w, clip= clip, quantiles = [0.1,0.25,0.5,0.75,0.9])
with open('parametr.npy', 'wb') as f:
    np.save(f, metrics)