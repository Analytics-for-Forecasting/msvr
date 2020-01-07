from model.MSVR import MSVR
from model.utility import create_dataset,rmse

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import argparse



dataPath = 'data/MackeyGlass_t17.txt'
rawData = np.loadtxt(dataPath)

parser = argparse.ArgumentParser(
    description='MSVR for Time Series Forecasting')
parser.add_argument('-inputDim', type=int, default=10, metavar='N',
                    help='steps for prediction (default: 1)')
parser.add_argument('-outputH', type=int, default=2)

if __name__ == "__main__":
    opt = parser.parse_args()
    
    dim = opt.inputDim
    h = opt.outputH

    ts = rawData.reshape(-1)
    segmentation = int(len(ts)*2/3)

    dataset = create_dataset(ts,dim,h)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset = scaler.fit_transform(dataset)
    X, Y = dataset[:, :(0 - h)], dataset[:, (0-h):]
    train_input = X[:segmentation, :]
    train_target = Y[:segmentation].reshape(-1, h)
    test_input = X[segmentation:, :]
    test_target = Y[segmentation:].reshape(-1, h)

    msvr = MSVR()
    msvr.fit(train_input,train_target)
    trainPred = msvr.predict(train_input)
    testPred = msvr.predict(test_input)

    trainMetric = rmse(train_target,trainPred)
    testMetric = rmse(test_target,testPred)

    print(trainMetric, testMetric)
