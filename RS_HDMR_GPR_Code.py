import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools


#### define a function that computes rmse
def rmse(ypred, ytest, scale_factor):
    Sqerr = np.power(ytest - ypred, 2)
    MSE = np.sum(Sqerr)
    rmse = scale_factor * np.sqrt(MSE / ytest.size)
    return rmse


#### define a function to sum columns
def sumcol(F, j):
    s = np.zeros(F.shape[0])
    for i in range(F.shape[1]):
        if i == j:
            i += 1
        else:
            s = s + F[:, i]
    return s


def GPRHDMR(X, y, X_train, y_train, X_test, y_test, order, noise_output, scale_factor, length_scale=0.3, sigma_f=1,
            number_cycles=1, init='naive', mixe='no', optimizer=None):
    """
    This function fits  a RS-HDMR-GPR to data using independent Gaussian Processes for component functions. GPR is used from the python package GaussianProcessRegressor from sklearn.
    :parameters of the function are:
    X_train: the training input data as returned from train_test_split function.
    y_train: the training output data as returned from train_test_split function.
    X_test: the test input data as returned from train_test_split function.
    y_test: the test output data as returned from train_test_split function.
    order: the order of the HDMR to use.
    noise_output: Value added to the diagonal of the kernel matrix during fitting.  Note that this is equivalent to adding a WhiteKernel with c=alpha.
    length_scale: length scale of the Gaussian kernel
    number of Guassian models to be used.
    :param matrices: list of 2D numpy.Arrays
    Represents the linear transformation applied to the data input (No bias term has yet been added).
    The list size must equal num_models.
         :param kernels: a list of Objects from sklearn.guassian_process.kernels
    the kernels to be used for training each HDMR.
    """
    all_combos = np.array(list(itertools.combinations(X_train, order)))
    mean = y_train.mean()
    if init == 'naive':
        yc = (1 / all_combos.shape[0]) * mean * np.ones((X_train.shape[0], all_combos.shape[
            0]))  # initialize the matrix for the of component functions to zeros, shape is n*D
    if init == 'poly':
        yc = np.ones((X_train.shape[0], all_combos.shape[
            0]))  # initialize the matrix for the of component functions to zeros, shape is n*D
        for i in range(0, all_combos.shape[0]):
            x = pd.DataFrame(X_train)[all_combos[i][0]]
            print(x.ndim)
            # f = interpolate.interp1d(x, y)
            f = np.polyfit(x, y_train, 3)
            xnew = x
            f = np.poly1d(f)
            yc[:, i] = f(x)  # use interpolation function returned by `interp1d`
    ###define at first one GPR to the D component functions
    l = length_scale
    rbf = ConstantKernel(constant_value=sigma_f, constant_value_bounds=(1e-2, 1e2)) * \
          RBF(length_scale=l, length_scale_bounds=(1e-2, 1e2))  # + WhiteKernel(noise_level=1e-05)
    GPR = [GaussianProcessRegressor(kernel=rbf, alpha=noise_output, optimizer=optimizer) for i in
           range(0, all_combos.shape[0])]
    for k in range(number_cycles):
        print('cycle number', k + 1)
        for i in range(0, all_combos.shape[0]):  # loop for in range the number of component functions

            vect = y_train - sumcol(yc, i)
            yc[:,
            i] = vect  # step1: output - inititializations ( but note here sir, for the zero initialization yc(1)=the output y)
            xx = pd.DataFrame(X_train)[all_combos[i]]  # just reshapint the input to be adequate with the model
            GPR[i].fit(xx, yc[:, i])  # fit
            if mixe == 'yes':
                yc[:, i] = (GPR[i].predict(xx) + vect) / 2
            else:
                yc[:, i] = GPR[i].predict(xx)  # predict to re use yc
            # print('hyper parameters are:', GPR[i].kernel_)
        rmse_train = rmse(y_train, sumcol(yc, 10000), scale_factor)
        print('train rmse', rmse(y_train, sumcol(yc, 10000), scale_factor))  ## compute rmse (it is computed on y_train)
    mean = y_train.mean()
    yct = np.zeros((X.shape[0], all_combos.shape[0]))
    for i in range(0, all_combos.shape[0]):  # loop for in range the number of component functions
        # yct[:,i]=y_test-sumcol(yct,i)# step1: output - inititializations
        xt = pd.DataFrame(X)[all_combos[i]]  # just reshapint the input to be adequate with the model
        yct[:, i] = GPR[i].predict(xt)  # predict to re use yct
    print('order of the HDMR is', order)
    print('test rmse', rmse(y, sumcol(yct, 10000), scale_factor))
    rmse_test = rmse(y_, sumcol(yct, 10000), scale_factor)
    fig, ax1 = plt.subplots()
    ax1.plot(y, sumcol(yct, 10000), 'bo', markersize=3)
    ax1.set_xlabel('Target')
    ax1.set_ylabel('Predictions')
    # ax1.set_title('predicted vs real output plot on test data')
    ax1.grid(True)
    fig, ax2 = plt.subplots()
    ax2.plot(sumcol(yc, 1000), y_train, 'bo', markersize=3)
    ax2.set_xlabel('Predicted output')
    ax2.set_ylabel('real output values')
    ax2.set_title('predicted vs real output plot on train data')
    ax2.grid(True)
    ypred = sumcol(yct, 10000)
    y_pred_scaled = ypred * scale_factor
    return rmse_train, rmse_test, sumcol(yct, 50000), GPR, y_pred_scaled


# 15D
df = pd.read_table('/project/6034927/dali/Nick50kQmax.dat', header=None, delimiter=r"\s+")
x = df.iloc[:, 0:15]
y = df.iloc[:, -1]
scale_factor = (y.max() - y.min())

X = (x - x.min()) / (x.max() - x.min())
# X=(x-x.mean())/x.std()
y = (y - y.min()) / (y.max() - y.min())
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.091, test_size=None, random_state=20)
print('scale_factor=', scale_factor)
# end 15D dataset read

HDMR = GPRHDMR(X, y, X_train, y_train, X_test, y_test, 4, 1e-5, scale_factor=scale_factor, length_scale=1.9,
               sigma_f=1 ** 2, number_cycles=100, init='poly', mixe='no', optimizer=None)
print('error out of 100 % is', (HDMR[1] / scale_factor) * 100, '%')

df[7] = HDMR[5]
df.to_numpy()
np.savetxt('/project/6034927/dali/UCL_4orderhdmr_5004trainpoints_rmse_isotropic.dat', df, fmt='%13.4f', delimiter='\t')
