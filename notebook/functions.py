import numpy as np
import math
#from keras.layers import Dense
#from keras.models import Sequential
from scipy.optimize import minimize
from pynverse import inversefunc

class NormalExample:
    def __init__(self, dim, mean, cov):
        mean = np.array(mean)
        cov = np.array(cov)

        self.dim = dim
        assert len(mean) == dim, "Mean dimension not good"
        self.mean = mean

        assert cov.shape[0] == cov.shape[1], "Matrix should be squared"
        assert cov.shape[0] == dim, "COV Dimension is not good"
        self.cov = cov

    def get_data(self, n):

        samples = np.random.multivariate_normal(self.mean, self.cov, n)

        return samples


def f_t_cesa(x, n, delta, t):
    C = np.log(2*n*(n+2)/delta)
    D = x + (11*C/3) * (np.log(n-t)+1) / (n-t) + 2 * np.sqrt((2*C*x)/(n-t))
    return D

"""
def f_t_Touati(x, n, delta, t):
    B = np.log(n*(n+2)/delta)
    C = np.log(2*n*(n+2)/delta)
    phi_t_x = np.sqrt(
        np.sqrt(1 + 4/9 * B * x/(n-t))
        - 1
    )
    D = x + (np.sqrt(C/2) * (np.sqrt(x) + np.sqrt((phi_t_x)**2/x))*np.sqrt(np.log(n-t)/n-t))
    return D
"""

def f_t_Touati(x, n, delta, t):
    C = np.log(2*n*(n+2)/delta)
    phi_t_x = np.sqrt(2.6*C*x/(n-t))
    D = x + np.sqrt(C/3) * np.sqrt(1+4*(x+phi_t_x)+(x+phi_t_x)**2)*np.sqrt(np.log(n-t)/(n-t))
    return D


def cesa_up_bd(x, n, delta):
    p1 = 36/n * np.log((2*n*(n + 3))/delta)
    p2 = 2*np.sqrt(x/n * np.log((2*n*(n+3) / delta)))
    up_bd = x + p1 + p2
    return up_bd


def Touati_up_bd_2(x, n, delta):
    p1 = 2/3 * np.log(2*n/delta)/n
    res = x + p1 + np.sqrt(1/3 * np.log(2*n/delta)/n *(2 + 4*x + 4/3 * np.log(2*n/delta)/n))
    return res


def cesa_up_final(x, n, delta):
    bound = cesa_up_bd(x, n, delta)
    D = f_t_cesa(bound, n, delta, 0)
    return D

# define baseline model
def keras_model(X):
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def C_a(a):
    C = (2*(1-(2*a)+2*np.sqrt(a*(a+1)))/(8*a-1))
    return C


def Touati_up_bd(x, n, delta, a):
    A = x + (a * C_a(a) * np.log(2 * n / delta))/n
    A1 = a*np.log(2*n/delta)/n
    C = 2+2*C_a(a)*x+(a*(C_a(a)**2)*np.log(2*n/delta))/n
    D = np.sqrt(A1*C)
    B = A+D
    return B


def Touati_up_final(x, n, delta, a):
    bound = Touati_up_bd(x, n, delta, a)
    D = f_t_Touati(bound, n, delta, 0)
    return D


def Touati_up_final_2(x, n, delta):

    bound = Touati_up_bd_bd(x, n , delta)
    D = f_t_Touati(bound, n, delta, 0)
    return D



def Touati_up_bd_bd(x, n , delta): 
    
    all_y = x
    all_x = []
    for y in all_y:
        error = lambda x: (function_touati(x,n,delta) - y)**2
        res = minimize(error, x0=0.5, bounds=[(0.01, 0.999)])
        all_x.append(res.x[0])

    all_x = np.array(all_x)
    return all_x




def function_touati(x, n , delta): 
    y = x - np.sqrt( ( (1-x)**2/np.log(1/x) ) * np.log(1/delta)/n)
    return y









