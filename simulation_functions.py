# Functions for generating simulation data
import numpy as np
import pandas as pd
from scipy.stats import gamma, norm, uniform, bernoulli
from math import exp
import patsy
import statsmodels.api as sm

# define inverse logit (expit) function
def expit(x): return(exp(x)/(1 + exp(x)))

# sim function
def sim(T, k, gam, theta, patid=0, lower=0, upper=float("inf"), prop=1):
    
    # Arguments: 
    # T: number of time points
    # k: check-ups every k time points
    # gam: parameters of MSM hazard function
    # theta: parameters of conditional distributions of treatment
    # lower/upper: limits below/above which positivity is violated
    # prop: proportion of doctors who are positivity compliant.

    # define lists for holding A, L, U and Y
    A = -1*np.ones(T + 2) # A[-1] (last value) holds A in t = -1
    L = np.zeros(T+1)
    U = np.zeros(T+1)
    Y = -1*np.ones(T + 2)
    eps = np.zeros(T+1)
    lam = np.zeros(T+1) # prob of failure at each time period
    delta = np.zeros(T+1)

    # set the first value of U, U[0], to a 
    # randomly generated value from a uniform
    # distribution a measure of general health
    U[0] = uniform.rvs()
    eps[0] = norm.rvs(0, 20)
    L[0] = gamma.ppf(U[0], 3, scale=154) + eps[0]

    # set A[-1] to 0: held in last value of A
    A[-1] = 0
    Y[0] = 0
    
    # set A[0]: If L[0] is less than lower, greater than upper change A[0]
    # and also introduce proportion of doctors who are positivity compliant
    uni = uniform.rvs()
    if prop < uni: # if not positivity compliant
        if L[0] <= lower:
            A[0] = 1
        elif L[0] >= upper:
            A[0] = 0
        else:
            A[0] = bernoulli.rvs(expit(theta[0] + theta[2] * (L[0] - 500)), size=1)
    else:
        A[0] = bernoulli.rvs(expit(theta[0] + theta[2] * (L[0] - 500)), size=1)
    
    # if treatment occurs at time t = 0, then Ts (time of first treatment) is zero
    if A[0] == 1:
        Ts = 0 
    else:
        Ts = -1
    
    # initial value of lambda
    lam[0] = expit(gam[0] + gam[2] * A[0])
    
    if lam[0] >= U[0]:
        Y[1] = 1
    else:
        Y[1] = 0
        
    # loop through each time period - stop when patient is dead or t = T + 1
    for t in range(1, T+1):
        if Y[t] == 0:
            delta[t] = norm.rvs(0, 0.05)
            U[t] = min(1, max(0, U[t-1] + delta[t]))
            if t % k != 0: # not a visit -> propogate previous values
                L[t] = L[t-1]
                A[t] = A[t-1]
            else:
                eps[t] = norm.rvs(100 * (U[t] - 2), 50)
                L[t] = max(0, L[t-1] + 150 * A[t-k] * (1-A[t-k-1]) + eps[t])
                if A[t-1] == 0:
                    if prop < uni: # if not positivity compliant
                        if L[t] <= lower:
                            A[t] = 1
                        elif L[t] >= upper:
                            A[t] = 0
                        else:
                            A[t] = bernoulli.rvs(expit(theta[0] + theta[2] * (L[0] - 500)), size=1)
                    else:
                        A[t] = bernoulli.rvs(expit(theta[0] + theta[2] * (L[0] - 500)), size=1)
                else:
                    A[t] = 1
                if A[t] == 1 and A[t-k] == 0: 
                    Ts = t
            lam[t] = expit(gam[0] + gam[1] * ((1 - A[t]) * t + A[t] * Ts) + gam[2] * A[t] + gam[3] * A[t] *(t - Ts))
            if (1 - np.prod(1 - lam)) >= U[0]:
                Y[t + 1] = 1
            else:
                Y[t+1] = 0
        else:
            break
    
    # we only need the data before death, so whatever value t is before
    # the end of the above loop 
    Y = np.ndarray.tolist(Y[1:(t+1)])
    U = np.ndarray.tolist(U[0:t])
    L = np.ndarray.tolist(L[0:t])
    A = np.ndarray.tolist(A[0:t])
    Ts = [Ts]*t

    df = np.vstack((Y, L, U, A, Ts))
    df = pd.DataFrame(df.T, columns=['Y', 'L', 'U', 'A', 'Ts'])
    df['Y'] = df['Y'].astype(int)
    df['A'] = df['A'].astype(int)
    df['patid'] = patid
    df.index.name = 'visit'
    return df.reset_index()

# function for generating data using the sim function for n patients
def sim_n(T, k, gam, theta, n = 1000, lower=0, upper=float("inf"), prop=1):

    frames = [sim(T, k, gam, theta, patid=i, lower=lower, upper=upper, prop=prop) for i in range(n)]
    df = pd.concat(frames)
    df = df.set_index(['patid', 'visit'])
    df = df.sort_index()
    return(df)

# for the IPW method we need to append weights to the data.
# The weights are calcualted using the same method as 
# those sent to use by Vanessa Didelez
def get_weights(df):

    # only need data when A is not 1 yet
    df["Ladj"] = df["L"] - 500
    df["As"] = df.groupby(level="patid")['A'].cumsum()
    df2 = df[df["As"] <= 1].copy(deep=True)
    df2 = df2.reset_index()
    df2 = df2[df2["visit"] % 5 == 0] # only need actual visits

    # numerator
    f = "A ~ visit"
    y, X = patsy.dmatrices(f, df2, return_type = "dataframe")
    n_logit = sm.Logit(y, X, missing="raise")
    n_result = n_logit.fit(disp=0, maxiter=100)
    df2["pn"] = n_result.predict()
    
    # denominator
    f = "A ~ visit + Ladj"
    y, X = patsy.dmatrices(f, df2, return_type = "dataframe")
    d_logit = sm.Logit(y, X, missing="raise")
    d_result = d_logit.fit(disp=0, maxiter=100)
    df2["pd"] = d_result.predict()

    # if A == 0, change probabilities to 1 - prob
    df2['pn2'] = np.where(df2['A']==0, (1 - df2["pn"]), df2["pn"])
    df2['pd2'] = np.where(df2['A']==0, (1 - df2["pd"]), df2["pd"])

    # construct stabilized weights, don't forget to group by
    df2['cpn'] = df2.groupby(level=0)['pn2'].cumprod()
    df2['cpd'] = df2.groupby(level=0)['pd2'].cumprod()
    df2['sw'] = df2['cpn']/df2['cpd']
    
    # set index
    df2 = df2.set_index(['patid', 'visit'])

    #combine df and df2
    df["sw"] = np.nan
    df.loc[df2.index, "sw"] = df2["sw"]
    df["sw"] = df["sw"].fillna(method="pad")
    return(df)