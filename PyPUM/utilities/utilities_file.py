# %% [markdown]
# # Utilities
# 
# This file collects a few useful functions for plotting, regression tables, computation, etc. 

# %%
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
import os
import sys
from numpy import linalg as la
from scipy import optimize
import scipy.stats as scstat
from matplotlib import pyplot as plt
import itertools as iter

# Files
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utilities.Logit_file import estimate_logit, logit_se, logit_t_p, q_logit, logit_score, logit_score_unweighted, logit_ccp, LogitBLP_estimator

# %%
# This function tests whether the utility parameters are identified, by looking at the rank of the stacked matrix of explanatory variables.

def rank_test(x):
    '''
    This function tests whether the utility parameters are identified, by looking at the rank of the stacked matrix of explanatory variables.

    Args:
        x: a dictionary (length T) of (J[t],K) numpy arrays or itself a numpy array (T,J,K) of the Covariates
    
    Returns.
        a print statement concerning the rank of x
    '''

    if (isinstance(x, (np.ndarray))):
        if (x.ndim == 3):
            T,J,K = x.shape
            xpsied = x.reshape((T*J,K))
        else:
            print('x is array of dim != 3')
    else:
        T = len(x)
        xpsied = np.concatenate([x[t] for t in np.arange(T)], axis = 0)
    
    eigs = la.eigvals(xpsied.T@xpsied)

    if np.min(eigs)<1.0e-8:
        print('x does not have full rank')
    else:
        print('x has full rank')

# %%
def kernel_estimate(x_cont, cont_vars, x_min, x_max, n_points, J, outside_option: bool = True):
    '''
    This function calculates kernel density estimates of each of the densities governing the distribution of the continuous covariates using a Gaussian kernel function.

    Args:
        x_cont: a numpy array (J[t],D_cont) of continuous covariates for a given market t
        cont_vars: a list of the labels of the continuous covariates
        x_min: a numpy array (D_cont,) of the minimum value within each characteristic of x_cont. Should be compatible with the value of outside_option (i.e. if outside_option = True, x_min should be the lowest value in x_cont when disregarding the outside_option)
        x_max: a numpy array (D_cont,) of the maximum value within each characteristic of x_cont. Should be compatible with the value of outside_option
        n_points_ an integer of the amount of points to evaluate the kernel estimator in.
        J: an integer describing the amount of alternatives in the given market t (i.e. J[t]). Should be compatible with the value of outside_option.
        outside_option: a boolean of whether outside is included in x_cont or not. If 'True' outside option is included in x_cont.
    
    Returns.
        f_hat: a (G,n_points) array of the kernel estimated density for each continuous variable in cont_vars 
    '''

    D_cont = len(cont_vars)
    
    if outside_option:
        z_cont = np.linspace(x_min, x_max, n_points).transpose()
        diff = z_cont[:,:,None]*np.ones((D_cont, n_points, J - 1)) - x_cont.transpose()[:,None,1:]
        IQR = scstat.iqr(x_cont[1:,:], axis = 0) # Compute interquartile range of each continuous variable
        sd = np.std(x_cont[1:,:], axis = 0) # Compute empirical standard deviation of each continuous variable
        h = 0.9*np.fmin(sd, IQR/1.34)/((J-1)**(1/5)) # Silverman's rule of thumb
    else:
        z_cont = np.linspace(x_min, x_max, n_points).transpose()
        diff = z_cont[:,:,None]*np.ones((D_cont, n_points, J)) - x_cont.transpose()[:,None,:]
        IQR = scstat.iqr(x_cont, axis = 0) # Compute interquartile range of each continuous variable
        sd = np.std(x_cont, axis = 0) # Compute empirical standard deviation of each continuous variable
        h = 0.9*np.fmin(sd, IQR/1.34)/(J**(1/5)) # Silverman's rule of thumb

    K = np.exp(-(diff**2)/(2*(h[:,None,None]**2))) / (np.sqrt(2*np.pi)*h[:,None,None]) # Use a gaussian kernel function

    f_hat = K.mean(axis=2)

    return f_hat

# %%
def numerical_grad(y, x, theta, sample_share, loglikelihood, specification, model = 'IPDL', delta = 1.0e-8):
    ''' 
    This function calculates the numerical and the analytical score functions at a given parameter \theta aswell the norm of their difference

    Args:
        Theta: a numpy array (K+G,) of parameters of (\beta', \lambda')',
        y: a dictionary of T numpy arrays (J[t],) of observed market shares in onehot encoding for each market t,
        x: a dictionary of T numpy arrays (J[t],K) of covariates for each market t,
        sample_share: a numpy array (T,) of the share of observations of each market t = 1,...,T
        model: a dictionary of the Similarity model specification as outputted by 'Similarity_specification'
        delta: the incremental change in the argument, a float, used in calculating numerical gradients

    Returns.
        normdiff: a float of the euclidean norm of the difference between the numerical and analytical score functions at \theta
        angrad: a numpy array (T,K+G) of analytical Similarity scores
        numgrad: a numpy array (T,K+G) of numerical Similarity scores
    '''

    T = len(x)
    K = x[0].shape[1]
    G = len(theta[K:])

    numgrad = np.empty((T, K+G))

    if model == 'IPDL':
        for i in np.arange(K+G):
            vec = np.zeros((K+G,))
            vec[i] = 1
            numgrad[:,i] = (loglikelihood(theta + delta*vec, y, x, sample_share, specification[0], specification[1]) - loglikelihood(theta, y, x, sample_share, specification[0], specification[1])) / delta

    else:
        for i in np.arange(K+G):
            vec = np.zeros((K+G,))
            vec[i] = 1
            numgrad[:,i] = (loglikelihood(theta + delta*vec, y, x, sample_share, specification) - loglikelihood(theta, y, x, sample_share, specification)) / delta
    
    return numgrad

# %%
def Reg_t_p(SE, Theta, N, Theta_hypothesis = 0):
    ''' 
    This function calculates t statistics and p values for characteristic and nest grouping parameters

    Args.
        SE: a numpy array (K+G,) of asymptotic Similarity MLE standard errors
        Theta: a numpy array (K+G,) of parameters of (\beta', \lambda')',
        N: an integer giving the number of observations
        Theta_hypothesis: a (K+G,) array or integer of parameter values to test in t-test. Default value is 0.
    
    Returns
        T: a (K+G,) array of estimated t tests
        p: a (K+G,) array of estimated asymptotic p values computed using the above t-tests
    '''

    T = np.abs(Theta - Theta_hypothesis) / SE # Compute two-sided t-tests
    p = 2*scstat.t.sf(T, df = N-1) # Compute p-values

    return T,p

# %%
def reg_table(theta, se, N, x_vars, nest_vars):
    '''
    This function constructs a regression table based on Similarity parameter standard error estimates

    Args:
        theta: a (K+G,) numpy array of estimated parameters
        se: a (K+G,) numpy array of estimated standard errors
        N: an integer; the number of observations
        x_vars: a list containing the names of the covariates
        nest_vars: a list containing the names of the nesting groups

    Returns.
        table: a pandas dataframe structured as a regression table w. parameter estiamtes, standard errors, t-tests, and p-values  
    '''
    Similarity_t, Similarity_p = Reg_t_p(se, theta, N) # Get t-test values and p values

    regdex = [*x_vars, *['group_' + var for var in nest_vars]] # Set the names of the covariates and the nesting groups as the index

    table  = pd.DataFrame({'theta': [ str(np.round(theta[i], decimals = 4)) + '***' if Similarity_p[i] <0.01 else str(np.round(theta[i], decimals = 3)) + '**' if Similarity_p[i] <0.05 else str(np.round(theta[i], decimals = 3)) + '*' if Similarity_p[i] <0.1 else np.round(theta[i], decimals = 3) for i in range(len(theta))], # Give stars to parameter estimates according to t-tests at levels of significance 0.1, 0.05, and 0.01
                'se' : np.round(se, decimals = 5),
                't (theta == 0)': np.round(Similarity_t, decimals = 3),
                'p': np.round(Similarity_p, decimals = 3)}, index = regdex).rename_axis(columns = 'variables')
    
    return table

# %% [markdown]
# ## Regularization for parameter bounds
# 
# As we see above, the least squares estimator is not guaranteed to respect the parameter bounds $\sum_g \hat \lambda_g<1$. We can use that if we replace $\hat q^0_t$ with the choice probabilities from the maximum likelihood estimator of the logit model, $\hat q^{logit}_t\propto \exp\{X_t\hat \beta^{logit}\}$, and plug these choice probabilities into the WLS estimator described above, it will return $\hat \theta=(\hat \beta^{logit},0,\ldots,0)$ as the parameter estimate. Let $\hat q_t(\alpha)$ denote the weighted average of the logit probabilites and the market shares,
# $$
# \hat q_t(\alpha) =(1-\alpha) \hat q^{logit}_t+\alpha \hat q^0_t.
# $$
#  Let $\hat \theta^0(\alpha)$ denote the resulting parameter vector. We perform a line search for values of $\alpha$, $(\frac{1}{2},\frac{1}{4},\frac{1}{8},\ldots)$ until $\hat \theta^0(\alpha)$ yields a feasible parameter vector.
# 

# %%
def LineSearch(Logit_Beta, q_obs, x, sample_share, estimator, model, N):
    '''
    This function performs a line search to find feasible lambda parameters

    Args:
        Logit_beta: a (K,) numpy array of estimated beta parameters from a corresponding Logit model
        q_obs: a dictionary of T numpy arrays (J[t],) of observed market shares in onehot encoding for each market t,
        x: a dictionary of T numpy arrays (J[t],K) of covariates for each market t,
        sample_share: a numpy array (T,) of the share of observations of each market t = 1,...,T,
        model: a dictionary of the Similarity model specification as outputted by 'Similarity_specification',
        N: an integer giving the number of observations

    Returns.
        theta_alpha: a (K+G,) numpy array of feasible parameters found by line search
    '''

    # Get dimensions of data
    T = len(x)
    K = x[0].shape[1]

    # Find probabilities
    q_logit = logit_ccp(Logit_Beta, x)

    # Search over alphas s.t. alpha = (1/2)^{k} for some positive integer k
    alpha0 = 0.5

    for k in np.arange(1,100):

        # Set alpha
        alpha = alpha0**k 
        
        # Compute convex combination of ccp's
        q_alpha = {t: (1 - alpha)*q_logit[t] + alpha*q_obs[t] for t in np.arange(T)}
        theta_alpha = estimator(q_alpha, x, sample_share, model, N) # Compute initial FKN parameters but using q_alpha ccp's 

        lambda_alpha = theta_alpha[K:] # Find lambda parameters
        pos_pars = np.array([theta for theta in lambda_alpha if theta > 0]) # Find positive lambda parameters

        if pos_pars.sum() <1:
            break # Break if positive parameters sum to less than 1

    return theta_alpha

# %%
def GridSearch(Logit_Beta, y, x, sample_share, mean_loglikelihood, estimator, model, N, num_alpha = 5):
    '''
    This function performs a grid search on the unit interval to find feasible parameters \theta

    Args:
        Logit_beta: a (K,) numpy array of estimated beta parameters from a corresponding Logit model
        y: a dictionary of T numpy arrays (J[t],) of observed market shares in onehot encoding for each market t,
        x: a dictionary of T numpy arrays (J[t],K) of covariates for each market t,
        sample_share: a numpy array (T,) of the share of observations of each market t = 1,...,T,
        model: a dictionary of the Similarity model specification as outputted by 'Similarity_specification',
        N: an integer giving the number of observations,
        num_alpha: an integer of the number of alphas for which the search is to be performed

    Returns.
        theta_star: a (K+G,) numpy array of feasible parameters found by grid search
    '''

    T = len(x)
    J0 = x[0].shape[0]
    psi_3d0 = model['psi_3d'][0]
    G = np.int64(psi_3d0.shape[0] - 1)
    K = x[0].shape[1]

    # Find probabilities
    q_logit = logit_ccp(Logit_Beta, x)
    q_obs = y

    # Search
    alpha_line = np.linspace(0, 1, num_alpha)
    LogL_alpha = np.empty((num_alpha,))
    theta_alpha = np.empty((num_alpha,K+G))

    for k in np.arange(len(alpha_line)):

        alpha = alpha_line[k]

        q_alpha = {t: (1 - alpha)*q_logit[t] + alpha*q_obs[t] for t in np.arange(T)}
        theta_alpha[k,:] = estimator(q_alpha, x, sample_share, model, N)

        #lambda_inout = theta_alpha[k,K]
        lambda_alpha = theta_alpha[k,K:] # theta_alpha[k,K+1:]
        pos_pars = np.array([theta for theta in lambda_alpha if theta > 0])

        if (pos_pars.sum() >= 1): #|(lambda_inout >= 1)
            LogL_alpha[k] = np.NINF
        else:
            LogL_alpha[k] = mean_loglikelihood(theta_alpha[k,:], y, x, sample_share, model)
    
    # Pick the best set of parameters
    alpha_star = np.argmax(LogL_alpha)
    theta_star = theta_alpha[alpha_star,:]

    return theta_star

# %%
def reg_comparison(estimator_names, variable_names, theta, se, p):
    ''' 
    This function construct a table comparing estimates, standard errors, and significances.

    Args:
        estimator_names: a list of names of varies models and/or estimators
        variable_names: a list of variables names for parameters
        theta: a dictionary of numpy arrays of parameters (ordered according to and congruently with variable names)
        se: a dictionary of numpy arrays of parameter standard errors (ordered according to and congruently with variable names)
        p: a dictionary of numpy arrays of parameter p-values (ordered according to and congruently with variable names)
    
    Returns.
        reg_comp:
    '''
    
    T = len(estimator_names)
    
    reg_comp_dict = {}
    
    # Find the dimension of the full parameter space
    d = np.max(np.array([theta[i].shape[0] for i in len(theta)])) 
    
    # Construct columns to show filling in empty nest variables where relevant with a dash '-'
    for t in np.arange(T):
        if theta[t].shape[0] == d:
            theta_show = [*theta[t]]
            se_show = [*se[t]]
            p_show = [*p[t]]
        else:
            theta_show = [*theta[t],*['-' for i in np.arange(d - theta[t].shape[0])]]
            se_show = [*se[t], *['-' for i in np.arange(d - theta[t].shape[0])]]
            p_show = [*p[t], *[ 1 for i in np.arange(d - theta[t].shape[0])]]

        reg_comp_dict[estimator_names[t]] = [par + '***' + ' (' + se + ')' if p < 0.01 else par + '**' + ' (' + se + ')' if p < 0.05 else par + '*' + ' (' + se + ')' if p < 0.1 else par + ' (' + se + ')' for p,par,se in zip(p_show, theta_show, se_show)]

    reg_comp = pd.Dataframe(reg_comp_dict, index = variable_names)

    return reg_comp


