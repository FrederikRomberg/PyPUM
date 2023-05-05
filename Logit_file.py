# %% [markdown]
# # Modeling Demand for Cars with the Multinomial Logit Model
# 
# In this notebook, we will explore the dataset used in
# Brownstone and Train (1999). We will estimate the Multinomial Logit Model
# model given the available data using the functions defined below.
# 

# %%
import numpy as np
import pandas as pd 
import os
from numpy import linalg as la
from scipy import optimize
from IPython import display
from matplotlib import pyplot as plt
import itertools as iter

# %% [markdown]
# Data
# ====
# 
# The data consists of a survey of households regarding their preferences
# for car purchase. Each household was given 6 options, but the
# characteristics that the respondents were asked about was varied. The
# surveys were originally conducted in order to illicit consumer
# preferences for alternative-fuel vehicles. The data is *stated
# preferences*, in the sense that consumers did not actually buy but just
# stated what they would hypothetically choose, which is of course a
# drawback. This is very common in marketing when historic data is either
# not available or does not provide satisfactory variation. The advantage
# of the stated preference data is therefore that the choice set can be
# varied greatly (for example, the characteristics includes the
# availability of recharging stations, which is important for purchase of
# electric cars).
# 
# The data has $N=4654$ respondents with $J=6$ cars to choose
# from.
# 
# Loading the dataset, `car_data.csv`, we get a dataframe with 
# $NJ = 27,924$ rows. The column `person_id` runs through $0,1,...,N-1$, and
# the column `j` is the index for the car, $\{0,1,...,5\}$. The variable 
# `binary_choice` is a dummy, =1 for the car chosen by the respondent. 
# A conveneint other variable, `y`, is the index for that car, repeated 
# and identical for all $J$ rows for each person. The x-variables describe 
# the characteristics of the 6 cars that the respondent was asked to choose 
# from. 
# 
# We also read in the dataset `car_labels.csv`, which contains the 
# variable labels and descriptions for all the variables. 
# The lists `x_vars` and `x_lab` will be used throughout as the list of 
# explanatory variables we want to work with. 
# 
# In order to get the data into a 3-dimensional array, we access 
# the underlying numpy arrays and resize them. For example 
# 
# > `x = dat[x_vars].values.resize((N,J,K))`
# 
# Note that this will only work because the data is sorted according to 
# first `person_id` and then `j`. You can test this by verifying that 
# `x[0,:,k]` prints the same as `dat.loc[dat.person_id == 0, x_vars[k]]`. 

# %%
# Load dataset and variable names
os.chdir('../GREENCAR_notebooks/')
input_path = os.getcwd() # Assigns input path as current working directory (cwd)
dat = pd.read_csv(os.path.join(input_path, 'car_data.csv'))
lab = pd.read_csv(os.path.join(input_path, 'car_labels.csv'), index_col = 'variable')

# %%
display.Image('brownstone_train_tab_1.PNG')

# %% [markdown]
# Table 1 from 'Forecasting new product penetration with flexible substitution patterns (1999), D. Brownstone, K. Train'

# %% [markdown]
# ## Scaling variables
# 
# To be consistent with the interpretation of estimates in 'Brownstone & Train (1999)' we rescale some of the explanatory variables. Furthermore, Logit models are most stable numerically if we ensure that variables are scaled near to $\pm 1$. 

# %%
dat['range'] = dat['range'] / 100                  # Hundreds of miles that the vehicle can travel between fuelings
dat['top_speed'] = dat['top_speed'] / 100          # Highest speed that the vehicle can attain, in hundreds of miles per hour
dat['size'] = dat['size'] / 10                     # Scaled categorical variable for numerical purposes
dat['acceleration'] = dat['acceleration'] / 10     # Measured in tens of seconds
dat['operating_cost'] = dat['operating_cost'] / 10 # Measured in tens of cents per mile

# %% [markdown]
# Since, respectively, 'EV'and 'Non-EV'and 'CNG' and 'Non-CNG' are equivalent we exclude the latter and keep all the other characteristics as explanatory variables.  

# %%
# variables to use as explanatory variables
x_vars = list(lab.iloc[3:-4].index.values) # variable names

# %%
# dimensions of data
N = dat.person_id.nunique()
J = dat.j.nunique()
K = len(x_vars)

# %% [markdown]
# Finally, we will primarily use numpy data types and numpy functions in this notebook. Hence we store our response variable 'y' and our explanatory variables 'x' as numpy arrays.

# %%
# response and explanatory variables as numpy arrays
a = dat['y'].values.reshape((N,J))
a = a[:, 0] # All values are equal along axis=1. Becomes an (N,) array i.e. it is a vector.
y = pd.get_dummies(a).to_numpy() # Convert y to an (N,J) array as the onehot encoding
x = dat[x_vars].values.reshape((N,J,K))

# %% [markdown]
# ## Estimating the logit model
# 
# We estimate a logit model on the data using maximum likelihood. In doing this we will see our first use of the numpy function 'einsum()' which quickly and easily computes matrix products, outer products, transposes, etc. 

# %%
def util(Beta, x):
    '''
    This function finds the deterministic utilities u = X*Beta.
    
    Args.
        Beta: (K,) numpy array of parameters
        x: (N,J,K) matrix of covariates

    Output
        u: (N,J) matrix of deterministic utilities
    '''

    assert Beta.ndim == 1
    assert x.ndim == 3

    u = np.einsum('njk,k->nj', x, Beta) # is the same as x @ Beta

    return u

# %%
def logit_loglikehood(Beta, a, x, MAXRESCALE: bool = True):
    '''
    This function calculates the likelihood contributions of a Logit model

    Args. 
        Beta: (K,) vector of parameters 
        x: (N,J,K) matrix of covariates 
        a: (N,) vector of outcomes (integers in 0, 1, ..., J-1)

    Returns
        ll_i: (N,) vector of loglikelihood contributions for a Logit
    '''
    assert Beta.ndim == 1 
    N,J,K = x.shape 

    # deterministic utility 
    v = util(Beta, x)

    if MAXRESCALE: 
        # subtract the row-max from each observation
        v -= v.max(axis=1, keepdims=True)  # keepdims maintains the second dimension, (N,1), so broadcasting is successful

    # denominator 
    denom = np.exp(v).sum(axis=1) # NOT keepdims! becomes (N,)

    # utility at chosen alternative for each individual i
    v_i = v[np.arange(N), a] # Becomes (N,)

    # likelihood 
    ll_i = v_i - np.log(denom) # difference between two 1-dimensional arrays 

    return ll_i


# %%
def q_logit(Beta, y, x):
    
    '''
    q: Criterion function, passed to estimate_logit().
    '''
    return -logit_loglikehood(Beta, y, x)

# %%
def estimate_logit(q, Beta0, y, x, options = {'disp': True}, **kwargs):
    ''' 
    Takes a function and returns the minimum, given start values and 
    variables to calculate the residuals.

    Args.
        q: a function to minimize,
        Beta0 : (K+G,) array of initial guess parameters,
        y: array of observed response variables (N,),
        x: array of observed explanatory variables (N,J,K),
        options: dictionary with options for the optimizer (e.g. disp=True,
            which tells it to display information at termination.)
    
    Returns
        res: Returns a dictionary with results from the estimation.
    '''

    # The objective function is the average of q(), 
    # but Q is only a function of one variable, theta, 
    # which is what minimize() will expect
    Q = lambda Theta: np.mean(q(Theta, y, x))

    # call optimizer
    result = optimize.minimize(Q, Beta0.tolist(), options=options, **kwargs)

    # collect output in a dict 
    res = {
        'beta': result.x, # vector of estimated parameters
        'success':  result.success, # bool, whether convergence was succesful 
        'nit':      result.nit, # no. algorithm iterations 
        'nfev':     result.nfev, # no. function evaluations 
        'fun':      result.fun # function value at termination 
    }

    return res

# %% [markdown]
# Estimating a Logit model via maximum likelihood with an initial guess of parameters $\hat \beta^0 = 0$ yields estimated parameters $\hat \beta^{\text{logit}}$ given as...

# %%
beta_0 = np.zeros((K,))

# Estimate the model
res_logit = estimate_logit(q_logit, beta_0, a, x)

# %%
logit_beta = res_logit['beta']
pd.DataFrame(logit_beta.reshape(1,len(logit_beta)))

# %% [markdown]
# ### We then compute the corresponding Logit choice probabilities

# %%
def logit_ccp(Beta, x, MAXRESCALE:bool=True):
    '''logit_ccp(): Computes the (N,J) matrix of choice probabilities from a logit model
    Args. 
        u: (N,J) matrix of  
    
    Returns
        ccp: (N,J) matrix of probabilities 
    '''
    
    # deterministic utility 
    v = util(Beta, x) # (N,J) 

    if MAXRESCALE: 
        # subtract the row-max from each observation
        v -= v.max(axis=1, keepdims=True)  # keepdims maintains the second dimension, (N,1), so broadcasting is successful
    
    # denominator 
    denom = np.exp(v).sum(axis=1, keepdims=True) # (N,1)
    
    # Conditional choice probabilites
    ccp = np.exp(v) / denom
    
    return ccp

# %% [markdown]
# Using our estimates $\hat \beta^{\text{logit}}$, the choice probabilities $\hat q_i^{logit}$ of products $\{0,1, \ldots , 5\}$ for individuals $i=0,1,\ldots , 4653$ thus becomes:

# %%
logit_q = logit_ccp(logit_beta, x)
pd.DataFrame(logit_q)

# %% [markdown]
# #### Logit elasticities
# 
# The logit (semi-)elasticities of the choice probabilities $q_i = P(u| \beta)$ for individual i wrt. to the $\ell$'th characteristic are given by the formula:
# $ \mathcal{E}_i= \nabla_x \ln P(u| \beta)= \left( I_J - \iota q_i'\right) \beta_\ell$
# where $()'$ denotes the transpose of a matrix, $\iota=(1, \ldots, 1)'$ is the all ones vector in $\mathbb{R}^{J}$,and $I_J$ is the identity matrix in $\mathbb{R}^{J\times J}$.

# %% [markdown]
# Lastly we compute the implied price-to-log-income elasticities for our logit model.

# %%
def logit_elasticity(q, Beta, char_number):
    ''' 
    This function calculates the logit elasticities of choice probabilities wrt. a given charateristic k

    Args.
        q: a (N,J) numpy array of choice probabilities
        Beta: a (K,) numpy array of parameters
        car_number: an integer k for the index of the characteristic

    Output:
        Epsilon: a (N,J,J) matrix of logit elasticities of choice probabilities wrt. the charateristic k
    '''

    assert q.ndim == 2
    assert Beta.ndim == 1

    N,J = q.shape

    iota_q = np.einsum('j,ni->nji', np.ones((J,)), q)
    Epsilon = (np.eye(J) - iota_q)*Beta[char_number]

    return Epsilon

# %% [markdown]
# Implemented on our datset, we thus find the elasticities as follows...

# %%
epsilon_logit = logit_elasticity(logit_q, logit_beta, 0)
pd.DataFrame(epsilon_logit[0,:,:])

# %% [markdown]
# In the above example for individual $i=0$, the $j\ell$'th entry corresponds to the elasticity of the choice probability of product $j$ with respect to the price-to-log-income (i.e. the $0$'th characteristic) of product $\ell$ for $j, \ell \in \{0,1, \ldots ,  5\}$. Note that the diagonal entries are negative, indicating that all products are normal, and that the cross-elasticities (i.e. $j \neq \ell$) with respect to any product $\ell$ are equal for all $j \neq \ell$. Our example thus validates the IIA property of the logit model. 

# %%
own_elasticities_logit = {j : (epsilon_logit.reshape((N, J**2))[:,j]).flatten() for j in np.arange(J**2)} # Finds j'th entry in each of the elasticity matrices of individuals i.  

j_pairs = iter.product(np.arange(J), np.arange(J))
num_bins = 25

fig, axes = plt.subplots(J, J)

for p, j in zip(j_pairs, np.arange(J**2)):
    axes[p].hist(own_elasticities_logit[j], num_bins)
    axes[p].vlines(0, 0, 1500, 'red', 'dotted')
    axes[p].get_xaxis().set_visible(False)
    axes[p].get_yaxis().set_visible(False)

fig.suptitle('Logit price-to-log-income elasticities')

plt.show()

# %% [markdown]
# #### Diversion Ratios for Logit
# 
# The diversion ratio to product j from product k wrt. to the $\ell$'th characteristic $-100 \cdot \frac{\partial P_j / \partial x_{k\ell}}{\partial P_k / \partial x_{k\ell}}$ for the standard logit model is given by equation:
# 
# $$
# D^i = -100 \cdot \nabla_{x_\ell} P(u|\beta)(\nabla_{x_\ell} P(u|\beta) \circ I_J)^{-1} = -100 \cdot \nabla_u P(u|\beta)(\nabla_u P(u|\beta) \circ I_J)^{-1}
# $$
# 
# Where '$\circ$' is the elementwise product of matrices and $\nabla_{x_\ell} P(u|\beta) = \beta_\ell(\mathrm{diag}(q) - qq')$ is the usual derivative of Logit choice probailities wrt. the $\ell$'th characteristic.

# %%
def logit_diversion_ratio(q, Beta):
    '''
    This function calculates the logit diversion ratios of choice probabilities wrt. a given charateristic k

    Args.
        q: a (N,J) numpy array of choice probabilities
        Beta: a (K,) numpy array of parameters
        car_number: an integer k for the index of the characteristic

    Output:
        DR: a (N,J,J) matrix of logit diversion ratios of choice probabilities wrt. the charateristic k
    '''

    assert q.ndim == 2
    assert Beta.ndim == 1

    N,J = q.shape

    diag_q = q[:,:,None] * np.eye(J,J)[None, :, :]
    qqT = np.einsum('nj,nk->njk', q, q)
    Grad = diag_q - qqT
    diag_Grad_mat = Grad * np.eye(J,J)[None, :, :]
    diag_Grad_vec = np.einsum('njk,k->nj', diag_Grad_mat, np.ones((J,)))
    DR = -100 * np.einsum('njk,nj->njk', Grad, 1./diag_Grad_vec)

    return DR
    


# %%
DR_logit_hat = logit_diversion_ratio(logit_q, logit_beta)
pd.DataFrame(DR_logit_hat[0,:,:])

# %%
own_DR_logit = {j : (DR_logit_hat.reshape((N, J**2))[:,j]).flatten() for j in np.arange(J**2)} # Finds j'th entry in each of the elasticity matrices of individuals i.  

j_pairs = iter.product(np.arange(J), np.arange(J))
num_bins = 25

fig, axes = plt.subplots(J, J)

for p, j in zip(j_pairs, np.arange(J**2)):
    axes[p].hist(own_DR_logit[j], num_bins)
    axes[p].vlines(0, 0, 1500, 'red', 'dotted')
    axes[p].get_xaxis().set_visible(False)
    axes[p].get_yaxis().set_visible(False)

fig.suptitle('Logit price-to-log-income diversion ratios')
plt.show()

