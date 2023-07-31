# %% [markdown]
# Data
# ====
# 
# The dataset consists of approximately 110 vehicle makes per year in the period 1970-1999 in five European markets (Belgium, France, Germany, Italy, and the United Kingdom). The data set includes 47 variables in total. The first four columns are market and product codes for the year, country, and make as well as quantity sold (No. of new registrations) which will be used in computing observed market shares. The remaining variables consist of car characteristics such as prices, horse power, weight and other physical car characteristics as well as macroeconomic variables such as GDP per capita which have been used to construct estimates of the average wage income and purchasing power.
# 
# We have in total 30 years and 5 countries, totalling $T=150$ year-country combinations, indexed by $t$, and we refer to each simply as market $t$. In market $t$, the choice set is $\mathcal{J}_t$ which includes the set of available makes as well as an outside option. Let $\mathcal{J} := \bigcup_{t=1}^T \mathcal{J}_t$ be the full choice set and 
#  $J:=\#\mathcal{J}$ the number of choices which were available in at least one market, for this data set there are $J=357$ choices.
#  
# 

# %% [markdown]
# Reading in the dataset `eurocars.csv` we thus have a dataframe of $\sum_{t=1}^T \#\mathcal{J}_t = 11459$ rows and $47$ columns. The `ye` column runs through $y=70,\ldots,99$, the `ma` column runs through $m=1,\ldots,M$, and the ``co`` column takes values $j\in \mathcal{J}$. 
# 
# Because we consider a country-year pair as the level of observation, we construct a `market` column taking values $t=1,\ldots,T$. In Python, this variable will take values $t=0,\ldots,T-1$. We construct an outside option $j=0$ in each market $t$ by letting the 'sales' of $j=0$ be determined as 
# 
# $$\mathrm{sales}_{0t} = \mathrm{pop}_t - \sum_{j=1}^J \mathrm{sales}_{jt}$$
# 
# where $\mathrm{pop}_t$ is the total population in market $t$, and the car characteristics of the outside option is set to zero. The market shares of each product in market $t$ can then be found as
# $$
# \textrm{market share}_{jt}=\frac{\mathrm{sales_{jt}}}{\mathrm{pop}_t}.
# $$
# We also read in the variable description of the dataset contained in `eurocars.dta`. We will use the list `x_vars` throughout to work with our explanatory variables. 

# %%
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import os
from numpy import linalg as la
from scipy import optimize
import scipy.stats as scstat
from matplotlib import pyplot as plt
import itertools as iter

# Files
import Logit_file as logit

# %%
# Load dataset and variable names
# os.chdir('../GREENCAR_notebooks/') # Assigns work directory

input_path = os.getcwd() # Assigns input path as current working directory (cwd)
descr = (pd.read_stata('eurocars.dta', iterator = True)).variable_labels() # Obtain variable descriptions
dat_file = pd.read_csv(os.path.join(input_path, 'eurocars.csv')) # reads in the data set as a pandas dataframe.
pd.DataFrame(descr, index=['description']).transpose().reset_index().rename(columns={'index' : 'variable names'}) # Prints data sets
# Choose which variables to include in the analysis, and assign them either as discrete variables or continuous.

# %%

x_discretevars = [ 'brand', 'home', 'cla']
x_contvars = ['cy', 'hp', 'we', 'le', 'wi', 'he', 'li', 'sp', 'ac', 'pr']
z_IV_contvars = ['xexr']
z_IV_discretevars = []
x_allvars =  [*x_contvars, *x_discretevars]

# Outside option is included if OO == True, otherwise analysis is done on the inside options only.
OO = True

# Print list of chosen variables as a dataframe
print(pd.DataFrame(descr, index=['description'])[x_allvars].transpose().reset_index().rename(columns={'index' : 'variable names'}))

# %% [markdown]
# We now clean the data to fit our setup

# %%
def Eurocars_cleandata(dat, x_contvars, x_discretevars, z_IV_contvars, z_IV_discretevars, outside_option = True):
    ''' 
    '''

    # Create the 'market' column of market index t

    dat = dat.sort_values(by = ['ye', 'ma'], ascending = True) # Sorts data set by year and market
    Used_cols = ['ye', 'ma', 'co', 'qu', 'pop', *x_contvars, *x_discretevars, *z_IV_contvars, *z_IV_discretevars]  
    dat = dat[Used_cols] # Leaves out unused macro variables
    market_vals = [*iter.product(dat['ye'].unique(), dat['ma'].unique())] # creates a list of ma-ye combinations
    market_vals = pd.DataFrame({'ye' : [val[0] for val in market_vals], 'ma' : [val[1] for val in market_vals]}) 
    market_vals = market_vals.reset_index().rename(columns={'index' : 'market'}) # Creates market index
    dat = dat.merge(market_vals, left_on=['ye', 'ma'], right_on=['ye', 'ma'], how='left') # Merges market index variable onto dat
    dat_org = dat # Save the original data with the 'market'-column added as 'dat_org'.

    # Create an inside/outside-option column if the outside option is included

    if outside_option:
        dat['in_out'] = 1

    # Drop rows which contain NaN values in any explanatory variable or in the response variable.

    dat = dat.dropna()

    # Convert discrete explanatory variables to integer valued variables and make sure continuous variables are floats.

    obj_columns = dat.select_dtypes(['object'])
    for col in obj_columns:
        if col in [*x_contvars, *z_IV_contvars]:
            dat[col] = dat[col].str.replace(',', '.').astype('float64')
        else:
            dat[col] = dat[col].astype('category').cat.rename_categories(np.arange(1, dat[col].nunique() + 1)).astype('int64')

    # Re-encode discrete variables such that only the outside option takes the value 0

    ###############################################################################
    x_0vars = [var for var in [*x_discretevars,*z_IV_discretevars] if len((dat[var].isin([0]))) > 0] # Picks out discrete variables where at least one car has category 0

    for col in x_0vars:
        dat[col] = dat[col].astype('category').cat.rename_categories(np.arange(1, dat[col].nunique() + 1)).astype('int64') # re-assigns category zero as category 1, and moves other categories up by one

    #################################################################################
    # Construct outside option for each market t
    if outside_option:
        outside_shares = dat.groupby('market', as_index=False)['qu'].sum() # sum of sales in each market
        outside_shares = outside_shares.merge(dat[['market', 'pop']], on = 'market', how='left').dropna().drop_duplicates(subset = 'market', keep = 'first')  # Adds population to dataframe
        outside_shares['qu'] = outside_shares['pop'] - outside_shares['qu'] # Assigns quantity for outside option as pop minus sum of sales
        keys_add = [key for key in dat.keys() if (key!='market')&(key!='qu')&(key!='pop')] 
        for key in keys_add:
            outside_shares[key] = 0 # Sets all variables other than market, qu and pop to zero for the outside option

        dat = pd.concat([dat, outside_shares]) # Add outside option to data set

    #################################################################################
    # Compute market shares for each product j in each market t 

    dat['ms'] = dat.groupby('market')['qu'].transform(lambda x: x/x.sum())

    #################################################################################
    T = dat['market'].nunique() # Assigns the total number of markets T
    J = np.array([dat[dat['market'] == t]['co'].nunique() for t in np.arange(T)]) # Array of number of choices in market t


    # Number of observations 
    if outside_option:
        N = np.array([dat[dat['market'] == t]['pop'].unique().sum() for t in np.arange(T)]).sum() # If outside option is included, number of observations in market t is the total population
    else:
        N = np.array([dat[dat['market'] == t]['qu'].sum() for t in np.arange(T)]).sum() # If outside option is not included, number of observations in market t is the total number of sales


    # Get each market's share of total population N
    pop_share = np.empty((T,))
    for t in np.arange(T):
        pop_share[t] = dat[dat['market'] == t]['qu'].sum() / N

    ##################################################################################
    dat[[*x_contvars, *z_IV_contvars]] = dat[[*x_contvars, *z_IV_contvars]] / dat[[*x_contvars, *z_IV_contvars]].abs().max() # Rescale continuous variables so that they lie in the interval [-1,1]. This is done for numerical stability.

    ###################################################################################
    # Construct dummies of discrete variables. For each variable, one of the columns is left out due to colinearity

    datx_disc = pd.get_dummies(dat[x_discretevars], prefix = x_discretevars, columns = x_discretevars, drop_first=True)
    if len(z_IV_discretevars) > 0:
        datz_disc = pd.get_dummies(dat[z_IV_discretevars], prefix = z_IV_discretevars, columns = z_IV_discretevars, drop_first=True)
    else:
        datz_disc = None

    # If outside option is included, then each variable results in a column which is 1 for the outside option, and zero for all other options. These columns are identical to the 'in_out' variable column,
    # so a second column must be dropped for each variable.
    if outside_option:
        datx_disc = datx_disc[[var for var in datx_disc.keys() if not var.endswith('1')]] # Drops a second column from discrete columns if outside option is included
        if len(z_IV_discretevars) > 0:
            datz_disc = datz_disc[[var for var in datz_disc.keys() if not var.endswith('1')]]

    # Add dummy variables onto the original DataFrame
    if len(z_IV_discretevars) > 0:
        dat = pd.concat([dat, datx_disc, datz_disc], axis = 1)
    else:
        dat = pd.concat([dat, datx_disc], axis = 1)

    # Record explanatory variables and IV regressors
    if outside_option:
        x_vars = ['in_out', *x_contvars, *datx_disc.keys() ]
    else:
        x_vars = [*x_contvars, *datx_disc.keys() ]

    if len(z_IV_discretevars) > 0:
        z_vars = [*z_IV_contvars, *datz_disc.keys()]
    else:
        z_vars = z_IV_contvars

    # Count the number of characteristics
    K = len(x_vars)

    return dat,dat_org,x_vars,z_vars,N,pop_share,T,J,K

# %%
dat, dat_org, x_vars, z_vars, N, pop_share, T, J, K = Eurocars_cleandata(dat_file, x_contvars, x_discretevars, z_IV_contvars, z_IV_discretevars, outside_option = OO)

# %%
# Create dictionaries of numpy arrays for each market. This allows the size of the data set to vary over markets.

dat = dat.reset_index(drop = True).sort_values(by = ['market', 'co']) # Sort data so that reshape is successfull

x = {t: dat[dat['market'] == t][x_vars].values.reshape((J[t],K)) for t in np.arange(T)} # Dict of explanatory variables
y = {t: dat[dat['market'] == t]['ms'].to_numpy().reshape((J[t])) for t in np.arange(T)} # Dict of market shares

# %%
# This function tests whether the utility parameters are identified, by looking at the rank of the stacked matrix of explanatory variables.

def rank_test(x):
    x_stacked = np.concatenate([x[t] for t in np.arange(T)], axis = 0)
    eigs=la.eig(x_stacked.T@x_stacked)[0]

    if np.min(eigs)<1.0e-8:
        print('x does not have full rank')
    else:
        print('x has full rank')

# %%
rank_test(x)


