# Importing needed library
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pandascharm as pc
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

from sklearn.preprocessing import OneHotEncoder
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import blosum as bl
from Bio.SubsMat.MatrixInfo import blosum62 as blosum
from Bio.SubsMat.MatrixInfo import blosum45 as blosum_new
from Bio import AlignIO
from Bio import SeqIO
from Bio.Align.Applications import MuscleCommandline
from Bio.Align import AlignInfo
from scipy.fft import fft, ifft

from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from pyaaisc import Aaindex
from Bio import AlignIO


import scipy.stats as stats

# Two versions of removing outliers
def removeoutliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_out = df[~((df < (Q1 - 1 * IQR)) |(df > (Q3 + 100 * IQR))).any(axis=1)]
#     df_out = df
    return df_out.dropna()

def removeoutlier_col(df,cols):
    Q1 = df[cols].quantile(0.25)
    Q3 = df[cols].quantile(0.75)
    IQR = Q3 - Q1

    df = df[~((df[[cols]] < (Q1 - 100 * IQR)) |(df[[cols]] > (Q3 + 100 * IQR))).any(axis=1)]
    return df.dropna()




# Machine Learning Tools

def LR(X_train, y_train, X_val):
    from sklearn.linear_model import LinearRegression
    lm = LinearRegression()
    lm.fit(X_train,y_train)
    lin_pred = lm.predict(X_val)
    pred_training= lm.predict(X_train)
    
    return lin_pred , pred_training

#Ordinary Least Squares Regression
def OLS(X_train, y_train, X_val):
    import statsmodels.api as sm
    from scipy import stats
    X2 = sm.add_constant(X_train)
    est = sm.OLS(y_train,X2)
    est2 = est.fit()
    print(est2.summary())
    coefficients=est2.params[1:]
    intercept=est2.params[0]
    OLS_pred=(coefficients*X_val).sum(axis = 1, skipna = True) + intercept
    
    return OLS_pred

#Lasso regression

def LASSO(X_train, y_train, X_val, jack = False):
    knife = len(X_train)-1
    param_grid={'alpha':np.arange(0,1.1,0.05).tolist()}
    lasso_reg=Lasso()
    if jack == False:
        grid_search=GridSearchCV(lasso_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    else:
        grid_search=GridSearchCV(lasso_reg, param_grid, cv=knife, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X_train,np.ravel(y_train))
    lasso_pred=grid_search.predict(X_val)
    pred_training= grid_search.predict(X_train)
    return lasso_pred, pred_training

#Partial Least Square Regression

def PLS(X_train, y_train, X_val, jack = False):
    knife = len(X_train)-1
    param_grid={'n_components':np.arange(1,10,1).tolist()}
    pls = PLSRegression()
    if jack == False:
        grid_search=GridSearchCV(pls, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    else:
        grid_search=GridSearchCV(pls, param_grid, cv=knife, scoring='neg_mean_squared_error', return_train_score=True)
    pls.fit(scale(X_train), y_train)
    PLS_pred=pls.predict(scale(X_val))
    pred_training= pls.predict(X_train)
    
    return PLS_pred , pred_training

#Random Forest

def RF(X_train, y_train, X_val, jack = False):
    knife = len(X_train)-1
    param_grid={'max_depth': [70], 'min_samples_leaf': [1], 'min_samples_split': [2], 'n_estimators': [50]}
    rf = RandomForestRegressor()
    if jack == False:
        grid_search=GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    else:
        grid_search=GridSearchCV(rf, param_grid, cv=knife, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X_train,np.ravel(y_train))
    RF_pred = grid_search.predict(X_val)
    pred_training= grid_search.predict(X_train)
    return RF_pred, pred_training

#Decision Tree

def DT(X_train, y_train, X_val, jack = False):
    knife = len(X_train)-1
    param_grid={ "max_features": [2], 'random_state':[20]}
    tree_reg=DecisionTreeRegressor()
    if jack == False:
        grid_search=GridSearchCV(tree_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    else:
        grid_search=GridSearchCV(tree_reg, param_grid, cv=knife, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X_train,np.ravel(y_train))
    DT_pred = grid_search.predict(X_val)
    pred_training= grid_search.predict(X_train)
    return DT_pred, pred_training

#SVM

def SVM(X_train, y_train, X_val, jack = False):
    knife = len(X_train)-1
    param_grid = {'C': [10], 'coef0': [0.01], 'degree': [3]}
 
    model = SVR()
    if jack == False:
        grid_search=GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    else:
        grid_search=GridSearchCV(model, param_grid, cv=knife, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X_train, y_train)
    SVM_pred = grid_search.predict(X_val)
    pred_training= grid_search.predict(X_train)
    return SVM_pred, pred_training

#Neural Network Regression

def NNR(X_train, y_train, X_val, jack = False):
    
    knife = len(X_train)-1
    param_grid = {'activation': ['tanh'], 'alpha': [0.0005], 'hidden_layer_sizes': [50], 'solver': ['sgd']}
    regr = MLPRegressor(random_state=101, max_iter=100).fit(X_train, y_train)
    if jack == False:
        grid_search=GridSearchCV(regr, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    else:
        grid_search=GridSearchCV(regr, param_grid, cv=knife, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X_train, y_train)
    NNR_pred = grid_search.predict(X_val)
    pred_training= grid_search.predict(X_train)
    return NNR_pred, pred_training

#Elastic Net

def EN(X_train, y_train, X_val, jack = False):
    knife = len(X_train)-1
    param_grid = {'alpha': [10], 'l1_ratio': [0.1], 'max_iter': [10]}
    ENmodel = ElasticNet()
    if jack == False:
        grid_search=GridSearchCV(ENmodel, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    else:
        grid_search=GridSearchCV(ENmodel, param_grid, cv=knife, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X_train, y_train)
    EN_pred = grid_search.predict(X_val)
    pred_training= grid_search.predict(X_train)

    return EN_pred, pred_training


def XGBR(X_train, y_train, X_val, jack = False):
    knife = len(X_train)-1
    
    param_grid = {'colsample_bytree': [0.8], 'gamma': [0.5], 'max_depth': [4], 'min_child_weight': [1], 'subsample': [0.6]}
    
    if jack == False:
        grid_search=GridSearchCV(xgb.XGBRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    else:
        grid_search=GridSearchCV(xgb.XGBRegressor(), param_grid, cv=knife, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X_train, y_train)
    XGB_pred = grid_search.predict(X_val)
    pred_training= grid_search.predict(X_train)
    return XGB_pred, pred_training


# BLOSUM score function

blosum.update(((b,a),val) for (a,b),val in list(blosum.items()))
blosum_new.update(((b,a),val) for (a,b),val in list(blosum_new.items()))

def score_pairwise(seq1, seq2, matrix, gap_s, gap_e, gap = True):
    for A,B in zip(seq1, seq2):
        diag = ('-'==A) or ('-'==B)
        yield (gap_e if gap else gap_s) if diag else matrix[(A,B)]
        gap = diag





def encode(encoding, output, df, aln, key = None):
    

    ClustalAlign = AlignIO.read(aln, 'clustal')
    summary_align = AlignInfo.SummaryInfo(ClustalAlign )
    
    dframe = pc.from_bioalignment(ClustalAlign)
    dframe = dframe.replace('U', '-')
    dframe = dframe.replace('O','-')
    
    main_output = removeoutliers(df[[output]])
    main_index = list(main_output.index)
    
    if encoding == 'One-Hot-Encoder':
        holder = np.nan
        X = df['Sequence'].iloc[main_index].dropna()
        rem_index = list(X.index)
        y = df[output].iloc[rem_index]
        one_hot = OneHotEncoder()
        temp_seq = np.array(X).reshape(-1,1)
        encoded = one_hot.fit(temp_seq)
        X = encoded.transform(temp_seq).toarray()
        
    if encoding == 'Bag-of-Words':
        holder = np.nan
        X = df['Sequence'].iloc[main_index].dropna()
        rem_index = list(X.index)
        y = df[output].iloc[rem_index]
        X = pd.DataFrame([ProteinAnalysis(i).count_amino_acids() for i in X])
        
        
        
    if encoding == 'bigram':
        holder = np.nan
        X = df['Sequence'].iloc[main_index].dropna()
        rem_index = list(X.index)
        y = df[output].iloc[rem_index]
        
        example = df['Sequence'][0]
        lst = ['E','G','L','Y','T','H','R','A','C','D','P','I','F','N','K','S','V','M','W','Q']
        all_dct = {}
        key = []
        for i in lst:
            for j in lst:
                st = i+j
                all_dct[st] = []

        for example, id in zip(X,range(len(X))):

            temp = list(example)
            temp_dct = dict.fromkeys(all_dct.keys(),0)
            for k in range(len(temp)-1):
                try:
                    check = temp[k] + temp[k+1]
                    temp_dct[check] += 1
                except:
                    pass
            for key, value in temp_dct.items():
                all_dct[key].append(value)
        X = pd.DataFrame.from_dict(all_dct)
        
    if encoding == 'trigram':
        holder = np.nan
        X = df['Sequence'].iloc[main_index].dropna()
        rem_index = list(X.index)
        y = df[output].iloc[rem_index]

        example = df['Sequence'][0]
        lst = ['E','G','L','Y','T','H','R','A','C','D','P','I','F','N','K','S','V','M','W','Q']
        all_dct = {}
        key = []
        for i in lst:
            for j in lst:
                for k in lst:
                    st = i+j+k
                    all_dct[st] = []

        for example, id in zip(X,range(len(X))):

            temp = list(example)
            temp_dct = dict.fromkeys(all_dct.keys(),0)
            for k in range(len(temp)-2):
                try:
                    check = temp[k] + temp[k+1]+temp[k+2]
                    temp_dct[check] += 1
                except:
                    pass
            for key, value in temp_dct.items():
                all_dct[key].append(value)
        X = pd.DataFrame.from_dict(all_dct)
        
    if encoding == 'quadrogram':
        holder = np.nan
        X = df['Sequence'].iloc[main_index].dropna()
        rem_index = list(X.index)
        y = df[output].iloc[rem_index]

        example = df['Sequence'][0]
        lst = ['E','G','L','Y','T','H','R','A','C','D','P','I','F','N','K','S','V','M','W','Q']
        all_dct = {}
        key = []
        for i in lst:
            for j in lst:
                for k in lst:
                    for l in lst:
                        st = i+j+k+l
                        all_dct[st] = []

        for example, id in zip(X,range(len(X))):

            temp = list(example)
            temp_dct = dict.fromkeys(all_dct.keys(),0)
            for k in range(len(temp)-3):
                try:
                    check = temp[k] + temp[k+1]+temp[k+2]+temp[k+3]
                    temp_dct[check] += 1
                except:
                    pass
            for key, value in temp_dct.items():
                all_dct[key].append(value)
        X = pd.DataFrame.from_dict(all_dct)

        
    if encoding == 'AAIndex':
        aaindex = Aaindex()
#         full_list = aaindex.get_all(dbkey='aaindex1')
        
        record = aaindex.get(key[0])
        index_data = record.index_data
        index_data['-'] = 0
        sequence_aligned=dframe.apply(lambda dframe : pd.Series(index_data[i] for i in dframe))

        temp1=sequence_aligned.transpose()
        temp2 = df[['Organism Name', output]]
        temp3 = removeoutlier_col(temp2,output).set_index('Organism Name')
        Z=pd.concat([temp1, temp3], axis=1).dropna()

        X=Z.loc[:, Z.columns != output]
        y=Z.loc[:, Z.columns == output]

        holder = key[0]
        
    if encoding == 'BLOSUM62':

        holder = np.nan
        temp1=dframe.transpose()
        temp2 = df[['Organism Name',output]]
        temp3 = removeoutlier_col(temp2,output).set_index('Organism Name')
        Z=pd.concat([temp1, temp3], axis=1).dropna()

        X=Z.loc[:, Z.columns != output]
        y=Z.loc[:, Z.columns == output]

        n = len(X)
        enc_seq = np.zeros((n,n))

        i = 0

        for a in list(X.index):
            j = 0
            for b in list(X.index):
                enc_seq[i,j] = sum(score_pairwise(X.loc[a], X.loc[b], blosum, -5, -1))
                j += 1
            i += 1

        X = enc_seq
        
    if encoding == 'BLOSUM45':

        holder = np.nan
        temp1=dframe.transpose()
        temp2 = df[['Organism Name',output]]
        temp3 = removeoutlier_col(temp2,output).set_index('Organism Name')
        Z=pd.concat([temp1, temp3], axis=1).dropna()

        X=Z.loc[:, Z.columns != output]
        y=Z.loc[:, Z.columns == output]

        n = len(X)
        enc_seq = np.zeros((n,n))

        i = 0

        for a in list(X.index):
            j = 0
            for b in list(X.index):
                enc_seq[i,j] = sum(score_pairwise(X.loc[a], X.loc[b], blosum_new, -5, -1))
                j += 1
            i += 1

        X = enc_seq
        
        
    if encoding == 'fft':
        aaindex = Aaindex()
        
        record = aaindex.get(key[0])
        index_data = record.index_data
        index_data['-'] = 0
        sequence_aligned=dframe.apply(lambda dframe : pd.Series(index_data[i] for i in dframe))

        temp1=sequence_aligned.transpose()
        temp2 = df[['Organism Name', output]]
        temp3 = removeoutlier_col(temp2,output).set_index('Organism Name')
        Z=pd.concat([temp1, temp3], axis=1).dropna()

        X=Z.loc[:, Z.columns != output]
        y=Z.loc[:, Z.columns == output]
        
        
        X = fft(X)
        holder = key[0]
        
    return X,y,holder






# Encoding With Temperature

def encode_temp(encoding, output, df, aln, key = None):
    

    ClustalAlign = AlignIO.read(aln, 'clustal')
    summary_align = AlignInfo.SummaryInfo(ClustalAlign )
    
    dframe = pc.from_bioalignment(ClustalAlign)
    dframe = dframe.replace('U', '-')
    dframe = dframe.replace('O','-')
    
    main_output = removeoutliers(df[[output]])
    main_index = list(main_output.index)
    scaler = preprocessing.StandardScaler()
    
    
    if encoding == 'One-Hot-Encoder':
        holder = np.nan
        X = df['Sequence'].iloc[main_index].dropna()
        rem_index = list(X.index)
        y = df[output].iloc[rem_index]
        one_hot = OneHotEncoder()
        temp_seq = np.array(X).reshape(-1,1)
        encoded = one_hot.fit(temp_seq)
        X = encoded.transform(temp_seq).toarray()
        temperature = 'Reaction Temperature'
        X1 = df[temperature].iloc[rem_index]
        temp1 = np.array(X1).reshape(-1,1)
        X = np.concatenate((X,temp1), axis =1)
        scaler.fit(X)
        X = scaler.transform(X)
        
#         holder = np.nan
#         temperature = 'Reaction Temperature'
#         temp1=dframe.transpose()
#         temp2 = df[['Organism Name',temperature,output]]
#         temp3 =removeoutlier_col(temp2,output).set_index('Organism Name')
#         Z=pd.concat([temp1, temp3], axis=1).dropna()
#         X=Z.loc[:, Z.columns != output ]
#         y=Z.loc[:, Z.columns == output]
#         temp_col = X.loc[:, X.columns == temperature ]
#         X = X.loc[:, X.columns != temperature ]
#         one_hot = OneHotEncoder()
#         temp_seq = np.array(X).reshape(-1,1)
#         encoded = one_hot.fit(temp_seq)
#         X = encoded.transform(temp_seq).toarray()
        
#         temp_col = np.array(temp_col).reshape(-1,1)
#         X = np.concatenate((X,temp_col), axis =1)
#         scaler.fit(X)
#         X = scaler.transform(X)
        
    
        
    if encoding == 'Bag-of-Words':
        holder = np.nan
        X = df['Sequence'].iloc[main_index].dropna()
        rem_index = list(X.index)
        y = df[output].iloc[rem_index]
        X = pd.DataFrame([ProteinAnalysis(i).count_amino_acids() for i in X])
        temperature = 'Reaction Temperature'
        X1 = df[temperature].iloc[rem_index]
        temp1 = np.array(X1).reshape(-1,1)
        X = np.concatenate((X,temp1), axis =1)
        scaler.fit(X)
        X = scaler.transform(X)
        
    if encoding == 'bigram':
        holder = np.nan
        X = df['Sequence'].iloc[main_index].dropna()
        rem_index = list(X.index)
        y = df[output].iloc[rem_index]
        
        example = df['Sequence'][0]
        lst = ['E','G','L','Y','T','H','R','A','C','D','P','I','F','N','K','S','V','M','W','Q']
        all_dct = {}
        key = []
        for i in lst:
            for j in lst:
                st = i+j
                all_dct[st] = []

        for example, id in zip(X,range(len(X))):

            temp = list(example)
            temp_dct = dict.fromkeys(all_dct.keys(),0)
            for k in range(len(temp)-1):
                try:
                    check = temp[k] + temp[k+1]
                    temp_dct[check] += 1
                except:
                    pass
            for key, value in temp_dct.items():
                all_dct[key].append(value)
        X = pd.DataFrame.from_dict(all_dct)
        temperature = 'Reaction Temperature'
        X1 = df[temperature].iloc[rem_index]
        temp1 = np.array(X1).reshape(-1,1)
        X = np.concatenate((X,temp1), axis =1)
        scaler.fit(X)
        X = scaler.transform(X)
    
    if encoding == 'trigram':
        holder = np.nan
        X = df['Sequence'].iloc[main_index].dropna()
        rem_index = list(X.index)
        y = df[output].iloc[rem_index]

        example = df['Sequence'][0]
        lst = ['E','G','L','Y','T','H','R','A','C','D','P','I','F','N','K','S','V','M','W','Q']
        all_dct = {}
        key = []
        for i in lst:
            for j in lst:
                for k in lst:
                    st = i+j+k
                    all_dct[st] = []

        for example, id in zip(X,range(len(X))):

            temp = list(example)
            temp_dct = dict.fromkeys(all_dct.keys(),0)
            for k in range(len(temp)-2):
                try:
                    check = temp[k] + temp[k+1]+temp[k+2]
                    temp_dct[check] += 1
                except:
                    pass
            for key, value in temp_dct.items():
                all_dct[key].append(value)
        X = pd.DataFrame.from_dict(all_dct)
        temperature = 'Reaction Temperature'
        X1 = df[temperature].iloc[rem_index]
        temp1 = np.array(X1).reshape(-1,1)
        X = np.concatenate((X,temp1), axis =1)
        scaler.fit(X)
        X = scaler.transform(X)
        
        
    if encoding == 'quadrogram':
        holder = np.nan
        X = df['Sequence'].iloc[main_index].dropna()
        rem_index = list(X.index)
        y = df[output].iloc[rem_index]

        example = df['Sequence'][0]
        lst = ['E','G','L','Y','T','H','R','A','C','D','P','I','F','N','K','S','V','M','W','Q']
        all_dct = {}
        key = []
        for i in lst:
            for j in lst:
                for k in lst:
                    for l in lst:
                        st = i+j+k+l
                        all_dct[st] = []

        for example, id in zip(X,range(len(X))):

            temp = list(example)
            temp_dct = dict.fromkeys(all_dct.keys(),0)
            for k in range(len(temp)-3):
                try:
                    check = temp[k] + temp[k+1]+temp[k+2]+temp[k+3]
                    temp_dct[check] += 1
                except:
                    pass
            for key, value in temp_dct.items():
                all_dct[key].append(value)
        X = pd.DataFrame.from_dict(all_dct)
        temperature = 'Reaction Temperature'
        X1 = df[temperature].iloc[rem_index]
        temp1 = np.array(X1).reshape(-1,1)
        X = np.concatenate((X,temp1), axis =1)
        scaler.fit(X)
        X = scaler.transform(X)
        
    if encoding == 'AAIndex':
        temperature = 'Reaction Temperature'
        aaindex = Aaindex()
#         full_list = aaindex.get_all(dbkey='aaindex1')
        
        record = aaindex.get(key[0])
        index_data = record.index_data
        index_data['-'] = 0
        sequence_aligned=dframe.apply(lambda dframe : pd.Series(index_data[i] for i in dframe))

        temp1=sequence_aligned.transpose()     
        temp2 = df[['Organism Name',temperature,output]]
        temp3 =removeoutlier_col(temp2,output).set_index('Organism Name')
        Z=pd.concat([temp1, temp3], axis=1).dropna()
        X=Z.loc[:, Z.columns != output ]
        y=Z.loc[:, Z.columns == output]
        temp_col = X.loc[:, X.columns == temperature ]
        X = X.loc[:, X.columns != temperature ]
        scaler.fit(X)
        X = scaler.transform(X)

        holder = key[0]
        
    if encoding == 'BLOSUM62':

        holder = np.nan
        temperature = 'Reaction Temperature'
        temp1=dframe.transpose()
        temp2 = df[['Organism Name',temperature,output]]
        temp3 =removeoutlier_col(temp2,output).set_index('Organism Name')
        Z=pd.concat([temp1, temp3], axis=1).dropna()
        X=Z.loc[:, Z.columns != output ]
        y=Z.loc[:, Z.columns == output]
        temp_col = X.loc[:, X.columns == temperature ]
        X = X.loc[:, X.columns != temperature ]
        
        
        n = len(X)
        enc_seq = np.zeros((n,n))

        i = 0

        for a in list(X.index):
            j = 0
            for b in list(X.index):
                enc_seq[i,j] = sum(score_pairwise(X.loc[a], X.loc[b], blosum, -5, -1))
                j += 1
            i += 1

        X = enc_seq
        temp_col = np.array(temp_col).reshape(-1,1)
        X = np.concatenate((X,temp_col), axis =1)
        scaler.fit(X)
        X = scaler.transform(X)
        
        
        
    if encoding == 'BLOSUM45':

        holder = np.nan
        temperature = 'Reaction Temperature'
        temp1=dframe.transpose()
        temp2 = df[['Organism Name',temperature,output]]
        temp3 =removeoutlier_col(temp2,output).set_index('Organism Name')
        Z=pd.concat([temp1, temp3], axis=1).dropna()
        X=Z.loc[:, Z.columns != output ]
        y=Z.loc[:, Z.columns == output]
        temp_col = X.loc[:, X.columns == temperature ]
        X = X.loc[:, X.columns != temperature ]
        
        
        n = len(X)
        enc_seq = np.zeros((n,n))

        i = 0

        for a in list(X.index):
            j = 0
            for b in list(X.index):
                enc_seq[i,j] = sum(score_pairwise(X.loc[a], X.loc[b], blosum_new, -5, -1))
                j += 1
            i += 1

        X = enc_seq
        temp_col = np.array(temp_col).reshape(-1,1)
        X = np.concatenate((X,temp_col), axis =1)
        scaler.fit(X)
        X = scaler.transform(X)
    
    if encoding == 'fft':
        aaindex = Aaindex()
        temperature = 'Reaction Temperature'
        record = aaindex.get(key[0])
        index_data = record.index_data
        index_data['-'] = 0
        sequence_aligned=dframe.apply(lambda dframe : pd.Series(index_data[i] for i in dframe))

        temp1=sequence_aligned.transpose()
        temp2 = df[['Organism Name',temperature,output]]
        temp3 =removeoutlier_col(temp2,output).set_index('Organism Name')
        Z=pd.concat([temp1, temp3], axis=1).dropna()
        X=Z.loc[:, Z.columns != output ]
        y=Z.loc[:, Z.columns == output]
        temp_col = X.loc[:, X.columns == temperature ]
        X = X.loc[:, X.columns != temperature ]
        
        
        X = fft(X)
        # adding magnitude
        X = np.abs(X)
        holder = key[0]
        scaler.fit(X)
        X = scaler.transform(X)
        
        
        
        
    return X,y, holder
    
def ml_process(encoding, output, df, aln, temp = False, jack = False ,  key = None):

    if temp == False:
        X,y,holder = encode(encoding, output, df, aln, key) 
    if temp == True:
        X,y,holder = encode_temp(encoding, output, df, aln, key)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=101)

    all_result = []

    lin_pred, lin_train = LR(X_train, y_train, X_val)
    lasso_pred , lasso_train = LASSO(X_train, y_train, X_val, jack)
    PLS_pred , PLS_train = PLS(X_train, y_train, X_val, jack)
    DT_pred , DT_train = DT(X_train, y_train, X_val, jack)
    RF_pred , RF_train = RF(X_train, y_train, X_val, jack)
    SVM_pred , SVM_train = SVM(X_train, y_train, X_val, jack)
    NNR_pred, NNR_train = NNR(X_train, y_train, X_val, jack)
    EN_pred , EN_train = EN(X_train, y_train, X_val, jack)
    XG_pred, XG_train = XGBR(X_train, y_train, X_val, jack)

    all_result.append([output, "Linear Regression", encoding, holder, np.sqrt(metrics.mean_squared_error(y_train, lin_train)), np.sqrt(metrics.mean_squared_error(y_val, lin_pred)), r2_score(y_train, lin_train),r2_score(y_val, lin_pred), stats.pearsonr(lin_pred,y_val)[0]])
    all_result.append([output, "LASSO Regression", encoding,holder, np.sqrt(metrics.mean_squared_error(y_train, lasso_train)), np.sqrt(metrics.mean_squared_error(y_val, lasso_pred)), r2_score(y_train, lasso_train),r2_score(y_val, lasso_pred),stats.pearsonr(lasso_pred,y_val)[0]]) 
#     all_result.append([output, "Partial Least Square", encoding,holder, np.sqrt(metrics.mean_squared_error(y_train, PLS_train)), np.sqrt(metrics.mean_squared_error(y_val, PLS_pred)), r2_score(y_train, PLS_train),r2_score(y_val,PLS_pred),stats.pearsonr(PLS_pred,y_val)[0]]) 
    all_result.append([output, "Decision Tree Regression", encoding,holder, np.sqrt(metrics.mean_squared_error(y_train, DT_train)), np.sqrt(metrics.mean_squared_error(y_val, DT_pred)), r2_score(y_train, DT_train),r2_score(y_val, DT_pred),stats.pearsonr(DT_pred,y_val)[0]]) 
    all_result.append([output, "Random Forest Regression", encoding,holder,np.sqrt(metrics.mean_squared_error(y_train, RF_train)), np.sqrt(metrics.mean_squared_error(y_val, RF_pred)), r2_score(y_train, RF_train),r2_score(y_val, RF_pred), stats.pearsonr(RF_pred,y_val)[0]]) 
    all_result.append([output, "Support Vector Machine Regression", encoding,holder, np.sqrt(metrics.mean_squared_error(y_train, SVM_train)), np.sqrt(metrics.mean_squared_error(y_val, SVM_pred)), r2_score(y_train, SVM_train),r2_score(y_val, SVM_pred),stats.pearsonr(SVM_pred,y_val)[0]]) 
    all_result.append([output,"Neural Network Regression",encoding,holder, np.sqrt(metrics.mean_squared_error(y_train, NNR_train)), np.sqrt(metrics.mean_squared_error(y_val, NNR_pred)), r2_score(y_train, NNR_train),r2_score(y_val, NNR_pred),stats.pearsonr(NNR_pred,y_val)[0]]) 
    all_result.append([output,"Elastic Net Regression",encoding,holder, np.sqrt(metrics.mean_squared_error(y_train, EN_train)), np.sqrt(metrics.mean_squared_error(y_val, EN_pred)), r2_score(y_train, EN_train),r2_score(y_val, EN_pred),stats.pearsonr(EN_pred,y_val)[0]])
    all_result.append([output,"XGB Regression",encoding,holder, np.sqrt(metrics.mean_squared_error(y_train, XG_train)), np.sqrt(metrics.mean_squared_error(y_val, XG_pred)), r2_score(y_train, XG_train),r2_score(y_val, XG_pred),stats.pearsonr(XG_pred,y_val)[0]]) 

    dfResults = pd.DataFrame(all_result, columns=['Output' , 'Algorithm','Encoding Method' ,'Code', "RMSE Training", 'RMSE Val',"R^2 train","R^2 Val", "Pearson r"])
        
    
    return all_result , dfResults




def temporary_non_scaled_encode(encoding, output, df, aln, key = None):
    
    ClustalAlign = AlignIO.read(aln, 'clustal')
    summary_align = AlignInfo.SummaryInfo(ClustalAlign )
    
    dframe = pc.from_bioalignment(ClustalAlign)
    dframe = dframe.replace('U', '-')
    dframe = dframe.replace('O','-')
    
    main_output = removeoutliers(df[[output]])
    main_index = list(main_output.index)
#     scaler = preprocessing.StandardScaler()
    
    if encoding == 'trigram':
        holder = np.nan
        X = df['Sequence'].iloc[main_index].dropna()
        rem_index = list(X.index)
        y = df[output].iloc[rem_index]

        example = df['Sequence'][0]
        lst = ['E','G','L','Y','T','H','R','A','C','D','P','I','F','N','K','S','V','M','W','Q']
        all_dct = {}
        key = []
        for i in lst:
            for j in lst:
                for k in lst:
                    st = i+j+k
                    all_dct[st] = []

        for example, id in zip(X,range(len(X))):

            temp = list(example)
            temp_dct = dict.fromkeys(all_dct.keys(),0)
            for k in range(len(temp)-2):
                try:
                    check = temp[k] + temp[k+1]+temp[k+2]
                    temp_dct[check] += 1
                except:
                    pass
            for key, value in temp_dct.items():
                all_dct[key].append(value)
        X = pd.DataFrame.from_dict(all_dct)
        temperature = 'Reaction Temperature'
        X1 = df[temperature].iloc[rem_index]
        temp1 = np.array(X1).reshape(-1,1)
        X = np.concatenate((X,temp1), axis =1)
        
        
    if encoding == 'quadrogram':
        holder = np.nan
        X = df['Sequence'].iloc[main_index].dropna()
        rem_index = list(X.index)
        y = df[output].iloc[rem_index]

        example = df['Sequence'][0]
        lst = ['E','G','L','Y','T','H','R','A','C','D','P','I','F','N','K','S','V','M','W','Q']
        all_dct = {}
        key = []
        for i in lst:
            for j in lst:
                for k in lst:
                    for l in lst:
                        st = i+j+k+l
                        all_dct[st] = []

        for example, id in zip(X,range(len(X))):

            temp = list(example)
            temp_dct = dict.fromkeys(all_dct.keys(),0)
            for k in range(len(temp)-3):
                try:
                    check = temp[k] + temp[k+1]+temp[k+2]+temp[k+3]
                    temp_dct[check] += 1
                except:
                    pass
            for key, value in temp_dct.items():
                all_dct[key].append(value)
        X = pd.DataFrame.from_dict(all_dct)
        temperature = 'Reaction Temperature'
        X1 = df[temperature].iloc[rem_index]
        temp1 = np.array(X1).reshape(-1,1)
        X = np.concatenate((X,temp1), axis =1)
        
    return X,y, holder



    

