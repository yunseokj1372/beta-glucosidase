# Importing needed library
import pandas as pd
import numpy as np

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
from Bio import AlignIO
from Bio import SeqIO
from Bio.Align.Applications import MuscleCommandline
from Bio.Align import AlignInfo

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

# Two versions of removing outliers
def removeoutliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_out = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df_out.dropna()

def removeoutlier_col(df,cols):
    Q1 = df[cols].quantile(0.25)
    Q3 = df[cols].quantile(0.75)
    IQR = Q3 - Q1

    df = df[~((df[[cols]] < (Q1 - 1.5 * IQR)) |(df[[cols]] > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df.dropna()




# Machine Learning Tools

def LR(X_train, y_train, X_test):
    from sklearn.linear_model import LinearRegression
    lm = LinearRegression()
    lm.fit(X_train,y_train)
    lin_pred = lm.predict(X_test)
    pred_training= lm.predict(X_train)
    
    return lin_pred , pred_training

#Ordinary Least Squares Regression
def OLS(X_train, y_train, X_test):
    import statsmodels.api as sm
    from scipy import stats
    X2 = sm.add_constant(X_train)
    est = sm.OLS(y_train,X2)
    est2 = est.fit()
    print(est2.summary())
    coefficients=est2.params[1:]
    intercept=est2.params[0]
    OLS_pred=(coefficients*X_test).sum(axis = 1, skipna = True) + intercept
    
    return OLS_pred

#Lasso regression

def LASSO(X_train, y_train, X_test):
    param_grid={'alpha':np.arange(0,1.1,0.05).tolist()}
    lasso_reg=Lasso()
    grid_search=GridSearchCV(lasso_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X_train,np.ravel(y_train))
    lasso_pred=grid_search.predict(X_test)
    pred_training= grid_search.predict(X_train)
    return lasso_pred, pred_training

#Partial Least Square Regression

def PLS(X_train, y_train, X_test):
    param_grid={'n_components':np.arange(1,10,1).tolist()}
    pls = PLSRegression()
    grid_search=GridSearchCV(pls, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    pls.fit(scale(X_train), y_train)
    PLS_pred=pls.predict(scale(X_test))
    pred_training= pls.predict(X_train)
    
    return PLS_pred , pred_training

#Random Forest

def RF(X_train, y_train, X_test):
    param_grid={"n_estimators": [100] , "max_features": [5], 'random_state':[20]}
    rf = RandomForestRegressor()
    grid_search=GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X_train,np.ravel(y_train))
    RF_pred = grid_search.predict(X_test)
    pred_training= grid_search.predict(X_train)
    return RF_pred, pred_training

#Decision Tree

def DT(X_train, y_train, X_test):
    param_grid={ "max_features": [5], 'random_state':[20]}
    tree_reg=DecisionTreeRegressor()
    grid_search=GridSearchCV(tree_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X_train,np.ravel(y_train))
    DT_pred = grid_search.predict(X_test)
    pred_training= grid_search.predict(X_train)
    return DT_pred, pred_training

#SVM

def SVM(X_train, y_train, X_test):
    param_grid = {'C': [100],
              'epsilon': [0.5],
              'kernel': ['poly']}
 
    model = SVR()
    grid = GridSearchCV(SVR(), param_grid, refit = True)
    grid.fit(X_train, y_train)
    SVM_pred = grid.predict(X_test)
    pred_training= grid.predict(X_train)
    return SVM_pred, pred_training

#Neural Network Regression

def NNR(X_train, y_train, X_test):
    param_grid = {"hidden_layer_sizes": [(1,),(50,)], "alpha": [0.5]}
    
    grid = GridSearchCV(MLPRegressor(random_state=101, max_iter=100), param_grid, refit = True)
    grid.fit(X_train, y_train)
    NNR_pred = grid.predict(X_test)
    pred_training= grid.predict(X_train)
    return NNR_pred, pred_training

#Elastic Net

def EN(X_train, y_train, X_test):
    param_grid = dict()
    param_grid['alpha'] = [1.0]
    param_grid['l1_ratio'] =[0.5]

    grid = GridSearchCV(ElasticNet(), param_grid,scoring='r2', refit = True)
    grid.fit(X_train, y_train)
    EN_pred = grid.predict(X_test)
    pred_training= grid.predict(X_train)

    return EN_pred, pred_training


def XGBR(X_train, y_train, X_test):
    params = { 'max_depth': [4],
           'n_estimators': [100],
           'colsample_bytree': [0.2],
           'min_child_weight': [3],
           'gamma': [0.3],
           'subsample': [0.4]}
    model = xgb.XGBRegressor()
    grid = GridSearchCV(estimator=XGBRegressor(), 
                       param_grid=params)
    grid.fit(X_train, y_train)
    XGB_pred = grid.predict(X_test)
    pred_training= grid.predict(X_train)
    return XGB_pred, pred_training


# BLOSUM score function

blosum.update(((b,a),val) for (a,b),val in list(blosum.items()))

def score_pairwise(seq1, seq2, matrix, gap_s, gap_e, gap = True):
    for A,B in zip(seq1, seq2):
        diag = ('-'==A) or ('-'==B)
        yield (gap_e if gap else gap_s) if diag else matrix[(A,B)]
        gap = diag



def ml_process(encoding, output, df, key = None):
    data=df[['Wild/Mutant', 'Kingdom', 'pH Optimum', 'Temperature Optimum',
       't1/2 (min)', 'kd (min-1)', 'pNP-Glc Km (mM)', 'pNP-Glc kcat (1/s)',
       'pNP-Glc kcat/Km (1/smM)', 'Cellobiose Km (mM)',
       'Cellobiose kcat (1/s)', 'Cellobiose kcat/Km (1/smM)', 'Cellobiose Ki (mM)',
       'pNP-Glc Ki (mM)', 'MW (kDa)']]
    

    ClustalAlign = AlignIO.read('seqs.aln', 'clustal')
    summary_align = AlignInfo.SummaryInfo(ClustalAlign )
    
    dframe = pc.from_bioalignment(ClustalAlign)
    dframe = dframe.replace('U', '-')
    dframe = dframe.replace('O','-')
    
    main_output = removeoutliers(df[[output]])
    main_index = list(main_output.index)
    
    all_result = []
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
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)
    
    
    lin_pred, lin_train = LR(X_train, y_train, X_test)
    lasso_pred , lasso_train = LASSO(X_train, y_train, X_test)
    PLS_pred , PLS_train = PLS(X_train, y_train, X_test)
    DT_pred , DT_train = DT(X_train, y_train, X_test)
    RF_pred , RF_train = RF(X_train, y_train, X_test)
    SVM_pred , SVM_train = SVM(X_train, y_train, X_test)
    NNR_pred, NNR_train = NNR(X_train, y_train, X_test)
    EN_pred , EN_train = EN(X_train, y_train, X_test)
    
    all_result.append([output, "Linear Regression", encoding, holder, np.sqrt(metrics.mean_squared_error(y_train, lin_train)), np.sqrt(metrics.mean_squared_error(y_test, lin_pred)), r2_score(y_train, lin_train),r2_score(y_test, lin_pred)]) 
    all_result.append([output, "LASSO Regression", encoding,holder, np.sqrt(metrics.mean_squared_error(y_train, lasso_train)), np.sqrt(metrics.mean_squared_error(y_test, lasso_pred)), r2_score(y_train, lasso_train),r2_score(y_test, lasso_pred)]) 
    all_result.append([output, "Partial Least Square", encoding,holder, np.sqrt(metrics.mean_squared_error(y_train, PLS_train)), np.sqrt(metrics.mean_squared_error(y_test, PLS_pred)), r2_score(y_train, PLS_train),r2_score(y_test, PLS_pred)]) 
    all_result.append([output, "Decision Tree Regression", encoding,holder, np.sqrt(metrics.mean_squared_error(y_train, DT_train)), np.sqrt(metrics.mean_squared_error(y_test, DT_pred)), r2_score(y_train, DT_train),r2_score(y_test, DT_pred)]) 
    all_result.append([output, "Random Forest Regression", encoding,holder,np.sqrt(metrics.mean_squared_error(y_train, RF_train)), np.sqrt(metrics.mean_squared_error(y_test, RF_pred)), r2_score(y_train, RF_train),r2_score(y_test, RF_pred)]) 
    all_result.append([output, "Support Vector Machine Regression", encoding,holder, np.sqrt(metrics.mean_squared_error(y_train, SVM_train)), np.sqrt(metrics.mean_squared_error(y_test, SVM_pred)), r2_score(y_train, SVM_train),r2_score(y_test, SVM_pred)]) 
    all_result.append([output,"Neural Network Regression",encoding,holder, np.sqrt(metrics.mean_squared_error(y_train, NNR_train)), np.sqrt(metrics.mean_squared_error(y_test, NNR_pred)), r2_score(y_train, NNR_train),r2_score(y_test, NNR_pred)]) 
    all_result.append([output,"Elastic Network Regression",encoding,holder, np.sqrt(metrics.mean_squared_error(y_train, EN_train)), np.sqrt(metrics.mean_squared_error(y_test, EN_pred)), r2_score(y_train, EN_train),r2_score(y_test, EN_pred)]) 


    dfResults = pd.DataFrame(all_result, columns=['Output' , 'Algorithm','Encoding Method' ,'Code', "RMSE Training", 'RMSE Test',"R^2 train","R^2 pred"])
    
    return dfResults


def encode(encoding, output, df, key = None):
    data=df[['Wild/Mutant', 'Kingdom', 'pH Optimum', 'Temperature Optimum',
           't1/2 (min)', 'kd (min-1)', 'pNP-Glc Km (mM)', 'pNP-Glc kcat (1/s)',
           'pNP-Glc kcat/Km (1/smM)', 'Cellobiose Km (mM)',
           'Cellobiose kcat (1/s)', 'Cellobiose kcat/Km (1/smM)', 'Cellobiose Ki (mM)',
           'pNP-Glc Ki (mM)', 'MW (kDa)']]
    

    ClustalAlign = AlignIO.read('seqs.aln', 'clustal')
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
        
        
    return X,y
    


