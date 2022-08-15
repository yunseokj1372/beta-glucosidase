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
import beta_glu



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




def rand_generate(inp, number_of_rand, active_site = None):
    ###
    # active_site : list of indices which indicate mutation
    ###
    lst = ['E','G','L','Y','T','H','R','A','C','D','P','I','F','N','K','S','V','M','W','Q']
    if active_site == None:
        index = np.arange(len(list(inp)))
        new_input = np.array(list(inp))
        for ind in np.random.choice(index, number_of_rand,replace = False):
            temp_lst = lst.copy()
            temp_lst.remove(new_input[ind])
            new_input[ind] = np.random.choice(temp_lst)
    else:
        index = np.arange(len(list(inp)))
        new_input = np.array(list(inp))
        for ind in active_site:
            temp_lst = lst.copy()
            temp_lst.remove(new_input[ind])
            new_input[ind] = np.random.choice(temp_lst)
        
    return ''.join(list(new_input))


def encode_input(inp, encoding, df,output, key = None, aln = None, temper=False):
    lst = ['E','G','L','Y','T','H','R','A','C','D','P','I','F','N','K','S','V','M','W','Q']
    all_dct = {}
    key = []
    
    scaler = preprocessing.StandardScaler()
    X,y,holder = beta_glu.encode_temp(encoding, output, df, aln, key = None)
    scaler.fit(X)
    
    
    if encoding == 'Bag-of-Words':
        str_seq = ''.join(inp)
        encoded_inp = pd.DataFrame([ProteinAnalysis(str_seq).count_amino_acids()])
    
    
    
    if encoding == 'bigram':
        str_seq = [inp]
        for i in lst:
            for j in lst:
                st = i+j
                all_dct[st] = []

        for example, id in zip(str_seq,range(len(str_seq))):
            
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
        encoded_inp = pd.DataFrame.from_dict(all_dct)
    
    
    if encoding == 'trigram':
        str_seq = [inp]
        
        for i in lst:
            for j in lst:
                for k in lst:
                    st = i+j+k
                    all_dct[st] = []

        for example, id in zip(str_seq,range(len(str_seq))):
            
            temp = list(example)
            temp_dct = dict.fromkeys(all_dct.keys(),0)
            for k in range(len(temp)-2):
                try:
                    check = temp[k] + temp[k+1] + temp[k+2]
                    temp_dct[check] += 1
                except:
                    pass
            for key, value in temp_dct.items():
                all_dct[key].append(value)
        encoded_inp = pd.DataFrame.from_dict(all_dct)
        
    if str.isnumeric(temper):
        encoded_inp_temp= np.append(encoded_inp,(int(temper)))
        encoded_inp_temp = scaler.transform(encoded_inp_temp.reshape(1,-1))
        
        
    return encoded_inp, encoded_inp_temp


def stochastic_optimizer(inp, encoding, df, output, model, key = None, aln = None, temper=False, epoch=100, number_of_rand =1, active_site = None):
    max = 0
    for i in range(epoch):
        rand_input = rand_generate(inp, number_of_rand, active_site)
        before, new_input =  encode_input(rand_input, encoding, df,output, key = None, aln = aln, temper=temper)
        compare_output = model.predict(new_input)
        if max <= compare_output:
            max = compare_output
            final_input = rand_input
        
    return max, final_input


    

def spotter(orig_inp, new_inp):
    empt = []
    for i in range(len(orig_inp)):
        if orig_inp[i] != new_inp[i]:
            ins = orig_inp[i]+str(i+1)+new_inp[i]
            empt.append(ins)
    return empt




