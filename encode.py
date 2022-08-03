import pandas as pd
import numpy as np
from sklearn import preprocessing


# Two versions of removing outliers
def removeoutliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_out = df[~((df < (Q1 - 100 * IQR)) |(df > (Q3 + 100 * IQR))).any(axis=1)]
#     df_out = df
    return df_out.dropna()

def removeoutlier_col(df,cols):
    Q1 = df[cols].quantile(0.25)
    Q3 = df[cols].quantile(0.75)
    IQR = Q3 - Q1

    df = df[~((df[[cols]] < (Q1 - 100 * IQR)) |(df[[cols]] > (Q3 + 100 * IQR))).any(axis=1)]
    return df.dropna()


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

        
        
        
        
    return X,y, holder