# -*- coding: utf-8 -*-
#conformal_skin_prediction.py

__doc__ = """ Description: 

With this script you can create and predict conformal models
of skin penetration.

The data is imported from a .csv file seperated using ;
Default option is to include data from Baba et al. 2015
The csv file includes the columns:
Compounds - compound names
Observed - Experimental value for skin perimabillity (log Kp)
Ref - Number that maps the experimental value to a refrence article
smiles - Smiles code that caractericies the compound

Use the flag -verbose to get 

Conformal predicdion is used to calculate prediction ranges

Martin Lindh 2016
"""
##########################################
# Import modules:

import sys
import argparse

import pandas
from pandas import DataFrame
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors

import numpy as np
from numpy.core.numeric import asanyarray
from numpy import mean

import nonconformist
from nonconformist.icp import IcpRegressor
from nonconformist.nc import NormalizedRegressorNc
from nonconformist.nc import RegressorNc, abs_error, abs_error_inv

import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

from math import sqrt

import random

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

import dill

import copy

# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-verbose', help='Get verbose output', action='store_true')
parser.add_argument('-i','-infile', help='Define the input file. Default:baba_jan_2015_smiles.csv', default='C:\Users\Martin\Documents\Magnus_Lundborg\Nya_data/baba_jan_2015_smiles.csv')
parser.add_argument('-c','-conformal_model', action ='store_true',  help='Create a new conformal predictions model')
parser.add_argument('-m', '-model_type', default = 'RF', help = 'Define model algorithm, RF (default) or SVM')
parser.add_argument('-t', '-test_set', default='random', help = 'Use: random (default), full_model, existing, reference')
parser.add_argument('-p', '-predict_model', action='store_true', help="Predict values for one or many smile-codes using single conformal model")
#parser.add_argument('-v', '-validate' , action='store_true')
parser.add_argument('-d','-data_file_name', help='Output file name. File with descriptor data', default='data_with_descriptors.csv')
parser.add_argument('-num_models', default='100', help='Number of models to create. Default = 100')
parser.add_argument('-reference', default=1, help='Use this reference as test_set. Default= 1')
parser.add_argument('-significance', default=0.2, help='Set the significance level of the conformal prediction. Default = 0.2')
parser.add_argument('-phys','-physiochemical_descriptors', default = False, help='Print out data about descriptors. Default = false')
parser.add_argument('-smile', default = 'CCO', help ='Smiles for which skin permeabillity prediction is requested')
args = parser.parse_args()

# -----------------------------------------------------------------------------


#Plot version numbers for Python, matplotlib, pandas and numby
if args.verbose:
    print(' ')
    print(' ################### Versions ################### ')
    print('Python version:\n{}'.format(sys.version))
    print('matplot lib version: {}'.format(matplotlib.__version__))
    print('pandas version: {}'.format(pandas.__version__))
    print('sklearn version: {}'.format(sklearn.__version__))
    print('numpy version: {}'.format(np.__version__))
    print('rdkit version: {}'.format(rdkit.__version__))
    print('nonconformist version: {}'.format(nonconformist.__version__))
    print('dill version: {}'.format(dill.__version__))


# -----------------------------------------------------------------------------




def read_data(infile):
    """
    Description - Reads CSV-file to create data frame
    """
    if args.verbose:
        print('########################## Read CSV-file to create data frame ##########################')
    file_data = pandas.read_csv(infile, sep = ';',header=2)
    
    if args.verbose:
        print(file_data.columns)

    

    return file_data
    
    
def calculate_descriptors(smiles):
    """
    Description - Calculate descriptors using rdkit
    smile','logP','PSA','MolWt','RingCount','HeavyAtomCount','NumRotatableBonds
    """
    
    if args.verbose:
        print('########################## Calculate_Descriptors ##########################')

    descriptors_df = DataFrame(columns=('smile','logP','PSA','MolWt','RingCount','HeavyAtomCount','NumRotatableBonds'))

    i = 0
    for smile in smiles:
        #print smile
        try:
            m = Chem.MolFromSmiles(smile)
        except:
            print(smile)
            print('error')
    

        try:
            logP = Chem.Descriptors.MolLogP(m)
            PSA = Chem.Descriptors.TPSA(m)
            MolWt = Chem.Descriptors.MolWt(m)
            RingCount = Chem.Descriptors.RingCount(m)
            HeavyAtomCount = Chem.Descriptors.HeavyAtomCount(m)
            NumRotatableBonds = Chem.Descriptors.NumRotatableBonds(m)
        except:
            print('Error computing descriptors')
            logP = 0
            PSA = 0
            MolWt = 0
            RingCount = 0
            HeavyAtomCount = 0
            NumRotatableBonds = 0
 
        descriptors_df.loc[i] = ([smile,logP,PSA,MolWt,RingCount,HeavyAtomCount,NumRotatableBonds])
        i +=1

    if args.verbose:
        print(descriptors_df.columns)  
 
    return descriptors_df

    
    
def create_indices_test_training_calibration(data):
    """
    Description - Create training, calibration and test indices
    
    Use existing test and training set
    (file trining)*0.8 training
    (file trining)*0.2 calibration set
    (file test) test
    
    Random selection - validation
    60 % train
    20 % calibrate
    20 % test

    Reference testset
    x/total % 
    (total - x)*0.8 train
    (total - x)*0.2 calibration


    Model creation 
    80 % train
    20 % calibration
        
            
    """
    train = []
    calibrate = []
    test = []
    
    
    if args.verbose:
        print('################## Setup training, calibration and test indices ##########')
        print(args.t)
         
    if args.t == 'existing':
        if args.verbose:
            print('Using existing sets')
            
        blob = []
        test = data.loc[data['Class'] == 'test'].index.tolist()
        blob = data.loc[data['Class'] == 'training'].index.tolist()
        calibrate = random.sample(blob, int(len(blob)*0.2))
        train = [x for x in blob if x not in calibrate]
    
    if args.t == 'random':
        if args.verbose:
            print('Creating random sets')
        idx = np.random.permutation(len(data))
        train = idx[:int(idx.size * 3 / 5)+1]
        calibrate = idx[int(idx.size * 3 / 5)+1:int(4 * idx.size / 5 )+1]
        test = idx[int(4 * idx.size / 5)+1:]
    

        
    if args.t == 'reference':
        if args.verbose:
            print('Creating test set from reference: '+str(args.reference))
        
        #print(data['Ref.'])
        
        test = data.loc[data['Ref.'] ==  int(args.reference)].index.tolist()
        print(len(test))
        blob = data.loc[data['Ref.'] != int(args.reference)].index.tolist()
        calibrate = random.sample(blob, int(len(blob)*0.2))
        train = [x for x in blob if x not in calibrate]   
        
    if args.t == 'full_model':
        if args.verbose:
            print('Creating sets for full model')
        test = []
        idx = np.random.permutation(len(data))
        train = idx[:int(idx.size * 4 / 5)]
        calibrate = idx[int(4 * idx.size / 5):]    
        
    if args.verbose:
        print('Size of sets:')
        print('Train: '+str(len(train)))
        print('Calibration: '+str(len(calibrate)))
        print('Test: '+str(len(test)))
    
    return train, calibrate, test 

def create_train_test_calibrate_sets(data, descriptors_df, train_i, calibrate_i, test_i):
    """
    Description - Create training and test sets. 
    
    Creates X (descriptors) and Y (permiability) values from indices and data. 
    """
    # print('########################## Create training and test sets. ##########################')
    
    # Create y with permiability data
    permiability_y = data['Observed']
    ytrain = permiability_y[train_i] #DEBUG    
    ytest = permiability_y[test_i] #DEBUG
    ycalibrate = permiability_y[calibrate_i] #DEBUG
    
    # Create X with calculated descriptor data
    data_X = descriptors_df.iloc[:,1:]  
    Xtrain =  data_X.iloc[train_i]
    Xtest =  data_X.iloc[test_i]
    Xcalibrate =  data_X.iloc[calibrate_i]
    
    return Xtrain, Xtest, Xcalibrate, ytrain, ytest, ycalibrate     


def write_csv_with_data(data, descriptors_df, newfilename):
    print('################## Write data to CSV-file #################')

    connected_data = pandas.concat([data, descriptors_df], axis=1)
    
    connected_data.to_csv(newfilename, sep=';')

    return True


def randomize_new_indices(train_i, calibrate_i, test_i, data, i):
    """
    Description - Create new indices for two indices arrays. 
    """
    #print('################## Setup new training and calibration and indices ##########')
    
    if  args.t == 'reference':                # or args.t == 'random':

            
        a = []    
        for each in train_i:
            a.append(each)
            
        b = []
        for each in calibrate_i:
            b.append(each)   
        
        c = []
        for each in test_i:
            c.append(each)            
                        
        A = a + b + c

    if args.t == 'full_model' or args.t == 'existing' or args.t == 'random':

        a = []    
        for each in train_i:
            a.append(each)
            
        b = []
        for each in calibrate_i:
            b.append(each) 

        A = a + b  

    idx = np.random.permutation(A)

    if args.t == 'random':

        #train = idx[:int(idx.size * 3 / 5)]
        #calibrate = idx[int(idx.size * 3 / 5):int(4 * idx.size / 5 )]
        #test = idx[int(4 * idx.size / 5):]
        
        #NEW RANDOM
        train = idx[:int(idx.size * 4 / 5)]
        calibrate = idx[int(4 * idx.size / 5):]
        test = test_i     

     

    if args.t == 'full_model' or args.t == 'existing':
        train = idx[:int(idx.size * 4 / 5)]
        calibrate = idx[int(4 * idx.size / 5):]
        test = test_i
    
    if args.t == 'reference':        
  
        test = data.loc[data['Ref.'] ==  (int(i) % data['Ref.'].max() + 1)].index.tolist()
  
        blob = data.loc[data['Ref.'] != int(i)].index.tolist()
        calibrate = random.sample(blob, int(len(blob)*0.2))
        train = [x for x in blob if x not in calibrate]  


        if args.verbose:
            #For DEBUG
            #print('Creating new test set from reference: '+str((int(i) % data['Ref.'].max() + 1)))
            #print('Compounds in new test set: '+str(test))
            pass
            
   
    return list(train), list(calibrate), list(test)
      
                
def calculate_prediction_y_and_error(median_values):    
    #Y-pred + osäkerhet median
    Y_pred_median = []
    error_median = []
    
    for each in median_values:
        Y_pred_median.append((each[0]+each[1])/2)
        error_median.append((abs(each[0])+abs(each[1]))/2 - abs(each[1]))
    #print('Mean value of errors using median: ' + str(np.mean(error_median)))
    #print(len(Y_pred_median))
    return Y_pred_median, error_median
    
def save_models(ICPs):
    """
    Description - Saves the classifiers using dill
    """
    if args.verbose:
        print('########################## Save Classifiers ##########################')
    
    outfile = 'filename.pkl'
    with open(outfile, 'wb') as out_strm: 
        s = dill.dump(ICPs, out_strm)

    return True 


def load_models():
    """
    Description - Load a previosly saved classifier from file using dill
    """
    if args.verbose:
        print('########################## Load Classifier ##########################')
    
    infile = 'filename.pkl'
    with open(infile, 'rb') as in_strm:
        icps = dill.load(in_strm)
 
    return icps

def predict_from_smiles_conformal_median(classifier_list, smiles):
    """
    Description - Predict value of penetration (kp) from a SMILES-description'
    of a molecule using the median of multiple conformal models.
    """
    print('########## Predict using multiple conformal models ##############')
    
    print smiles
    
    descriptors_df= calculate_descriptors(smiles)
    Xvalues = []
    Xvalues = asanyarray(descriptors_df.iloc[:,1:])
    
    print(len(Xvalues))
   
    A = pandas.DataFrame(index = range(len(Xvalues)))
    B = pandas.DataFrame(index = range(len(Xvalues)))
    C = pandas.DataFrame(index = range(len(Xvalues)))


    index = list(xrange(len(smiles)))

    i = 0    
    for classifier in classifier_list:
        
        predicted_skin_permiabillity = classifier.predict(Xvalues, significance = 0.2)
        predicted_values = pandas.DataFrame(predicted_skin_permiabillity)

        A[i] = predicted_values[0]
        B[i] = predicted_values[1]     
        
        i +=1 
        #print(predicted_values) DEBUG

    C['median_prediction_0'] = A.median(axis=1)
    C['median_prediction_1'] = B.median(axis=1)
    C['median_prediction'] = (C['median_prediction_0'] + C['median_prediction_1'])/2
    C['median_prediction_size'] = C['median_prediction'] - C['median_prediction_0']

    #Y_pred_median_test = C['median_prediction'].dropna()
    #median_prediction_size = C['median_prediction_size'].dropna().tolist()
           
    if args.verbose:
        print('Number of conformal models used: '+ str(i))
        print('Predicted range (first entry): '+str(C['median_prediction_0'][0])+' to '+str(C['median_prediction_1'][0]))
        print('Predicted value (first entry): '+str(C['median_prediction'][0]))
        print('Predicted range (first entry): '+str(C['median_prediction_size'][0]))
    #print('Predicted range (second entry): '+str(C['median_prediction_0'][1])+' - '+str(C['median_prediction_1'][1]))

    return C

def create_conformal_model():

    #Read data from file
    data = read_data(args.i)
	
    #Calculate descriptors using RD-kit
    descriptors_df = calculate_descriptors(data['smiles']) 
    
    #Assign indices
    train_i, calibrate_i, test_i  = create_indices_test_training_calibration(data) # Create indices for test,training, calibration sets          
    test_index_total = [x for x in test_i]
    calibrate_index_total = [x for x in calibrate_i]

    #Create inductive conformal prediction regressor
   
    if args.m == 'RF':
        icp = IcpRegressor(NormalizedRegressorNc(RandomForestRegressor, KNeighborsRegressor, abs_error, abs_error_inv, model_params={'n_estimators': 100}))

    if args.m == 'SVM':
        #No support vector regressor
        print('error - no SVM-regressor avliable')
        icp = IcpRegressor(NormalizedRegressorNc(SVR, KNeighborsRegressor, abs_error, abs_error_inv, model_params={'n_estimators': 100}))
           
    #Create DataFrames to store data
    A = pandas.DataFrame(index = range(len(data)))
    B = pandas.DataFrame(index = range(len(data)))
    C = pandas.DataFrame(index = range(len(data)))

    iA = pandas.DataFrame(index = range(len(data)))
    iB = pandas.DataFrame(index = range(len(data)))
    iC = pandas.DataFrame(index = range(len(data)))

    if args.verbose:
	print('Number of models to create: '+args.num_models)
	print('############## Starting calculations ##############')
    
    icp_s = []


    for i in range(int(args.num_models)): #DEBUG 100
        Xtrain, Xtest, Xcalibrate, ytrain, ytest, ycalibrate = create_train_test_calibrate_sets(data, descriptors_df,  train_i, calibrate_i, test_i)

        #Create nornal model
        icp.fit(Xtrain, ytrain)
    
        #Calibrate normal model               
        icp.calibrate(asanyarray(Xcalibrate), asanyarray(ycalibrate))
            
        #Predrict test and training sets
        prediction_test = icp.predict(asanyarray(Xtest), significance = args.significance) # 0.2
        prediction_calibrate = icp.predict(asanyarray(Xcalibrate), significance = args.significance)

        #Create DF with data
        blob = pandas.DataFrame(prediction_test, index=test_i)
        iblob = pandas.DataFrame(prediction_calibrate, index=calibrate_i)
        
        A[i] = blob[0]
        B[i] = blob[1]

        iA[i] = iblob[0]
        iB[i] = iblob[1]
        

        #Create new indices for next model
        test_index_total = np.unique(np.concatenate((test_index_total, test_i), axis=0))
        calibrate_index_total = np.unique(np.concatenate((calibrate_index_total, calibrate_i), axis=0)) 
        
        train_i, calibrate_i, test_i  = randomize_new_indices(train_i, calibrate_i, test_i, data, i)
        
        #temp = sklearn.base.clone(icp)
        icp_s.append(copy.copy(icp))
    

    ### Save models ###
    save_models(icp_s)
    

 

    if args.verbose:
        print('################## Loop finished, model created, test set predicted #################')
        
    experimental_values = data['Observed'][test_index_total]
    iexperimental_values = data['Observed'][calibrate_index_total] 


    C['median_prediction_0'] = A.median(axis=1)
    C['median_prediction_1'] = B.median(axis=1)
    C['median_prediction'] = (C['median_prediction_0'] + C['median_prediction_1'])/2
    C['median_prediction_size'] = C['median_prediction'] - C['median_prediction_0']

    Y_pred_median_test = C['median_prediction'].dropna()
    median_prediction_size = C['median_prediction_size'].dropna().tolist()
        
    num_outside_median = 0
    for i in range(len(data)):
        try:
            if  C['median_prediction_0'].dropna()[i] < experimental_values[i] < C['median_prediction_1'].dropna()[i]:
                pass
            else:
                num_outside_median +=1
                #print('Outside range')
        except:
            pass #print('error')
    
    #Internal prediction
    iC['median_prediction_0'] = iA.median(axis=1)
    iC['median_prediction_1'] = iB.median(axis=1)
    iC['median_prediction'] = (iC['median_prediction_0'] + iC['median_prediction_1'])/2
    iC['median_prediction_size'] = iC['median_prediction'] - iC['median_prediction_0']
    
    iY_pred_median_test = iC['median_prediction'].dropna()
    imedian_prediction_size = iC['median_prediction_size'].dropna().tolist()

    inum_outside_median = 0
    for i in range(len(data)):
        try:
            if  iC['median_prediction_0'].dropna()[i] < iexperimental_values[i] < iC['median_prediction_1'].dropna()[i]:
                pass
            else:
                inum_outside_median +=1
                #print('Outside range')
        except:
            pass #print('error')


    if args.verbose:
        print('########################## Prediction statistics external test ##########################')
        print('')
    

       
    print('Number of compounds predicted in test set: '+ str(C['median_prediction'].notnull().sum()))   
    
    if args.t != 'full_model':         
        ex_r2_score= r2_score(experimental_values, Y_pred_median_test)
        print('R^2 (coefficient of determination):  %.3f' % ex_r2_score)

        ex_mean_squared_error = mean_squared_error(experimental_values, Y_pred_median_test)
        ex_rmse = sqrt(ex_mean_squared_error)               
        print('RMSE:  %.3f' % ex_rmse)
        
        ex_MAE = mean_absolute_error(experimental_values, Y_pred_median_test)
        print('Mean absolute error:  %.3f' % ex_MAE)
 
        print('Mean squared error: %.3f' % ex_mean_squared_error)

        #Average prediction range   
        print('Mean of median prediction range: %.3f' % mean(median_prediction_size))

        percent_num_outside_median = 100*float(num_outside_median)/float(len(experimental_values))
        print('Number of compounds outside of prediction range: '+str(num_outside_median))
        print('% of compounds predicted outside of prediction range: '+str(percent_num_outside_median) +' %')
        print(' ')

        #####Internal Prediction ########
    
        print('Number of compounds predicted in training set: '+ str(iC['median_prediction'].notnull().sum()))   
          
        iex_r2_score= r2_score(iexperimental_values, iY_pred_median_test)
        print('R^2 (coefficient of determination):  %.3f' % iex_r2_score)

        iex_mean_squared_error = mean_squared_error(iexperimental_values, iY_pred_median_test)
        iex_rmse = sqrt(iex_mean_squared_error)               
        print('RMSE:  %.3f' % iex_rmse)
        
        print('Mean squared error: %.3f' % iex_mean_squared_error)
        
       
        iex_MAE = mean_absolute_error(iexperimental_values, iY_pred_median_test)
        print('Mean absolute error:  %.3f' % iex_MAE)

        #Average prediction range   
        print('Mean of median prediction range: %.3f' % mean(imedian_prediction_size))



        ipercent_num_outside_median = 100*float(inum_outside_median)/float(len(iexperimental_values))
        print('Number of compounds outside of prediction range: '+str(inum_outside_median))
        print('% of compounds predicted outside of prediction range: '+str(ipercent_num_outside_median) +' %')
        print(' ')   

        #### Plot results - plot test set
        if args.verbose:
            print(' ################ Plotting testset #################')
        fig, ax = plt.subplots()

        ax.errorbar(experimental_values, Y_pred_median_test, yerr=median_prediction_size,
        fmt='o', markeredgecolor = 'black', markersize = 6,
        mew=1, ecolor='black', elinewidth=0.3, capsize = 3, capthick=1, errorevery = 1)
    
        #Set the size
        ax.set_ylim([-10,-3])
        ax.set_xlim([-10,-3])
    

        # Plot title and lables
        #plt.title('Median predictions with prediction ranges for the testset')
        plt.ylabel('Predicted log Kp')
        plt.xlabel('Experimental log Kp')
    
        # Draw line 
        fit = np.polyfit(experimental_values, Y_pred_median_test, 1)
    
        x = [-10,-3]
    
        #Regression line
        #ax.plot(experimental_values, fit[0]*asanyarray(experimental_values)+ fit[1], color='black')
        #ax.plot(x, fit[0]*asanyarray(x)+ fit[1], color='black')
    

    
        #Creating colored dots for ref 10
    
        #ref10_experimental = data.loc[data['Ref.'] == 10]['Observed']
        #ref10_predicted = C['median_prediction'][ref10_experimental.index]
        #ax.scatter(ref10_experimental, ref10_predicted,marker = 'o', color ='red', s = 100)
    


        ax.plot(x, x, color='black')
    
        plt.show()

    #Print data in CSV-file
    
    descriptors_df['Median prediction low range'] = C['median_prediction_0']
    descriptors_df['Median prediction high range'] = C['median_prediction_1'] 
    descriptors_df['Median prediction'] = C['median_prediction']
    descriptors_df['size prediction range'] = C['median_prediction_1'] - C['median_prediction_0']
    write_csv_with_data(data,descriptors_df, args.d)


    #Calculate min, max and mean values for descriptors
    if args.phys:
        print(args.phys)
        print('Min: ')
        print(descriptors_df.min())
        print('Max: ')
        print(descriptors_df.max())
        print('Mean:')    
        print(descriptors_df.mean()) 


# -----------------------------------------------------------------------------

if args.verbose:
    print(" Verbose output turned on")
    
    print(' Model Type is: '+args.m) # m = model_type

    #print(parser)



    
if args.c:
    #Create conformal models
    if args.verbose:
        print(' Input data file: ')
        print(' '+args.i) # i = input_file

    create_conformal_model()
       
if args.p:
    if args.verbose:
        print('Predict using earlier models')
    try:
        classifier_list = load_models()
    except:
        print('No conformal model found')
    
    smile_code = [args.smile] #args.smiles
    predict_from_smiles_conformal_median(classifier_list, smile_code)
    
    