#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 14:32:24 2018

@author: joe
"""

# This code provides functions for reading in directional reflectance data obtained
# via ground spectroscopy, reformatting into a pandas dataframe of features and labels,
# optimising and training a series of machine learning algorithms on the ground spectra
# then using the trained model to predict the surface type in each pixel of a UAV
# image. The UAV image has been preprocessed by stitching individual images together and
# converting to reflectance using panels on the ground before being saved as a multiband TIF which
# is then loaded here. A narrowband to broadband conversion (Knap 1999) is applied to the
# data to create an albedo map, and this is then used to create a large dataset of surface 
# type and associated broadband albedo


###########################################################################################
############################# IMPORT MODULES #########################################

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing, cross_validation, neighbors, svm
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import gdal
import rasterio
from sklearn.grid_search import GridSearchCV
plt.style.use('ggplot')

HCRF_file = '//home//joe//Code//HCRF_master_machine.csv'
img_name = '/media/joe/FDB2-2F9B/uav_refl.tif'

#######################################################################################
############################ DEFINE FUNCTIONS ###################################


def create_dataset(HCRF_file):
# Read in raw HCRF data to DataFrame. This version pulls in HCRF data from 2016 and 2017
    hcrf_master = pd.read_csv(HCRF_file)
    HA_hcrf = pd.DataFrame()
    LA_hcrf = pd.DataFrame()
    CI_hcrf = pd.DataFrame()
    CC_hcrf = pd.DataFrame()
    WAT_hcrf = pd.DataFrame()
    
    # Group site names
    
    HAsites = ['13_7_SB2','13_7_SB4','14_7_S5','14_7_SB1','14_7_SB5','14_7_SB10',
    '15_7_SB3','21_7_SB1','21_7_SB7','22_7_SB4','22_7_SB5','22_7_S3','22_7_S5',
    '23_7_SB3','23_7_SB5','23_7_S3','23_7_SB4','24_7_SB2','HA_1', 'HA_2','HA_3',
    'HA_4','HA_5','HA_6','HA_7','HA_8','HA_10','HA_11','HA_12','HA_13','HA_14',
    'HA_15','HA_16','HA_17','HA_18','HA_19','HA_20','HA_21','HA_22','HA_24',
    'HA_25','HA_26','HA_27','HA_28','HA_29','HA_30','HA_31',
    # the following were reclassified from LAsites due to their v low reflectance
    '13_7_S2','14_7_SB9','MA_11','MA_14','MA_15','MA_17','21_7_SB2','22_7_SB1',
    'MA_4','MA_7','MA_18'
    ]
    # These have been removed completely from HAsites: '21_7_S3', '23_7_S5', 'HA_32'
    # '24_7_S1','25_7_S1','HA_9', 'HA_33','13_7_SB1', '13_7_S5', 'HA_23'
    
    LAsites = [
    '14_7_S2','14_7_S3','14_7_SB2','14_7_SB3','14_7_SB7','15_7_S2','15_7_SB4',
    '20_7_SB1','20_7_SB3','21_7_S1','21_7_S5','21_7_SB4','22_7_SB2','22_7_SB3','22_7_S1',
    '23_7_S1','23_7_S2','24_7_S2','MA_1','MA_2','MA_3','MA_5','MA_6','MA_8','MA_9',
    'MA_10','MA_12','MA_13','MA_16','MA_19',
    #these have been moved from CI
    '13_7_S1','13_7_S3','14_7_S1','15_7_S1','15_7_SB2','20_7_SB2','21_7_SB5','21_7_SB8','25_7_S3'
    ]
    # These have been removed competely from LA sites
    # '13_7_S2','13_7_SB1','14_7_SB9', '15_7_S3' ,'MA_11',' MA_14','MA15','MA_17',
    # '13_7_S5', '25_7_S2','25_7_S4','25_7_S5'
    
    CIsites =['13_7_SB3','13_7_SB5','15_7_S4','15_7_SB1','15_7_SB5','21_7_S2',
    '21_7_S4','21_7_SB3','22_7_S2','22_7_S4','23_7_SB1','23_7_SB2','23_7_S4',
    'WI_1','WI_2','WI_3','WI_4','WI_5','WI_6','WI_7','WI_8','WI_9','WI_10','WI_11',
    'WI_12','WI_13']
    
    CCsites = ['DISP1','DISP2','DISP3','DISP4','DISP5','DISP6','DISP7','DISP8',
               'DISP9','DISP10','DISP11','DISP12','DISP13','DISP14']
    
    WATsites = ['21_7_SB5','21_7_SB7','21_7_SB8',
             '25_7_S3', 'WAT_1','WAT_3','WAT_4','WAT_5','WAT_6']
    
    #REMOVED FROM WATER SITES 'WAT_2'
    
    # Create dataframes for ML algorithm
    for i in HAsites:
        hcrf_HA = np.array(hcrf_master[i])
        HA_hcrf['{}'.format(i)] = hcrf_HA
    
    for ii in LAsites:
        hcrf_LA = np.array(hcrf_master[ii])
        LA_hcrf['{}'.format(ii)] = hcrf_LA
         
    for iii in CIsites:   
        hcrf_CI = np.array(hcrf_master[iii])
        CI_hcrf['{}'.format(iii)] = hcrf_CI   
    
    for iv in CCsites:   
        hcrf_CC = np.array(hcrf_master[iv])
        CC_hcrf['{}'.format(iv)] = hcrf_CC   
    
    for v in WATsites:   
        hcrf_WAT = np.array(hcrf_master[v])
        WAT_hcrf['{}'.format(v)] = hcrf_WAT  
    
    # Make dataframe with column for label, columns for reflectancxe at key wavelengths
    # select wavelengths to use - currently set to 8 Sentnel 2 bands
    
    
    X = pd.DataFrame()
    
    X['R125'] = np.array(HA_hcrf.iloc[125])
    X['R210'] = np.array(HA_hcrf.iloc[210])
    X['R318'] = np.array(HA_hcrf.iloc[315])
    X['R367'] = np.array(HA_hcrf.iloc[367])
    X['R490'] = np.array(HA_hcrf.iloc[490])
    
    X['label'] = 'HA'
    
    
    Y = pd.DataFrame()
    Y['R125'] = np.array(LA_hcrf.iloc[125])
    Y['R210'] = np.array(LA_hcrf.iloc[210])
    Y['R318'] = np.array(LA_hcrf.iloc[318])
    Y['R367'] = np.array(LA_hcrf.iloc[367])
    Y['R490'] = np.array(LA_hcrf.iloc[490])
    
    Y['label'] = 'LA'
    
    
    Z = pd.DataFrame()
    
    Z['R125'] = np.array(CI_hcrf.iloc[125])
    Z['R210'] = np.array(CI_hcrf.iloc[210])
    Z['R318'] = np.array(CI_hcrf.iloc[318])
    Z['R367'] = np.array(CI_hcrf.iloc[367])
    Z['R490'] = np.array(CI_hcrf.iloc[490])
    
    Z['label'] = 'CI'
    
    
    P = pd.DataFrame()
    
    P['R125'] = np.array(CC_hcrf.iloc[125])
    P['R210'] = np.array(CC_hcrf.iloc[210])
    P['R318'] = np.array(CC_hcrf.iloc[318])
    P['R367'] = np.array(CC_hcrf.iloc[367])
    P['R490'] = np.array(CC_hcrf.iloc[490])
    
    P['label'] = 'CC'
    
    
    Q = pd.DataFrame()
    Q['R125'] = np.array(WAT_hcrf.iloc[125])
    Q['R210'] = np.array(WAT_hcrf.iloc[210])
    Q['R318'] = np.array(WAT_hcrf.iloc[318])
    Q['R367'] = np.array(WAT_hcrf.iloc[367])
    Q['R490'] = np.array(WAT_hcrf.iloc[490])
    
    Q['label'] = 'WAT'
    


    Zero = pd.DataFrame()
    Zero['R125'] = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    Zero['R210'] = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    Zero['R318'] = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    Zero['R367'] = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    Zero['R490'] = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    
    Zero['label'] = 'UNKNOWN'
    
    # Join dataframes into one continuous DF
    
    X = X.append(Y,ignore_index=True)
    X = X.append(Z,ignore_index=True)
    X = X.append(P,ignore_index=True)
    X = X.append(Q,ignore_index=True)
    X = X.append(Zero,ignore_index=True)    

    # Create features and labels (XX = features - all data but no labels, YY = labels only)
    
    XX = X.drop(['label'],1)
    YY = X['label']
    
    return X, XX, YY



def optimise_train_model(X,XX,YY, error_selector):
    
    # Function to split the dataste into training and test sets, then test the performance of 
    # a range of models on the training data. The final model selected is then evaluated
    # using the test set. The function automatically selects the model that performs best on
    # the training sets, then tests it on the test set. All performance metrics are 
    # printed to ipython. The error type to use to select the model is determined in the 
    # function call. Options for error_selector are: 'accuracy', 'F1', 'recall', 'average_all_metric' 
    
    # X, XX, YY are the datasets with and without labels. error selector determines which error metric
    # the code should use to choose the best classifier, as different models might outperforms others
    # depending upon the error metric used. The options are strings 'F1', 'accuracy' or 'recall'
    # empty lists to append to
    Naive_Bayes = []
    KNN = []
    SVM = []
    
    # split data into test and train sets. Random_state = 42 ensures the ransomly selected values in tets and
    # training sets are consistent throughout the script (can be set to any aribitrary integer value - 42 chosen
    # because 42 = meaning of life, the universe and everything)
    
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(XX,YY,test_size = 0.2)

    # optimise params for support vector machine using cross validation grid search
    # (GridSearchCV) to find optimal set of values for best model performance.
    # Apply to three kernel types and wide range of C and gamma values. 
    # Print best set of params.  
    
    tuned_parameters = [
            {'kernel': ['linear'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
                         'C': [0.1, 1, 10, 100, 1000, 10000]},
                        {'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
                         'C': [0.1, 1, 10, 100, 1000, 10000]},
                        {'kernel':['poly'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
                         'C':[0.1,1,10,100,1000,10000]},
                        {'kernel':['sigmoid'],'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
                         'C':[0.1,1,10,100,1000,10000]}
                        ]
    
    clf_svm = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5)
    clf_svm.fit(X_train, Y_train)
    
    print()
    print("Best parameters set found on development set:")
    print(clf_svm.best_params_)
    print() #line break
    
    kernel = clf_svm.best_estimator_.get_params()['kernel']
    C = clf_svm.best_estimator_.get_params()['C']
    gamma = clf_svm.best_estimator_.get_params()['gamma']
    
    
    # test different classifers and report performance metrics using traning data only
    
    # 1. Try Naive Bayes
    clf_NB = GaussianNB()
    clf_NB.fit(X_train,Y_train)
    accuracy_NB = clf_NB.score(X_train,Y_train) #calculate accuracy
    Y_predict_NB = clf_NB.predict(X_train) # make nre prediction
    conf_mx_NB = confusion_matrix(Y_train,Y_predict_NB) # calculate confusion matrix
    recall_NB = recall_score(Y_train,Y_predict_NB,average="macro")
    f1_NB = f1_score(Y_train, Y_predict_NB, average="macro") # calculate f1 score
    average_metric_NB = (accuracy_NB+recall_NB+f1_NB)/3
    
    # 2. Try K-nearest neighbours
    clf_KNN = neighbors.KNeighborsClassifier()
    clf_KNN.fit(X_train,Y_train)
    accuracy_KNN = clf_KNN.score(X_train,Y_train)
    Y_predict_KNN = clf_KNN.predict(X_train)
    conf_mx_KNN = confusion_matrix(Y_train,Y_predict_KNN)
    recall_KNN = recall_score(Y_train,Y_predict_KNN,average="macro")
    f1_KNN = f1_score(Y_train, Y_predict_KNN, average="macro")
    average_metric_KNN = (accuracy_KNN + recall_KNN + f1_KNN)/3
    
    
    # 3. Try support Vector Machine with best params from optimisation
    clf_svm = svm.SVC(kernel=kernel, C=C, gamma = gamma)
    clf_svm.fit(X_train,Y_train)
    accuracy_svm = clf_svm.score(X_train,Y_train)
    Y_predict_svm = clf_svm.predict(X_train)
    conf_mx_svm = confusion_matrix(Y_train,Y_predict_svm)
    recall_svm = recall_score(Y_train,Y_predict_svm,average="macro")
    f1_svm = f1_score(Y_train, Y_predict_svm, average="macro")
    average_metric_svm = (accuracy_svm + recall_svm + f1_svm)/3
    
    print('*** MODEL TEST SUMMARY ***')
    print('KNN accuracy = ',accuracy_KNN, 'KNN_F1_Score = ', f1_KNN)
    print('Naive Bayes accuracy = ', accuracy_NB, 'Naive_Bayes_F1_Score = ',f1_NB)
    print('SVM accuracy = ',accuracy_svm, 'SVM_F1_Score = ', f1_svm)
    
            # PLOT CONFUSION MATRICES
    plt.figure()    
    plt.imshow(conf_mx_NB)
    plt.title('NB Model Confusion matrix')
    plt.colorbar()
    
    plt.figure()
    plt.imshow(conf_mx_KNN)
    plt.title('KNN Model Confusion matrix')
    plt.colorbar()
    
    plt.figure()
    plt.imshow(conf_mx_svm)
    plt.title('SVM Model Confusion matrix')
    plt.colorbar()
    
    print() #line break
    
    if error_selector == 'accuracy':
        
        if np.mean(KNN) > np.mean(Naive_Bayes) and np.mean(KNN) > np.mean(SVM):
            clf = neighbors.KNeighborsClassifier()
            clf.fit(X_train,Y_train)
            print('KNN model chosen')
        elif np.mean(Naive_Bayes) > np.mean(KNN) and np.mean(Naive_Bayes) > np.mean(SVM):
            clf = GaussianNB()
            clf.fit(X_train,Y_train)
            print('Naive Bayes model chosen')
        else:
            clf = svm.SVC(kernel=kernel, C=C, gamma = gamma)
            clf.fit(X_train,Y_train)
            print('SVM model chosen')
            print('SVM Params: C = ',C,' gamma = ',gamma,' kernel = ',kernel )

    elif error_selector == 'recall':
        
        if recall_KNN > recall_NB and recall_KNN > recall_svm:
            clf = neighbors.KNeighborsClassifier()
            clf.fit(X_train,Y_train)
            print('KNN model chosen')
        elif recall_NB > recall_KNN and recall_NB > recall_svm:
            clf = GaussianNB()
            clf.fit(X_train,Y_train)
            print('Naive Bayes model chosen')
        else:
            clf = svm.SVC(kernel=kernel, C=C, gamma = gamma)
            clf.fit(X_train,Y_train)
            print('SVM model chosen')
            print('SVM Params: C = ',C,' gamma = ',gamma,' kernel = ',kernel )        
        
    elif error_selector == 'F1':
        if f1_KNN > f1_NB and f1_KNN > f1_svm:
            clf = neighbors.KNeighborsClassifier()
            clf.fit(X_train,Y_train)
            print('KNN model chosen')
        elif f1_NB > f1_KNN and f1_NB > f1_svm:
            clf = GaussianNB()
            clf.fit(X_train,Y_train)
            print('Naive Bayes model chosen')
        else:
            clf = svm.SVC(kernel=kernel, C=C, gamma = gamma)
            clf.fit(X_train,Y_train)
            print('SVM model chosen')
            print('SVM Params: C = ',C,' gamma = ',gamma,' kernel = ',kernel )


    elif error_selector == 'average_all_metric':
        if f1_KNN > f1_NB and f1_KNN > f1_svm:
            clf = neighbors.KNeighborsClassifier()
            clf.fit(X_train,Y_train)
            print('KNN model chosen')
        elif f1_NB > f1_KNN and f1_NB > f1_svm:
            clf = GaussianNB()
            clf.fit(X_train,Y_train)
            print('Naive Bayes model chosen')
        else:
            clf = svm.SVC(kernel=kernel, C=C, gamma = gamma)
            clf.fit(X_train,Y_train)
            print('SVM model chosen')
            print('SVM Params: C = ',C,' gamma = ',gamma,' kernel = ',kernel )



# Now that model has been selected using error metrics from training data, the final
# model can be evaluated on the test set. The code below therefore measures the f1, recall,
# confusion matrix and accuracy  for the final selected model and prints to ipython.
            
    Y_test_predicted = clf.predict(X_test)
    final_conf_mx = confusion_matrix(Y_test, Y_test_predicted)

    plt.figure()
    plt.imshow(final_conf_mx)
    plt.title('Final Model Confusion Matrix')
    plt.colorbar()
    
    # Normalise confusion matrix to show errors
    row_sums = final_conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = final_conf_mx / row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    plt.figure()
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.colorbar()

    final_recall = recall_score(Y_test,Y_test_predicted,average="macro")
    final_f1 = f1_score(Y_test, Y_test_predicted, average="macro")
    final_accuracy = clf.score(X_test,Y_test)
    final_average_metric = (final_recall + final_accuracy + final_f1)/3

    print() #line break
    print ('*** FINAL MODEL SUMMARY ***')
    print('Final Model Accuracy = ', final_accuracy)
    print('Final Model Recall = ', final_recall)
    print('Final model F1 = ', final_f1)
    print('Final Model Average metric = ', final_average_metric)


    return clf


def ImageAnalysis(img_name,clf):

    # set up empty lists to append into
    lyr1 = []
    lyr2 = []
    lyr3 = []
    lyr4 = []
    lyr5 = []
    predicted = []
    test_array = []
    arrays = []
    albedo_array = []
    
    # use gdal to open image and assign each layer to a separate numpy array
    ds = gdal.Open(img_name, gdal.GA_ReadOnly)
    for i in range(1, ds.RasterCount+1):
        arrays.append(ds.GetRasterBand(i).ReadAsArray())
    
    # get the length and width of the image from numpy.shape
    lenx, leny = np.shape(arrays[0])
    
    # Loop through each pixel and append the pixel value from each layer to a 1D list
    for i in range(0,lenx,1):
        for j in range(0,leny,1):
            lyr1.append(arrays[0][i,j])
            lyr2.append(arrays[1][i,j])
            lyr3.append(arrays[2][i,j])
            lyr4.append(arrays[3][i,j])
            lyr5.append(arrays[4][i,j])        
    
    # create an array of arrays. 1st order array has an array per pixel. The sub-arrays
    # contain reflectance value for each band - result is 5 reflectance values per pixel
            
    for i in range(0,len(lyr1),1):
        test_array.append([lyr1[i], lyr2[i], lyr3[i], lyr4[i], lyr5[i] ])
        albedo_array.append(0.726*(lyr2[i]) - 0.322*(lyr2[i])**2 - 0.015*(lyr5[i]) + 0.581*(lyr5[i]))
        
    # apply ML algorithm to 5-value array for each pixel - predict surface type
        
    predicted = clf.predict(test_array)
    
    # convert surface class (string) to a numeric value for plotting
    predicted[predicted == 'UNKNOWN'] = float(0)
    predicted[predicted == 'WAT'] = float(1)
    predicted[predicted == 'CC'] = float(2)
    predicted[predicted == 'CI'] = float(3)
    predicted[predicted == 'LA'] = float(4)
    predicted[predicted == 'HA'] = float(5)
    
    # ensure array data type is float (required for imshow)
    predicted = predicted.astype(float)
    # reshape 1D array back into original image dimensions
    predicted = np.reshape(predicted,[lenx,leny])
    albedo_array = np.reshape(albedo_array,[lenx,leny])
    
    #plot classified surface
    plt.figure(figsize = (30,30)),plt.imshow(predicted),plt.colorbar()
    plt.figure(figsize = (30,30)),plt.imshow(albedo_array),plt.colorbar()
    
    # Calculate coverage stats
    numHA = (predicted==5).sum()
    numLA = (predicted==4).sum()
    numCI = (predicted==3).sum()
    numCC = (predicted==2).sum()
    numWAT = (predicted==1).sum()
    numUNKNOWN = (predicted==0).sum()
    noUNKNOWNS = (predicted !=0).sum()
    
    tot_alg_coverage = (numHA+numLA)/noUNKNOWNS *100
    HA_coverage = (numHA)/noUNKNOWNS * 100
    LA_coverage = (numLA)/noUNKNOWNS * 100
    CI_coverage = (numCI)/noUNKNOWNS * 100
    CC_coverage = (numCC)/noUNKNOWNS * 100
    WAT_coverage = (numWAT)/noUNKNOWNS * 100
    
    # Print coverage summary
    
    print('**** SUMMARY ****')
    print('% algal coverage (Hbio + Lbio) = ',np.round(tot_alg_coverage,2))
    print('% Hbio coverage = ',np.round(HA_coverage,2))
    print('% Lbio coverage = ',np.round(LA_coverage,2))
    print('% cryoconite coverage = ',np.round(CC_coverage,2))
    print('% clean ice coverage = ',np.round(CI_coverage,2))
    print('% water coverage = ',np.round(WAT_coverage,2))

    return predicted, albedo_array, HA_coverage, LA_coverage, CI_coverage, CC_coverage, WAT_coverage


def albedo_report(predicted,albedo_array):

    alb_WAT = []
    alb_CC = []
    alb_CI = []
    alb_LA = []
    alb_HA = []
    
    predicted = predicted.ravel()
    albedo_array = albedo_array.ravel()
    
    idx_WAT = np.where(predicted ==1)[0]
    idx_CC = np.where(predicted ==2)[0]
    idx_CI = np.where(predicted ==3)[0]
    idx_LA = np.where(predicted ==4)[0]
    idx_HA = np.where(predicted ==5)[0]
    
    for i in idx_WAT:
        alb_WAT.append(albedo_array[i])
    for i in idx_CC:
        alb_CC.append(albedo_array[i])
    for i in idx_CI:
        alb_CI.append(albedo_array[i])
    for i in idx_LA:
        alb_LA.append(albedo_array[i])
    for i in idx_HA:
        alb_HA.append(albedo_array[i])
    
    
    # Calculate summary stats
    mean_CC = np.mean(alb_CC)
    std_CC = np.mean(alb_CC)
    max_CC = np.max(alb_CC)
    min_CC = np.min(alb_CC)

    mean_CI = np.mean(alb_CI)
    std_CI = np.mean(alb_CI)
    max_CI = np.max(alb_CI)
    min_CI = np.min(alb_CI)
    
    mean_LA = np.mean(alb_LA)
    std_LA = np.mean(alb_LA)
    max_LA = np.max(alb_LA)
    min_LA = np.min(alb_LA)

    mean_HA = np.mean(alb_HA)
    std_HA = np.mean(alb_HA)
    max_HA = np.max(alb_HA)
    min_HA = np.min(alb_HA)

    mean_WAT = np.mean(alb_WAT)
    std_WAT = np.mean(alb_WAT)
    max_WAT = np.max(alb_WAT)
    min_WAT = np.min(alb_WAT)
        
    ## FIND IDX WHERE CLASS = Hbio..
    ## BIN ALBEDOS FROM SAME IDXs
    print('mean albedo WAT = ', mean_WAT)
    print('mean albedo CC = ', mean_CC)
    print('mean albedo CI = ', mean_CI)
    print('mean albedo LA = ', mean_LA)
    print('mean albedo HA = ', mean_HA)

    albedo_DF = pd.DataFrame(columns=['albedo','class'])
    albedo_DF['class'] = predicted
    albedo_DF['albedo'] = albedo_array
    albedo_DF.to_csv('UAV_albedo_dataset.csv')

    return alb_WAT, alb_CC, alb_CI, alb_LA, alb_HA, mean_CC,std_CC,max_CC,min_CC,mean_CI,std_CI,max_CI,min_CI,mean_LA,mean_HA,std_HA,max_HA,min_HA,mean_WAT,std_WAT,max_WAT,min_WAT

################################################################################
################################################################################



############### RUN ENTIRE SEQUENCE ###################

# create dataset
X,XX,YY = create_dataset(HCRF_file)
#optimise and train model
clf = optimise_train_model(X,XX,YY, error_selector = 'F1')
# apply model to Sentinel2 image
# predicted, albedo_array, HA_coverage, LA_coverage, CI_coverage, CC_coverage, WAT_coverage = ImageAnalysis(img_name,clf)
#obtain albedo summary stats
#alb_WAT, alb_CC, alb_CI, alb_LA, alb_HA, mean_CC,std_CC,max_CC,min_CC,mean_CI,std_CI,max_CI,min_CI,mean_LA,mean_HA,std_HA,max_HA,min_HA,mean_WAT,std_WAT,max_WAT,min_WAT = albedo_report(predicted,albedo_array)
