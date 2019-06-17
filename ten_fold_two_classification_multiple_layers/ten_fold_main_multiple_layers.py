import os.path
import re
import mat4py
import scipy.io as scio
import numpy as np

from MyCNN_Core_vn3_multiple_layers import MyCNN_vn3_multiple_layers

def readFile(filenames):
    datamatrix = scio.loadmat(filenames)
    # print(datamatrix['final_Matrix'])
    finalmatrix=datamatrix['final_Matrix']
    return finalmatrix

def train_test_split_two(posmatrix,negmatrix, posmatrixlabel,negmatrixlabel,foldrate,iteration):

    if iteration==0:
        posbegin=0
        negbegin=0
    else:
        posbegin=int(len(posmatrix)*foldrate*iteration)
        negbegin=int(len(negmatrix)*foldrate*iteration)

    posend=int(len(posmatrix)*foldrate*(iteration+1))
    negend=int(len(negmatrix)*foldrate*(iteration+1))
    count_pos=len(posmatrix)-(posend-posbegin)
    if posend>len(posmatrix):
        posend=len(posmatrix)
    if negend>len(negmatrix):
        negend=len(negmatrix)

    X_test =np.vstack([posmatrix[posbegin:posend], negmatrix[negbegin:negend]])
    y_test=np.hstack([posmatrixlabel[posbegin:posend], negmatrixlabel[negbegin:negend]])

    X_train= np.vstack([posmatrix[0:posbegin], posmatrix[posend:len(posmatrix)],negmatrix[0:negbegin], negmatrix[negend:len(negmatrix)]])
    y_train = np.hstack([posmatrixlabel[0:posbegin], posmatrixlabel[posend:len(posmatrixlabel)],negmatrixlabel[0:negbegin], negmatrixlabel[negend:len(negmatrixlabel)]])
    a=0

    return X_train, X_test, y_train, y_test,count_pos

def train_test_split_three(normalmatrix, medicatedmatrix,unmedicatedmatrix,normalmatrixlabel,medicatedmatrixlabel, unmedicatedmatrixlabel,foldrate,iteration):

    if iteration==0:
        normalbegin=0
        medicatedbegin=0
        unmedicatedbegin=0
    else:
        normalbegin=int(len(normalmatrix)*foldrate*iteration)
        medicatedbegin=int(len(medicatedmatrix)*foldrate*iteration)
        unmedicatedbegin = int(len(unmedicatedmatrix) * foldrate * iteration)

    normalend=int(len(normalmatrix)*foldrate*(iteration+1))
    medicatedend=int(len(medicatedmatrix)*foldrate*(iteration+1))
    unmedicatedend = int(len(unmedicatedmatrix) * foldrate * (iteration + 1))

    if normalend>len(normalmatrix):
        normalend=len(normalmatrix)
    if medicatedend>len(medicatedmatrix):
        medicatedend=len(medicatedmatrix)
    if unmedicatedend>len(unmedicatedmatrix):
        unmedicatedend=len(unmedicatedmatrix)

    count_normal=len(normalmatrix)-(normalend-normalbegin)
    count_medicated = len(medicatedmatrix) - (medicatedend - medicatedbegin)
    count_unmedicated = len(unmedicatedmatrix) - (unmedicatedend - unmedicatedbegin)

    X_test =np.vstack([normalmatrix[normalbegin:normalend], medicatedmatrix[medicatedbegin:medicatedend], unmedicatedmatrix[unmedicatedbegin:unmedicatedend]])
    y_test=np.hstack([normalmatrixlabel[normalbegin:normalend], medicatedmatrixlabel[medicatedbegin:medicatedend], unmedicatedmatrixlabel[unmedicatedbegin:unmedicatedend]])

    X_train= np.vstack([normalmatrix[0:normalbegin], normalmatrix[normalend:len(normalmatrix)],
                        medicatedmatrix[0:medicatedbegin], medicatedmatrix[medicatedend:len(medicatedmatrix)],
                        unmedicatedmatrix[0:unmedicatedbegin], unmedicatedmatrix[unmedicatedend:len(unmedicatedmatrix)]])
    y_train = np.hstack([normalmatrixlabel[0:normalbegin], normalmatrixlabel[normalend:len(normalmatrixlabel)],
                         medicatedmatrixlabel[0:medicatedbegin], medicatedmatrixlabel[medicatedend:len(medicatedmatrixlabel)],
                         unmedicatedmatrixlabel[0:unmedicatedbegin],unmedicatedmatrixlabel[unmedicatedend:len(unmedicatedmatrixlabel)]])
    a=0

    return X_train, X_test, y_train, y_test,count_normal,count_medicated,count_unmedicated

if __name__ == "__main__":
# this code is for parallel running
    #for two classifications
    posmatrix = np.load('C:/python/codes/ten_fold_two_classification_multiple_layers/ten fold validation for two classifications/posdata.npy')
    posmatrixlabel = np.load('C:/python/codes/ten_fold_two_classification_multiple_layers/ten fold validation for two classifications/posdatalabel.npy')
    negmatrix = np.load('C:/python/codes/ten_fold_two_classification_multiple_layers/ten fold validation for two classifications/negdata.npy')
    negmatrixlabel = np.load('C:/python/codes/ten_fold_two_classification_multiple_layers/ten fold validation for two classifications/negdatalabel.npy')


    count_pos = 3300
    fold=10
    foldrate=1/10
    final_result = []
    final_result_p = []
    final_result_r = []
    final_result_AUC = []
    final_result_confusion=[]
    for i in range(fold):

        #for two classifications
        X_train, X_test, y_train, y_test,count_pos = train_test_split_two(posmatrix,negmatrix, posmatrixlabel,negmatrixlabel,foldrate,i)
        resultACC, confusion_result = MyCNN_vn3_multiple_layers(X_train, y_train, X_test,y_test, count_pos)
        final_result.append(resultACC)
        final_result_confusion.append(confusion_result)
        #final_result_p.append(resultPrecision)
        #final_result_r.append(final_result_r)
        print('accuracy:', resultACC[len(resultACC) - 1])
        np.save('final_result_two_classifications_multiple_layers.npy', final_result)
        np.save('final_result_confusion_two_classifications_multiple_layers.npy', final_result_confusion)


    #np.save('final_result_p_two_classifications_multiple_layers.npy', final_result_p)
    #np.save('final_result_r_two_classifications_multiple_layers.npy', final_result_r)
    a=0