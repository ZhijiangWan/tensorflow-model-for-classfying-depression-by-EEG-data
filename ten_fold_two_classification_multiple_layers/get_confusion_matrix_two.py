import tensorflow as tf
import random
import numpy as np
def get_confusion_matrix_two_classification(result_label, result_predict):
    '''
        prediction
            0  1
real     0  A1 A2
         1  B1 B2



    '''
    A1=0
    A2 = 0

    B1=0
    B2 = 0

    length=len(result_label)
    for i in range(length):

        temp_real=result_label[i]
        temp_pre=result_predict[i]
        if temp_real==0 and temp_pre==0:
            A1=A1 + 1
        elif temp_real==0 and temp_pre==1:
            A2 = A2 + 1
        elif temp_real == 1 and temp_pre == 0:
            B1 = B1 + 1
        elif temp_real == 1 and temp_pre == 1:
            B2 = B2 + 1
        else:
            debug=0




    return A1,A2,B1,B2
