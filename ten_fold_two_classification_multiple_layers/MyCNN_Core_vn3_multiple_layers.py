import tensorflow as tf
import random
import numpy as np
from tensorflow.python import debug as tf_debug
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as metric
from get_confusion_matrix_two import get_confusion_matrix_two_classification
def compute_accuracy(xs, ys, X, y, keep_prob, sess, prediction):
    # y_pre=debug_sess.run(prediction, feed_dict={xs: X, keep_prob: 1})  # dropout值定义为0.5
    y_pre = sess.run(prediction, feed_dict={xs: X, keep_prob: 1.0})  # 预测，这里的keep_prob是dropout时用的，防止过拟合
    #print('predication results:',y_pre)
    #print('labels:', y)
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y,
                                                                 1))  # tf.argmax 给出某个tensor对象在某一维上的其数据最大值所在的索引值,即为对应的数字，tf.equal 来检测我们的预测是否真实标签匹配
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 平均值即为准确度
    result = sess.run(accuracy)
    real_label = tf.argmax(y, 1)
    result_label = sess.run(real_label)
    result_predict = sess.run(tf.argmax(y_pre, 1))
    return result,result_label,result_predict


'''权重初始化函数'''


def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)  # 使用truncated_normal进行初始化
    return tf.Variable(inital)


'''偏置初始化函数'''


def bias_variable(shape):
    inital = tf.constant(0.1, shape=shape)  # 偏置定义为常量
    return tf.Variable(inital)


'''卷积函数'''


def conv2d(x, W,a,b):  # x是图片的所有参数，W是此卷积层的权重
    return tf.nn.conv2d(x, W, strides=[1, a, b, 1],
                        padding='SAME')  # strides[0]和strides[3]的两个1是默认值，中间两个1代表padding时在x方向运动1步，y方向运动1步


'''池化函数1'''
def max_pool_6x6(x):
    return tf.nn.max_pool(x, ksize=[1, 6,6, 1],
                          strides=[1, 6, 6, 1],
                          padding='SAME')  # 池化的核函数大小为1x8，因此ksize=[1,1,8,1]，步长为8，因此strides=[1,1,8,1]

def max_avg_1x16(x):
    return tf.nn.avg_pool(x, ksize=[1, 1, 16, 1],
                          strides=[1, 1, 16, 1],
                          padding='SAME')  # 池化的核函数大小为1x6，因此ksize=[1,1,6,1]，步长为6，因此strides=[1,1,6,1]

def max_avg_1x32(x):
    return tf.nn.avg_pool(x, ksize=[1, 1, 32, 1],
                          strides=[1, 1, 32, 1],
                          padding='SAME')  # 池化的核函数大小为1x6，因此ksize=[1,1,6,1]，步长为6，因此strides=[1,1,6,1]
'''池化函数2'''


def max_pool_1x16(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 16, 1],
                          strides=[1, 1, 16, 1],
                          padding='SAME')  # 池化的核函数大小为1x8，因此ksize=[1,1,8,1]，步长为8，因此strides=[1,1,8,1]

def max_pool_1x4(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 4, 1],
                          strides=[1, 1, 4, 1],
                          padding='SAME')  # 池化的核函数大小为1x8，因此ksize=[1,1,8,1]，步长为8，因此strides=[1,1,8,1]

def max_pool_1x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],
                          strides=[1, 1, 2, 1],
                          padding='SAME')  # 池化的核函数大小为1x8，因此ksize=[1,1,8,1]，步长为8，因此strides=[1,1,8,1]

def random_select_vn2(matrix, matrixlabel, stride, size,count_pos):
    posselecttrainmatrix = []
    posselecttrainmatrixlabel = []

    negselecttrainmatrix = []
    negselecttrainmatrixlabel = []
    for i in range(0, int(size/2)):
        posuplimit = count_pos / stride
        uplimit = len(matrix) / stride
        posrannum = random.randint(0, posuplimit-1)
        negrannum= random.randint(posuplimit, uplimit-1)
        tempposmatrix = matrix[(posrannum * stride):(posrannum * stride + stride), :]
        tempnegmatrix = matrix[(negrannum * stride):(negrannum * stride + stride), :]
        if matrixlabel[posrannum * stride] == 1:
            tempposmatrixlabel = [0, 1]
        else:
            tempposmatrixlabel = [1, 0]

        if matrixlabel[negrannum * stride] == 0:
            tempnegmatrixlabel = [1, 0]
        else:
            tempnegmatrixlabel = [0, 1]
        # tempselectmatrixlabel=matrixlabel[rannum*stride]
        posserialmatrix = []
        negserialmatrix = []
        for j in range(0, stride):
            if len(posserialmatrix) == 0:
                posserialmatrix = tempposmatrix[j, :]
            else:
                # print(len(serialmatrix))
                posserialmatrix = np.hstack([posserialmatrix, tempposmatrix[j, :]])

            if len(negserialmatrix) == 0:
                negserialmatrix = tempnegmatrix[j, :]
            else:
                # print(len(serialmatrix))
                negserialmatrix = np.hstack([negserialmatrix, tempnegmatrix[j, :]])

        if len(posselecttrainmatrix) == 0:
            posselecttrainmatrix = posserialmatrix
            posselecttrainmatrixlabel = tempposmatrixlabel
        else:
            posselecttrainmatrix = np.vstack([posselecttrainmatrix, posserialmatrix])
            posselecttrainmatrixlabel = np.vstack([posselecttrainmatrixlabel, tempposmatrixlabel])

        if len(negselecttrainmatrix) == 0:
            negselecttrainmatrix = negserialmatrix
            negselecttrainmatrixlabel = tempnegmatrixlabel
        else:
            negselecttrainmatrix = np.vstack([negselecttrainmatrix, negserialmatrix])
            negselecttrainmatrixlabel = np.vstack([negselecttrainmatrixlabel, tempnegmatrixlabel])

    selecttrainmatrix=np.vstack([posselecttrainmatrix,negselecttrainmatrix])
    selecttrainmatrixlabel=np.vstack([posselecttrainmatrixlabel,negselecttrainmatrixlabel])
    return selecttrainmatrix, selecttrainmatrixlabel

def random_select(matrix, matrixlabel, stride, size):
    selecttrainmatrix = []
    selecttrainmatrixlabel = []
    for i in range(0, size):
        uplimit = len(matrix) / stride
        rannum = random.randint(0, uplimit - 1)
        tempselectmatrix = matrix[(rannum * stride):(rannum * stride + stride), :]
        if matrixlabel[rannum * stride] == 1:
            tempselectmatrixlabel = [0, 1]
        else:
            tempselectmatrixlabel = [0, 0]
        # tempselectmatrixlabel=matrixlabel[rannum*stride]
        serialmatrix = []
        for j in range(0, stride):
            if len(serialmatrix) == 0:
                serialmatrix = tempselectmatrix[j, :]
            else:
                # print(len(serialmatrix))
                serialmatrix = np.hstack([serialmatrix, tempselectmatrix[j, :]])

        if len(selecttrainmatrix) == 0:
            selecttrainmatrix = serialmatrix
            selecttrainmatrixlabel = tempselectmatrixlabel
        else:
            selecttrainmatrix = np.vstack([selecttrainmatrix, serialmatrix])
            selecttrainmatrixlabel = np.vstack([selecttrainmatrixlabel, tempselectmatrixlabel])

    return selecttrainmatrix, selecttrainmatrixlabel


def format_testmatrix(testmatrix, testmatrixlabel):
    format_testmatrixlabel = []
    format_testmatrix = []
    for j in range(0, len(testmatrixlabel), 6):
        temp = testmatrix[j:j + 6, :]
        temptestmatrix = np.hstack([temp[0, :], temp[1, :], temp[2, :], temp[3, :], temp[4, :], temp[5, :]])
        if len(format_testmatrix) == 0:
            format_testmatrix = temptestmatrix
        else:
            format_testmatrix = np.vstack([format_testmatrix, temptestmatrix])
    for i in range(0, len(testmatrixlabel), 6):
        if testmatrixlabel[i] == 1:
            temp = [0, 1]
        else:
            temp = [1, 0]
        if len(format_testmatrixlabel) == 0:
            format_testmatrixlabel = temp
        else:
            format_testmatrixlabel = np.vstack([format_testmatrixlabel, temp])

    return format_testmatrix, format_testmatrixlabel


def MyCNN_vn3(trainmatrix, trainmatrixlabel, testmatrix, testmatrixlabel,count_pos):
    xs = tf.placeholder(tf.float32, [None, 6 * 3072])  # 输入图片的大小，28x28=784
    ys = tf.placeholder(tf.float32, [None, 2])  # 输出0-9共10个数字
    global_step=tf.Variable(0)
    keep_prob = tf.placeholder(tf.float32)  # 用于接收dropout操作的值，dropout为了防止过拟合
    x_image = tf.reshape(xs, [-1, 6, 3072,
                              1])  # -1代表先不考虑输入的图片例子多少这个维度，后面的1是channel的数量，因为我们输入的图片是黑白的，因此channel是1，例如如果是RGB图像，那么channel就是3
    '''第一层卷积，池化'''
    W_conv1 = weight_variable([6, 1, 1, 6])  # 卷积核定义为5x5,1是输入的通道数目，32是输出的通道数目
    b_conv1 = bias_variable([6])  # 每个输出通道对应一个偏置
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1,1,1) + b_conv1)  # 卷积运算，并使用ReLu激活函数激活
    h_pool1 = max_pool_6x6(h_conv1)  # pooling操作

    '''第二层卷积，池化'''
    W_conv2 = weight_variable([1, 2, 6, 12])  # 卷积核还是5x5,32个输入通道，64个输出通道
    b_conv2 = bias_variable([12])  # 与输出通道一致
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2,1,1) + b_conv2)
    h_pool2 = max_avg_1x32(h_conv2)

    '''全连接层'''
    # h_pool2_flat = tf.reshape(h_pool3, [-1,  15 * 64])  # 将最后操作的数据展开
    # W_fc1 = weight_variable([15 * 64, 1024])  # 下面就是定义一般神经网络的操作了，继续扩大为1024
    h_pool2_flat = tf.reshape(h_pool2, [-1, 16 * 12])  # 将最后操作的数据展开
    W_fc1 = weight_variable([16 * 12, 32])  # 下面就是定义一般神经网络的操作了，继续扩大为1024
    b_fc1 = bias_variable([32])  # 对应的偏置
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 运算、激活（这里不是卷积运算了，就是对应相乘）

    '''全连接层'''
    # h_pool2_flat = tf.reshape(h_pool3, [-1,  15 * 64])  # 将最后操作的数据展开
    # W_fc1 = weight_variable([15 * 64, 1024])  # 下面就是定义一般神经网络的操作了，继续扩大为1024
    h_pool3_flat = tf.reshape(h_fc1, [-1, 32])  # 将最后操作的数据展开
    W_fc2 = weight_variable([32,16])  # 下面就是定义一般神经网络的操作了，继续扩大为1024
    b_fc2 = bias_variable([16])  # 对应的偏置
    h_fc2 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc2) + b_fc2)  # 运算、激活（这里不是卷积运算了，就是对应相乘）

    '''dropout'''
    h_fc1_drop = tf.nn.dropout(h_fc2, keep_prob)  # dropout操作
    '''最后一层全连接'''
    W_fc3 = weight_variable([16, 2])  # 最后一层权重初始化
    b_fc3 = bias_variable([2])  # 对应偏置

    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)  # 使用softmax分类器
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # 交叉熵损失函数来定义cost function
    # cross_entropy =tf.losses.sparse_softmax_cross_entropy(labels=ys, logits=prediction)

    #cross_entropy = tf.reduce_sum(tf.square(tf.subtract(ys, prediction)))
    learning_rate=tf.train.exponential_decay(0.1,global_step,1000,0.96,staircase=True)
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 调用梯度下降
    #train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)

    '''下面就是tf的一般操作，定义Session，初始化所有变量，placeholder传入值训练'''
    with tf.Session() as sess:
        # sess = tf.Session()
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.initialize_all_variables())
        # debug_sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
        # print(sess.run(h_conv1))
        step = len(trainmatrix) / 6
        resultACC = []
        resultPrecision = []
        resultRecall = []
        '''this function is for parallel matrix for training the model'''
        formated_testmatrix, formated_testmatrixlabel = format_testmatrix(testmatrix, testmatrixlabel)

        for i in range(1000):
            selecttrainmatrix, selecttrainmatrixlabel = random_select_vn2(trainmatrix, trainmatrixlabel, 6, 300,count_pos)

            # (selecttrainmatrix==formated_testmatrix).any()
            # a=(trainmatrix[0:204,0:100])
            # b=(testmatrix[0:6,0:100])
            ss = StandardScaler()
            std_selecttrainmatrix = ss.fit_transform(selecttrainmatrix.transpose())
            final_std_selecttrainmatrix = std_selecttrainmatrix.transpose()

            std_formated_testmatrix = ss.fit_transform(formated_testmatrix.transpose())
            final_std_formated_testmatrix = std_formated_testmatrix.transpose()
            sess.run(train_step, feed_dict={xs: final_std_selecttrainmatrix, ys: selecttrainmatrixlabel,
                                            keep_prob: 0.5})  # dropout值定义为0.5
            acc = 0
            #if i % 10 == 0:
            acc, result_label, result_predict = compute_accuracy(xs, ys, final_std_formated_testmatrix, formated_testmatrixlabel, keep_prob, sess,
                                       prediction)
            precision = metric.precision_score(result_label, result_predict, pos_label=1)
            recall = metric.recall_score(result_label, result_predict, pos_label=1)
            print(acc)
            # print(final_std_selecttrainmatrix-final_std_selecttrainmatrix)
            '''
            if acc > 0.6:
                break
            else:
                continue
            '''
            # a = sess.graph.get_tensor_by_name(' w:h_conv1')
            # print(sess.run(h_conv1))
            resultACC.append(acc)
            resultPrecision.append(precision)
            resultRecall.append(recall)
            # resultAUC.append(auc)
        return resultACC, resultPrecision, resultRecall


'''
        if i % 10 == 0:
            print(compute_accuracy(xs, ys, formated_testmatrix, formated_testmatrixlabel, keep_prob, sess,
                                   prediction))  # 每50次输出一下准确度
'''


def MyCNN_vn3_multiple_layers(trainmatrix, trainmatrixlabel, testmatrix, testmatrixlabel,count_pos):
    xs = tf.placeholder(tf.float32, [None, 6 * 3072])  # 输入图片的大小，28x28=784
    ys = tf.placeholder(tf.float32, [None, 2])  # 输出0-9共10个数字
    keep_prob = tf.placeholder(tf.float32)  # 用于接收dropout操作的值，dropout为了防止过拟合
    global_step = tf.Variable(0)
    x_image = tf.reshape(xs, [-1, 6, 3072,
                              1])  # -1代表先不考虑输入的图片例子多少这个维度，后面的1是channel的数量，因为我们输入的图片是黑白的，因此channel是1，例如如果是RGB图像，那么channel就是3
    '''第一层卷积，池化'''
    W_conv1 = weight_variable([6, 8, 1, 6])  # 卷积核定义为5x5,1是输入的通道数目，32是输出的通道数目
    b_conv1 = bias_variable([6])  # 每个输出通道对应一个偏置
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 1, 1) + b_conv1)  # 卷积运算，并使用ReLu激活函数激活
    h_pool1 = max_pool_1x2(h_conv1)  # pooling操作

    '''第二层卷积，池化'''
    W_conv2 = weight_variable([6, 8, 6, 6])  # 卷积核还是5x5,32个输入通道，64个输出通道
    b_conv2 = bias_variable([6])  # 与输出通道一致
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 1, 1) + b_conv2)
    h_pool2 = max_pool_1x2(h_conv2)

    '''第三层卷积，池化'''
    W_conv3 = weight_variable([6, 8, 6, 6])  # 卷积核还是5x5,32个输入通道，64个输出通道
    b_conv3 = bias_variable([6])  # 与输出通道一致
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, 1, 1) + b_conv3)
    h_pool3 = max_pool_1x2(h_conv3)

    '''第四层卷积，池化'''
    W_conv4 = weight_variable([6, 8, 6, 12])  # 卷积核还是5x5,32个输入通道，64个输出通道
    b_conv4 = bias_variable([12])  # 与输出通道一致
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4, 1, 1) + b_conv4)
    h_pool4 = max_pool_1x2(h_conv4)

    '''第五层卷积，池化'''
    W_conv5 = weight_variable([6, 8, 12, 12])  # 卷积核还是5x5,32个输入通道，64个输出通道
    b_conv5 = bias_variable([12])  # 与输出通道一致
    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5, 1, 1) + b_conv5)
    h_pool5 = max_pool_1x2(h_conv5)

    '''第六层卷积，池化'''
    W_conv6 = weight_variable([6, 8, 12, 12])  # 卷积核还是5x5,32个输入通道，64个输出通道
    b_conv6 = bias_variable([12])  # 与输出通道一致
    h_conv6 = tf.nn.relu(conv2d(h_pool5, W_conv6, 1, 1) + b_conv6)
    h_pool6 = max_pool_1x2(h_conv6)

    '''第七层卷积，池化'''
    W_conv7 = weight_variable([6, 8, 12, 12])  # 卷积核还是5x5,32个输入通道，64个输出通道
    b_conv7 = bias_variable([12])  # 与输出通道一致
    h_conv7 = tf.nn.relu(conv2d(h_pool6, W_conv7, 1, 1) + b_conv7)
    h_pool7 = max_pool_1x2(h_conv7)

    '''第八层卷积，池化'''
    W_conv8 = weight_variable([6, 8, 12, 12])  # 卷积核还是5x5,32个输入通道，64个输出通道
    b_conv8 = bias_variable([12])  # 与输出通道一致
    h_conv8 = tf.nn.relu(conv2d(h_pool7, W_conv8, 1, 1) + b_conv8)
    h_pool8 = max_pool_1x2(h_conv8)

    '''全连接层'''
    # h_pool2_flat = tf.reshape(h_pool3, [-1,  15 * 64])  # 将最后操作的数据展开
    # W_fc1 = weight_variable([15 * 64, 1024])  # 下面就是定义一般神经网络的操作了，继续扩大为1024
    h_pool2_flat = tf.reshape(h_pool8, [-1, 6 * 12 * 12])  # 将最后操作的数据展开
    W_fc1 = weight_variable([6 * 12 * 12, 32])  # 下面就是定义一般神经网络的操作了，继续扩大为1024
    b_fc1 = bias_variable([32])  # 对应的偏置
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 运算、激活（这里不是卷积运算了，就是对应相乘）

    '''全连接层'''
    # h_pool2_flat = tf.reshape(h_pool3, [-1,  15 * 64])  # 将最后操作的数据展开
    # W_fc1 = weight_variable([15 * 64, 1024])  # 下面就是定义一般神经网络的操作了，继续扩大为1024
    h_pool3_flat = tf.reshape(h_fc1, [-1, 32])  # 将最后操作的数据展开
    W_fc2 = weight_variable([32, 16])  # 下面就是定义一般神经网络的操作了，继续扩大为1024
    b_fc2 = bias_variable([16])  # 对应的偏置
    h_fc2 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc2) + b_fc2)  # 运算、激活（这里不是卷积运算了，就是对应相乘）

    '''dropout'''
    h_fc1_drop = tf.nn.dropout(h_fc2, keep_prob)  # dropout操作
    '''最后一层全连接'''
    W_fc3 = weight_variable([16, 2])  # 最后一层权重初始化
    b_fc3 = bias_variable([2])  # 对应偏置

    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)  # 使用softmax分类器
    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # 交叉熵损失函数来定义cost function
    # cross_entropy =tf.losses.sparse_softmax_cross_entropy(labels=ys, logits=prediction)

    # cross_entropy = tf.reduce_sum(tf.square(tf.subtract(ys, prediction)))
    # learning_rate = tf.train.exponential_decay(0.1, global_step, 1000, 0.96, staircase=True)

    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 调用梯度下降
    # train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)

    '''下面就是tf的一般操作，定义Session，初始化所有变量，placeholder传入值训练'''
    with tf.Session() as sess:
        # sess = tf.Session()
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.initialize_all_variables())
        # debug_sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
        # print(sess.run(h_conv1))
        step = len(trainmatrix) / 6
        resultACC = []
        resultPrecision = []
        resultRecall = []
        confusion_result=[]
        formated_testmatrix, formated_testmatrixlabel = format_testmatrix(testmatrix, testmatrixlabel)
        for i in range(1000):
            selecttrainmatrix, selecttrainmatrixlabel = random_select_vn2(trainmatrix, trainmatrixlabel, 6, 150,count_pos)

            # (selecttrainmatrix==formated_testmatrix).any()
            # a=(trainmatrix[0:204,0:100])
            # b=(testmatrix[0:6,0:100])

            ss = StandardScaler()
            std_selecttrainmatrix = ss.fit_transform(selecttrainmatrix.transpose())
            final_std_selecttrainmatrix = std_selecttrainmatrix.transpose()

            std_formated_testmatrix = ss.fit_transform(formated_testmatrix.transpose())
            final_std_formated_testmatrix = std_formated_testmatrix.transpose()

            sess.run(train_step, feed_dict={xs: final_std_selecttrainmatrix, ys: selecttrainmatrixlabel,
                                            keep_prob: 0.5})  # dropout值定义为0.5
            acc = 0
            #if i % 10 == 0:
            acc, result_label, result_predict = compute_accuracy(xs, ys, final_std_formated_testmatrix, formated_testmatrixlabel, keep_prob, sess,
                                       prediction)
            A1, A2, B1, B2,  = get_confusion_matrix_two_classification(result_label, result_predict)
            #precision = metric.precision_score(result_label, result_predict, pos_label=1)
            #recall = metric.recall_score(result_label, result_predict, pos_label=1)
            # auc = metric.roc_auc_score(result_label, result_predict,pos_label=1)
            print(acc)
            confusion_result.append([A1, A2, B1, B2])
            # print(final_std_selecttrainmatrix-final_std_selecttrainmatrix)
            '''
            if acc > 0.6:
                break
            else:
                continue
            '''
            # a = sess.graph.get_tensor_by_name(' w:h_conv1')
            # print(sess.run(h_conv1))
            resultACC.append(acc)
            #resultPrecision.append(precision)
            #resultRecall.append(recall)
    return resultACC,confusion_result


'''
        if i % 10 == 0:
            print(compute_accuracy(xs, ys, formated_testmatrix, formated_testmatrixlabel, keep_prob, sess,
                                   prediction))  # 每50次输出一下准确度
'''