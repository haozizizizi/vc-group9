import glob
import os
import cv2
import pickle # 保存模型
import csv
# from sklearn.model_selection import train_test_split


import feature_extracter
import classifier_choose


def LOAD_trian(path, mode: int):  # mode : 1-彩色 0-灰色 -1 附加alpha通道

    X = []  # img数据集
    Y = []  # label数据集

    for file_name in os.listdir(path):
        # print(file_name) 是乱序打开文件夹，不能使用递增写入label
        img_path = os.path.join(path,file_name)
        print(img_path)
        img_path = img_path + '\\'
        for i in glob.glob(img_path + '*.png'):
            # e.g.: 'C:\Users\fkbgr\Desktop\VC\Finall\D2cby\Dataset_2\Train\0\00000_00000_00000.png'
            # e.g.: 'C:\Users\fkbgr\Desktop\VC\Finall\D2cby\Dataset_2\Train\10'
            # e.g.:'
            '''将label加载进入Y'''
            AfterTrainPath= i.split("Train")[1]
            label = AfterTrainPath.split("\\")[2][3:5]
            if label[0] == '0':
                Y.append(label[1])
            else:
                Y.append(label)
            print(label)

            '''将img加载进入X'''
            img = cv2.imread(i, mode)  # 没有设置转换灰色
            X.append(img)
        print(label)
    # print(len(X))
    return X, Y


def LOAD_test(test_path,csv_path,mode:int):
    '''
    注意此处Y并不是label，而是img的id，id作为img主键关联到csv，使用csv读取label
    :param path:
    :param mode:
    :return:
    '''
    X = [] # img数据集
    # ID = [] # label数据集
    Y = [] # img标签集

    # csv_path = os.path.join(csvpath + 'Test.csv')

    with open(csv_path,'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        headers = next(csv_reader)
        for row in csv_reader:
            # print('标签：',row[6],'ID:',row[7][5:])
            i = os.path.join(test_path + row[7][5:])
            '''将label加载进入Y'''
            Y.append(row[6])
            '''将img加载进入X'''
            X.append(cv2.imread(i,mode))
        # print(Y)


    # for i in glob.glob(path + '*.png'):
    #     # e.g. 'C:\Users\fkbgr\Desktop\VC\Finall\D2cby\Dataset_2\Test\00001.png'
    #     '''将label加载进入Y'''
    #     AfterDataset_2_Path = i.split("Dataset_2")[1][1:]
    #     # label = AfterTrainPath.split("\\")[1][3:5]
    #     print(AfterDataset_2_Path)
    #     ID.append(AfterDataset_2_Path)
    #     '''将img加载进入X'''
    #     img = cv2.imread(i,mode) # 没有设置转换灰色
    #     X.append(img)

    # print(len(X))

    return X, Y


def PREPROCESS(X):
    X_preprocess = []

    for x in X:
        X_temp = cv2.resize(x, (48, 48))
        X_preprocess.append(X_temp)
    return X_preprocess


# def trainANDtest(X_feature, Y):
#     X_train, X_test, Y_train, Y_test = train_test_split(X_feature, Y, test_size=0.2, shuffle=True)
#     return X_train, X_test, Y_train, Y_test


def model_save(classifier,model_path):
    with open(model_path,'wb') as f:
        pickle.dump(classifier,f) # 将训练好的模型classifier保存在变量f中，且保存到本地

def model_load(model_path):
    with open(model_path,'rb') as f:
        classifier = pickle.load(f) #将模型存储在变量classifier上
    return classifier

def main_train():
    path =  'C:\\Users\\fkbgr\\Desktop\\VC\\Finall\\Dataset_2\\Train'
    X, Y = LOAD_trian(path= path, mode = 0)  # mode : 1-彩色 0-灰色 -1 附加alpha通道
    print('-------------LOAD finish-------------\n')
    '''预处理'''
    X_prepro = PREPROCESS(X)  # 将大小统一为48*48
    # randomshow_gray(X_prepro) # 随机展示一张预处理图片
    print('-------------size:48*48 finish-------------\n')
    '''特征提取'''
    X_feature = feature_extracter.feature_HOG(X_prepro)
    print('-------------feature extract finish-------------\n')
    # '''8：2比例分trian test'''
    # X_train, X_test, Y_train, Y_test = trainANDtest(X_feature, Y)
    # '''分类'''
    classifier = classifier_choose.class_SVC(X_feature, Y)
    print('-------------classifier finish-------------\n')

    return classifier



def main_test(classifier,csv_path,test_path,mode:int):
    print('-------------test start-------------\n')
    path = 'C:\\Users\\fkbgr\\Desktop\\VC\\Finall\\Dataset_2\\Test\\'
    X_test, Y_test = LOAD_test(csv_path=csv_path ,test_path= test_path,mode= mode)  # mode : 1-彩色 0-灰色 -1 附加alpha通道
    '''预处理'''
    X_test_prepro = PREPROCESS(X_test)  # 将大小统一为48*48
    # randomshow_gray(X_prepro) # 随机展示一张预处理图片
    '''特征提取'''
    X_test_feature = feature_extracter.feature_HOG(X_test_prepro)
    # '''准确率'''
    ACC = classifier.score(X_test_feature, Y_test)
    print('-------------test finish-------------\n')
    print('accuracy = ',ACC)

# def main_predict(path , classifier):
#     X_pred , Y_pred =

if __name__ == '__main__':
    model_path = 'C:\\Users\\fkbgr\\Desktop\\VC\\Finall\\D2cby\\clsmodel\\gray_hog_svc.pickle'
    # classifier = main_train()
    # model_save(classifier,model_path)

    csv_path = 'C:\\Users\\fkbgr\\Desktop\\VC\\Finall\\Dataset_2\\Test.csv'
    test_path = 'C:\\Users\\fkbgr\\Desktop\\VC\\Finall\\Dataset_2\\Test\\'
    # LOAD_test(csv_path=csv_path ,test_path= test_path,mode= 0)

    classifier_load = model_load(model_path)
    main_test(classifier_load,csv_path=csv_path ,test_path= test_path,mode= 0)



    # path =  'C:\\Users\\fkbgr\\Desktop\\VC\\Finall\\D2cby\\Dataset_2\\Train'
    #     C:\Users\fkbgr\Desktop\VC\Finall\D2cby\Dataset_2\Train
    # LOAD(path,0)
