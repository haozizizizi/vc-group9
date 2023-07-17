from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def class_SVC(X,Y):
    classifier = SVC()
    classifier.fit(X,Y)

    return classifier

def class_LR(X,Y):
    '''
    LogisticRegression分类器
    :param X:
    :param Y:
    :return:
    '''
    classifier = LogisticRegression(max_iter=500)  # 初始化分类器
    classifier.fit(X, Y)  # 使用训练数据训练分类器

    return classifier

def class_DT(X,Y):
    '''
    decision tree决策树
    :param X:
    :param Y:
    :return:
    '''
    classifier = DecisionTreeClassifier()  # 初始化分类器
    classifier.fit(X,Y)  # 使用训练数据训练分类器

    return classifier

def class_KNN(X,Y):
    '''
    KNeighborsClassifier K临近
    :param X:
    :param Y:
    :return:
    '''
    classifier = KNeighborsClassifier()  # 初始化分类器
    classifier.fit(X,Y)  # 使用训练数据训练分类器

    return classifier

def class_RF(X,Y):
    '''
    RandomForest
    :param X:
    :param Y:
    :return:
    '''
    classifier = RandomForestClassifier()
    classifier.fit(X,Y)

    return classifier

def class_MLP(X,Y):
    '''
    M
    :param X:
    :param Y:
    :return:
    '''
    classifier = MLPClassifier()
    classifier.fit(X,Y)

    return classifier
