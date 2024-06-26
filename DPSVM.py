import pandas as pd
import numpy as np
import cvxopt
cvxopt.solvers.options['show_progress']=False
cvxopt.solvers.options['maxiters']=200
from numpy import linalg
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold


df = pd.read_csv('diabetes.csv')


X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

mean = X_train.mean()
sd = X_train.std()

X_train_normalized = (X_train - mean) / sd
X_test_normalized = (X_test - mean) / sd

class SVM:
    def __init__(self, C, kernel='linear', degree=2, gamma=0.2):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma

    def gen_kernel(self, X1, X2):
        if self.kernel == 'linear':
            return np.dot(X1, X2)
        elif self.kernel == 'poly':
            return (1 + np.dot(X1, X2)) ** self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * (np.linalg.norm(X1-X2) ** 2))

    def compute2(self, X, y):
        # Solving using CVXOPT solver
        self.X = X
        self.y = y
        self.get_samples, self.get_features = X.shape
        self.K = np.zeros((self.get_samples, self.get_samples))
        for i in range(self.get_samples):
            for j in range(self.get_samples):
                self.K[i, j] = self.gen_kernel(X[i],X[j])
        P = cvxopt.matrix(np.outer(y, y) * self.K)
        q = cvxopt.matrix(-1 * np.ones(self.get_samples))
        A = cvxopt.matrix(y, (1, self.get_samples))
        A = A * 1.0
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(np.vstack((np.diag(np.ones(self.get_samples)*-1), np.identity(self.get_samples))))
        h = cvxopt.matrix(np.hstack((np.zeros(self.get_samples),np.ones(self.get_samples) * self.C)))


        res = cvxopt.solvers.qp(P, q, G, h, A, b)

        self.alpha = np.ravel(res['x'])

        sv = self.alpha > 1e-5
        t = np.arange(len(self.alpha))[sv]
        self.num_sv = sum(sv)
        self.support_vectors = self.X[sv]
        self.support_vectors_y = self.y[sv]
        self.alpha = self.alpha[sv]
        self.bias = 0
        for i in range(self.num_sv):
            self.bias += self.support_vectors_y[i]
            self.bias -= np.sum(self.alpha * self.support_vectors_y * self.K[t[i], sv])
        self.bias /= self.num_sv
        if self.kernel == 'linear':
            self.weights = np.zeros(self.get_features)
            for i in range(self.num_sv):
                self.weights += self.alpha[i] * self.support_vectors_y[i] * self.support_vectors[i]
        else:
            self.weights = None

    def compute(self, X, y):
        # Solving using Scipy.optimize.minimize
        self.X = X
        self.y = y

        self.get_samples, self.get_features = X.shape
        self.K = np.zeros((self.get_samples, self.get_samples))
        for i in range(self.get_samples):
            for j in range(self.get_samples):
                self.K[i, j] = self.gen_kernel(X[i], X[j])
        def objective(alpha):
            t1 = np.sum(alpha)
            t2 = 0
            for i in range(self.get_samples):
                for j in range(self.get_samples):
                    t2 += alpha[i]*alpha[j]*y[i]*y[j]*self.K[i, j]
            t2 = t2*0.5
            t3 = np.dot(alpha, alpha.T)
            t3 = (t3*0.5)/self.C
            return t2+t3-t1

        def stat_cond(alpha):
            return np.dot(alpha, self.y)

        bounds = [(0, self.C) for i in range(self.get_samples)]
        alpha_guess = np.zeros(self.get_samples)
        constraint = {'type': 'eq', 'fun': stat_cond}
        res = minimize(objective, alpha_guess, method='SLSQP', bounds=bounds, constraints=constraint)

        self.alpha = res.x

        sv = self.alpha > 1e-5
        t = np.arange(len(self.alpha))[sv]
        self.num_sv = sum(sv)
        self.support_vectors = self.X[sv]
        self.support_vectors_y = self.y[sv]
        self.alpha = self.alpha[sv]
        self.bias = 0
        for i in range(self.num_sv):
            self.bias += self.support_vectors_y[i]
            self.bias -= np.sum(self.alpha * self.support_vectors_y * self.K[t[i], sv])
        self.bias /= self.num_sv
        if self.kernel == 'linear':
            self.weights = np.zeros(self.get_features)
            for i in range(self.num_sv):
                self.weights += self.alpha[i] * self.support_vectors_y[i] * self.support_vectors[i]
        else:
            self.weights = None

    def predict(self, X):
        if self.weights is not None:
            y_pred = np.dot(X, self.weights) + self.bias
        else:
            y_pred = np.zeros((len(X)))
            for i in range(len(X)):
                val = 0
                for a, sv_y, sv in zip(self.alpha, self.support_vectors_y, self.support_vectors):
                    val += a * sv_y * self.gen_kernel(X[i], sv)
                y_pred[i] = val
            y_pred = y_pred + self.bias

        for i in range(len(y_pred)):
            if y_pred[i] <= 0:
                y_pred[i] = 0
            else:
                y_pred[i] = 1
        return y_pred



X_train_train, X_train_validation, y_train_train, y_train_validation = train_test_split(X_train_normalized, y_train, stratify=y_train, test_size=0.3, random_state=42)

skf = StratifiedKFold(n_splits=5)

print('Linear Kernel')
for C in [0.1, 1, 10, 100]:
    print(f'C={C}')
    fold_no = 1
    for train_index, test_index in skf.split(X_train_normalized, y_train):
        train_data = X_train_normalized.values[train_index]
        validate_data = X_train_normalized.values[test_index]
        train_data_y = y_train.values[train_index]
        validate_data_y = y_train.values[test_index]
        svm = SVM(C=C, kernel="linear")
        svm.compute(train_data, train_data_y)
        y_pred = np.array(svm.predict(validate_data))
        accuracy = accuracy_score(y_pred, validate_data_y)
        print('For Fold {} the accuracy is :'.format(str(fold_no)), accuracy)
        fold_no += 1

print('Polynomial Kernel')
for C in [0.1, 1, 10, 100]:
    for deg in [2,3]:
        print(f'C={C}, degree={deg}')
        fold_no = 1
        for train_index, test_index in skf.split(X_train_normalized, y_train):
            train_data = X_train_normalized.values[train_index]
            validate_data = X_train_normalized.values[test_index]
            train_data_y = y_train.values[train_index]
            validate_data_y = y_train.values[test_index]
            svm = SVM(C=C, kernel="poly", degree=deg)
            svm.compute(train_data, train_data_y)
            y_pred = np.array(svm.predict(validate_data))
            accuracy = accuracy_score(y_pred, validate_data_y)
            print('For Fold {} the accuracy is :'.format(str(fold_no)), accuracy)
            fold_no += 1

print('RBF Kernel')
for C in [0.1, 1, 10, 100]:
    for gam in [0.1,0.2]:
        print(f'C={C}, gamma={gam}')
        fold_no = 1
        for train_index, test_index in skf.split(X_train_normalized, y_train):
            train_data = X_train_normalized.values[train_index]
            validate_data = X_train_normalized.values[test_index]
            train_data_y = y_train.values[train_index]
            validate_data_y = y_train.values[test_index]
            svm = SVM(C=C, kernel="rbf", gamma=gam)
            svm.compute(train_data, train_data_y)
            y_pred = np.array(svm.predict(validate_data))
            accuracy = accuracy_score(y_pred, validate_data_y)
            print('For Fold {} the accuracy is :'.format(str(fold_no)), accuracy)
            fold_no += 1

svm_L = SVM(C=10, kernel="linear")
svm_L.compute(X_train_normalized.values, y_train.values)
pred_L = svm_L.predict(X_test_normalized.values)
accuracy_L = accuracy_score(pred_L, y_test)
print(classification_report(pred_L, y_test))
print(f'Accuracy of our svm for linear kernel = {accuracy_L}')

svm_inbuilt = LinearSVC(C=10)
svm_inbuilt.fit(X_train_normalized, y_train)
y_pred_inbuilt = svm_inbuilt.predict(X_test_normalized)
accuracy_inbuilt = accuracy_score(y_pred_inbuilt, y_test)
print(classification_report(y_pred_inbuilt, y_test))
print(f'Accuracy using svm from sklearn = {accuracy_inbuilt}')


svm_P = SVM(C=10, kernel="poly", degree=2)
svm_P.compute(X_train_normalized.values, y_train.values)
pred_P = svm_P.predict(X_test_normalized.values)
accuracy_P = accuracy_score(pred_P, y_test)
print(classification_report(pred_P, y_test))
print(f'Accuracy of our svm for polynomial kernel= {accuracy_P}')


svm_R = SVM(C=10, kernel="rbf", gamma=0.1)
svm_R.compute(X_train_normalized.values, y_train.values)
pred_R = svm_R.predict(X_test_normalized.values)
accuracy_R = accuracy_score(pred_R, y_test)
print(classification_report(pred_R, y_test))
print(f'Accuracy of our svm for rbf kernel= {accuracy_R}')

if __name__ == "__main__":
    def split_train(X1, y1, X2, y2):
        X1_train = X1[:190]
        y1_train = y1[:190]
        X2_train = X2[:190]
        y2_train = y2[:190]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train


    def gen_lin_separable_overlap_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 300)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 300)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2


    def split_test(X1, y1, X2, y2):
        X1_test = X1[190:]
        y1_test = y1[190:]
        X2_test = X2[190:]
        y2_test = y2[190:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test


    def test_linear():
        X1, y1, X2, y2 = gen_lin_separable_overlap_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)
        # X_train, X_test, y_train, y_test =

        clf = SVM(C=1000.1, kernel="linear")
        clf.compute(X_train, y_train)
        for i in range(len(y_test)):
            if (y_test[i] == -1):
                y_test[i] = 0
        y_predict = clf.predict(X_test)
        print(classification_report(y_predict, y_test))
        accuracy = accuracy_score(y_predict, y_test)
        print(f'Accuracy of our svm on data generated randomly= {accuracy}')
        # correct = np.sum(y_predict == y_test)
        # print("%d out of %d predictions correct" % (correct, len(y_predict)))

test_linear()