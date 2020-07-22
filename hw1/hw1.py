from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import sys

def read_data(filename):
    train_X, train_Y = [], []
    f = open(filename, 'r')
    for line in f:
        data = list(map(float, line.split()))
        train_X.append(data[1:])
        train_Y.append(data[0])
    return np.array(train_X), np.array(train_Y)

def convert_label(train_Y, number):
    y = train_Y.copy()
    y[y != number] = -1
    y[y == number] = 1
    return y

def plot(x, y, x_axis, y_axis, title, hist = False, lst = [], filename = ''):
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    if not hist:
        plt.plot(x, y, '-go')
    else:
        axis = plt.gca()
        axis.set_ylim([min(y), max(y)+5])
        plt.hist(lst, [-1.5 + 1 * i for i in range(6)], histtype = 'bar', rwidth = 0.5, color = 'green')
 
    if len(filename): plt.savefig(filename)
    else: plt.show()

def p11(train_X, train_Y):
    x, y = train_X, convert_label(train_Y, 0)
    C = [-5, -3, -1, 1, 3]
    w = []
    for power in C:
        clf = svm.SVC(C = 10**power, kernel = 'linear', shrinking = False)
        clf.fit(x, y)
        w.append(np.sqrt(np.sum(clf.coef_ ** 2)))
    plot(C, w, r'$log{C}$', r'$||w||$', 'Problem 11', filename = './p11.png')

def p12(train_X, train_Y):
    x, y = train_X, convert_label(train_Y, 8)
    C = [-5, -3, -1, 1, 3]
    E_in = []
    for power in C:
        print(f'running {power}')
        clf = svm.SVC(C = 10**power, kernel = 'poly', degree = 2, coef0 = 1, gamma = 1, shrinking = False)
        clf.fit(x, y)
        E_in.append(1 - clf.score(x, y))
    plot(C, E_in, r'$log{C}$', r'$E_{in}$', 'Problem 12', filename = './p12.png')

def p13(train_X, train_Y):
    x, y = train_X, convert_label(train_Y, 8)
    C = [-5, -3, -1, 1, 3]
    num_sv = []
    for power in C:
        print(f'running {power}')
        clf = svm.SVC(C = 10**power, kernel = 'poly', degree = 2, coef0 = 1, gamma = 1, shrinking = False)
        clf.fit(x, y)
        num_sv.append(len(clf.support_))
    plot(C, num_sv, r'$log{C}$', f'number of SVs', 'Problem 13', filename = './p13.png')



def p14(train_X, train_Y):

    def kernel(x1, x2, gamma):
        return np.exp(-gamma * np.sum((x1 - x2)**2))

    x, y = train_X, convert_label(train_Y, 0)
    C = [-3, -2, -1, 0, 1]
    dis = []
    for power in C:
        clf = svm.SVC(C = 10**power, kernel = 'rbf', gamma = 80, shrinking = False)
        clf.fit(x, y)
        sv = clf.support_vectors_
        sy = y[clf.support_]
        coef = np.abs(clf.dual_coef_.reshape(-1))
        w = 0
        for i in range(len(sv)):
            for j in range(len(sv)):
                w += coef[i] * coef[j] * sy[i] * sy[j] * kernel(sv[i], sv[j], 80)
        dis.append(1 / np.sqrt(w))
    plot(C, dis, r'$log{C}$', f'distance', 'Problem 14', filename = './p14.png')

def p15(train_X, train_Y, test_X, test_Y):
    x, y = train_X, convert_label(train_Y, 0)
    tx, ty = test_X, convert_label(test_Y, 0)
    gammas = [0, 1, 2, 3, 4]
    Eout = []
    for gamma in gammas:
        clf = svm.SVC(C = 0.1, kernel = 'rbf', gamma = 10**gamma, shrinking = False)
        clf.fit(x, y)
        Eout.append(1 - clf.score(tx, ty))
    
    plot(gammas, Eout, r'$log{\gamma}$', 'Eout', 'Problem 15', filename = './p15.png')

def p16(train_X, train_Y):
    x, y = train_X, convert_label(train_Y, 0)
    gammas = [-1, 0, 1, 2, 3]
    count, lst = [0 for _ in range(len(gammas))], []
    for t in range(100):
        np.random.seed(t)
        shuf = np.arange(len(x))
        np.random.shuffle(shuf)
        x, y = x[shuf], y[shuf]
        vx, vy, tx, ty = x[:1000], y[:1000], x[1000:], y[1000:]
        best, pick = np.inf, np.inf
        for r in range(len(gammas)):
            clf = svm.SVC(C = 0.1, kernel = 'rbf', gamma = 10**gammas[r], shrinking = False)
            clf.fit(tx, ty)
            Eout = 1 - clf.score(vx, vy)
            if Eout < best:
                best = Eout
                pick = r
        count[pick] += 1
        lst.append(gammas[pick])
    
    print(count)
    plot(gammas, count, r'$log{\gamma}$', 'Count', 'Problem 16', True, lst, filename = './p16.png')
    
problem_no = int(sys.argv[1])
train_X, train_Y = read_data('features.train')
test_X, test_Y = read_data('features.test')
if problem_no == 11: p11(train_X, train_Y)
elif problem_no == 12: p12(train_X, train_Y)
elif problem_no == 13: p13(train_X, train_Y)
elif problem_no == 14: p14(train_X, train_Y)
elif problem_no == 15: p15(train_X, train_Y, test_X, test_Y)
elif problem_no == 16: p16(train_X, train_Y)
