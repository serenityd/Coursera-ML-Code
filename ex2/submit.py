import numpy as np

from Submission import Submission
from Submission import sprintf

__all__ = ['submit']

homework = 'logistic-regression'

part_names = [
    'Sigmoid Function',
    'Logistic Regression Cost',
    'Logistic Regression Gradient',
    'Predict',
    'Regularized Logistic Regression Cost',
    'Regularized Logistic Regression Gradient',
    ]

srcs = [
    'ex2.py',
    'ex2.py',
    'ex2.py',
    'ex2.py',
    'ex2.py',
    'ex2.py',
    ]


def output(part_id):
    X = np.column_stack((np.ones(20),
                          (np.exp(1) * np.sin(np.linspace(1, 20, 20))),
                          (np.exp(0.5) * np.cos(np.linspace(1, 20, 20)))))
    Y = np.sin(X[:,0] + X[:,1]) > 0

    fname = srcs[part_id-1].rsplit('.',1)[0]
    print(fname)
    mod = __import__(fname)
    print(mod)
    func1 = getattr(mod, 'sigmoid')
    func2 = getattr(mod, 'costFunction')
    func3 = getattr(mod, 'gradientFunction')
    func4 = getattr(mod, 'predict')
    func5 = getattr(mod, 'test')
    func6 = getattr(mod, 'test')
    X=np.mat(X)
    Y=np.mat(Y).T
    Y = np.where(Y == True, 1, 0)
    if part_id == 1:
        return sprintf('%0.5f ', func1(X))
    elif part_id == 2:
        return sprintf('%0.5f ', func2(np.array([0.25, 0.5, -0.5]), X, Y))
    elif part_id == 3:
        return sprintf('%0.5f ', func3(np.array([0.25, 0.5, -0.5]), X, Y))
    elif part_id == 4:
        return sprintf('%0.5f ', func4(np.array([0.25, 0.5, -0.5]), X,Y))
    elif part_id == 5:
        return sprintf('%0.5f ', func5(np.array([0.25, 0.5, -0.5]), X, Y, 0.1))
    elif part_id == 6:
        return sprintf('%0.5f ', func6(np.array([0.25, 0.5, -0.5]), X, Y, 0.1))

s = Submission(homework, part_names, srcs, output)
try:
    s.submit()
except Exception as ex:
    template = "An exception of type {0} occured. Messsage:\n{1!r}"
    message = template.format(type(ex).__name__, ex.args)
    print(message)
