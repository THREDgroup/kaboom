#helper functions
import numpy as np
import copy
import scipy

# A collection of useful Helper Functions

#make a deep copy instead of reference to a variable
def cp(x):
    return copy.deepcopy(x)

def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def isNaN(num):
    return num != num

def mean(x):
    return np.mean(x)

def norm(x):
    return float(np.linalg.norm(x))

def dist(x,y):
    return np.linalg.norm(np.array(x)-np.array(y))

#constrain a value to within a lower and upper bound
def bounds(x,low,high):
    if x > high:
        return high
    if x < low:
        return low
    return x

def pScore(A,B):#ANOVA of 2 independent samples: are differences significant?
    _, p = scipy.stats.ttest_ind(A,B)
    return p

def dotNorm(a,b): #return normalized dot product (how parallel 2 vectors are, -1 to 1)
    if norm(a) <= 0 or norm(b)<= 0:
#         print("vector was length zero")
        return 0
    a = np.array(a)
    b = np.array(b)
    dotAB = np.sum(a*b)
    normDotAB = dotAB / (norm(a)*norm(b))
    return normDotAB

def aiColor(ai): #generate color: red for innovators to blue for adaptors
    ai01 = bounds((ai - 40)/ 120,0,1)
    red = ai01
    blue = 1 - ai01
    return (red,0,blue)
