"""
Collection of non-model specific helper functions for the KABOOM model
"""
import numpy as np
import copy
import scipy

def cp(x):
    """make a deep copy instead of reference to a variable """
    return copy.deepcopy(x)

def isNumber(s):
    """ check if argument is numeric """
    try:
        float(s)
        return True
    except ValueError:
        return False

def isNaN(num):
    """ check if argument is NaN """
    return num != num

def mean(x):
    """ find Euclidean mean """
    return np.mean(x)

def norm(x):
    """ Find the Euclidian norm (2-norm) of a vector """
    return float(np.linalg.norm(x))

def dist(x,y):
    """ Find the Euclidean distance between two n-dimensional points """
    return np.linalg.norm(np.array(x)-np.array(y))

def bounds(x,low,high):
    """ constrain a value to within a lower and upper bound """
    if x > high:
        return high
    if x < low:
        return low
    return x

def pScore(A,B):
    """ ANOVA of 2 independent samples: are differences significant? """
    _, p = scipy.stats.ttest_ind(A,B)
    return p

def dotNorm(a,b):
    """return normalized dot product (how parallel 2 vectors are, -1 to 1) """
    if norm(a) <= 0 or norm(b)<= 0:
#         print("vector was length zero")
        return 0
    a = np.array(a)
    b = np.array(b)
    dotAB = np.sum(a*b)
    normDotAB = dotAB / (norm(a)*norm(b))
    return normDotAB

def aiColor(ai):
    """generate color along spectrum: red for innovators->blue for adaptors """
    ai01 = bounds((ai - 40)/ 120,0,1)
    red = ai01
    blue = 1 - ai01
    return (red,0,blue)

def effectSize(a,b):
    delta = np.mean(a)-np.mean(b)
    sd_pooled_num = (len(a)-1)*(np.std(a)**2)+ (len(b)-1)*(np.std(b)**2)
    sd_pooled_denom = len(a)+len(b)-2
    sd_pooled = np.sqrt(sd_pooled_num/sd_pooled_denom)
    return delta/sd_pooled
