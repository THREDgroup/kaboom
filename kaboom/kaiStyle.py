#this module creates a Virtual Population with KAI scores representative of the
#US population, and implements logic for manipulating/using KAI scores

import os   

import numpy as np
import pandas as pd
from kaboom import helperFunctions as h
#import dataset of kai scores and subscores from real study


my_path = os.path.dirname(__file__)
path = my_path + "/KAI/KAI_DATA_2018_07_09.csv"
kaiDF_DATASET = pd.read_csv(path) # "/Users/samlapp/SAE_ABM/kaboom/kaboom)
kaiDF_DATASET.columns = ["KAI","SO","E","RG"]

#create a normally distributed set of n KAI scores with subscores, reflecting the dataset
#parameters: n = size of population to create, asDF=bool,
    # return a pandas df (True) or numpy array (False)
def makeKAI(n=1,asDF=True):
    pop = np.random.multivariate_normal(kaiDF_DATASET.mean(),kaiDF_DATASET.cov(),n)
    if asDF:
        popDF = pd.DataFrame(pop)
        popDF.columns = kaiDF_DATASET.columns
        return popDF.round() if n>1 else popDF.loc[0]
    else:
        return pop.round() if n>1 else pop[0]

#KAI scores are objects
class KAIScore():
    def __init__(self,subscores=None):
        if subscores is None:
            subscores = makeKAI(1,True)
        self.KAI = subscores.KAI
        self.SO = subscores.SO
        self.E = subscores.E
        self.RG = subscores.RG

#calculate standardized KAI score and subscores (rescale to mean 0, std 1)
def standardizedAI(ai):
    return (ai - kaiDF_DATASET.mean().KAI)/kaiDF_DATASET.std().KAI
def standardizedRG(rg):
    return (rg - kaiDF_DATASET.mean().RG)/kaiDF_DATASET.std().RG
def standardizedE(E):
    return (E - kaiDF_DATASET.mean().E)/kaiDF_DATASET.std().E
def standardizedSO(SO):
    return (SO - kaiDF_DATASET.mean().SO)/kaiDF_DATASET.std().SO

#calculate speed and temperature parameters based on agents' KAI scores
def calcAgentSpeed(kai, p):
    speed = h.bounds(np.exp(standardizedAI(kai))* p.AVG_SPEED, p.MIN_SPEED ,np.inf)
    return speed

def calcAgentTemp(E, p):
    return np.exp(standardizedE(E)*p.AVG_TEMP)

#calculate the gemoetric decay ratio for temp and speed based on steps, initial temp and final temp
def calculateDecay(steps,T0=1.0,Tf=0.01):
    if T0<=Tf or T0<=0:
        return 0
    return (Tf / float(T0) ) ** (1/steps)

#agents with high (innovative) E score don't decay in Temperature as much as low-E score agents
def calculateAgentDecay(agent, steps):
    E_N = standardizedE(agent.kai.E)
    E_transformed = np.exp((E_N*-1)+2)
    startEndRatio = h.bounds(1/E_transformed, 1E-10,1)
    T0 = agent.temp
    TF = T0 * startEndRatio
    return calculateDecay(steps,T0,TF)

#locate a requested KAI score from the virtual population
#chooses randomly from all agents in population with desired total KAI score
#returns KAI object
def findAiScore(kai,kaiPopulation):
    kai = int(kai)
    a = kaiPopulation.loc[kaiPopulation['KAI'] == kai]
    ind = np.random.choice(a.index)
    me = kaiPopulation.loc[ind]
    return KAIScore(me) #this is a KAIScore object
