"""
Logic for creating KAI population and manipulating KAI scores

This module creates a Virtual Population with KAI scores representative of the
US population, and implements logic for manipulating/using KAI scores.
"""

import os
import numpy as np
import pandas as pd
from kaboom import helperFunctions as h

#import dataset of kai scores and subscores from real study
my_path = os.path.dirname(__file__)
path = my_path + "/KAI/KAI_DATA_2018_07_09.csv"
kaiDF_DATASET = pd.read_csv(path) # "/Users/samlapp/SAE_ABM/kaboom/kaboom)
kaiDF_DATASET.columns = ["KAI","SO","E","RG"]

#
def makeKAI(n=1,asDF=True):
    """
    create a normally distributed set of n KAI scores & subscores, using dataset

    parameters:
    ----------
    n : int, size of population to create
    asDF : boolean, return a pandas df (True) or numpy array (False)

    returns:
    -------
    pop : the virtual population containing [n] individual KAI scores
    return a pandas df if asDF==True, otherwise a numpy array of shape [n, 4]
    """
    pop = np.random.multivariate_normal(kaiDF_DATASET.mean(),kaiDF_DATASET.cov(),n)
    if asDF:
        popDF = pd.DataFrame(pop)
        popDF.columns = kaiDF_DATASET.columns
        return popDF.round() if n>1 else popDF.loc[0]
    else:
        return pop.round() if n>1 else pop[0]

class KAIScore():
    """KAI scores are objects containing the total score and subscores"""
    def __init__(self,subscores=None):
        """ Initialize KAI score object (drawn from realistic population)"""
        if subscores is None:
            subscores = makeKAI(1,True)
        self.KAI = subscores.KAI #Total KAI score
        self.SO = subscores.SO #Sufficiency of Originality subscore
        self.E = subscores.E #Efficiency subscore
        self.RG = subscores.RG #Rule Group Conformity subscore

#
def standardizedAI(ai):
    """Calculate standardized KAI score (rescale to mean 0, std 1)"""
    return (ai - kaiDF_DATASET.mean().KAI)/kaiDF_DATASET.std().KAI
def standardizedRG(rg):
    """
    Calculate standardized Rule-Group Conformity score
    (rescale to mean 0, std 1)
    """
    return (rg - kaiDF_DATASET.mean().RG)/kaiDF_DATASET.std().RG
def standardizedE(E):
    """
    Calculate standardized Efficiency score (rescale to mean 0, std 1)"""
    return (E - kaiDF_DATASET.mean().E)/kaiDF_DATASET.std().E
def standardizedSO(SO):
    """
    Calculate standardized Sufficiency of Originality score
    (rescale to mean 0, std 1)
    """
    return (SO - kaiDF_DATASET.mean().SO)/kaiDF_DATASET.std().SO

#calculate speed and temperature parameters based on agents' KAI scores
def calcAgentSpeed(kai, p):
    """
    Calculate starting speed of agent based on KAI score and parameters

    parameters:
    ----------
    kai: KAIScore object, for this agent
    p : Params object, contains current model Parameters

    returns:
    -------
    speed : float, agent's starting speed
    """
    speed = h.bounds(np.exp(standardizedAI(kai))* p.AVG_SPEED, p.MIN_SPEED ,np.inf)
    return speed

def calcAgentTemp(E, p):
    """
    Calculate starting temperature of agent based on KAI score and parameters

    parameters:
    ----------
    E: float, Efficiency subscore for this agent
    p : Params object, contains current model Parameters

    returns:
    -------
    speed : float, agent's starting temperature
    """
    return np.exp(standardizedE(E)*p.AVG_TEMP)

def calculateDecay(steps,T0=1.0,Tf=0.01):
    """
    calculate the gemoetric decay ratio for temperature and speed

    parameters:
    ----------
    steps: int, number of steps (iterations) in simulation run
    T0 : float, starting Temperature
    Tf : float, final Temperature

    returns:
    -------
    ratio : geometric decay ratio for temperature and speed
    Warning: if T0 < Tf or T0<=0, will return zero
    """
    if T0<=Tf or T0<=0:
        return 0
    ratio = (Tf / float(T0) ) ** (1/steps)
    return return ratio

def calculateAgentDecay(agent, steps):
    """
    Calculate the gemoetric decay ratio of speed and temp for a specific agentTeams

    Agents with high (innovative) E score don't decay in Temperature as
    much as low-E score agents. Therefore, adaptive agents will converge / cool
    quickly while innovative agents may never converge or cool down.

    parameters:
    ----------
    agent: Agent object
    steps: number of steps in the simulation (stored in Params as p.steps)

    returns:
    -------
    ratio : geometric decay ratio for temperature and speed
    Warning: if T0 < Tf or T0<=0, will return zero
    """
    E_N = standardizedE(agent.kai.E)
    E_transformed = np.exp((E_N*-1)+2)
    startEndRatio = h.bounds(1/E_transformed, 1E-10,1)
    T0 = agent.temp
    TF = T0 * startEndRatio
    ratio = calculateDecay(steps,T0,TF)
    return ratio

#
#
def findAiScore(kai,kaiPopulation):
    """
    Locate a requested KAI score from the virtual population.

    Chooses randomly from all agents in the virtual population with desired
    total KAI score

    parameters:
    ----------
    kai : int, desired total KAI score
    kaiPopulation : Pandas dataframe of KAI virtual population

    returns:
    -------
    kaiScore: a KAIScore object with total KAI equal to the first parameter
    """
    kai = int(kai)
    a = kaiPopulation.loc[kaiPopulation['KAI'] == kai]
    ind = np.random.choice(a.index) #randomly choose from all matching entries
    me = kaiPopulation.loc[ind] #draw from pandas DF
    kaiScore = KAIScore(me) #convert to a KAIScore object
    return kaiScore #this is a KAIScore object
