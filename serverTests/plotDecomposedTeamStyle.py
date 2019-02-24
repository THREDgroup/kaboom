#saving the session workspace varibles to serverTests/individualTeamStyles
import numpy as np
import time as timer
import multiprocessing
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import itertools
import os
import scipy

from kaboom import helperFunctions as h
from kaboom import kaiStyle as kai
from kaboom.params import Params
from kaboom import modelFunctions as m
#from kaboom.carMakers import carTeamWorkProcess
from kaboom.kaboom import saveResults
from kaboom.kaboom import createTeam
from kaboom.CarDesigner import CarDesignerWeighted
#

allTeams45 = [] # Creates an empty list
for root, dirs, files in os.walk("/Users/samlapp/SAE_ABM/kaboom/serverTests/decomposedServerResultsA/styleResults45"):
    for fl in files:
        print(fl)
        f = open(root+"/"+fl,'rb')
        team = pickle.load(f)
        allTeams45.append(team)
        
allTeams70 = [] # Creates an empty list
for root, dirs, files in os.walk("/Users/samlapp/SAE_ABM/kaboom/serverTests/decomposedServerResultsA/styleResults70"):
    for fl in files:
        print(fl)
        f = open(root+"/"+fl,'rb')
        team = pickle.load(f)
        allTeams70.append(team)
        
allTeams120 = [] # Creates an empty list
for root, dirs, files in os.walk("/Users/samlapp/SAE_ABM/kaboom/serverTests/decomposedServerResultsA/styleResults120"):
    for fl in files:
        print(fl)
        f = open(root+"/"+fl,'rb')
        team = pickle.load(f)
        allTeams120.append(team)
        
allTeams145 = [] # Creates an empty list
for root, dirs, files in os.walk("/Users/samlapp/SAE_ABM/kaboom/serverTests/decomposedServerResultsA/styleResults145"):
    for fl in files:
        print(fl)
        f = open(root+"/"+fl,'rb')
        team = pickle.load(f)
        allTeams145.append(team)
#        
allTeams = [allTeams45, allTeams70,allTeams120,allTeams145]
kais = [45,70,120,145]
carTeams = ['brk', 'c', 'e', 'ft', 'fw', 'ia','fsp','rsp', 'rt', 'rw', 'sw']

controlScores = np.array([-43697.83046474878,
 -53539.600755698295,
 -51475.71127076317,
 -42580.22982036467,
 -55799.1550135683,
 -48337.7879095872,
 -53567.45303120078,
 -45038.94097887598,
 -54929.55670206938,
 -45854.197285548806,
 -53866.196861206874,
 -56265.54094660408,
 -55839.239946908034,
 -50304.24237876096,
 -46157.816942948375,
 -41412.904269363855])*-1

#plot team0:

subTeamStyles = []
#plot for each subteam:
for st in range(len(carTeams)):
    print("Modified subteam: "+carTeams[st])
    subteamScores = []
    listOfStyles = []
    plt.subplot(3,4,st+1)

    for i, teams in enumerate(allTeams):
        subteamSet = [t for t in teams if t.agents[st].kai.KAI != 95]
        style = [kais[i] for t in subteamSet]
        scores = [t.getBestScore()*-1 for t in subteamSet]
        for t in subteamSet:
            listOfStyles.append(kais[i])
            subteamScores.append(t.getBestScore()*-1)
        plt.scatter(style, scores,c=[.9,.9,.9])
        plt.scatter([95 for c in controlScores],controlScores,c=[.9,.9,.9])
    for c in controlScores:
        listOfStyles.append(95)
        subteamScores.append(c)
    qm = np.polyfit(listOfStyles,subteamScores, 2)
    qmodel = np.poly1d(qm)
    x = np.linspace(45,145,101)
    y = qmodel(x)
    bestStyle = x[np.argmin(y)]
    print("Best style:" +str(bestStyle))
    subTeamStyles.append(bestStyle)
    mns = m.plotCategoricalMeans(listOfStyles,subteamScores)
    plt.plot(x, qmodel(x),c='red')
    plt.ylim([30000,60000])
    plt.yticks(np.arange(30000,60000,10000))
#    plt.
#    print("p value: "+str(lm.pvalue)  )
#    print("R squared: "+str(lmqmrvalue**2))
    plt.title(carTeams[st])# + ", R^2 = "+str(lm.rvalue**2))
plt.savefig("./plots/compositeSubteams.pdf")
#    plt.show()
    
plt.show()
#for st in [4]:#range(len(carTeams)):
#    print("Modified subteam: "+carTeams[st])
#    subteamScores = []
#    listOfStyles = []
#    for i, teams in enumerate(allTeams):
#        subteamSet = [t for t in teams if t.agents[st].kai.KAI != 95]
#        style = [kais[i] for t in subteamSet]
#        scores = [t.getBestScore() for t in subteamSet]
#        for t in subteamSet:
#            listOfStyles.append(kais[i])
#            subteamScores.append(t.getBestScore())
#        plt.scatter(style, scores)
#        plt.scatter([95 for c in controlScores],controlScores)
#    for c in controlScores:
#        listOfStyles.append(95)
#        subteamScores.append(c)
#    qm = np.polyfit(listOfStyles,subteamScores, 2)
#    qmodel = np.poly1d(qm)
#    x = np.linspace(45,145,100)
#    plt.plot(x, qmodel(x))
##    print("p value: "+str(lm.pvalue)  )
##    print("R squared: "+str(lm.rvalue**2))
##    plt.title("Modified subteam: "+carTeams[st] + ", R^2 = "+str(lm.rvalue**2))
#    plt.savefig("./plots/subteamQuadratic"+str(st)+"_styleEffect.png",dpi=300)
#    plt.show()
#
#
#    
    
    