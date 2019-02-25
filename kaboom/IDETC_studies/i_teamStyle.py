"""
This computational experiment tests the question:
Does changing the KAI style of a homogeneous team affect perfomance on the
car design problem?

This will reproduce the results from IDETC paper experiment 1

The results are saved to a folder in the results directory, and a pdf of the
plotted results is also saved to that directory
"""
import numpy as np
import time as timer
import multiprocessing
import pandas as pd
from matplotlib import pyplot as plt
import itertools
import os

from kaboom.params import Params
from kaboom import modelFunctions as m
from kaboom.carMakers import carTeamWorkProcess
from kaboom.kaboom import saveResults, teamWorkSharing
from kaboom.CarDesigner import CarDesignerWeighted
from kaboom.BeamDesigner import BeamDesigner


def run():
    """ Experiment to test how KAI style affects car design performance 
    and beam design performance """
    
    t0 = timer.time()
    p=Params()
    
    #change team size and specialization
    p.nAgents = 33
    p.nDims = 56
    p.steps = 100
    p.reps = 16
    
    myPath = os.path.dirname(__file__)
    parentPath = os.path.dirname(myPath)
    paramsDF = pd.read_csv(parentPath+"/SAE/paramDBreduced.csv")
    paramsDF = paramsDF.drop(["used"],axis=1)
    paramsDF.head()
    
    teams = ['brk', 'c', 'e', 'ft', 'fw', 'ia','fsp','rsp', 'rt', 'rw', 'sw']
    paramTeams = paramsDF.team
    p.nTeams = len(teams)
    #in the semantic division of the problem, variables are grouped by parts of
    #the car (eg, wheel dimensions; engine; brakes)
    teamDimensions_semantic = [[ 1 if paramTeam == thisTeam else 0 for paramTeam in paramTeams] for thisTeam in teams]
    
    p.agentTeams = m.specializedTeams(p.nAgents,p.nTeams)
    p.teamDims = teamDimensions_semantic
       
    
    styleTeams = []
    flatTeamObjects = []
    aiScores = np.linspace(45,145,9)
    for aiScore in aiScores:
        #use a homogeneous team with KAI style of [aiScore]
        p.aiScore = aiScore
        p.aiRange = 0
        
        teamObjects = [] #save results
        if __name__ == '__main__' or 'kaboom.IDETC_STUDIES.i_teamStyle':
            pool = multiprocessing.Pool(processes = 4)
            allTeams = pool.starmap(carTeamWorkProcess, zip(range(p.reps),itertools.repeat(p),itertools.repeat(CarDesignerWeighted)))
            for t in allTeams:
                teamObjects.append(t)
                flatTeamObjects.append(t)
            pool.close()
            pool.join()
        print("time to complete: "+str(timer.time()-t0))
        styleTeams.append(teamObjects)
        
#    saveResults(flatTeamObjects, p, "carProblem_KaiStyle")
    
    #invert the scores *-1 to show a maximization (rather than minimization) 
    #objective. (Then, in this plot, higher scores are better)
    allScores = [t.getBestScore()*-1 for s in styleTeams for t in s]
        
    allkai = [kai for kai in aiScores for i in range(p.reps)]
    m.plotCategoricalMeans(allkai,allScores)
    plt.scatter(allkai,allScores,c=[0.9,0.9,0.9])
    qFit = np.polyfit(allkai,allScores,2)
    q = np.poly1d(qFit)
    x = np.linspace(45,145,100)
    plt.plot(x,q(x), c='red')
    plt.xticks([int(i) for i in aiScores])
    plt.xlabel("KAI score of homogeneous team")
    plt.ylabel("Car Design Performance")
    plt.savefig(myPath+'/results/i_teamStyle_carProblem.pdf')
    plt.clf()
    
    
    #Now test the performance on the beam design problem
    p= Params()
    p.nAgents = 8
    p.nDims = 4
    p.nTeams = 2
    p.reps = 16
    p.steps = 100
    p.agentTeams = m.specializedTeams(p.nAgents,p.nTeams)
    p.teamDims = m.teamDimensions(p.nDims,p.nTeams)
    
    beamTeams = []
    for aiScore in aiScores:
        teamSet = []
        for i in range(p.reps):
            t= teamWorkSharing(p,BeamDesigner)
            teamSet.append(t)
        beamTeams.append(teamSet)
        print('next')
    
    
    #flip scores so that higher is better in the plot
    allScores = [ [t.getBestScore()*-1 for t in teams] for teams in beamTeams] 
    allAiScores = [ai for ai in aiScores for i in range(p.reps)]
    allScoresFlat= [ s for r in allScores for s in r]    
    
    plt.scatter(allAiScores,allScoresFlat,c=[.9,.9,.9])
    m.plotCategoricalMeans(allAiScores,allScoresFlat)
    
    #quadratic fit
    qm = np.polyfit(allAiScores,allScoresFlat, 2)
    qmodel = np.poly1d(qm)
    x = np.linspace(45,145,101)
    plt.plot(x, qmodel(x), c='red')
    plt.xlabel("KAI score of homogeneous team")
    plt.ylabel("Beam Design Performance")
    
    plt.savefig(myPath + '/results/i_teamStyle_beamProblem.pdf')
    plt.show()
    plt.clf()
        
