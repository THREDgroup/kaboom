# KABOOM: KAI agent based organizational optimization model

#Import Packages

import numpy as np
import os
import matplotlib.pyplot as plt
import time as timer
import pickle

from kaboom import helperFunctions as h
from kaboom.params import Params
from kaboom import kaiStyle as kai
from kaboom.agent import Steinway
from kaboom.team import Team
#from kaboom.CarDesigner import CarDesigner

#from kaboom.CarDesigner import CarDesigner

#our virtual population of KAI scores
kaiPopulation = kai.makeKAI(100000)

#create defaults parameters Params object
defaultParams = Params()

# machinery to save results in object files
class Result:
    def __init__(self):
        self.bestScore = np.inf
        self.bestCurrentScore = np.inf
        self.nMeetings = 0
        self.agentKAIs = []

def saveResults(teams,p,dirName='',url='./results'):
    directory = url+'/'+str(timer.time())+dirName
    os.mkdir(directory)

    paramsURL = directory+'/'+'parameters.txt'
    np.savetxt(paramsURL,[p.makeParamString()], fmt='%s')

    teamResults = []
    for team in teams:
        result = Result()
        result.bestScore = team.getBestScore()
        result.bestCurrentScore = team.getBestCurrentScore()
        result.nMeetings = team.nMeetings
        result.agentKAIs = [a.kai for a in team.agents]
        teamResults.append(result)
    rFile = directory+'/'+'results.obj'
    rPickle = open(rFile, 'wb')
    pickle.dump(teamResults, rPickle)

    return directory

#example of saving and re-loading results:
# directory = saveResults(saveTeams, 'trial')
# filehandler = open(directory+'/results.obj', 'rb')
# returnedObj = pickle.load(filehandler)

# # Individual Exploration
def work(AgentConstructor,p=defaultParams, color = 'red',speed=None,temp=None,startPosition=None):
    a = AgentConstructor(p.nDims)
    if p.aiScore is not None:
        a.kai = kai.findAiScore(p.aiScore,kaiPopulation)
        a.speed = h.bounds(p.AVG_SPEED + kai.standardizedAI(a.kai.KAI) * p.SD_SPEED, p.MIN_SPEED ,np.inf)
        a.temp = h.bounds(p.AVG_TEMP + kai.standardizedE(a.kai.E) * p.SD_TEMP, 0 ,np.inf)
    if p.startPositions is not None:
        a.startAt(startPosition)
    if temp is not None:
        a.temp = temp
    if speed is not None:
        a.speed = speed
    a.startTemp = h.cp(a.temp)
    a.startSpeed = h.cp(a.speed)

    a.decay = kai.calculateAgentDecay(a, p.steps)

    scores = []

    for i in range(p.steps):
        didMove = a.move(p,teamPosition = None)
        if didMove:
            scores.append(h.cp(a.score))
            if(p.showViz and a.nmoves>0):
#                     plt.scatter(a.rNorm[0],a.rNorm[1],c=color)
                plt.scatter(a.r[0],a.score,c=color)
        a.speed *= a.decay
        a.temp *= a.decay

    return a

# # Team Work
#create a team with desired composition of agent styles
def createTeam(p,agentConstructor = Steinway):
    np.random.seed()
#    p.agentTeams = specializedTeams(p.nAgents,p.nTeams)
#    p.teamDims = teamDimensions(p.nDims,p.nTeams)
#     print(teamDims)
    squad = Team(agentConstructor,p,kaiPopulation)
    for i in range(len(squad.agents)):
        a = squad.agents[i]
        aTeam = p.agentTeams[i]
        a.team = aTeam
        a.myDims = p.teamDims[aTeam]
        a.decay = kai.calculateAgentDecay(a,p.steps)

    if p.curatedTeams and p.aiRange is not None and p.aiScore is not None:
        for team in range(len(p.teamDims)):
            teamAgents=[a for a in squad.agents if a.team == team]
            if len(teamAgents)<2:
                a = teamAgents[0]
                a.kai = kai.findAiScore(p.aiScore,kaiPopulation)
                a.speed = kai.calcAgentSpeed(a.kai.KAI,p)
                a.temp = kai.calcAgentTemp(a.kai.E,p)
            else:
                for i in range(len(teamAgents)):
                    myKai = p.aiScore - p.aiRange/2.0 + p.aiRange*(float(i)/(len(teamAgents)-1))
                    a = teamAgents[i]
                    a.kai = kai.findAiScore(myKai,kaiPopulation)
                    a.speed = kai.calcAgentSpeed(a.kai.KAI,p)
                    a.temp = kai.calcAgentTemp(a.kai.E,p)
    for a in squad.agents:
        a.startSpeed = h.cp(a.speed)
        a.startTemp = h.cp(a.temp)

    return squad

# run the simulation for the team, for p.steps iterations
def teamWorkSharing(p,agentConstructor = Steinway):

    np.random.seed()
    squad = createTeam(p,agentConstructor)

    i = 0 #not for loop bc we need to increment custom ammounts inside loop
    while i < p.steps:
        squad.nMeetings += squad.step(p)
        score = squad.getBestCurrentScore() #getBestCurrentScore
        squad.scoreHistory.append(score)
        if p.showViz:
            plt.scatter(i,score,marker='o',s=100,c='black')
        if (i+1)%p.meetingTimes == 0:
            cost = squad.haveInterTeamMeeting(p)
            i += cost #TEAM_MEETING_COST
#             if showViz:
#                 plt.show()
        i += 1
    if p.showViz: plt.show()

    return squad

# have the same team solve a set of different problems, resetting before each
def robustnessTest(p,problemSet=[[.2,.5,1,2,5],[.001,.005,.01,.02,.05]],agentConstructor = Steinway): #for one team composition, run multiple problems
    np.random.seed()
    roughnesses = problemSet[0]
    speeds = problemSet[1]
    scoreForEachProblem = []
    team = createTeam(p,agentConstructor)
    for i in range(len(roughnesses)):
        p.AVG_SPEED = speeds[i]
        p.amplitude = roughnesses[i]
        team.reset(p) #this also recalculates agents start speeds based on the new AVG_SPEED
        team.run(p)
        scoreForEachProblem.append(team.getBestScore())
    return scoreForEachProblem, team

#wrapper for teamWorkSharing used for multiprocessing
def teamWorkProcess(processID,p,agentConstructor = Steinway):
    np.random.seed()
    team = teamWorkSharing(p,agentConstructor)
    return team
