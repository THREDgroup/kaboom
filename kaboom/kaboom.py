"""
This module contains functions for running the agent based model kaboomDir

KABOOM stands for KAI agent based organizational optimization model
The functions in this module are called to run the simulation and save results.
"""
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

#our virtual population of KAI scores
kaiPopulation = kai.makeKAI(100000)

#create defaults parameters Params object
defaultParams = Params()

class Result:
    """
    A Results object stores results of the simulation for one team.

    Only the best score, best final score, number of meetings, and agent
    KAI scores are retained. Other information is discarded.
    """
    def __init__(self):
        self.bestScore = np.inf
        self.bestCurrentScore = np.inf
        self.nMeetings = 0
        self.agentKAIs = []

def saveResults(teams,p,dirName='',url='./results'):
    """
    Save a list of simulation Result objects to a .obj file in a new directory.

    Creates a directory with a timestamp for the results .obj files.
    Later, we will save the parameters summary and any output figures to the
    same directory.

    Parameters
    ----------
    teams : flat list of all Team objects returned from simulation

    p : Params object, contains current model Parameters

    Returns
    -------
    directory : string, relative path to the directory created for .obj file
    """
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
    """
    Simulate one agent solving a problem independently, without a team.

    Parameters
    ----------
    AgentConstructor : constructor class for the desired agent type

    p : Params object, contains current model Parameters

    color : string or rgb color, color to plot the path of the agent
    (only plots path if p.showViz == True)

    speed : float (default = None)
    If given, will give the agent a specific speed. Else drawn randomly.

    temp : float (default = None)
    If given, will give the agent a specific temperature. Else drawn randomly.

    startPosition : list of float, shape = [nDims] (default = none)
    If given, will start the agent a specific position. Else drawn randomly.

    Returns
    -------
    a : AgentConstructor, an agent of class given by the first argument
    this agent contains it's history in agent.memory
    you can access the agent's best score using a.getBestScore()
    """
    a = AgentConstructor(p.nDims)
    if p.aiScore is not None:
        a.kai = kai.findAiScore(p.aiScore,kaiPopulation)
        a.speed = h.bounds(p.AVG_SPEED + kai.standardizedAI(a.kai.KAI) * p.SD_SPEED, p.MIN_SPEED ,np.inf)
        a.temp = h.bounds(p.AVG_TEMP + kai.standardizedE(a.kai.E) * p.SD_TEMP, 0 ,np.inf)
    if p.startPositions is not None:
        a.startAt(p.startPositions[0])
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
    """
    Create a team of agents before running it through the simulation.

    Design the team with a specific composition of KAI scores,
    and subdivision of a problem into specialized subteams

    Parameters
    ----------
    p : Params object, contains current model Parameters
    including p.nAgents, p.nTeams, p.nDims, p.AVG_SPEED, p.AVG_TEMP

    AgentConstructor : constructor class (default = Steinway)

    Returns
    -------
    myTeam: Team object with desired characteristics, ready to run in the model
    """

    np.random.seed()
#    p.agentTeams = specializedTeams(p.nAgents,p.nTeams)
#    p.teamDims = teamDimensions(p.nDims,p.nTeams)
#     print(teamDims)
    myTeam = Team(agentConstructor,p,kaiPopulation)
    for i in range(len(myTeam.agents)):
        a = myTeam.agents[i]
        aTeam = p.agentTeams[i]
        a.team = aTeam
        a.myDims = p.teamDims[aTeam]
        a.decay = kai.calculateAgentDecay(a,p.steps)

    if p.curatedTeams and p.aiRange is not None and p.aiScore is not None:
        for team in range(len(p.teamDims)):
            teamAgents=[a for a in myTeam.agents if a.team == team]
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
    for a in myTeam.agents:
        a.startSpeed = h.cp(a.speed)
        a.startTemp = h.cp(a.temp)

    return myTeam

def teamWorkSharing(p,agentConstructor = Steinway):
    """
    Run the simulation for the team, for p.steps iterations

    This is the main method used to run the simulation. It creates a team,
    simulates problem solving with interactions and team meetings, and returns
    the Team object containing the history and results of the simulation.

    Parameters
    ----------
    p : Params object, contains current model Parameters
    including p.nAgents, p.nTeams, p.nDims, p.AVG_SPEED, p.AVG_TEMP

    AgentConstructor : constructor class (default = Steinway)

    Returns
    -------
    myTeam: Team object post-simulation containing simulation results & history
    including myTeam.agents (list of Agent objects)
    find the best score with myTeam.getBestScore()
    """
    np.random.seed()
    myTeam = createTeam(p,agentConstructor)

    i = 0 #not for loop bc we need to increment custom ammounts inside loop
    while i < p.steps:
        myTeam.nMeetings += myTeam.step(p)
        score = myTeam.getBestCurrentScore() #getBestCurrentScore
        myTeam.scoreHistory.append(score)
        if p.showViz:
            plt.scatter(i,score,marker='o',s=100,c='black')
        if (i+1)%p.meetingTimes == 0:
            cost = myTeam.haveInterTeamMeeting(p)
            i += cost #TEAM_MEETING_COST
#             if showViz:
#                 plt.show()
        i += 1
    if p.showViz: plt.show()

    return myTeam


def robustnessTest(p,problemSet=[[.2,.5,1,2,5],[.001,.005,.01,.02,.05]]): #for one team composition, run multiple problems
    """
    Have the same team solve a set of different problems, resetting before each

    This is an alternative to teamWorkSharing. It creates a team, then runs
    the simulation for the exact same team but with different problems. After
    each problem, the team is reset to its initial conditions. It is
    specifically designed for the variable-amplitude sinusoid problem that
    Steinway agents solve.

    Parameters
    ----------
    p : Params object, contains current model Parameters
    including p.nAgents, p.nTeams, p.nDims, p.AVG_SPEED, p.AVG_TEMP

    problemSet : a list of values for p.amplitude and p.AVG_SPEED
    (shape is [number of problems , 2])

    AgentConstructor : constructor class (default = Steinway)

    Returns
    -------
    scoreForEachProblem: list of floats, shape: [number of problems]
    a list of best-ever scores for the teams performance on each of the problems
    given in the problemSet

    team: Team object post-simulation (note that the history of the team only
    reflects the last problem solved, not all of the problems)
    """
    np.random.seed()
    roughnesses = problemSet[0]
    speeds = problemSet[1]
    scoreForEachProblem = []
    team = createTeam(p,Steinway)
    for i in range(len(roughnesses)):
        p.AVG_SPEED = speeds[i]
        p.amplitude = roughnesses[i]
        team.reset(p) #this also recalculates agents start speeds based on the new AVG_SPEED
        team.run(p)
        scoreForEachProblem.append(team.getBestScore())
    return scoreForEachProblem, team

def teamWorkProcess(processID,p,agentConstructor = Steinway):
    """
    A wrapper for teamWorkSharing() that contains a processID argument

    This is useful when calling the method from multiprocessing.pool, which
    always passes a processID as the first argument.
    """
    np.random.seed()
    team = teamWorkSharing(p,agentConstructor)
    return team
