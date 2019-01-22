#the Team class holds agents of the Agent class, and has the agents work together
#by dividing into subteams, having pair-wise collaborations, and having team meetings

import numpy as np
from kaboom import modelFunctions as m
from kaboom import kaiStyle as kai
import scipy

class Team(): #a group of agents working on the same dimension and objective function
    def __init__(self, agentConstructor,p,kaiPopulation):

        #initialize the team with the desired agents
        self.agents = []
        aiScores = np.zeros(p.nAgents)
        #curated team composition:
        if (p.aiScore is not None) and (p.aiRange is not None):
            minScore = np.max([40, p.aiScore-p.aiRange/2.0])
            maxScore = np.min([150,p.aiScore+p.aiRange/2.0])
            aiScores = np.linspace(minScore,maxScore,p.nAgents)
            np.random.shuffle(aiScores) #randomly assign these to agents, not in order...
        for i in range(p.nAgents):
            a = agentConstructor(p) #creates an agent with random kai score
            if p.startPositions is not None:
                a.startAt(p.startPositions[i])
            #if necessary, give it a specific style (kai score):
            if (p.aiScore is not None) and (p.aiRange is not None):
                aiScore = aiScores[i]
                a.kai = kai.findAiScore(aiScore,kaiPopulation)
                a.speed = kai.calcAgentSpeed(a.kai.KAI,p)
                a.temp = kai.calcAgentTemp(a.kai.E,p)
            a.startSpeed = a.speed
            a.startTemp = a.temp
            a.myDims = np.ones(p.nDims) #default: all dimensions owned by every agent
            self.agents.append(a)
        self.nAgents = p.nAgents
        aiScores = [a.kai.KAI for a in self.agents]
        self.dAI = np.max(aiScores)- np.min(aiScores)

        #record the team's process and behaviors:
        self.nMeetings = 0
        self.shareHistory = []
        self.nTeamMeetings = 0
        self.subTeamMeetings = 0
        self.meetingDistances = []
        self.scoreHistory = []

        #if there are subteams owning certain dimensions, each subteams dimensions are listed in a matrix
        self.specializations = p.teamDims

    #reset the team to its starting condition
    def reset(self,p):
        self.nMeetings = 0
        self.shareHistory = []
        self.nTeamMeetings = 0
        self.subTeamMeetings = 0
        self.meetingDistances = []
        for a in self.agents:
            a.reset(p)
        self.scoreHistory = []

    #run the teamwork simulatino for p.steps iterations
    def run(self,p):
        np.random.seed()
        i = 0 #not for loop bc we need to increment custom ammounts inside loop
        while i < p.steps:
            self.nMeetings += self.step(p)
            if (i+1)%p.meetingTimes == 0:
                cost = self.haveInterTeamMeeting(p)
                i += cost #TEAM_MEETING_COST
            i += 1

    #mean solution (position) of all agents
    def getSharedPosition(self): #this is in the normalized space
        positions = np.array([a.r for a in self.agents])
        return [np.mean(positions[:,i]) for i in range(len(positions[0]))]

    #mean solution (position) of all agents on a subteam
    def getSubTeamPosition(self,team): #this is in the normalized space
        positions = np.array([a.r for a in self.agents if a.team == team])
        return [np.mean(positions[:,i]) for i in range(len(positions[0]))]

    # find best score of any agent on the team from entire history
    def getBestScore(self):
        return np.min([a.getBestScore() for a in self.agents])

    # find best current score of any agent on the team
    def getBestCurrentScore(self):
        return np.min([a.score for a in self.agents])

    # find best Solution (object) of any agent on the team from entire history
    def getBestSolution(self):
        allSolns = [a.getBestSolution() for a in self.agents]
        allScores = [s.score for s in allSolns]
        return allSolns[np.argmin(allScores)]

    # find best current Solution (object) of any agent on the team
    def getBestCurrentSolution(self):
        allSolns = [a.memory[-1] for a in self.agents]
        allScores = [s.score for s in allSolns]
        return allSolns[np.argmin(allScores)]

    # find best Solution (object) of any agent on the team from entire history
    def getBestTeamSolution(self,team=-1): #returns a Solution object
        bestIndividualSolns = [a.getBestSolution() for a in self.agents if a.team == team ]
        bestScoreLocation = np.argmin([s.score for s in bestIndividualSolns])
        return bestIndividualSolns[bestScoreLocation]

    # find best current score of any agent on the team
    def getBestCurrentTeamSolution(self,team=-1): #returns a Solution object
        individualSolns = [a.memory[-1] for a in self.agents if a.team == team ]
        bestScoreLocation = np.argmin([s.score for s in individualSolns])
        return individualSolns[bestScoreLocation]

    #pair up agents who try to collaborate on a turn
    def haveMeetings(self,talkers,p):
        nMeetings = 0
        for i in np.arange(0,len(talkers)-1,2):
            a1 = talkers[i]
            a2 = talkers[i+1]
            didShare = m.tryToShare(a1,a2,p)
            if didShare:
#                 print(str(a1.id) + ' and '+str(a2.id)+' shared!')
                nMeetings +=1
        self.nMeetings += nMeetings
        self.shareHistory.append(nMeetings)
        return nMeetings

    #team meeting for all agents on the team: end with unified solution
    def haveInterTeamMeeting(self,p):
        allPositions = [a.r for a in self.agents]
        teamDistance_Sum = sum(scipy.spatial.distance.pdist(allPositions))
#         print(teamDistance_Sum)
        #how much does the meeting cost? increases with distance
        cost = min(int(teamDistance_Sum / p.nDims),15)

        consensusPosition = np.zeros(p.nDims)
        #get the best solution from each specialized subteam, and extract their specialized dimensions
        for team in range(len(self.specializations)):
            bestTeamSoln = self.getBestCurrentTeamSolution(team)
            specializedInput = bestTeamSoln.r * self.specializations[team]
            consensusPosition += specializedInput
        consensusPosition = self.agents[0].constrain(consensusPosition,p)
#        consensusScore = self.agents[0].fr(consensusPosition)

        #calculate how far everyone had to move
        individualDistances = allPositions-consensusPosition
        meetingDistance = np.mean(scipy.spatial.distance.pdist(individualDistances))
        kaiDistance = np.mean(scipy.spatial.distance.pdist([[a.kai.KAI] for a in self.agents]))/100
        self.meetingDistances.append(meetingDistance*kaiDistance)

        #now move all agents to this consensus position
        for a in self.agents:
            a.moveTo(consensusPosition)

        self.nTeamMeetings += 1

        return cost #[consensusScore, consensusPosition]


    #perform one iteration of the simulation
    #this function is called for each turn (300 times in one simulation)
    #during each step, each agent can explore a new solution or communicate
    def step(self,p):
        #for speed, pre-calculate the positions of each sub-team
        subTeamPositions = [None for i in range(len(self.specializations))]
        if p.groupConformityBias:
            subTeamPositions = [self.getSubTeamPosition(i) for i in range(len(self.specializations))]

        nMeetings = 0
        talkers = []
        for a in self.agents:
            if a.wantsToTalk(p.pComm):
                talkers.append(a)
            else:
                teamPosition = subTeamPositions[a.team]
                a.move(p,teamPosition)
        if len(talkers)%2>0: #odd number, have last one explore instead of share
            a = talkers.pop()
            teamPosition = subTeamPositions[a.team]
            a.move(p,teamPosition)
        nMeetings += self.haveMeetings(talkers,p)

        self.updateTempSpeed()
        return nMeetings

    #update speed and temperature of all agents on the team
    def updateTempSpeed(self):
        for a in self.agents:
            a.temp *= a.decay
            a.speed *= a.decay
