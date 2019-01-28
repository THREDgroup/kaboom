"""
The Team class holds agents of the Agent class,

The agents work together by dividing into subteams, completing pair-wise
collaborations, and having team meetings.
"""
import numpy as np
from kaboom import modelFunctions as m
from kaboom import kaiStyle as kai
import scipy

class Team():
    """
    Create a group of agents working on the same objective functionself.

    Teams may have specialized subteams that take on different dimensions
    of the problem, and may be composed of agents with similar or dissimilar
    KAI styles. The Team class has methods to simulate a simulation step with
    agent exploration and sharing, and to have team-wide meetings that unify
    the team to one common solution.
    """
    def __init__(self, agentConstructor,p,kaiPopulation):
        """
        Create a Team of given Agent class according to model parameters.

        Parameters
        ----------
        agentConstructor : costructor Class for desired agents (eg, Steinway)
        p : Params object, contains current model Parameters
        kaiPopulation: list of KAIScore, virtual population to draw from

        Returns
        -------
        self : Team object
        """
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
            #if necessary, give the agent a specific style (kai score):
            if (p.aiScore is not None) and (p.aiRange is not None):
                aiScore = aiScores[i]
                a.kai = kai.findAiScore(aiScore,kaiPopulation)
                a.speed = kai.calcAgentSpeed(a.kai.KAI,p)
                a.temp = kai.calcAgentTemp(a.kai.E,p)
            a.startSpeed = a.speed
            a.startTemp = a.temp
            #by default, all dimensions are owned by every agent
            a.myDims = np.ones(p.nDims)
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

        #if there are subteams owning certain dimensions,
        #each subteams dimensions are listed in a matrix
        self.specializations = p.teamDims

    def reset(self,p):
        """reset the team to its starting condition"""
        self.nMeetings = 0
        self.shareHistory = []
        self.nTeamMeetings = 0
        self.subTeamMeetings = 0
        self.meetingDistances = []
        for a in self.agents:
            a.reset(p)
        self.scoreHistory = []

    def run(self,p):
        """run the teamwork simulatino for p.steps iterations"""
        np.random.seed()
        i = 0 #not for loop bc we need to increment custom ammounts inside loop
        while i < p.steps:
            self.nMeetings += self.step(p)
            if (i+1)%p.meetingTimes == 0:
                cost = self.haveInterTeamMeeting(p)
                i += cost #TEAM_MEETING_COST
            i += 1

    def getSharedPosition(self):
        """Find the mean solution (position) of all agents on this team."""
        positions = np.array([a.r for a in self.agents])
        return [np.mean(positions[:,i]) for i in range(len(positions[0]))]

    def getSubTeamPosition(self,team):
        """Find the mean solution (position) of all agents on a subteam."""
        positions = np.array([a.r for a in self.agents if a.team == team])
        return [np.mean(positions[:,i]) for i in range(len(positions[0]))]

    def getBestScore(self):
        """Find best score of any agent on this team from entire history."""
        return np.min([a.getBestScore() for a in self.agents])

    def getBestCurrentScore(self):
        """Find best current score of any agent on this team."""
        return np.min([a.score for a in self.agents])

    def getBestSolution(self):
        """Find best Solution of any agent on team from its entire history."""
        allSolns = [a.getBestSolution() for a in self.agents]
        allScores = [s.score for s in allSolns]
        return allSolns[np.argmin(allScores)]

    def getBestCurrentSolution(self):
        """Find best current Solution of any agent on this team."""
        allSolns = [a.memory[-1] for a in self.agents]
        allScores = [s.score for s in allSolns]
        return allSolns[np.argmin(allScores)]

    #
    def getBestTeamSolution(self,subteam=-1):
        """Find best Solution of agents on a subteam from entire history."""
        bestIndividualSolns = [a.getBestSolution() for a in self.agents
                            if a.team == subteam ]
        bestScoreLocation = np.argmin([s.score for s in bestIndividualSolns])
        return bestIndividualSolns[bestScoreLocation]

    def getBestCurrentTeamSolution(self,subteam=-1):
        """Find best current Solution of agents on a subteam """
        individualSolns = [a.memory[-1] for a in self.agents
                        if a.team == subteam ]
        bestScoreLocation = np.argmin([s.score for s in individualSolns])
        return individualSolns[bestScoreLocation]

    def haveMeetings(self,talkers,p):
        """
        Execute team meetings during a simulation iteration.

        First, pair up agents who want to collaborate on this turn. If there's
        an odd one out, it explores instead of collaborating this turn. Then
        have the pairs of agents communicate and count how many of these
        pairwise collaborations were successful (i.e. did not fail due to
        cognitive gap).

        Parameters
        ----------
        talkers : a list of Agents that decided to communicate this iteration
        p : Params object, contains current model Parameters

        Returns
        -------
        nMeetings : int, the number of successful (non-failed) meetings
        """
        nMeetings = 0
        for i in np.arange(0,len(talkers)-1,2):
            a1 = talkers[i]
            a2 = talkers[i+1]
            didShare = m.tryToShare(a1,a2,p)
            if didShare:
                nMeetings +=1
        self.nMeetings += nMeetings
        self.shareHistory.append(nMeetings)
        return nMeetings

    def haveInterTeamMeeting(self,p):
        """
        Conduct a team meeting across the entire Team

        In a team meeting, each subteam finds its best solution and contributes
        that to the aggregate team solution. The aggregate solution will copy
        the values of each parameter's specialized dimensions. For example, if
        team1 specialized in x1 and has solution [1,1] while team2 specializes
        in dimension x2 and has solution [10,10], the aggregate solution takes
        the specialized dimension of each subteam's solution and is [1,10].
        Once the aggregate solution is established, all agents on the team
        move to (accept) the aggregate solution regardless of whether it
        improves their current solution or not.

        Parameters
        ----------
        p : Params object, contains current model Parameters

        Returns
        -------
        cost : float, a measure of the meeting's difficulty based on how far
        apart the agents' previous solutions were. This is not used in the
        current model but may be useful in future models.
        """

        allPositions = [a.r for a in self.agents]
        teamDistance_Sum = sum(scipy.spatial.distance.pdist(allPositions))

        #how much does the meeting cost? increases with distance
        #WARNING: This is not used in the current model.
        cost = min(int(teamDistance_Sum / p.nDims),15)

        consensusPosition = np.zeros(p.nDims)
        #get the best solution from each specialized subteam,
        #and extract their specialized dimensions
        for team in range(len(self.specializations)):
            bestTeamSoln = self.getBestCurrentTeamSolution(team)
            specializedInput = bestTeamSoln.r * self.specializations[team]
            consensusPosition += specializedInput
        consensusPosition = self.agents[0].constrain(consensusPosition,p)

        #calculate how far everyone had to move
        # individualDistances = allPositions-consensusPosition
        # meetingDistance = np.mean(scipy.spatial.distance.pdist(individualDistances))
        # kaiDistance = np.mean(scipy.spatial.distance.pdist(
        #             [[a.kai.KAI] for a in self.agents]))/100
        # self.meetingDistances.append(meetingDistance*kaiDistance)

        #now move all agents to this consensus position
        for a in self.agents:
            a.moveTo(consensusPosition)

        self.nTeamMeetings += 1

        return cost #[consensusScore, consensusPosition]

    def step(self,p):
        """
        Perform one iteration of the simulationself.

        This method is called for each turn, aka iteration in the simulation.
        During each turn, each agent can explore a new solution or communicate
        with another agent. Every p.meetingTimes steps, all the agents attend
        a team meeting rather than exploring or communicating.

        Parameters
        ----------
        p : Params object, contains current model Parameters

        Returns
        -------
        nMeetings : int, the number of successful pairwise meetings this turn
        """
        #for speed, pre-calculate the positions of each sub-team
        subTeamPositions = [None for i in range(len(self.specializations))]
        if p.groupConformityBias:
            subTeamPositions = [self.getSubTeamPosition(i) for i in range(len(self.specializations))]

        nMeetings = 0
        talkers = []
        #Each agent is either added to a list of communicators or explores
        for a in self.agents:
            if a.wantsToTalk(p.pComm):
                talkers.append(a)
            else:
                #get the team position of this agent's subteam
                teamPosition = subTeamPositions[a.team]
                a.move(p,teamPosition) #explore a new solution
        if len(talkers)%2>0: #odd number, have last one explore instead of share
            a = talkers.pop()
            teamPosition = subTeamPositions[a.team]
            a.move(p,teamPosition)
        #now the agents who want to communicate are paired up and share
        nMeetings += self.haveMeetings(talkers,p)

        self.updateTempSpeed() #geometric decay of teamp & speed for each agent
        return nMeetings

    def updateTempSpeed(self):
        """Update the speed and temperature of all agents on this team"""
        for a in self.agents:
            a.temp *= a.decay
            a.speed *= a.decay
