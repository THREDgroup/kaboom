"""
This module contains the base Agent class and derived classes.

Agents form the basis of the agent based model, and have methods for
exploring a solution space according to a simulated annealing paradigm.
"""
import numpy as np
from kaboom import helperFunctions as h
from kaboom import modelFunctions as m
from kaboom import kaiStyle as kai
from kaboom.solution import Solution

class Agent:
    """
    The base class for agents in the agent based model

    Warning: Call derived classes (Steinway) rather than calling this  directly.
    """
    def __init__(self, p):
        """
        Initialize the agent.
        """
        self.score = np.inf
        self.r = np.random.uniform(-1,1,size=p.nDims)
        self.nmoves = 0
        self.kai = kai.KAIScore()
        self.speed = kai.calcAgentSpeed(self.kai.KAI,p)
        self.temp = kai.calcAgentTemp(self.kai.E,p)
        self.memory = [Solution(self.r,self.score,type(self))]
        self.team = -1
        self.decay = kai.calculateAgentDecay(self,100)
        self.startTemp = h.cp(self.temp)
        self.startSpeed = h.cp(self.speed)

    def reset(self,p):
        """ Reset the agent to it's initial birth conditions. """
        self.temp = self.startTemp
        self.speed = kai.calcAgentSpeed(self.kai.KAI,p)
        self.r = np.random.uniform(-1,1,size=p.nDims)
        self.memory = [Solution(self.r,self.score,type(self))]
        self.nmoves = 0
        self.score = np.inf


    def move(self,p,teamPosition=None):
        """
        Make the agent explore a new solutionself.

        On each turn (iteration), each agent calls move() to explore a new
        solution (unless they are collaborating that turn). The agent randomly
        chooses a direction of travel in the solution space, projects that
        direciton onto the dimensions that they control (self.myDims), and
        scales the movement by a travel distance. It then evaluates the new
        solution and decides whether to accept or reject it.

        Parameters
        ----------
        p : Params object, contains current model Parameters
        teamPosition : list, shape = [nDims] (default=None)

        Returns
        -------
        returns True if agent accepts new solution, else False
        """

        #randomly pick direction of new solution
        d = np.random.uniform(-1,1,p.nDims)
        d = d * self.myDims #project onto the dimensions I can move
        dn = np.linalg.norm(d)
        #scale the distance moved by random Poissan distribution and current speed
        distance = m.travelDistance(self.speed) * p.nDims
        d = d / dn * distance

        candidateSolution = (self.r + d)
        candidateSolution = self.constrain(candidateSolution,p) #constrain to feasible space

        #evaluate the new solution and decide whether to move there
        acceptsNewSolution = self.evaluate(candidateSolution,p,teamPosition=teamPosition)
        if acceptsNewSolution:
            self.moveTo(candidateSolution) #updates solution, memory, and score
            return True
        return False

    #update the current solution and score, then add it to agent memory
    # takes argument r: the new solution (vector with length nDims)
    def moveTo(self, r):
        """
        Change the agent's current solution to r and update it's score.

        Also add the new solution to the agent's memory and increment the
        agent's move countself.

        Parameters:
        ----------
        r : list, shape = [nDims]
            the new solution to move to
        """

        self.r = r
        self.score = self.f()
        self.memory.append(Solution(self.r,self.score,type(self)))
        self.nmoves += 1

    def startAt(self,position):
        """
        Define the initial position of the agent and wipe its memory.

        Parameters:
        ----------
        position : list, the initial solution to start at
        """
        self.r = position
        self.memory = [Solution(r=self.r,score=self.f(),agent_class=type(self))]

    #agent decides to communicate on a turn with probability pComm
    def wantsToTalk(self,pComm):
        """
        Decide (stochastically) whether this agent will collaborate this turn.

        Parameters:
        ----------
        pComm : float
            probability of communication (collaboration) on any turn
        """
        if(np.random.uniform() < pComm):
            return True
        return False

    #
    # returns score
    def getBestScore(self):
        """
        Search an agent's memory for its best solution, and return the score.

        Returns:
        -------
        bestScore : float, best score of any past solution the agent has had
        """
        bestScore = self.score
        for s in self.memory:
            if s.score < bestScore:
                bestScore = s.score
        return bestScore

    def getBestSolution(self):
        """
        Search an agent's memory for its best solution, and return the Solution.

        Returns:
        -------
        bestScore : Solution object, best of any past solution the agent has had
        """
        bestSolution = h.cp(self.memory[0])
        for mem in self.memory:
            if mem.score < bestSolution.score:
                bestSolution = mem
        return bestSolution


    def soBias(self,currentPosition,candidatePosition,p):
        """
        Calculate the Sufficiency of Originality bias for a solution.

        An agent's perception of solution quality is modified by the
        Sufficiency of Originality bias. Adaptors (low KAI.SO) prefer solutions
        towards solutions existing in their memory, while innovators
        (high KAI.SO) prefer solutions away from those in their memory.

        The returned value is added to the agent's percieved solution quality
        when evaluating a solution.

        Parameters:
        ----------
        currentPosition : list, size = [nDims]
            the agent's current solution
        candidatePosition : list, size = [nDims]
            the new solution being considered (evaluated)
        p : Params object, contains current model Parameters

        Returns:
        -------
        sufficiency_of_originality : float,
            positive or negative preference of the agent for the new solution
        """
        #Sufficiency of Originality, standardized (mean 0, std 1)
        soNorm = kai.standardizedSO(self.kai.SO)
        memSize = len(self.memory)
        if memSize < 2: return 0 #agent doesn't have enough memories for this

        candidateDirection = candidatePosition - currentPosition

        # the percieved memory direction is the the weighted mean position of
        # past memories
        memDirection = 0
        #memory weights are based on temporal order (Recency and Primacy Bias)
        weights = m.memoryWeightsPrimacy(memSize)
        #loop through memories, but don't include current soln
        for i in range(memSize-1):
            past_soln = self.memory[i]
            pairwiseDiff = past_soln.r - currentPosition
            memDirection += pairwiseDiff * weights[i]

        #now check if the new solution is in the direction of the memories
        #or away from the memories by taking the dot product
        paradigmRelatedness = h.dotNorm(memDirection, candidateDirection)
        raw_PR_score = soNorm * paradigmRelatedness
        sufficiency_of_originality = raw_PR_score*p.SO_STRENGTH

        return sufficiency_of_originality

    #Agent's perception of solution quality is modified by group conformity bias (RG score)
    #adaptors prefer solutions towards group, innovators prefer solutions away from group
    #the returned value is added to the agent's percieved solution quality
    def groupConformityBias(self,teamPosition,currentPosition,candidatePosition,p): #influences preference for new solutions, f(A-I)
        """
        Calculate the Group Conformity bias for a solution.

        An agent's perception of solution quality is modified by the
        Group Conformity bias. Adaptors (low KAI.RG) prefer solutions
        towards their team's solutions, while innovators
        (high KAI.RG) prefer solutions away from their team's solutions.

        The returned value is added to the agent's percieved solution quality
        when evaluating a solution.

        Parameters:
        ----------
        teamPosition : list, size = [nDims]
            the mean (centroid) of all teammate's solutions
        currentPosition : list, size = [nDims]
            the agent's current solution
        candidatePosition : list, size = [nDims]
            the new solution being considered (evaluated)
        p : Params object, contains current model Parameters

        Returns:
        -------
        groupConformity : float,
            positive or negative preference of the agent for the new solution
        """
        rgNorm = kai.standardizedRG(self.kai.RG)
        candidateDirection = candidatePosition - currentPosition
        teamDirection = teamPosition - currentPosition

        #Check if the new solution is towards the team or away from the team
        groupConformity = h.dotNorm(teamDirection, candidateDirection)
        groupConformity = (groupConformity)*rgNorm*p.RG_STRENGTH
        return groupConformity

    def evaluate(self,candidateSolution,p,teamPosition=None):
        """
        Evaluate a solution and decide whether to accept or reject it.

        if soBias or groupConformityBias are False, they don't affect perception

        Parameters:
        ----------
        candidatePosition : list, size = [nDims]
            the new solution being considered (evaluated)
        p : Params object, contains current model Parameters
        teamPosition : list, size = [nDims] (default = None)
            the mean (centroid) of all teammate's solutions
            required if groupCoformityBias = true

        Returns:
        -------
        returns True if agent accepts candidate solution, else False
        """
        candidateScore = self.fr(candidateSolution) #true objective function value of solution
        if p.soBias: #so bias is on
            candidateScore += self.soBias(self.r,candidateSolution,p)
        if p.groupConformityBias: #rg bias is on
            gcB = self.groupConformityBias(teamPosition,self.r,candidateSolution,p)
            candidateScore += gcB
        #if solution is better, always accept
        if candidateScore < self.score:
            return True
        #accept worse solution with some probability: exp((old-new )/temp)
        elif m.pickWorseScore(self.score,candidateScore,self.temp):
            self.score = candidateScore #its worse, but we go there anyways
            return True
        return False
    
    #constrain solution [x] to the feasible space bounded by [p.spaceSize]
    #returns constrained solution
    def constrain(self,x,p):
        for i in range(len(x)):
            x[i] = h.bounds(x[i],-1*p.spaceSize,1*p.spaceSize)
        return x
#
class Steinway(Agent):
    """
    These Agents solve a tuneable roughness objective function.

    The objective function has a quadratic and sinusoid with variable amplitude.
    To use a different objective function, create a different class with
    different f() and fr() method.
    """
    def __init__(self, p):
        """
        Create a Steinway agentTeams

        Parameters:
        ----------
        p : Params object, contains current model Parameters
        """
        Agent.__init__(self,p)
        self.myDims = np.ones(p.nDims)
        self.roughness=p.amplitude
    def f(self): #evaluate objective function for this agent's current solution
        return m.objectiveTune(self.r,self.roughness)
    def fr(self,r): #evaluate objective function for a given solution r
        return m.objectiveTune(r,self.roughness)
    
    #constrain solution [x] to the feasible space bounded by [p.spaceSize]
    #returns constrained solution
    def constrain(self,x,p):
        for i in range(len(x)):
            x[i] = h.bounds(x[i],-1*p.spaceSize,1*p.spaceSize)
        return x
