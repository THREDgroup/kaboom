"""
Inherits the Agent class, and creates agents that design an I-beam.
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from kaboom.helperFunctions import bounds
from kaboom.agent import Agent
from kaboom.params import Params
from kaboom.solution import Solution
#from kaboom.kaboom import teamWorkSharing
from kaboom import kaiStyle as kai


from kaboom import modelFunctions as m


class Beam: #a symmetrical wide-flange beam
    def __init__(self,r=[0.23,.25,.1,.02]):
        #r=[.29,.32,.165,.0075]): #default values
        self.innerHeight = r[0] #m, height to inside of flange
        self.outerHeight = r[1] #m, height to outside of flange
        self.width = r[2] #m, width of the flanges
        self.thickness = r[3] #m, thickness of the vertical connector
        self.length = 5 #m
        self.Emodulus = 200E9
        
    def calcMaxStress(self,load = 50000):
        momentOfInertia = 1/12 * (self.width * 
                                  (self.outerHeight**3 - self.innerHeight**3) 
                                  + self.thickness*self.innerHeight**3) 
        extremeDistance = self.outerHeight / 2
        maxS = abs(load*self.length / 4 * extremeDistance / momentOfInertia)
        return maxS
    
    def calcMaxDisplacement(self,load = 50000):
        momentOfInertia = 1/12 * (self.width * (self.outerHeight**3 - 
                                                self.innerHeight**3) + 
                                 self.thickness*self.innerHeight**3)
        maxDisplacement = -1 * load * self.length**3/(48 * self.Emodulus*momentOfInertia)
        return maxDisplacement
    
    def calcArea(self):
        area = (self.width * (self.outerHeight-self.innerHeight) + 
                            self.thickness * self.innerHeight )
        return area
    
    def normVector(self):
        r = np.array([self.innerHeight, self.outerHeight, self.width, self.thickness])
        return r
    
    def constrain(self, minDim=.007, maxDim=1):
        #constrain all dimensions of the beam to a min and max dimension
        self.thickness = bounds(self.thickness,minDim, maxDim)
        self.innerHeight = bounds(self.innerHeight, minDim, maxDim)

        self.outerHeight = bounds(self.outerHeight, minDim, maxDim)
        self.width = bounds(self.width, minDim, maxDim)
        if (self.outerHeight-self.innerHeight) < (2*minDim):
            self.innerHeight = self.outerHeight - minDim*2
            
    def draw(self):
        fig,ax = plt.subplots(1)
        rect = patches.Rectangle((0,0),self.width,self.outerHeight,facecolor='black')
        ax.add_patch(rect)
        rect2 = patches.Rectangle((0,(self.outerHeight-self.innerHeight)/2),self.width,self.innerHeight,facecolor='white')
        ax.add_patch(rect2)
        rect3 = patches.Rectangle((self.width/2-self.thickness/2,0),self.thickness,self.outerHeight,facecolor='black')
        ax.add_patch(rect3)
        
        plt.show()
        
        
def beamObjective(beam): #minimize
    if not feasibleBeam(beam):
        return np.inf
    maxD = beam.calcMaxDisplacement()
    maxStress = beam.calcMaxStress()
#    print(abs(maxD)*1E3)
#    print(maxStress/1E6/10)
    return abs(maxD)*1E3 + maxStress/1E6/10

def feasibleBeam(beam,maxArea=.007):
    #constraint on the cross-sectional area
    if beam.calcArea()>maxArea:
        return False
    return True

class BeamDesigner(Agent):
    """
    These Agents design a wide-flange Beam for a center load and simple support

    """
    def __init__(self, p):
        """
        Create a BeamDesigner agent

        Parameters:
        ----------
        p : Params object, contains current model Parameters
        """
        Agent.__init__(self,p)
        self.myDims = np.ones(p.nDims)
        self.beam = Beam()
        self.r = self.beam.normVector()

    def reset(self,p):
        """ Reset the agent to it's initial birth conditions. """
        self.temp = self.startTemp
        self.speed = kai.calcAgentSpeed(self.kai.KAI,p)
        self.memory = []

        self.beam = Beam()
        self.r = self.beam.normVector()
        
        self.nmoves = 0
        self.score = np.inf

    def moveTo(self,r):
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
        self.beam = Beam(r)
        self.score = self.f()
        self.memory.append(Solution(self.r,self.score,type(self)))
        self.nmoves += 1

    def startAt(self,r):
        """
        Define the initial position of the agent and wipe its memory.

        Parameters:
        ----------
        position : list, the initial solution to start at
        """
        self.r = r
        self.beam = Beam(r)
        self.memory = [Solution(r=self.r,score=self.f(),agent_class=type(self))]


    def f(self): #evaluate objective function for this agent's current solution
        self.beam = Beam(self.r)
        return beamObjective(self.beam)
    def fr(self,new_r):
        newBeam = Beam(new_r)
        return beamObjective(newBeam)

    #constrain vector solution [r] to the feasible space
    #returns constrained solution
    def constrain(self,r,p):
        beam = Beam(r)
        beam.constrain()
        rConstrained = beam.normVector()
        return rConstrained #[-1 for i in range(len(x))]


#TEST
#p = Params()
#p.nDims = 4
#p.groupConformityBias = False
#a = BeamDesigner(p)
##a.kai = kai.findAiScore(130,
#a.score = a.f()
#
#print(a.r)
#print(a.score)
#for i in range(100):
#    a.move(p)
#    b=a.beam
#    if i%10== 0:
#        b.draw()
#print("final score:")
#print(a.r)
#print(a.score)


#            
#def initViz(self):
#    fig,ax = plt.subplots(1)
#    rect = patches.Rectangle((0,0),self.width,self.outerHeight,facecolor='black')
#    ax.add_patch(rect)
#    rect2 = patches.Rectangle((0,(self.outerHeight-self.innerHeight)/2),self.width,self.innerHeight,facecolor='white')
#    ax.add_patch(rect2)
#    rect3 = patches.Rectangle((self.width/2-self.thickness/2,0),self.thickness,self.outerHeight,facecolor='black')
#    ax.add_patch(rect3)
#
#def animate(i):
#    r = a.memory[i].r
#    beam = Beam(r)
#    rect1.setExtent
##    
#    return rect, rect2, rect3

#
#from matplotlib import animation
##import matplotlib.image as mpimg
#
#fig = plt.figure()
#fig.set_dpi(300)
#fig.set_size_inches(7, 6.5)
#
#anim = animation.FuncAnimation(fig, animate, 
#                               repeat = False,
#                               frames=len(c.memory), 
#                               interval=50,
#                               blit=True,
#                               init_func=initViz, 
#                               )
#
#
#plt.show()