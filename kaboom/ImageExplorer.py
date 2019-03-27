"""
This module contains the Agent class to explore an image as optimization space.
"""
import numpy as np
from kaboom.agent import Agent
from kaboom import helperFunctions as h
from matplotlib import pyplot as plt
#from kaboom import modelFunctions as m
#from kaboom import kaiStyle as kai
#from kaboom.solution import Solution

class Nittany(Agent):
    """
    These Agents solve a tuneable roughness objective function.

    The objective function has a quadratic and sinusoid with variable amplitude.
    To use a different objective function, create a different class with
    different f() and fr() method.
    """
    def __init__(self, p,imageURL):
        """
        Create a Nittany agent

        Parameters:
        ----------
        p : Params object, contains current model Parameters
        """
        Agent.__init__(self,p)
        self.myDims = np.ones(p.nDims)
        
        image = np.array(plt.imread(imageURL))
        reds = image[:,:,0]
        greens = image[:,:,1]
        blues = image[:,:,2]
        self.saturation = reds + greens + blues
        
    def f(self):
        """ evaluate objective function for this agent's current solution """
        imageSize = np.shape(self.saturation)
        return self.saturation[int(self.r[0]*imageSize[0]),int(self.r[1]*imageSize[1])]
    def fr(self,r):
        imageSize = np.shape(self.saturation)
        """ evaluate objective function for a given solution r """
        return self.saturation[int(r[0]*imageSize[0]),int(r[1]*imageSize[1])]

    def constrain(self,x,p):
        """
        constrain solution [x] to the feasible space bounded by [p.spaceSize]

        Parameters:
        ----------
        p : Params object, contains current model Parameters

        Returns:
        -------
        x : constrained solution
        """
        x[0] = h.bounds(x[0],0,.9999)
        x[1] = h.bounds(x[1],0,.9999)
        return x
