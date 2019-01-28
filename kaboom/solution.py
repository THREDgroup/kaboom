"""This class creates objects that contains solution parameters and scores."""

from kaboom.helperFunctions import cp

class Solution():
    """
    Solution objects hold solution parameters and scores

    An agent's solutions are stored in a list at Agent.memory
    """
    def __init__(self, r,  score, agent_class=None):
        """ create a Solution object to hold parameters and score"""
        self.r = cp(r) #note : cp() is a deep copy
        self.score = cp(score)
        self.agent_class = cp(agent_class)
