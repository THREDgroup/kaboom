#Solutions are objects
from kaboom.helperFunctions import cp
class Solution():
    def __init__(self, r,  score, agent_class=None):
        self.r = cp(r)
#         self.rNorm = self.r / scalingVector
        self.score = cp(score)
        self.agent_class = cp(agent_class)
