#class for model Parameters
#a Param object stores model parameters and sets default values
import kaboom.modelFunctions as m

class Params():
    def __init__(self):
        #Objective Function Parameters
        self.amplitude = .025 #sinusoidal amplitude, alpha
        self.w_global = 100 #omega, sinusoidal frequency
        self.SO_STRENGTH = 2 #weight of sufficiency of originality preference
        self.RG_STRENGTH = 2 #weight of group conformity preference
        self.steps = 300 #iterations in simulation
        self.spaceSize = 1 #feasible solution space is cube of +- space size

        #Agent parameters
        self.AVG_SPEED = 7.0E-3 #effective solution space size (beta)
        self.SD_SPEED = 7.0E-4
        self.MIN_SPEED = 1.0E-4
        self.AVG_TEMP = 1
        self.SD_TEMP = 0.5
        self.startPositions = None
        self.startRange = 1
        self.groupConformityBias = True #turn group conformity effects on/off
        self.soBias = True #turn sufficiency of originality effects on/off

        #Communication Parameters
        self.pComm = 0.2 #probability of comm on each step (called c in paper)
        self.meetingTimes = 50 #have one meeting every 50 iterations
        self.TEAM_MEETING_COST = 1 #1 iteration (turn)
        self.selfBias = 0 # self bias >0: agents percieve own solutions as better than they are
        self.commBonus = 10 #increasing the communication bonus makes successful communication more likely
        self.commRange = 180 #for modifying the slope of successful communication probability

        #Team parameters
        self.nAgents = 6
        self.nDims = 10
        self.nTeams = 2
        self.teamDims = m.teamDimensions(self.nDims,self.nTeams)
        self.agentTeams = m.specializedTeams(self.nAgents,self.nTeams)

        #default team has organic style composition
        self.curatedTeams = False #for Heterogeneous Linearly Distributed teams
        self.aiScore = None #for a custom mean kai score
        self.aiRange = None #for a custom range of kai scores (linearl distributed)

        #Other
        self.showViz = False #for visualizing simulation
        self.reps = 16 #experiment repetitions

    #make a string (to save to a .txt file) listing Param object's parameters
    def makeParamString(self):
        s= ""
        s+= "steps: "+ str(self.steps) + " \n"
        s+= "self-bias: " +str(self.selfBias)+ " \n"
        s+= "num agents: " +str(self.nAgents)+ " \n"
        s+= "num teams: " +str(self.nTeams)+ " \n"
        s+= "num dimensions: " +str(self.nDims)+ " \n"
        s+= "rg strength: " +str(self.RG_STRENGTH)+ " \n"
        s+= "so strength: " +str(self.SO_STRENGTH)+ " \n"
        s+= "repeats: " +str(self.reps)+ " \n"
        s+= "avg speed: " +str(self.AVG_SPEED) + " \n"
        s+= "sd speed: " + str(self.SD_SPEED)+ " \n"
        s+= "min speed: " +str(self.MIN_SPEED)+ " \n"
        s+= "avg temp: "+ str(self.AVG_TEMP)+ " \n"
        s+= "sd temp: " +str(self.SD_TEMP)+ " \n"
        s+= "roughness: " +str(self.amplitude)+ " \n"
        s+= "ai score: "+ str(self.aiScore)+" \n"
        # s+= "ai range: "+ str(self.Range)+" \n"
        return s
