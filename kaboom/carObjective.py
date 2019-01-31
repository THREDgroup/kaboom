"""
This module implements a contextualized design problem of designing a racecar.

The racecar design problem is from Zurita et al [1]. It parameterizes the design
of a racecar into a 56-variable problem, with variables separated into 11
sub-design problems, for instance the motor or cabin design. The paper
introduces an objective function which evaluates the expected performance of the
car based on metrics such as accelleration and turning speed. To this end, 11
sub objectives are defined that correspond to different aspects of performance.
The weighted sum of these sub objectives gives the total value of the objective
function (weights from Zurita [1] and uniform weights are implemented here.) See
Zurita [1] for details on the design problem.


[1] Zurita, N., Colby, M., Tumer, I., Hoyle, C., & Tumer, K. (2017).
"Design of Complex Engineered Systems Using Multi-Agent Coordination."
Journal of Computing and Information Science in Engineering, 18(October), 1–13.
https://doi.org/10.1115/1.4038158

"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

from kaboom import helperFunctions as h

kaboomDir = os.path.dirname(__file__)

#load the list of car parameters from a csv file
carParamsDF = pd.read_csv(kaboomDir +"/SAE/paramDB.csv")
carParamsDF.columns = ['paramID','name','variable','team','kind','minV','maxV','used']
# carParamsDF.at[3,"maxV"] = np.pi/4
# carParamsDF.at[10,"maxV"] = np.pi/4
carParamsDF.at[17,"maxV"] = np.pi /4
carParamsDF.used = pd.to_numeric(carParamsDF.used)
carParamsDF = carParamsDF.drop(columns=["paramID"],axis=1)
#remove unused variables...
carParamsDF = carParamsDF.loc[carParamsDF.used > 0 ]
carParamsDF.to_csv(kaboomDir +"/SAE/paramDBreduced.csv")
carParamsDF = pd.read_csv(kaboomDir +"/SAE/paramDBreduced.csv")
carParamsDF = carParamsDF.drop(["used"],axis=1)

#logical vector: (TRUE) if car parameter has fixed min/max values,
# (FALSE) if min/max values are f(car)
hasNumericBounds = [True if h.isNumber(row.minV) and h.isNumber(row.maxV) else False for i, row in carParamsDF.iterrows()]

#tweak the original to jitter equivalent values
# materialsDF = pd.read_csv("/Users/samlapp/Documents/THRED Lab/SAE/materials.csv")
# materialsDF.q = [int(1 + np.random.uniform(0.98,1.02)*materialsDF.iloc[i]['q']) for i in range(len(materialsDF))]
# materialsDF.to_csv("/Users/samlapp/Documents/THRED Lab/SAE/materialsTweaked.csv")
materialsDF = pd.read_csv(kaboomDir +"/SAE/materialsTweaked.csv")
materialsDF.head()

tiresDF = pd.read_csv(kaboomDir +"/SAE/tires.csv")

# motorsDF = pd.read_csv("/Users/samlapp/Documents/THRED Lab/SAE/motors.csv")
# # first time: we want to make motors with the same power slightly different:
# motorsDF.Power = [int(1 + np.random.uniform(0.98,1.02)*motorsDF.iloc[i]['Power']) for i in range(len(motorsDF))]
# motorsDF.to_csv("/Users/samlapp/Documents/THRED Lab/SAE/motorsTweaked.csv")
enginesDF = pd.read_csv(kaboomDir +"/SAE/motorsTweaked.csv")
print("unique" if len(enginesDF)-len(np.unique(enginesDF.Power)) == 0 else "not uniuqe")
enginesDF.columns = ["ind","id","name","le","we","he","me","Phi_e","T_e"]
enginesDF.at[0,'T_e'] = 12.4
enginesDF.head()

# susDF = pd.read_csv("/Users/samlapp/Documents/THRED Lab/SAE/suspension.csv")
# susDF.krsp = [int(np.random.uniform(0.98,1.02)*susDF.iloc[i]['krsp']) for i in range(len(susDF))]
# susDF.kfsp = susDF.krsp
# susDF.to_csv("/Users/samlapp/Documents/THRED Lab/SAE/suspensionTweaked.csv")
susDF = pd.read_csv(kaboomDir +"/SAE/suspensionTweaked.csv")
print("unique" if len(susDF)-len(np.unique(susDF.krsp)) == 0 else "not uniuqe")
susDF = susDF.drop(columns=[susDF.columns[0]])
susDF.head()

# brakesDF = pd.read_csv("/Users/samlapp/Documents/THRED Lab/SAE/brakes.csv")
# brakesDF.columns = [a.strip() for a in brakesDF.columns]
# brakesDF.rbrk = [np.random.uniform(0.98,1.02)*brakesDF.iloc[i]['rbrk'] for i in range(len(brakesDF))]
# brakesDF.to_csv("/Users/samlapp/Documents/THRED Lab/SAE/brakesTweaked.csv")
brakesDF = pd.read_csv(kaboomDir +"/SAE/brakesTweaked.csv")
print("unique" if len(brakesDF)-len(np.unique(brakesDF['rbrk'])) == 0 else "not uniuqe")
brakesDF = brakesDF.drop(columns=[brakesDF.columns[0]])
brakesDF.head()


class CarParams:
    """ A class to hold the parameters for car design

    Some of the parameters take discrete values (Eg, choosing a material). These are mapped onto a continuous axis based on one property (eg, modululs of elasticity for materials or Power for engines). In the problem space, agents explore the continuous dimension. Then the discrete value for the final solution is chosen as the closest option to the continuous position.
    """
    def __init__(self,v = carParamsDF):
        """ initialize a CarParam object holding all variables of a Solution"""
        self.vars = v.variable
        self.team = v.team
        for i, row in v.iterrows():
            setattr(self, row.variable.strip(),-1)

#example of creating a CarParams object and modifying values:
car = CarParams()
for v in car.vars:
    value = np.random.uniform()
    setattr(car,v,value)
# extracting a value:
# carParamsDF.loc[carParamsDF.variable=="hrw"]["team"][0]

# Make a dictionary of the sub-teams and their dimensions
teams = np.unique(carParamsDF.team)
teamDimensions = [[row.team == t for i, row in carParamsDF.iterrows()] for t in teams]
teamDictionary = {}
for i in range(len(teams)):
    teamDictionary[teams[i]] = teamDimensions[i]
paramList = np.array(carParamsDF.variable)

pNames = carParamsDF.variable
blankParameterObject = CarParams()
def asCarParameters(carList):
    """convert a parameter vector (list of float) to CarParameter object"""
    car = h.cp(blankParameterObject)
    for i in range(len(carList)):
        setattr(car,pNames[i],carList[i])
    return car

numberParameters = len(carParamsDF)
def asVector(car):
    """convert a CarParameter object to a parameter vector (list of float)"""
    carList = np.zeros(numberParameters)
    for i in range(numberParameters):
        pName = pNames[i]
        carList[i] = getattr(car,pName)
    return carList

### Objective Subfunctions

# constants
#
# The car’s top velocity vcar is 26.8 m/s (60 mph).
#
# The car’s engine speed x_e is 3600 rpm.
#
# The density of air q_air during the race is 1.225 kg/m3.
#
# The track radio of curvature r_track is 9 m.
#
# The pressure applied to the brakes Pbrk is 1x10^7 Pa
#

#store max values of each parameter
paramMaxValues = []
#(later, scale parameters to go between unit cube (approximately) and SI units)

v_car = 26.8 #m/s (60 mph)
w_e = 3600 * 60 * 2 *np.pi #rpm  to radians/sec
rho_air = 1.225 #kg/m3.
r_track = 9 #m
P_brk = 10**7 #Pascals
C_dc = 0.04 #drag coefficient of cabin
gravity = 9.81 #m/s^2

#mass (minimize)
def mrw(car):
    """ Calculate mass of rear wing """
    return car.lrw * car.wrw *car.hrw * car.qrw
def mfw(car):
    """ calculate mass of front wing """
    return car.lfw * car.wfw *car.hfw * car.qfw
def msw(car):
    """ Calculate mass of side wing """
    return car.lsw * car.wsw *car.hsw * car.qsw
def mia(car):
    """ Calculate mass of impact attenuator """
    return car.lia * car.wia *car.hia * car.qia
def mc(car):
    """ Calculate mass of cabin """
    return 2*(car.hc*car.lc*car.tc + car.hc*car.wc*car.tc + car.lc*car.hc*car.tc)*car.qc
def mbrk(car):
    """ Calculate mass of breaks"""
    return car.lbrk * car.wbrk * car.hbrk * car.qbrk
def mass(car):
    """ Calculate total mass of car (want to minimize) """
    mass = mrw(car) + mfw(car) + 2 * msw(car) + 2*car.mrt + 2*car.mft + car.me + mc(car) + mia(car) + 4*mbrk(car) + 2*car.mrsp + 2*car.mfsp
    return mass

def cGy(car):
    """ Calculate center of gravity height (minimize)"""
    t1 =  (mrw(car)*car.yrw + mfw(car)*car.yfw+ car.me*car.ye + mc(car)*car.yc + mia(car)*car.yia) / mass(car)
    t2 = 2* (msw(car)*car.ysw + car.mrt*car.rrt + car.mft*car.rft + mbrk(car)*car.rft + car.mrsp*car.yrsp + car.mfsp*car.yfsp) / mass(car)

    return t1 + t2

#Drag (minimize) and downforce (maximize)
def AR(w,alpha,l):
    """ Calculate aspect ratio of a wing """
    return w* np.cos(alpha) / l

def C_lift(AR,alpha):
    """ Calculate lift coefficient of a wing"""
    return 2*np.pi* (AR / (AR + 2)) * alpha

def C_drag(C_lift, AR):
    """ Calculate drag coefficient of wing"""
    return C_lift**2 / (np.pi * AR)

def F_down_wing(w,h,l,alpha,rho_air,v_car):
    """ Calculate total downward force of wing"""
    wingAR = AR(w,alpha,l)
    C_l = C_lift(wingAR, alpha)
    return 0.5 * alpha * h * w * rho_air * (v_car**2) * C_l

def F_drag_wing(w,h,l,alpha,rho_air,v_car):
    """ Calculate total drag force on a wing """
    wingAR = AR(w,alpha,l)
    C_l = C_lift(wingAR, alpha)
    C_d = C_drag(C_l,wingAR)
    return F_drag(w,h,rho_air,v_car,C_d)

def F_drag(w,h,rho_air,v_car,C_d):
    """ Calculate drag force on a body """
    return 0.5*w*h*rho_air*v_car**2*C_d

def F_drag_total(car):
    """ Calculate total drag on vehicle (minimize)"""
    cabinDrag = F_drag(car.wc,car.hc,rho_air,v_car,C_dc)
    rearWingDrag = F_drag_wing(car.wrw,car.hrw,car.lrw,car.arw,rho_air,v_car)
    frontWingDrag = F_drag_wing(car.wfw,car.hfw,car.lfw,car.afw,rho_air,v_car)
    sideWingDrag = F_drag_wing(car.wsw,car.hsw,car.lsw,car.asw,rho_air,v_car)
    return rearWingDrag + frontWingDrag + 2* sideWingDrag + cabinDrag

def F_down_total(car):
    """ Calculate total downforce (maximize)"""
    downForceRearWing = F_down_wing(car.wrw,car.hrw,car.lrw,car.arw,rho_air,v_car)
    downForceFrontWing = F_down_wing(car.wfw,car.hfw,car.lfw,car.afw,rho_air,v_car)
    downForceSideWing = F_down_wing(car.wsw,car.hsw,car.lsw,car.asw,rho_air,v_car)
    return downForceRearWing + downForceFrontWing + 2*downForceSideWing

def rollingResistance(car,tirePressure,v_car):
    """ Calculate rolling resistance of a tire """
    C = .005 + 1/tirePressure * (.01 + .0095 * (v_car**2))
    return C * mass(car) * gravity

def acceleration(car):
    """ Calculate acceleration (maximize) """
    mTotal = mass(car)
    tirePressure = car.Prt #CHRIS should it be front or rear tire pressure?
    total_resistance = F_drag_total(car) + rollingResistance(car, tirePressure,v_car)

    w_wheels = v_car / car.rrt #rotational speed of rear tires
    efficiency = total_resistance * v_car / car.Phi_e
    torque = car.T_e
    #converted units of w_e from rpm to rad/s
    F_wheels = torque * efficiency * w_e /(car.rrt * w_wheels)

    return (F_wheels - total_resistance) / mTotal


def crashForce(car):
    """ Calculate crash force (minimize) """
    return np.sqrt(mass(car) * v_car**2 * car.wia * car.hia * car.Eia / (2*car.lia))

def iaVolume(car):
    """ Calculate impact attenuator volume (minimize) """
    return car.lia*car.wia*car.hia

y_suspension = 0.05 # m
dydt_suspension = 0.025 #m/s
def suspensionForce(k,c):
    """ Calculate suspension force """
    return k*y_suspension + c*dydt_suspension

def cornerVelocity(car):
    """ Calculate corner velocity (maximize) """
    F_fsp = suspensionForce(car.kfsp,car.cfsp)
    F_rsp = suspensionForce(car.krsp,car.crsp)
    downforce = F_down_total(car)
    mTotal = mass(car)

    C = rollingResistance(car,car.Prt,v_car)
    forces = downforce+mTotal*gravity-2*F_fsp-2*F_rsp
    if forces < 0:
        return 0
    return np.sqrt( forces * C * r_track / mTotal )

p = 0
def breakingDistance(car):
    """ Calculate breaking distance (minimize) """
    mTotal = mass(car)
    C = rollingResistance(car,car.Prt,v_car)

    A_brk = car.hbrk * car.wbrk
    c_brk = .37 #standard brake pad is usually in the range of 0.35 to 0.42
    Tbrk = 2 * c_brk * P_brk * A_brk * car.rbrk

    #y forces:
    F_fsp = suspensionForce(car.kfsp,car.cfsp)
    F_rsp = suspensionForce(car.krsp,car.crsp)
    Fy = mTotal*gravity + F_down_total(car) - 2 * F_rsp - 2*F_fsp
    if Fy<=0: Fy = 1E-10

    a_brk = Fy * C / mTotal + 4*Tbrk*C/(car.rrt*mTotal) #breaking accelleration

    #breaking distance
    return v_car**2 / (2*a_brk)

def suspensionAcceleration(car):
    """ Calculate suspension acceleration (minimize) """
    Ffsp = suspensionForce(car.kfsp,car.cfsp)
    Frsp = suspensionForce(car.krsp,car.crsp)
    mTotal = mass(car)
    Fd = F_down_total(car)
    return (2*Ffsp - 2*Frsp - mTotal*gravity - Fd)/mTotal

def pitchMoment(car):
    """ Calculate pitch moment (minimize) """
    Ffsp = suspensionForce(car.kfsp,car.cfsp)
    Frsp = suspensionForce(car.krsp,car.crsp)

    downForceRearWing = F_down_wing(car.wrw,car.hrw,car.lrw,car.arw,rho_air,v_car)
    downForceFrontWing = F_down_wing(car.wfw,car.hfw,car.lfw,car.afw,rho_air,v_car)
    downForceSideWing = F_down_wing(car.wsw,car.hsw,car.lsw,car.asw,rho_air,v_car)
    # assuming lcg is lc and lf is 0.5
    lcg = car.lc
    lf = 0.5
    return 2*Ffsp*lf + 2*Frsp*lf + downForceRearWing*(lcg - car.lrw) - downForceFrontWing*(lcg-car.lfw) - 2*downForceSideWing*(lcg-car.lsw)

#Global objective: linear sum of objective subfunctions
#sub-objectives to maximize will be mirrored *-1 to become minimizing
subObjectives = [mass,cGy,F_drag_total,F_down_total,acceleration,crashForce,iaVolume,cornerVelocity,breakingDistance,suspensionAcceleration,pitchMoment]
alwaysMinimize = [1,1,1,-1,-1,1,1,-1,1,1,1] #1 for minimizing, -1 for maximizing
weightsNull = np.ones(len(subObjectives)) / len(subObjectives)
weights1 = np.array([14,1,20,30,10,1,1,10,10,2,1])/100
weights2 = np.array([25,1,15,20,15,1,1,15,5,1,1])/100
weights3 = np.array([14,1,20,15,25,1,1,10,10,2,1])/100

#pitch moment is zero bc incorrect eqn
weightsCustom = np.array([14,1,20,30,11,1,1,10,10,2,0])/100

def objectiveDetailedNonNormalized(car,weights):
    score = 0
    subscores = []
    for i in range(len(subObjectives)):
        obj = subObjectives[i]
        subscore = obj(car)
        subscores.append(subscore)
        score += weights[i]*alwaysMinimize[i]*subscore
    return score,subscores

subscoreMean = np.zeros(len(subObjectives))
subscoreSd = np.ones(len(subObjectives))

def objective(car,weights=weightsNull):
    """
    Compute the total objective function for the car's performance

    The objective is framed as a minimization problem so lower scores are better.

    Parameters:
    ----------


    Returns:
    -------
    score : float, total objective function score (minimization problem)
    """
    score = 0
    for i in range(len(subObjectives)):
        obj = subObjectives[i]
        subscore= obj(car)
        normalizedSubscore = (subscore - subscoreMean[i]) / subscoreSd[i]
        score += weights[i]*alwaysMinimize[i]*normalizedSubscore
    return score

def objectiveDetailed(car,weights=weightsNull):
    score = 0
    subscores = []
    for i in range(len(subObjectives)):
        obj = subObjectives[i]
        subscore= obj(car)
        normalizedSubscore = (subscore - subscoreMean[i]) / subscoreSd[i]
        subscores.append(normalizedSubscore)
        score += weights[i]*alwaysMinimize[i]*normalizedSubscore
    return score, subscores


# ## Objectives

# ## Constraints

# In[1041]:


#a list with all the min-max functions (!) which can be called to return max and min value as f(car)
minMaxParam = [None for i in range(len(carParamsDF))]
def wrw(car):
    minV = 0.300
    maxV = r_track - 2 * car.rrt

    return minV, maxV
minMaxParam[carParamsDF.loc[carParamsDF.variable=="wrw"].index[0]] = wrw

# def xrw(car):
#     minV = car.lrw / 2
#     maxV = .250 - minV
#     return minV, maxV
# minMaxParam[carParamsDF.loc[carParamsDF.variable=="xrw"].index[0]] = xrw

def yrw(car):
    minV = .5 + car.hrw / 2
    maxV = 1.2 - car.hrw / 2
    return minV, maxV
minMaxParam[carParamsDF.loc[carParamsDF.variable=="yrw"].index[0]] = yrw
wheelSpace = .1 #?? don't have an equation for this rn, min is .075

aConst = wheelSpace
def lfw(car):
    minV = .05
    maxV = .7 - aConst
    return minV, maxV
minMaxParam[carParamsDF.loc[carParamsDF.variable=="lfw"].index[0]] = lfw

f_track = 3 # bounds: 3, 2.25 m
def wfw(car):
    minV = .3
    maxV = f_track
    return minV, maxV
minMaxParam[carParamsDF.loc[carParamsDF.variable=="wfw"].index[0]] = wfw

# def xfw(car):
#     minV = car.lrw + car.rrt + car.lc + car.lia + car.lfw/2
#     maxV = .25 + car.rrt + car.lc + car.lia + car.lfw/2
#     return minV, maxV
# minMaxParam[carParamsDF.loc[carParamsDF.variable=="xfw"].index[0]] = xfw

xConst = .030 #ground clearance 19 to 50 mm
def yfw(car):
    minV = xConst + car.hfw / 2
    maxV = .25 - car.hfw/2
    return minV, maxV
minMaxParam[carParamsDF.loc[carParamsDF.variable=="yfw"].index[0]] = yfw

# def xsw(car):
#     minV = car.lrw + 2*car.rrt + aConst + car.lsw / 2
#     maxV = .250 + 2*car.rrt + aConst + car.lsw / 2
#     return minV, maxV
# minMaxParam[carParamsDF.loc[carParamsDF.variable=="xsw"].index[0]] = xsw

def ysw(car):
    minV = xConst + car.hsw/2
    maxV = .250 - car.hsw/2
    return minV, maxV
minMaxParam[carParamsDF.loc[carParamsDF.variable=="ysw"].index[0]] = ysw

# def xrt(car):
#     minV = car.lrw + car.rrt
#     maxV = .250 + car.rrt
#     return minV, maxV
# minMaxParam[carParamsDF.loc[carParamsDF.variable=="xrt"].index[0]] = xrt

# def xft(car):
#     minV = car.lrw + car.rrt + car.lc
#     maxV = .250 + car.rrt  + car.lc
#     return minV, maxV
# minMaxParam[carParamsDF.loc[carParamsDF.variable=="xft"].index[0]] = xft

# def xe(car):
#     minV = car.lrw + car.rrt - car.le / 2
#     maxV = car.lrw + aConst + car.rrt - car.le / 2
#     return minV, maxV
# minMaxParam[carParamsDF.loc[carParamsDF.variable=="xe"].index[0]] = xe

def ye(car):
    minV = xConst + car.he / 2
    maxV = .5 - car.he / 2
    return minV, maxV
minMaxParam[carParamsDF.loc[carParamsDF.variable=="ye"].index[0]] = ye

def hc(car):
    minV = .500
    maxV = 1.200 - xConst
    return minV, maxV
minMaxParam[carParamsDF.loc[carParamsDF.variable=="hc"].index[0]] = hc

# def xc(car):
#     minV = car.lrw + car.rrt + car.lc / 2
#     maxV = .250 + car.rrt + car.lc / 2
#     return minV, maxV
# minMaxParam[carParamsDF.loc[carParamsDF.variable=="xc"].index[0]] = xc

def yc(car):
    minV = xConst + car.hc / 2
    maxV = 1.200 - car.hc / 2
    return minV, maxV
minMaxParam[carParamsDF.loc[carParamsDF.variable=="yc"].index[0]] = yc

def lia(car):
    minV = .2
    maxV = .7  - car.lfw # what is l_fr?
    return minV, maxV
minMaxParam[carParamsDF.loc[carParamsDF.variable=="lia"].index[0]] = lia

# def xia(car):
#     minV = car.lrw + car.rrt + car.lc + car.lia / 2
#     maxV = .250 + car.rrt + car.lc + car.lia/ 2
#     return minV, maxV
# minMaxParam[carParamsDF.loc[carParamsDF.variable=="xia"].index[0]] = xia

def yia(car):
    minV = xConst + car.hia / 2
    maxV = 1.200 - car.hia / 2
    return minV, maxV
minMaxParam[carParamsDF.loc[carParamsDF.variable=="yia"].index[0]] = yia

def yrsp(car):
    minV = car.rrt
    maxV = car.rrt * 2
    return minV, maxV
minMaxParam[carParamsDF.loc[carParamsDF.variable=="yrsp"].index[0]] = yrsp

def yfsp(car):
    minV = car.rft
    maxV = car.rft * 2
    return minV, maxV
minMaxParam[carParamsDF.loc[carParamsDF.variable=="yfsp"].index[0]] = yfsp

#test:
# for f in minMaxParam:
#     if f is not None:
#         print(f(car))


# In[1042]:


def getAttr(obj):
    return [a for a in dir(obj) if not a.startswith('__')]


# In[1043]:


def findMaterialByDensity(rho):
    differences = abs(np.array(materialsDF.q) - rho)
    material = materialsDF.iloc[np.argmin(differences)]
    return material.Code, material.q, material.E
def findTireByRadius(radius):
    differences = abs(np.array(tiresDF.radius) - radius)
    tire = tiresDF.iloc[np.argmin(differences)]
    return tire.ID, tire.radius, tire.mass

def findEngineByPower(power):
    differences = abs(np.array(enginesDF.Phi_e) - power)
    engine = enginesDF.loc[np.argmin(differences)]
    return engine

def findSuspensionByK(k):
    differences = abs(np.array(susDF.krsp) - k)
    sus = susDF.loc[np.argmin(differences)]
    return sus

def findBrakesByR(r): #what is the driving variable for brakes??? r?
    differences = abs(np.array(brakesDF.rbrk) - r)
    brakes = brakesDF.loc[np.argmin(differences)]

    return brakes


# In[1044]:


def constrain(car,dimsToConstrain=np.ones(len(carParamsDF))):
#     p_attribute = getAttr(car)
    paramIndices = [i for i in range(len(dimsToConstrain)) if dimsToConstrain[i] ==1]
    for i in paramIndices: #range(len(carParamsDF)): # we need to do the equations bounds last
#         if not dimsToConstrain[i]: #we don't need to check this dimension, it didn't change
#             continue
        param = carParamsDF.loc[i]
        variable = param.variable
        value = getattr(car,variable)
        if param.kind == 1: #continuous param with min and max
            if hasNumericBounds[i]:
                newValue = h.bounds(value,float(param["minV"]),float(param["maxV"]))
                setattr(car,variable,newValue)
            #do the equation ones after setting all other parameters

        elif param.kind == 2: #choose a material based on density
            materialID,density,modulusE = findMaterialByDensity(value)
            setattr(car,variable,density)
            #find other variables that are driven by this one:
            for i, otherParam in carParamsDF[carParamsDF.team == variable].iterrows():#dimension is driven by this one
                setattr(car,otherParam.variable,modulusE)
        elif param.kind == 3: #choose tires
            tireID,radius,weight = findTireByRadius(value)
            setattr(car,variable,radius)
            #find other variables that are driven by this one:
            for i, otherParam in carParamsDF[carParamsDF.team == variable].iterrows():
                setattr(car,otherParam.variable,weight)
        elif param.kind == 4: #choose motor
            tableRow= findEngineByPower(value) #Phi_e,l_e,w_e,h_e,T_e,m_e
            setattr(car,variable,tableRow[variable])
            #find other variables that are driven by this one:
            for i, otherParam in carParamsDF[carParamsDF.team == variable].iterrows():
                setattr(car,otherParam.variable,tableRow[otherParam.variable])
        elif param.kind == 5: #choose brakes
            tableRow = findBrakesByR(value) # r is driving variable

            setattr(car,variable,tableRow[variable]) #df columns need to be same as variable names
            #find other variables that are driven by this one:
            for i, otherParam in carParamsDF[carParamsDF.team == variable].iterrows():
            #their "team" is THIS VAR: dimension is driven by this
                setattr(car,otherParam.variable,tableRow[otherParam.variable])

        elif param.kind == 6: #choose suspension
            tableRow = findSuspensionByK(value) #kfsp, cfsp, mfsp
            setattr(car,variable,tableRow[variable])
            #find other variables that are driven by this one:
            for i, otherParam in carParamsDF[carParamsDF.team == variable].iterrows():#their "team" is THIS VAR: dimension is driven by this
                setattr(car,otherParam.variable,tableRow[otherParam.variable])

    #now we can do the ones that depend on other variables
    for i in paramIndices:
        param = carParamsDF.loc[i]
        variable = param.variable
        value = getattr(car,variable)
        if param.kind == 1 and not hasNumericBounds[i]:
            f = minMaxParam[i] #list of minMax functions for each variable
            minV, maxV = f(car)
            newValue = h.bounds(value,minV,maxV)
            setattr(car,variable,newValue)
    return car


# ## create the scaling vector
carVmax = asCarParameters([1e15 for i in range(len(carParamsDF))])
maxVals = constrain(carVmax)
scalingVector = asVector(maxVals) #this acts as a scaling vector to map SI unit
# values to ~unit cube

carVmin = asCarParameters([-1E10 for i in range(len(carParamsDF))])
minVals = constrain(carVmin)
# scalingVector1 = asVector(maxVals) #this acts as a scaling vector to map SI unit values to ~unit cube
# carParamsDF.iloc[11]
n = range(len(asVector(minVals)))
plt.plot(n,asVector(maxVals)/scalingVector)
plt.plot(n,asVector(minVals)/scalingVector)
# plt.plot(n,asVector(maxVals)/scalingVector)


# In[1047]:


mins = asVector(minVals)/scalingVector


def startCarParams(returnParamObject=True):
    """Create feasible starting values for a car design """
    nParams = len(carParamsDF)
    carV = np.random.uniform(0,1,nParams) * scalingVector
    car = constrain(asCarParameters(carV))
#     print(asVector(car))
    return car if returnParamObject else asVector(car)
# objective(car,weightsCustom)

def normalizedCarVector(car):
    """Convert CarParams object to a normalized vector of floats"""
    carV = asVector(car)
    normalizedCarVector = carV / scalingVector
    return normalizedCarVector

def normCarVector_to_car(normalizedCarVector):
    """Convert normalized vector of floats to a CarParams object"""
    carV = normalizedCarVector * scalingVector
    car = asCarParameters(carV)
    return car

# ### run random possible start values through objectives to get distribution of outputs
# subscores = []
# for i in range(3000):
#     car = startCarParams()
#     _,ss = objectiveDetailedNonNormalized(car,weightsNull)
#     subscores.append(ss)
# s = np.array(subscores)

# x = s[:,7]
# s[:,7 ] = [3000 if isNaN(x[i]) else x[i] for i in range(len(x)) ]
# for i in range(len(s[0,:])):
#     print(np.mean(s[:,i]))


# ### capture the mean and standard deviations of subscores
# so that we can normalize them assuming Normal dist.
# Now, the objective function will fairly weight sub-objectives using custom weights

# # FIRST TIME, need to run this if we don't have the file with saved values
# subscoreMean = []
# subscoreSd = []
# for i in range(len(subscores[0])):
#     subscoreMean.append(np.mean(s[:,i]))
#     subscoreSd.append(np.std(s[:,i]))

# subscoreStatsDF = pd.DataFrame(columns=["subscoreMean","subscoreSD"],data=np.transpose([subscoreMean,subscoreSd]))
# subscoreStatsDF.to_csv(kaboomDir +"/SAE/subscoreStatsDF.csv")

#     plt.hist((np.array(s[:,i]) - subscoreMean[i])/subscoreSd[i])
#     plt.show()

#subscoreStatsDF = pd.read_csv(kaboomDir +"/SAE/subscoreStatsDF.csv")
#subscoreMean = list(subscoreStatsDF.subscoreMean)
#subscoreSd = list(subscoreStatsDF.subscoreSD)
