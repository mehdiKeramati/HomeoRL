import numpy
import scipy
import pylab
import cmath

#################################################################
#                        Functions
#################################################################

# --------------------   Setting the probability [transitionProbability] for transiting from a state [state] to another state [nextState] by taking an action [action] 
def setTransition(state,action,nextState,transitionProbability):
    T[state*actionsNum + action][nextState] = transitionProbability
    return 

# --------------------   Returns the probability [transitionProbability] of transiting from a state [state] to another state [nextState] by taking an action [action] 
def getTransition(state,action,nextState):
    return T[state*actionsNum + action][nextState]

# --------------------   Returns "true" if action "action" is available in state "state"; returns "false", otherwise
def isActionAvailable(state,action):
    probSum = 0 ;
    for i in range(0,statesNum):
        probSum = probSum + getTransition(state,action,i)
    if probSum == 1:
        return 1
    elif probSum == 0:
        return 0
    else:
        print "Error: There seems to be a problem in defining the transition function"        
        return
# --------------------   Setting the reward [reward] that will be received by baking an action [action] from a state [state] 
def setOutcome(state,action,outcome):
    O[state*actionsNum + action] = outcome
    return 

#---------------------   Action Selection: SoftMax
def actionSelectionSoftmax(state):
    
    alternatives = Q [ state*actionsNum : (state+1)*actionsNum ]

    sumEQ = 0
    for i in range(0,actionsNum):
        if isActionAvailable(state,i):
            sumEQ = sumEQ + abs(cmath.exp( (bias+alternatives[i]) / beta ))
    
    index = numpy.random.uniform(0,sumEQ)
    probSum=0
    for i in range(0,actionsNum):
        if isActionAvailable(state,i):
            probSum = probSum + abs(cmath.exp( (bias+alternatives[i]) / beta ))
            if probSum >= index:
                return i

    print "Error: An unexpected (strange) problem has occured in action selection..."
    return 0

#---------------------   Action Selection: SoftMax
def actionSelectionEpsilonGreedy(state):
        
    index = numpy.random.uniform(0,1)
    if index <= epsilon :
        numOfAlternatives = 0
        for i in range(0,actionsNum):
            if isActionAvailable(state,i):
                numOfAlternatives = numOfAlternatives + 1
     #   dice = numpy.random.uniform(0,numOfAlternatives) - 0.0001
    #    for i in range(0,actionsNum):
   #         if isActionAvailable(state,i):
  #              if dice < (i+1) :
 #                   retuen i
                
#        numOfAlternatives = numOfAlternatives + 1
        
    

    print "Error: An unexpected (strange) problem has occured in action selection..."
    return 0

#---------------------   Transition Function: returns the next state after taking an action [action] in an state [state]
def transition(state,action):
    alternatives = T[state*actionsNum + action]
    index=numpy.random.uniform(0,1)
    probSum=0
    for i in range(0,statesNum):
        probSum = probSum + alternatives[i]
        if probSum >= index:
            return i
    print "Error: No transition is possible from the chosen state-action pair (%d,%d) to any new state..."%(state,action)
    return 0

#---------------------   Outcome Function: returns the outcome of taking an action [action] in an state [state]
def outcome(state,action):
    return O[state*actionsNum + action]

#---------------------   Reward Function: returns the reward of taking an action [action] in an state [state]
def reward(inState,outcome,energyPerStep):
#    d1 = numpy.square(optimalInState-inState)
#    d2 = numpy.square(optimalInState-inState-outcome+energyPerStep)
    d1 = numpy.absolute(optimalInState-inState)
    d2 = numpy.absolute(optimalInState-inState-outcome+energyPerStep)
    return d1-d2

#---------------------   Update internal state
def updateInState(inState,outcome):
    return inState+outcome

#---------------------   Update : Using TD(0) algorithm
def updateQ_TD0(state,action,nextState,reward):
    counter = 0
    QNext = numpy.zeros(actionsNum,float)
    for i in range(0,actionsNum):
        if isActionAvailable(nextState,i):
            QNext[counter] = Q[ nextState*actionsNum + i ]
            counter = counter+1

    VNext = max(QNext[0:counter])
    delta = reward + gamma*VNext - Q[state*actionsNum + action] 
    Q[state*actionsNum + action] = Q[state*actionsNum + action] + alpha*delta
    return 

# --------------------   Initialize Q-values to zero
def initializeQ():
    Q = numpy.zeros(statesNum * actionsNum , float)
    return Q
    
# --------------------   Initialize External state in a random way
def initializeExState():
    index = numpy.random.uniform(0,2)
    if index<1:
        return 0
    return 1
    
#---------------------   Logging
def logging(trial,inState,exState,action,outcome,reward):
    aveQ[trial] = aveQ[trial] + Q
    aveInState[trial] = aveInState[trial] + inState
    aveReward[trial] = aveReward[trial] + reward
    aveOutcome[trial] = aveOutcome[trial] + outcome
    aveAction[trial][exState* actionsNum + action ] = aveAction[trial][ exState* actionsNum + action ] + 1
    aveExState[trial][exState] = aveExState[trial][exState] + 1
    return 

#---------------------   Averaging the added up variables ove the total number of runs
def loggingFinalization():
    for i in range(0,trialsNum):        
        for j in range(0,statesNum * actionsNum):
            aveQ[i][j] = aveQ[i][j] / runsNum
            aveAction[i][j] = aveAction[i][j] / runsNum
        aveInState[i] = aveInState[i] / runsNum
        aveReward[i] = aveReward[i] / runsNum
        aveOutcome[i] = aveOutcome[i] / runsNum
        for j in range(0,statesNum):
            aveExState[i][j] = aveExState[i][j] / runsNum
    return

#---------------------   Plotting some desired variables
def plotting():

    aveActionT = numpy.transpose (aveAction)
    aveQT = numpy.transpose (aveQ)
    aveRewardT = numpy.transpose (aveReward)
    aveOutcomeT = numpy.transpose (aveOutcome)
    aveInStateT = numpy.transpose (aveInState)
    aveExStateT = numpy.transpose (aveExState)

    pylab.figure()
    S0 = pylab.plot(aveExStateT[0])
    S1 = pylab.plot(aveExStateT[1])
    pylab.legend((S0, S1), ('Presence in s0', 'Presence in S1'))

    pylab.figure()
    h = pylab.plot(aveInStateT)
    pylab.legend((h),('homeostatic point'))

    pylab.figure()
    R = pylab.plot(aveRewardT)
    pylab.legend((R),('Obtained Reward'))

    pylab.figure()
    Q0 = pylab.plot(aveQT[0])
    Q1 = pylab.plot(aveQT[1])
    Q2 = pylab.plot(aveQT[2])
    Q3 = pylab.plot(aveQT[3])
    pylab.legend((Q0,Q1,Q2,Q3), ('Q(s0,a0)','Q(s0,a1)','Q(s1,a0)','Q(s1,a1)'))
    
    pylab.show()    

    return


#==================================================================================================================================================
#======   The code starts running from here   =====================================================================================================
#==================================================================================================================================================

#################################################################
#                        Environment's Dynamics
#################################################################

# --------------------   Number of the States
statesNum  = 2

# --------------------   Number of the Actions
actionsNum = 2

# --------------------   Initial External State
#initialExState = 0

# --------------------   Final External States
#finalExStates = [6,8]

# --------------------   Transition Function : (from state s, by action a, going to state s', by probability p)
T = numpy.zeros( [statesNum * actionsNum, statesNum] , float)
setTransition(0,0,0,1)
setTransition(0,1,1,1)
setTransition(1,0,1,1)
setTransition(1,1,0,1)

# --------------------   Outcome Function : (from state s, by action a, recieving reward r)
smallOutcome = 1
bigOutcome   = 4

O = numpy.zeros (statesNum * actionsNum , float)

setOutcome(0,0,smallOutcome)
setOutcome(0,1,bigOutcome)
setOutcome(1,0,bigOutcome)
setOutcome(1,1,smallOutcome)

#################################################################
#                        Agent's Dynamics
#################################################################

energyExpenditurePerStep = 2

# --------------------   Initial internal State
initialInState =-50

# --------------------   Optimal internal State = Homeostatic setpoint
optimalInState = 0

# -------------------- Meta Variables
alpha = 0.3         # Learning rate
gamma = 0.9         # Discount factor

beta = 1           # Rate of exploration
bias = 0           # exploration bias

#################################################################
#                        Simulation Initializations
#################################################################

# -------------------- Simulations Variables
trialsNum = 400
runsNum = 50

# --------------------   Q-values
Q = numpy.zeros(statesNum * actionsNum , float)

# --------------------   Average-over-runs Variables
aveQ = numpy.zeros([trialsNum , statesNum * actionsNum] , float)
aveAction = numpy.zeros([trialsNum , statesNum * actionsNum] , float)
aveInState = numpy.zeros([trialsNum] , float)
aveExState = numpy.zeros([trialsNum , statesNum] , float)
aveReward = numpy.zeros([trialsNum] , float)
aveOutcome = numpy.zeros([trialsNum] , float)

#################################################################
#                        Main Code
#################################################################


for run in range(0,runsNum):
    Q = initializeQ()
    s = initializeExState()     # in a random way
    h = initialInState
    
    print "Run number: %d" %(run)

    trial = 0
    while (trial<trialsNum):

        a = actionSelectionSoftmax(s)
        nextS = transition(s,a)
        o = outcome(s,a)
        r = reward(h,o,energyExpenditurePerStep)

        logging(trial,h,s,a,o,r)    

        updateQ_TD0(s,a,nextS,r)            
        
        h = updateInState(h,o) - energyExpenditurePerStep

        s=nextS

        trial = trial + 1

loggingFinalization()
plotting()

