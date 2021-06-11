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
            sumEQ = sumEQ + abs(cmath.exp( (alternatives[i]) / beta ))
    
    index = numpy.random.uniform(0,sumEQ)
    probSum=0
    for i in range(0,actionsNum):
        if isActionAvailable(state,i):
            probSum = probSum + abs(cmath.exp( (alternatives[i]) / beta ))
            if probSum >= index:
                return i

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
    d1 = numpy.power(numpy.absolute(numpy.power(optimalInState-inState,n*1.0)),(1.0/m))
    d2 = numpy.power(numpy.absolute(numpy.power(optimalInState-inState-outcome+energyPerStep,n*1.0)),(1.0/m))
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
    return 0
    
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
def loggingFinalization1():

    for i in range(0,trialsNum):        
        for j in range(0,statesNum * actionsNum):
            aveQ[i][j] = aveQ[i][j] / runsNum
            aveAction[i][j] = aveAction[i][j] / runsNum
        aveInState[i] = aveInState[i] / runsNum
        aveReward[i] = aveReward[i] / runsNum
        aveOutcome[i] = aveOutcome[i] / runsNum
        for j in range(0,statesNum):
            aveExState[i][j] = aveExState[i][j] / runsNum
 
    aveQ1 = aveQ
    aveAction1 = aveAction
    for i in range(0,trialsNum):        
        aveInState1[i] = aveInState[i]
        aveOutcome1[i] = aveOutcome[i]
    aveReward1 = aveReward
    aveExState1 = aveExState
    
    return

def loggingFinalization2():
    for i in range(0,trialsNum):        
        for j in range(0,statesNum * actionsNum):
            aveQ[i][j] = aveQ[i][j] / runsNum
            aveAction[i][j] = aveAction[i][j] / runsNum
        aveInState[i] = aveInState[i] / runsNum
        aveReward[i] = aveReward[i] / runsNum
        aveOutcome[i] = aveOutcome[i] / runsNum
        for j in range(0,statesNum):
            aveExState[i][j] = aveExState[i][j] / runsNum
    
    aveQ2 = aveQ
    aveAction2 = aveAction
    for i in range(0,trialsNum):        
        aveInState2[i] = aveInState[i]
        aveOutcome2[i] = aveOutcome[i]

    aveReward2 = aveReward
    aveExState2 = aveExState
    
    return

#---------------------   Plotting some desired variables
def plotting():

    aveActionT1 = numpy.transpose (aveAction1)
    aveQT1 = numpy.transpose (aveQ1)
    aveRewardT1 = numpy.transpose (aveReward1)
    aveOutcomeT1 = numpy.transpose (aveOutcome1)
    aveInStateT1 = numpy.transpose (aveInState1)
    aveExStateT1 = numpy.transpose (aveExState1)

    aveActionT2 = numpy.transpose (aveAction2)
    aveQT2 = numpy.transpose (aveQ2)
    aveRewardT2 = numpy.transpose (aveReward2)
    aveOutcomeT2 = numpy.transpose (aveOutcome2)
    aveInStateT2 = numpy.transpose (aveInState2)
    aveExStateT2 = numpy.transpose (aveExState2)


    pylab.rc('xtick', labelsize=9)
    pylab.rc('ytick', labelsize=9)

#-----------
    fig1 = pylab.figure( figsize=(2,1.5) )
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)
    h = ax1.plot(aveInStateT1 , linewidth = 2 , color = 'green')
    h = ax1.plot(aveInStateT2 , linewidth = 2 , color = 'red')
    pylab.ylim((-55, 40))
    pylab.yticks(pylab.arange(-60, 41, 20))
    pylab.xticks(pylab.arange(0, 301, 100))
    pylab.savefig('interalStates.eps', format='eps')

#----------- plotting accumulated consumed quantity
    aveQuantityT1 = numpy.zeros([trialsNum] , float)
    aveQuantityT2 = numpy.zeros([trialsNum] , float)
    
    aveQuantityT1[0] = aveOutcomeT1[0]
    aveQuantityT2[0] = aveOutcomeT2[0]

    for i in range(1,trialsNum):        
        aveQuantityT1[i] = aveQuantityT1[i-1] + aveOutcomeT1[i]
        aveQuantityT2[i] = aveQuantityT2[i-1] + aveOutcomeT2[i]

    fig2 = pylab.figure( figsize=(2,1.5) )
    fig2.subplots_adjust(bottom=0.2)
    ax1 = fig2.add_subplot(111)
    k = ax1.plot(aveQuantityT1 , linewidth = 2 , color = 'green')
    k = ax1.plot(aveQuantityT2 , linewidth = 2 , color = 'red')
#    pylab.ylim((-55, 40))
#    pylab.yticks(pylab.arange(-60, 41, 20))
    pylab.xticks(pylab.arange(0, 301, 100))
    pylab.savefig('QuantityConsumed.eps', format='eps')
    
    print "Normal: %d" %(aveQuantityT1[299])
    print "Tasty: %d"  %(aveQuantityT2[299])
    
    pylab.show()    

    return


#==================================================================================================================================================
#======   The code starts running from here   =====================================================================================================
#==================================================================================================================================================

#################################################################
#                        Environment's Dynamics
#################################################################

# --------------------   Number of the States
statesNum  = 1

# --------------------   Number of the Actions
actionsNum = 2

# --------------------   Initial External State
#initialExState = 0

# --------------------   Final External States
#finalExStates = [6,8]

# --------------------   Transition Function : (from state s, by action a, going to state s', by probability p)
T = numpy.zeros( [statesNum * actionsNum, statesNum] , float)
setTransition(0,0,0,1)
setTransition(0,1,0,1)

# --------------------   Outcome Function : (from state s, by action a, recieving reward r)
food = 4

tasteInducedReward = 100

O = numpy.zeros (statesNum * actionsNum , float)

setOutcome(0,0,food)
setOutcome(0,1,0)

#################################################################
#                        Agent's Dynamics
#################################################################

energyExpenditurePerStep = 1

# --------------------   Initial internal State
initialInState = -50

# --------------------   Optimal internal State = Homeostatic setpoint
optimalInState = -5

# -------------------- Meta Variables
alpha = 0.7        # Learning rate
gamma = 0.8         # Discount factor

beta = 50           # Rate of exploration

m = 2
n = 4

#################################################################
#                        Simulation Initializations
#################################################################

# -------------------- Simulations Variables
trialsNum = 300
runsNum = 200

# --------------------   Q-values
Q = numpy.zeros(statesNum * actionsNum , float)

# --------------------   Average-over-runs Variables
aveQ = numpy.zeros([trialsNum , statesNum * actionsNum] , float)
aveAction = numpy.zeros([trialsNum , statesNum * actionsNum] , float)
aveInState = numpy.zeros([trialsNum] , float)
aveExState = numpy.zeros([trialsNum , statesNum] , float)
aveReward = numpy.zeros([trialsNum] , float)
aveOutcome = numpy.zeros([trialsNum] , float)


aveQ1 = numpy.zeros([trialsNum , statesNum * actionsNum] , float)
aveAction1 = numpy.zeros([trialsNum , statesNum * actionsNum] , float)
aveInState1 = numpy.zeros([trialsNum] , float)
aveExState1 = numpy.zeros([trialsNum , statesNum] , float)
aveReward1 = numpy.zeros([trialsNum] , float)
aveOutcome1 = numpy.zeros([trialsNum] , float)

aveQ2 = numpy.zeros([trialsNum , statesNum * actionsNum] , float)
aveAction2 = numpy.zeros([trialsNum , statesNum * actionsNum] , float)
aveInState2 = numpy.zeros([trialsNum] , float)
aveExState2 = numpy.zeros([trialsNum , statesNum] , float)
aveReward2 = numpy.zeros([trialsNum] , float)
aveOutcome2 = numpy.zeros([trialsNum] , float)


#################################################################
#                        Main Code
#################################################################

#===================================================================
tasteInducedReward = 0

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
        
        if a==0:
            r = r + tasteInducedReward

        logging(trial,h,s,a,o,r)    

        updateQ_TD0(s,a,nextS,r)            
        
        h = updateInState(h,o) - energyExpenditurePerStep

        s=nextS

        trial = trial + 1

loggingFinalization1()

#===================================================================
tasteInducedReward = 100

aveQ = numpy.zeros([trialsNum , statesNum * actionsNum] , float)
aveAction = numpy.zeros([trialsNum , statesNum * actionsNum] , float)
aveInState = numpy.zeros([trialsNum] , float)
aveExState = numpy.zeros([trialsNum , statesNum] , float)
aveReward = numpy.zeros([trialsNum] , float)
aveOutcome = numpy.zeros([trialsNum] , float)
aveQuantity = numpy.zeros([trialsNum] , float)


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
        
        if a==0:
            r = r + tasteInducedReward

        logging(trial,h,s,a,o,r)    

        updateQ_TD0(s,a,nextS,r)            
        
        h = updateInState(h,o) - energyExpenditurePerStep

        s=nextS

        trial = trial + 1

loggingFinalization2()

#===================================================================

plotting()

