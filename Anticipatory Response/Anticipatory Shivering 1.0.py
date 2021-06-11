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
def outcome(state,action,h):
    outcome = O[state*actionsNum + action]
    if (state == 0) or (state==1): 
        adj = normalPlaceTemperature-h
    if (state == 2) or (state==3): 
        adj = coldPlaceTemperature-h
    outcome = outcome + tempAdjustmentRatio * adj
    return outcome

#---------------------   Reward Function: returns the reward of taking an action [action] in an state [state]
def reward(inState,outcome):
    d1 = numpy.power(numpy.absolute(numpy.power(optimalInState-inState,n*1.0)),(1.0/m))
    d2 = numpy.power(numpy.absolute(numpy.power(optimalInState-inState-outcome,n*1.0)),(1.0/m))
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
     
#---------------------   Logging
def logging(tS1,tS0,tTotal,inState,exState,action,outcome,reward):
    aveQ[tTotal] = aveQ[tTotal] + Q
    aveInState[tTotal] = aveInState[tTotal] + inState
    aveDeviation[tTotal] = aveDeviation[tTotal] + numpy.abs(inState-optimalInState)
    aveReward[tTotal] = aveReward[tTotal] + reward
    aveOutcome[tTotal] = aveOutcome[tTotal] + outcome
    if (exState==0):
        aveAction[tS0][exState* actionsNum + action ] = aveAction[tS0][exState* actionsNum + action ] + 1
    if (exState==1) or (exState==2) or (exState==3):
        aveAction[tS1][exState* actionsNum + action ] = aveAction[tS1][exState* actionsNum + action ] + 1
#    aveExState[trial][exState] = aveExState[trial][exState] + 1
    return 

#---------------------   Averaging the added up variables ove the total number of runs
def loggingFinalization():

    for i in range(0,trialsNumTotal):        
            aveAction[i][4] = aveAction[i][4] + aveAction[i][6] 
            aveAction[i][5] = aveAction[i][5] + aveAction[i][7] 
    
    for i in range(0,trialsNumTotal):        
        for j in range(0,statesNum * actionsNum):
            aveQ[i][j] = aveQ[i][j] / runsNum
            aveAction[i][j] = aveAction[i][j] / runsNum
        aveInState[i] = aveInState[i] / runsNum
        aveDeviation[i] = aveDeviation[i] / runsNum
        aveReward[i] = aveReward[i] / runsNum
        aveOutcome[i] = aveOutcome[i] / runsNum

        aveAction[trialsNumS0+1][0] = aveAction[trialsNumS0][0]
        aveAction[trialsNumS0+1][1] = aveAction[trialsNumS0][1]
        aveAction[trialsNumS1+1][2] = aveAction[trialsNumS1][2]
        aveAction[trialsNumS1+1][3] = aveAction[trialsNumS1][3]
        aveAction[trialsNumS1+1][4] = aveAction[trialsNumS1][4]
        aveAction[trialsNumS1+1][5] = aveAction[trialsNumS1][5]
        aveInState[trialsNumTotal] = aveInState[trialsNumTotal-1]
        aveInState[trialsNumTotal+1] = aveInState[trialsNumTotal]
        
                
#        for j in range(0,statesNum):
#            aveExState[i][j] = aveExState[i][j] / runsNum
    return

#---------------------   Plotting some desired variables
def plotting():

    aveActionT = numpy.transpose (aveAction)
    aveQT = numpy.transpose (aveQ)
    aveRewardT = numpy.transpose (aveReward)
    aveOutcomeT = numpy.transpose (aveOutcome)
    aveInStateT = numpy.transpose (aveInState)
    aveDeviationT = numpy.transpose (aveDeviation)
    aveExStateT = numpy.transpose (aveExState)


    pylab.rc('xtick', labelsize=9)
    pylab.rc('ytick', labelsize=9)

#-----------
    fig1 = pylab.figure( figsize=(2,1.5) )
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)
    S0 = ax1.plot(aveActionT[0] , linewidth = 2 , color='red' )
    S1 = ax1.plot(aveActionT[1] , linewidth = 2 , color='blue' )
    pylab.yticks(pylab.arange(0, 1.01, 0.2))
    pylab.ylim((-0.03,1.03))
    pylab.xlim((-10,trialsNumS0-1))
    pylab.xticks(pylab.arange(0, trialsNumS0+1, 250))
    for line in ax1.yaxis.get_ticklines():
        line.set_markeredgewidth(1)
#        line.set_color('green')
#        line.set_markersize(25)
#    pylab.set_ylabel('volts')
#    pylab.set_xlabel('volts X')
#    pylab.set_title('a sine wave')
    fig1.savefig('s0-actions.eps', format='eps')

#-----------
    fig1 = pylab.figure( figsize=(2,1.5) )
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)
    S0 = ax1.plot(aveActionT[2] , linewidth = 2 , color='red' )
    S1 = ax1.plot(aveActionT[3] , linewidth = 2 , color='blue' )
    pylab.yticks(pylab.arange(0, 1.01, 0.2))
    pylab.ylim((-0.03,1.03))
    pylab.xlim((-1,trialsNumS1-1))
    pylab.xticks(pylab.arange(0, trialsNumS1+1, 25))
    for line in ax1.yaxis.get_ticklines():
        line.set_markeredgewidth(1)
#        line.set_color('green')
#        line.set_markersize(25)
#    pylab.set_ylabel('volts')
#    pylab.set_xlabel('volts X')
#    pylab.set_title('a sine wave')
    fig1.savefig('s1-actions.eps', format='eps')

#-----------
    fig1 = pylab.figure( figsize=(2,1.5) )
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)
    S0 = ax1.plot(aveActionT[4] , linewidth = 2 , color='red' )
    S1 = ax1.plot(aveActionT[5] , linewidth = 2 , color='blue' )
    pylab.yticks(pylab.arange(0, 1.01, 0.2))
    pylab.ylim((-0.03,1.03))
    pylab.xlim((-1,trialsNumS1-1))
    pylab.xticks(pylab.arange(0, trialsNumS1+1, 25))
    for line in ax1.yaxis.get_ticklines():
        line.set_markeredgewidth(1)
#        line.set_color('green')
#        line.set_markersize(25)
#    pylab.set_ylabel('volts')
#    pylab.set_xlabel('volts X')
#    pylab.set_title('a sine wave')
    fig1.savefig('s2-actions.eps', format='eps')
#----------
    fig1 = pylab.figure( figsize=(2,1.5) )
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)
    h = ax1.plot(aveInStateT , linewidth = 2 , color = 'black')
    pylab.ylim((31,41))
    pylab.yticks(pylab.arange(31, 41.1, 2))
    pylab.xlim((-1,trialsNumTotal-1))
    pylab.xticks(pylab.arange(0, trialsNumTotal+1, 300))
    pylab.savefig('interalState.eps', format='eps')    
#----------
    fig1 = pylab.figure( figsize=(2,1.5) )
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)
    h = ax1.plot(aveDeviationT , linewidth = 2 , color = 'black')
#    pylab.ylim((31,41))
#    pylab.yticks(pylab.arange(31, 41.1, 2))
    pylab.xlim((-1,trialsNumTotal-1))
    pylab.xticks(pylab.arange(0, trialsNumTotal+1, 300))
    pylab.savefig('deviation.eps', format='eps')

    pylab.show()    
    
    return 
"""

    pylab.rc('xtick', labelsize=9)
    pylab.rc('ytick', labelsize=9)


    line = numpy.zeros([trialsNumTotal] , float)
    for i in range(0,trialsNumTotal):        
         line[i] = optimalInState 

    fig1 = pylab.figure( figsize=(2,1.5) )
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)
    l = ax1.plot(line , linewidth = 1.5 , color = 'green', linestyle = 'dashed' )
    h = ax1.plot(aveInStateT , linewidth = 2 , color = 'black')
    pylab.ylim((20,45))
    pylab.yticks(pylab.arange(20, 45.1, 5))
    pylab.xlim((50,100))
#    pylab.xticks(pylab.arange(0, trialsNumTotal+1, 300))
    pylab.savefig('interalEarly2.eps', format='eps')

    fig1 = pylab.figure( figsize=(2,1.5) )
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)
    l = ax1.plot(line , linewidth = 1.5 , color = 'green', linestyle = 'dashed' )
    h = ax1.plot(aveInStateT , linewidth = 2 , color = 'black')
    pylab.ylim((20,45))
    pylab.yticks(pylab.arange(20, 45.1, 5))
    pylab.xlim((900,950))
#    pylab.xticks(pylab.arange(0, trialsNumTotal+1, 300))
    pylab.savefig('interalLate2.eps', format='eps')


    pylab.show()   


    return  

""" 
   
"""
    pylab.figure()
    Q0 = pylab.plot(aveQT[0])
    Q1 = pylab.plot(aveQT[1])
    Q2 = pylab.plot(aveQT[2])
    Q3 = pylab.plot(aveQT[3])
    Q4 = pylab.plot(aveQT[4])
    Q5 = pylab.plot(aveQT[5])
    Q6 = pylab.plot(aveQT[6])
    Q7 = pylab.plot(aveQT[7])
    pylab.legend((Q0,Q1,Q2,Q3,Q4,Q5,Q6,Q7), ('Q(s0,S)','Q(s0,NS)','Q(s1,S)','Q(s1,NS)','Q(s2,S)','Q(s2,NS)','Q(s3,S)','Q(s3,NS)'))
    pylab.show()  
"""


#==================================================================================================================================================
#======   The code starts running from here   =====================================================================================================
#==================================================================================================================================================

#################################################################
#                        Environment's Dynamics
#################################################################

heatFromShivering = 6

coldProbability = 0.10

normalPlaceTemperature  = 37

coldPlaceTemperature = -20

# --------------------   Number of the States
statesNum  = 4

# --------------------   Number of the Actions
actionsNum = 2

# --------------------   Initial External State
initialExState = 0

# --------------------   Final External States
#finalExStates = [6,8]

# --------------------   Transition Function : (from state s, by action a, going to state s', by probability p)
T = numpy.zeros( [statesNum * actionsNum, statesNum] , float)
setTransition(0,0,0,1-coldProbability)
setTransition(0,0,1,coldProbability)
setTransition(0,1,0,1-coldProbability)
setTransition(0,1,1,coldProbability)
setTransition(1,0,2,1)
setTransition(1,1,3,1)
setTransition(2,0,0,1)
setTransition(2,1,0,1)
setTransition(3,0,0,1)
setTransition(3,1,0,1)

# --------------------   Outcome Function : (from state s, by action a, recieving reward r)
O = numpy.zeros (statesNum * actionsNum , float )

setOutcome(0,0,heatFromShivering)
setOutcome(1,0,heatFromShivering)
setOutcome(2,0,heatFromShivering)
setOutcome(3,0,heatFromShivering)

#################################################################
#                        Agent's Dynamics
#################################################################

# --------------------    internal State
initialInState = 37

optimalInState = 37

# -------------------- Meta Variables
alpha = 0.25         # Learning rate
gamma = 0.8         # Discount factor

beta = 5           # Rate of exploration

tempAdjustmentRatio = 0.3

m = 2
n = 4



#################################################################
#                        Simulation Initializations
#################################################################

# -------------------- Simulations Variables

trialsNumS1 = 100
trialsNumS0 = trialsNumS1 * (1.0/coldProbability)
trialsNumTotal = trialsNumS1 * (1.0/coldProbability) + 2*trialsNumS1
trialsNumBig = ( trialsNumS1 * (1.0/coldProbability) + 2*trialsNumS1 ) * 2

runsNum = 20


# --------------------   Q-values
Q = numpy.zeros(statesNum * actionsNum , float)

# --------------------   Average-over-runs Variables
aveQ = numpy.zeros([trialsNumBig , statesNum * actionsNum] , float)
aveAction = numpy.zeros([trialsNumBig , statesNum * actionsNum] , float)
aveInState = numpy.zeros([trialsNumBig] , float)
aveDeviation = numpy.zeros([trialsNumBig] , float)
aveExState = numpy.zeros([trialsNumBig , statesNum] , float)
aveReward = numpy.zeros([trialsNumBig] , float)
aveOutcome = numpy.zeros([trialsNumBig] , float)

#################################################################
#                        Main Code
#################################################################


for run in range(0,runsNum):
    Q = initializeQ()
    s = initialExState
    h = initialInState
#    Q[0]=-50
#    Q[2]=-50
#    Q[4]=-50
#    Q[6]=-50
    print "Run number: %d" %(run)

    trialsS1 = 0
    trialsS0 = 0
    trialsTotal = 0
    
    while (trialsS1<=trialsNumS1) or (trialsS0<=trialsNumS0) or (trialsTotal<=trialsNumTotal):

        a = actionSelectionSoftmax(s)
        nextS = transition(s,a)
        o = outcome(s,a,h)
        r = reward(h,o)

        logging(trialsS1,trialsS0,trialsTotal,h,s,a,o,r)    

        updateQ_TD0(s,a,nextS,r)            
        
        h = updateInState(h,o)

        if (s==2) or (s==3):
            trialsS1 = trialsS1 + 1

        if (s==0):
            trialsS0 = trialsS0 + 1

        trialsTotal = trialsTotal + 1

        s=nextS



loggingFinalization()
plotting()

