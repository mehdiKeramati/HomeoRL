'''
-----------------------------------------------------------------------------------------------------------------------------------
--                                                                                                                               --
--                                                                                                                               --
--                                    Escalation of Cocaine-Seeking                                                              --
--                                              in the                                                                           -- 
--                             Homeostatic Reinforcement Learning Framewrok                                                      --  
--                                                                                                                               --
--                                                                                                                               --
--                                                                                                                               --
--      Programmed in : Python 2.6                                                                                               --
--      By            : Mehdi Keramati                                                                                           -- 
--      Date          : March 2013                                                                                            --
--                                                                                                                               --
-----------------------------------------------------------------------------------------------------------------------------------
'''

import scipy
import numpy
import pylab
import cmath

'''
###################################################################################################################################
###################################################################################################################################
#                                                         Functions                                                               #
###################################################################################################################################
###################################################################################################################################
'''
'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Setting the transition function of the MDP   --------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''
def setTransition(state,action,nextState,transitionProbability):
    transition [state][action][nextState] = transitionProbability
    return 

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Setting the outcome function of the MDP   -----------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''
def setOutcome(state,action,nextState,out):
    outcome [state][action][nextState] = out
    return 

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Setting the non-homeostatic reward function of the MDP   --------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''
def setNonHomeostaticReward(state,action,nextState,rew):
    nonHomeostaticReward [state][action][nextState] = rew
    return 

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Return the probability of the transitions s-a->s'  -------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def getTransition(s,a,nextS):
    return transition[s][a][nextS]

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Return the next state that the animal fell into  ---------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def getRealizedTransition(state,action):
           
    index = numpy.random.uniform(0,1)
    probSum = 0
    for nextS in range(0,statesNum):
        probSum = probSum + getTransition(state,action,nextS)
        if index <= probSum:
            return nextS    

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Obtained outcome   ---------------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def getOutcome(state,action,nextState):
    return outcome[state,action,nextState]

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Obtained non-homeostatic reward    ------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''
def getNonHomeostaticReward(state,action,nextState):
    return nonHomeostaticReward [state][action][nextState] 

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Homeostatically-regulated Reward   ------------------------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def driveReductionReward(inState,outcome):
    d1 = numpy.power(numpy.absolute(numpy.power(setpoint-inState,n*1.0)),(1.0/m))
    d2 = numpy.power(numpy.absolute(numpy.power(setpoint-inState-outcome,n*1.0)),(1.0/m))
    return d1-d2

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Create a new animal   ------------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def initializeAnimal():    
            
    animalState[0] = initialExState
    animalState[1] = initialInState
    animalState[3] = 0 
        
    for i in range(0,statesNum):
        for j in range(0,actionsNum):
            for k in range(0,statesNum):
                estimatedTransition[i][j][k] = (1.0)/(statesNum*1.0)
                estimatedOutcome[i][j][k] = 0.0
                estimatedNonHomeostaticReward[i][j][k] = 0.0
    
#    Assuming that the animals know the energy cost (fatigue) of pressing a lever 
    for i in range(0,statesNum):
        for j in range(0,statesNum):
            estimatedNonHomeostaticReward[i][1][j] = -leverPressCost
    
    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Is action a available is state s?   ----------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def isActionAvailable(state,action):
    probSum = 0 ;
    for i in range(0,statesNum):
        probSum = probSum + getTransition(state,action,i)
    if probSum == 1:
        return 1
    elif probSum == 0:
        return 0
    else:
        print "Error: There seems to be a problem in defining the transition function of the environment"        
        return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Goal-directed Value estimation   -------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def valueEstimation(state,inState,depthLeft):

    values = numpy.zeros ( [actionsNum] , float )

    # If this is the last depth that should be searched :
    if depthLeft==1:
        for action in range(0,actionsNum):
            for nextState in range(0,statesNum):
                homeoReward    = driveReductionReward(inState,estimatedOutcome[state][action][nextState])
                nonHomeoReward = estimatedNonHomeostaticReward[state][action][nextState]
                transitionProb = estimatedTransition[state][action][nextState]
                values[action] = values[action] +  transitionProb * ( homeoReward + nonHomeoReward )
        return values
    
    # Otherwise :
    for action in range(0,actionsNum):
        for nextState in range(0,statesNum):
            if estimatedTransition[state][action][nextState] < pruningThreshold :
                VNextStateBest = 0
            else:    
                VNextState = valueEstimation(nextState,inState,depthLeft-1)
                VNextStateBest = maxValue (VNextState)
            homeoReward    = driveReductionReward(inState,estimatedOutcome[state][action][nextState])
            nonHomeoReward = estimatedNonHomeostaticReward[state][action][nextState]
            transitionProb = estimatedTransition[state][action][nextState]
            values[action] = values[action] + transitionProb * ( homeoReward + nonHomeoReward + gamma*VNextStateBest ) 
            
    return values
    
'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Max ( Value[nextState,a] ) : for all a  ------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def maxValue(V):
    maxV = V[0]
    for action in range(0,actionsNum):
        if V[action]>maxV:
            maxV = V[action]    
    return maxV
    
'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Action Selection : Softmax   ------------------------------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def actionSelectionSoftmax(state,V):

    # Normalizing values, in order to be overflow due to very high values
    maxV = V[0]
    if maxV==0:
        maxV=1        
    for action in range(0,actionsNum):
        if maxV < V[action]:
            maxV = V[action]
    for action in range(0,actionsNum):
        V[action] = V[action]/maxV


    sumEV = 0
    for action in range(0,actionsNum):
        sumEV = sumEV + abs(cmath.exp( V[action] / beta ))

    index = numpy.random.uniform(0,sumEV)

    probSum=0
    for action in range(0,actionsNum):
            probSum = probSum + abs(cmath.exp( V[action] / beta ))
            if probSum >= index:
                return action

    print "Error: An unexpected (strange) problem has occured in action selection..."
    return 0
    
'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Update internal state upon consumption   ------------------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def updateInState(inState,outcome):
#    interS = inState + outcome - cocaineDegradationRate*(inState-inStateLowerBound)
    interS = inState + outcome - cocaineDegradationRate
    if interS<inStateLowerBound:
        interS=inStateLowerBound
    return interS

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Update the expected-outcome function  --------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def updateOutcomeFunction(state,action,nextState,out):
    estimatedOutcome[state][action][nextState] = (1.0-updateOutcomeRate)*estimatedOutcome[state][action][nextState] + updateOutcomeRate*out
    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Update the expected-non-homeostatic-reward function  ------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def updateNonHomeostaticRewardFunction(state,action,nextState,rew):
    estimatedNonHomeostaticReward[state][action][nextState] = (1.0-updateRewardRate)*estimatedNonHomeostaticReward[state][action][nextState] + updateRewardRate*rew
    return


'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Update the expected-transition function  ------------------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def updateTransitionFunction(state,action,nextState):

    #---- First inhibit all associations
    for i in range(0,statesNum):
        estimatedTransition[state][action][i] = (1.0-updateTransitionRate)*estimatedTransition[state][action][i]
    
    #---- Then potentiate the experiences association
    estimatedTransition[state][action][nextState] = estimatedTransition[state][action][nextState] + updateTransitionRate
            
    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Cocaine Seeking Sessions  --------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def cocaineSeeking  (sessionNum ):

    exState     = animalState[0]
    inState     = animalState[1]
    setpointS   = animalState[2]
    trialCount  = animalState[3]
    cocBuffer   = 0
        
    setOutcome(0,1,0,cocaine)       # At state s, by doing action a and going to state s', we receive the outcome 
    for trial in range(trialCount,trialCount+seekingTrialsNumShA):
        
        estimatedActionValues   = valueEstimation(exState,inState,searchDepth)
        valueLP                 = estimatedActionValues[1]
        action                  = actionSelectionSoftmax(exState,estimatedActionValues)
        nextState               = getRealizedTransition(exState,action)
        out                     = getOutcome(exState,action,nextState)
        nonHomeoRew             = getNonHomeostaticReward(exState,action,nextState)
        HomeoRew                = driveReductionReward(inState,out)
        
        logging(trial,action,inState,out,valueLP)    
        print "Rat number: %d / %d     Session Number: %d / %d     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,sessionNum+1,sessionsNum,trial-trialCount+1,seekingTrialsNumShA)

        updateOutcomeFunction(exState,action,nextState,out)
        updateNonHomeostaticRewardFunction(exState,action,nextState,nonHomeoRew)
        updateTransitionFunction(exState,action,nextState)            
        
        inState     = updateInState(inState,out)
        exState   = nextState

    animalState[0]    = exState
    animalState[1]    = inState
    animalState[3]    = trialCount+seekingTrialsNumShA

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Extinction Sessions  -------------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def extinction  ( sessionNum ):

    exState     = animalState[0]
    inState     = animalState[1]
    trialCount  = animalState[3]
    cocBuffer   = 0
        
    setOutcome(0,1,0,0)       # At state s, by doing action a and going to state s', we receive the outcome 

    for trial in range(trialCount,trialCount+extinctionTrialsNum):

        estimatedActionValues   = valueEstimation(exState,inState,searchDepth)
        valueLP                 = estimatedActionValues[1]
        action                  = actionSelectionSoftmax(exState,estimatedActionValues)
        nextState               = getRealizedTransition(exState,action)
        out                     = getOutcome(exState,action,nextState)
        nonHomeoRew             = getNonHomeostaticReward(exState,action,nextState)
        HomeoRew                = driveReductionReward(inState,out)

        logging(trial,action,inState,out,valueLP)    
        print "Rat number: %d / %d     Session Number: %d / %d     trial: %d / %d      Extinction" %(animal+1,animalsNum,sessionNum+1,sessionsNum,trial-trialCount+1,extinctionTrialsNum)

        updateOutcomeFunction(exState,action,nextState,out)
        updateNonHomeostaticRewardFunction(exState,action,nextState,nonHomeoRew)
        updateTransitionFunction(exState,action,nextState)            
        
        inState     = updateInState(inState,out)
        exState   = nextState

    animalState[0]    = exState
    animalState[1]    = inState
    animalState[3]    = trialCount+extinctionTrialsNum

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Logging the current information for the Short-access group  ----------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def logging(trial,action,inState,coca,valueLP):
   
    if action==0: 
        nulDoing[trial]              = nulDoing[trial] + 1
    elif action==1: 
        leverPress[trial]            = leverPress[trial] + 1
    internalState[trial]             = internalState[trial] + inState
    outcomeExpectency [trial]        = outcomeExpectency [trial] + estimatedOutcome[0][1][0]
    leverPressValue [trial]          = leverPressValue [trial] + valueLP
    
    return


'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Wrap up all the logged data   ----------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def loggingFinalization():
    
    for trial in range(0,totalTrialsNum):
        nulDoing[trial]             = nulDoing[trial]/animalsNum
        leverPress[trial]           = leverPress[trial]/animalsNum
        internalState[trial]        = internalState[trial]/animalsNum
        outcomeExpectency [trial]   = outcomeExpectency [trial]/animalsNum
        leverPressValue [trial]     = leverPressValue [trial]/animalsNum

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the internal state of the last session  -------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotInternalState():

    font = {'family' : 'normal', 'size'   : 16}
    pylab.rc('font', **font)
    pylab.rcParams.update({'legend.fontsize': 16})
        
    fig1 = pylab.figure( figsize=(8,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)


    ax1 = fig1.add_subplot(111)
    ax1.axhline(100,  color='0.25',ls='--', lw=1 )
    S0 = ax1.plot(internalState , linewidth = 2.5 , color='black' )


    pylab.ylim((-10 , 150))
    pylab.yticks(pylab.arange(0, 151, 25))
    pylab.xlim((-50,trialsPerDay+50))

    p = pylab.axvspan( 8*60*15 , trialsPerDay+50 , facecolor='0.75',edgecolor='none', alpha=0.5)        
 
    tick_lcs = []
    tick_lbs = []
    for i in range ( 0 , 14 ):
        tick_lcs.append( i*60*15 ) 
        tick_lbs.append(i)
    pylab.xticks(tick_lcs, tick_lbs)

    for line in ax1.get_yticklines() + ax1.get_xticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

 
    ax1.set_ylabel('Internal State')
    ax1.set_xlabel('Time (hour)')
    fig1.savefig('internalState.eps', format='eps')

    return
'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the internal state of the last session  -------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotInternalStateFocused():

    font = {'family' : 'normal', 'size'   : 16}
    pylab.rc('font', **font)
    pylab.rcParams.update({'legend.fontsize': 16})
        
    fig1 = pylab.figure( figsize=(8,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)


    ax1 = fig1.add_subplot(111)
    ax1.axhline(100,  color='0.25',ls='--', lw=1 )
    S0 = ax1.plot(internalState[8*60*15 - 100 : 8*60*15 + 100 ] , linewidth = 2.5 , color='black' )


    for line in ax1.get_yticklines() + ax1.get_xticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

 
    ax1.set_ylabel('Lever\npress')
    ax1.set_xlabel('Time (hour)')
    fig1.savefig('internalStateFOCUSED.eps', format='eps')

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the Outcome Expectancy                  -------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotOutcomeExpectency():

    font = {'family' : 'normal', 'size'   : 16}
    pylab.rc('font', **font)
    pylab.rcParams.update({'legend.fontsize': 16})
        
    fig1 = pylab.figure( figsize=(8,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)


    ax1 = fig1.add_subplot(111)
    ax1.axhline(5,  color='0.25',ls='--', lw=1 )
    S0 = ax1.plot(outcomeExpectency , linewidth = 2.5 , color='black' )


    pylab.ylim((0 , 6))
#    pylab.yticks(pylab.arange(0, 7, 1))
    pylab.xlim((-50,trialsPerDay+50))

    p = pylab.axvspan( 8*60*15 , trialsPerDay+50 , facecolor='0.75',edgecolor='none', alpha=0.5)        
 
    tick_lcs = []
    tick_lbs = []
    for i in range ( 0 , 14 ):
        tick_lcs.append( i*60*15 ) 
        tick_lbs.append(i)
    pylab.xticks(tick_lcs, tick_lbs)

    tick_lcs = []
    tick_lbs = []
    for i in range ( 0 , 6 ):
        tick_lcs.append( i ) 
        tick_lbs.append(i*0.2)
    pylab.yticks(tick_lcs, tick_lbs)


    for line in ax1.get_yticklines() + ax1.get_xticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

 
    ax1.set_ylabel('Outcome Expectancy')
    ax1.set_xlabel('Time (hour)')
    fig1.savefig('Outcome Expectancy.eps', format='eps')

    return


'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the infusions for the last session ------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotLeverPress():

        
    fig1 = pylab.figure( figsize=(8,2) )
#    fig1.subplots_adjust(top=0.65)
    fig1.subplots_adjust(bottom=0.3)
    fig1.subplots_adjust(left=0.16)

    ax1 = fig1.add_subplot(111)
    S0 = ax1.plot(leverPress , linewidth = 2 , color='black' )
    
#    pylab.yticks(pylab.arange(0, 1.01, 0.2))
    pylab.ylim((0,2.5))
    pylab.xlim((-50,trialsPerDay+50))
 
    tick_lcs = []
    tick_lbs = []
    pylab.yticks(tick_lcs, tick_lbs)
    for i in range ( 0 , 7 ):
        tick_lcs.append( 50 + i*300 ) 
        tick_lbs.append(i*20)
#    pylab.xticks(tick_lcs, tick_lbs)

    for line in ax1.get_yticklines() + ax1.get_xticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    ax1.set_ylabel('Infusion')
    ax1.set_xlabel('Time (min)')
    fig1.savefig('leverPress.eps', format='eps')

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the infusions per 10 minutes for the Short-Access group ---------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotLeverPressRate():
    
    trialsPerBlock = 20*(60/4)  # => every block is five minutes
    numberOfBlocks = trialsPerDay / trialsPerBlock
    
    leverPressRate  = numpy.zeros( [numberOfBlocks] , float)    

    for i in range(0,numberOfBlocks):
        for j in range(i*trialsPerBlock,  + (i+1)*trialsPerBlock ):
            leverPressRate[i] = leverPressRate[i] + leverPress[j]

    fig1 = pylab.figure( figsize=(8,3.5) )
    fig1.subplots_adjust(bottom=0.2)
    fig1.subplots_adjust(left=0.16)
    ax1 = fig1.add_subplot(111)
    S1 = ax1.plot(leverPressRate , '-o', ms=8, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )

    pylab.yticks(pylab.arange(20, 121, 20))
    pylab.ylim((10,130))
    pylab.xlim((-1,39))
    
    p = pylab.axvspan( 23.5 , 39 , facecolor='0.75',edgecolor='none', alpha=0.5)        


    tick_lcs = []
    tick_lbs = []
    for i in range ( 0 , 14 ):
        tick_lcs.append( i*3 ) 
        tick_lbs.append(i)
    pylab.xticks(tick_lcs, tick_lbs)

    
    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    ax1.set_ylabel('Lever Press / 20 min')
    ax1.set_xlabel('Time (hour)')
    fig1.savefig('LeverPressRate.eps', format='eps')

    return


'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the internal state of the last session  -------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotLeverPressValue():

    font = {'family' : 'normal', 'size'   : 16}
    pylab.rc('font', **font)
    pylab.rcParams.update({'legend.fontsize': 16})
        
    fig1 = pylab.figure( figsize=(8,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)


    ax1 = fig1.add_subplot(111)
    S0 = ax1.plot(leverPressValue , linewidth = 2.5 , color='black' )


#    pylab.ylim((-10 , 150))
#    pylab.yticks(pylab.arange(0, 151, 25))
    pylab.xlim((-50,trialsPerDay+50))
 
    tick_lcs = []
    tick_lbs = []
    for i in range ( 0 , 7 ):
        tick_lcs.append( 50 + i*300 ) 
        tick_lbs.append(i*20)
#    pylab.xticks(tick_lcs, tick_lbs)

    for line in ax1.get_yticklines() + ax1.get_xticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

 
    ax1.set_ylabel('Lever Press Value')
    ax1.set_xlabel('Time (min)')
    fig1.savefig('LeverPressValue.eps', format='eps')

    return


'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the inter-nfusion intervals for the last session of the Long-Access group ---------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotInterInfusionIntervals():
 
#--------------------------------------- Compute III For Short-Access
    iiiShA  = []   # inter-infusion intervals
    
    for j in range(trialsPerDay + (sessionsNum-1)*(trialsPerDay),trialsPerDay + (sessionsNum-1)*(trialsPerDay)+seekingTrialsNumShA):
        if infusionShA[j]==1:
            previousInfTime = j
            break

    for j in range( j+1 , trialsPerDay + (sessionsNum-1)*(trialsPerDay)+seekingTrialsNumShA):
        if infusionShA[j]==1:
            interInf = (j - previousInfTime) * 4        # x*4 , because every trial is 4 seconds
            iiiShA.append(interInf)
            previousInfTime = j

    infusionsNumShA = len(iiiShA)
    xShA = numpy.arange(1, infusionsNumShA+1, 1)
           
    
    iiimax = 0
    for j in range( 0 , 10 ):
        if iiimax<iiiShA[j]:
            iiimax = iiiShA[j]

            
    fig1 = pylab.figure( figsize=(8,3.5) )
    fig1.subplots_adjust(bottom=0.2)
    fig1.subplots_adjust(left=0.16)
    ax1 = fig1.add_subplot(111)
    S0 = ax1.plot(xShA,iiiShA[0:infusionsNumShA], '-o', ms=5, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )
        
    pylab.ylim((-20,iiimax+200))
    pylab.yticks(pylab.arange(0, 601, 200))
    pylab.xlim((0,41))
    
    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    ax1.set_ylabel('Inter-infusion interval (sec)')
    ax1.set_xlabel('Infusion number')
    fig1.savefig('interInfusionIntervals.eps', format='eps')

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot all the results  ------------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotting():

    loggingFinalization()

    plotInternalState()
    plotInternalStateFocused()
    plotLeverPress()
    plotLeverPressRate()
    plotOutcomeExpectency()
    plotLeverPressValue()

#    plotInterInfusionIntervals()
    
    pylab.show()   
    
    return

'''
###################################################################################################################################
###################################################################################################################################
#                                                             Main                                                                #
###################################################################################################################################
###################################################################################################################################
'''
'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Definition of the Markov Decison Process FR1 - Timeout 20sec  ---------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

cocaine         = 5             # Dose of self-administered drug
leverPressCost  = 5              # Energy cost for pressing the lever

statesNum       = 1              # number of stater 
actionsNum      = 2              # number of action   action 0 = Null     action 1 = Lever Press    action 2 = Enter Magazine
initialExState  = 0

transition = numpy.zeros( [statesNum , actionsNum, statesNum] , float)
setTransition(0,0,0,1)          # From state s, and by taking a, we go to state s', with probability p
setTransition(0,1,0,1)

outcome = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
setOutcome(0,1,0,cocaine)       # At state s, by doing action a and going to state s', we receive the outcome 

nonHomeostaticReward = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
setNonHomeostaticReward(0,1,0,-leverPressCost)

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Definition of the Animal   --------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

#------------ Homeostatic System
initialInState          = 0
setpoint                = 105
inStateLowerBound       = 0
cocaineDegradationRate  = 1.5    # Dose of cocaine that the animal loses in every time-step

#------------ Drive Function
m                       = 2     # Parameter of the drive function : m-th root
n                       = 2     # Parameter of the drive function : n-th pawer

#------------ Goal-directed system
updateOutcomeRate       = 0.0025  # Learning rate for updating the outcome function
updateTransitionRate    = 0.5  # Learning rate for updating the transition function
updateRewardRate        = 0.5  # Learning rate for updating the non-homeostatic reward function
gamma                   = 0.85     # Discount factor
beta                    = 2     # Rate of exploration
searchDepth             = 1     # Depth of going into the decision tree for goal-directed valuation of choices
pruningThreshold        = 0.1   # If the probability of a transition like (s,a,s') is less than "pruningThreshold", cut it from the decision tree 

estimatedTransition              = numpy.zeros( [statesNum , actionsNum, statesNum] , float)
estimatedOutcome                 = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
estimatedNonHomeostaticReward    = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )

animalState                            = numpy.zeros ( [4] , float )     # a vector of the external state, internal state, setpoint, and trial

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Simulation Parameters   -----------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

animalsNum          = 20                                  # Number of animals

sessionsNum         = 1                                  # Number of sessions of cocain seeking, followed by rest in home-cage
seekingHoursShA     = 8            
extinctionHours     = 5

trialsPerHour       = 60*60/4                            # Number of trials during one hour (as each trial is supposed to be 4 seconds)
trialsPerDay        = (seekingHoursShA+extinctionHours)*trialsPerHour

seekingTrialsNumShA = seekingHoursShA * trialsPerHour    # Number of trials for each cocaine seeking session
extinctionTrialsNum = extinctionHours*trialsPerHour      # Number of trials for each extinction session

totalTrialsNum      = trialsPerDay

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Logging Parameters   --------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

nulDoing            = numpy.zeros( [totalTrialsNum] , float)
leverPress          = numpy.zeros( [totalTrialsNum] , float)
internalState       = numpy.zeros( [totalTrialsNum] , float)
outcomeExpectency   = numpy.zeros( [totalTrialsNum] , float)
leverPressValue     = numpy.zeros( [totalTrialsNum] , float)

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Simulation   ----------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

for animal in range(0,animalsNum):    
    
    initializeAnimal          (                         )
    for session in range(0,sessionsNum):

        cocaineDegradationRate  = 1.5
        cocaineSeeking        (  session )

        cocaineDegradationRate  = 0.05
        extinction            (  session )

plotting()

