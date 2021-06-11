
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
def driveReductionReward(inState,setpointS,outcome):
    d1 = numpy.power(numpy.absolute(numpy.power(setpointS-inState,n*1.0)),(1.0/m))
    d2 = numpy.power(numpy.absolute(numpy.power(setpointS-inState-outcome,n*1.0)),(1.0/m))
    return d1-d2

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Create a new animal   ------------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def initializeAnimal():    
            
    state[0] = initialExState
    state[1] = initialInState
    state[2] = setpoint 
        
    for i in range(0,statesNum):
        for j in range(0,actionsNum):
            for k in range(0,statesNum):
                estimatedTransition            [i][j][k] = 0.0
                estimatedOutcome               [i][j][k] = 0.0
                estimatedNonHomeostaticReward  [i][j][k] = 0.0
    
#    Assuming that the animals know the energy cost (fatigue) of pressing a lever 
    for i in range(0,statesNum):
        for j in range(0,statesNum):
            estimatedNonHomeostaticReward      [i][1][j] = -leverPressCost
            estimatedNonHomeostaticReward      [i][2][j] = -leverPressCost
    
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
---------------------------------   Goal-directed Value estimation, Assuming the animal is under Cocaine  ------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def valueEstimation(state,inState,setpointS,depthLeft):

    values = numpy.zeros ( [actionsNum] , float )

    # If this is the last depth that should be searched :
    if depthLeft==1:
        for action in range(0,actionsNum):
            for nextState in range(0,statesNum):
                homeoReward    = driveReductionReward(inState,setpointS,water)*estimatedOutcome[state][action][nextState]/water
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
                VNextState = valueEstimation(nextState,setpointS,inState,depthLeft-1)
                VNextStateBest = maxValue (VNextState)
            homeoReward    = driveReductionReward(inState,setpointS,cocaine)*estimatedOutcome[state][action][nextState]/cocaine
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

    interS = inState + outcome - waterLosePerTrial
    
    return interS


'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Update the expected-outcome function  --------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def updateOutcomeFunction(state,action,nextState,out):

    estimatedOutcome [state][action][nextState] = (1.0-updateOutcomeRate)*estimatedOutcome[state][action][nextState] +     updateOutcomeRate*out
    
    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Update the expected-non-homeostatic-reward function  ------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def updateNonHomeostaticRewardFunction(state,action,nextState,rew):
        
    estimatedNonHomeostaticReward [state][action][nextState] = (1.0-updateRewardRate)*estimatedNonHomeostaticReward[state][action][nextState] +     updateRewardRate*rew
    
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
---------------------------------   Water  Seeking  Sessions  --------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def seeking (sessionNum):

    exState     = state[0]
    inState     = state[1]
    setpointS   = state[2]
    trialCount  = state[3]
    
    for trial in range(trialCount,trialCount+seekingTrialsNum):

        estimatedActionValues           = valueEstimation           ( exState, inState, setpointS, searchDepth  )
        action                          = actionSelectionSoftmax    ( exState , estimatedActionValues           )
        nextState                       = getRealizedTransition     ( exState , action                          )
        out                             = getOutcome                ( exState , action    , nextState           )
        nonHomeoRew                     = getNonHomeostaticReward   ( exState , action    , nextState           )
        HomeoRew                        = driveReductionReward      ( inState , setpointS , out                 )

        logging(trial,action,inState,out)    
        print "Animal number: %d / %d     Session Number: %d / %d     trial: %d / %d      animal seeking water" %(animal+1,animalsNum,sessionNum+1,sessionsNum,trial-trialCount+1,seekingTrialsNum)

        if sessionNum == 3:
            updateOutcomeFunction           ( exState , action , nextState , 0            )
        else:
            updateOutcomeFunction           ( exState , action , nextState , out          )   
                     
        updateNonHomeostaticRewardFunction  ( exState , action , nextState , nonHomeoRew  )
        
        updateTransitionFunction            ( exState , action , nextState                )            
        
        inState     = updateInState         ( inState , out                               )

        exState   = nextState

    state[0]    = exState
    state[1]    = inState
    state[2]    = setpointS
    state[3]    = trialCount+seekingTrialsNum

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Logging the current information for the Short-access group  ----------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def logging(trial,action,inState,outy):
   
    if action==0: 
        nulDoing[trial]             = nulDoing[trial] + 1
    elif action==1: 
        greenLeverPress[trial]     = greenLeverPress[trial] + 1
    elif action==2: 
        yellowLeverPress[trial]     = yellowLeverPress[trial] + 1
        
    internalStateLog[trial]         = internalStateLog[trial] + inState

    estimatedOutcomeOnGreenLog [trial]     = estimatedOutcomeOnGreenLog [trial] + estimatedOutcome[0][1][0]
    estimatedOutcomeOnYellowLog [trial]     = estimatedOutcomeOnYellowLog [trial] + estimatedOutcome[0][2][0]

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Wrap up all the logged data   ----------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def loggingFinalization():
    
    for trial in range(0,totalTrialsNum):
        
        nulDoing[trial]                = nulDoing[trial]/animalsNum
        greenLeverPress[trial]         = greenLeverPress[trial]/animalsNum
        yellowLeverPress[trial]        = yellowLeverPress[trial]/animalsNum
        internalStateLog[trial]        = internalStateLog[trial]/animalsNum
        estimatedOutcomeOnGreenLog[trial]     = estimatedOutcomeOnGreenLog[trial]/animalsNum
        estimatedOutcomeOnYellowLog[trial]    = estimatedOutcomeOnYellowLog[trial]/animalsNum

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the internal state  ---------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotInternalStateOral():
 
    font = {'family' : 'normal', 'size'   : 16}
    pylab.rc('font', **font)
    pylab.rcParams.update({'legend.fontsize': 16})
        
    fig1 = pylab.figure( figsize=(5,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)

    ax1 = fig1.add_subplot(111)
    S0 = ax1.plot(internalStateLog [totalTrialsNum/4 : totalTrialsNum/2] , linewidth = 2 , color='black' )

    ax1.axhline(setpoint, color='0.25',ls='--', lw=1 )

    pylab.yticks(pylab.arange(0, 51, 10))
    pylab.ylim((-10,60))
    pylab.xlim( ( -50  , totalTrialsNum/4 +50 ) )
    
    tick_lcs = []
    tick_lbs = []
    for i in range ( 0 , 9 ):
        tick_lcs.append( i*150 ) 
        tick_lbs.append(i*5)
    pylab.xticks(tick_lcs, tick_lbs)

    for i in range ( 0 , 8 ):
        if i%2==0:
            p = pylab.axvspan( i*150 + i, (i+1)*150 + i , facecolor='0.75',edgecolor='none', alpha=0.5)        

    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    ax1.set_ylabel('Internal State')
    ax1.set_xlabel('Time (min)')
    ax1.set_title('Oral reward')
    fig1.savefig('internalStateOral.eps', format='eps')

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the internal state  ---------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotInternalStateFistula():
 
    font = {'family' : 'normal', 'size'   : 16}
    pylab.rc('font', **font)
    pylab.rcParams.update({'legend.fontsize': 16})
        
    fig1 = pylab.figure( figsize=(5,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)

    ax1 = fig1.add_subplot(111)
    S0 = ax1.plot(internalStateLog [totalTrialsNum*3/4 : totalTrialsNum] , linewidth = 2 , color='black' )

    ax1.axhline(setpoint, color='0.25',ls='--', lw=1 )

    pylab.yticks(pylab.arange(0, 51, 10))
    pylab.ylim((-10,60))
    pylab.xlim( ( -50  , totalTrialsNum/4 +50 ) )
    
    tick_lcs = []
    tick_lbs = []
    for i in range ( 0 , 9 ):
        tick_lcs.append( i*150 ) 
        tick_lbs.append(i*5)
    pylab.xticks(tick_lcs, tick_lbs)

    for i in range ( 0 , 8 ):
        if i%2==0:
            p = pylab.axvspan( i*150 + i, (i+1)*150 + i , facecolor='0.75',edgecolor='none', alpha=0.5)        

    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    ax1.set_ylabel('Internal State')
    ax1.set_xlabel('Time (min)')
    ax1.set_title('Fistula reward')
    fig1.savefig('internalStateFistula.eps', format='eps')

    return


'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   plot Etimated Cocaine Probability for ShA rats ------------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotEtimatedWaterProbabilityOral():
 
    font = {'family' : 'normal', 'size'   : 16}
    pylab.rc('font', **font)
    pylab.rcParams.update({'legend.fontsize': 16})
        
    fig1 = pylab.figure( figsize=(5,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)

    ax1 = fig1.add_subplot(111)
    S0 = ax1.plot(estimatedOutcomeOnGreenLog  [totalTrialsNum/4 : totalTrialsNum/2] , linewidth = 2 , color='green' )
    S1 = ax1.plot(estimatedOutcomeOnYellowLog [totalTrialsNum/4 : totalTrialsNum/2] , linewidth = 2 , color='red' )
    
    leg=fig1.legend((S0, S1), ('green key','yellow key'), loc = (0.52,0.45))
    leg.draw_frame(False)

    ax1.axhline(0, color='0.25',ls='--', lw=1 )
    ax1.axhline(1, color='0.25',ls='--', lw=1 )

    pylab.yticks(pylab.arange(0, 1.001, 0.2))
    pylab.ylim((-0.1,1.1))
    pylab.xlim( ( -50  , totalTrialsNum/4 +50 ) )
    
    tick_lcs = []
    tick_lbs = []
    for i in range ( 0 , 9 ):
        tick_lcs.append( i*150 ) 
        tick_lbs.append(i*5)
    pylab.xticks(tick_lcs, tick_lbs)

    for i in range ( 0 , 8 ):
        if i%2==0:
            p = pylab.axvspan( i*150 + i, (i+1)*150 + i , facecolor='0.75',edgecolor='none', alpha=0.5)        

    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    ax1.set_ylabel('Subjective probability')
    ax1.set_xlabel('Time (min)')
    ax1.set_title('Oral reward')
    fig1.savefig('etimatedWaterProbabilityOral.eps', format='eps')

    return    

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   plot Etimated Cocaine Probability for ShA rats ------------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotEtimatedWaterProbabilityFistula():
 
    font = {'family' : 'normal', 'size'   : 16}
    pylab.rc('font', **font)
    pylab.rcParams.update({'legend.fontsize': 16})
        
    fig1 = pylab.figure( figsize=(5,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)

    ax1 = fig1.add_subplot(111)
    S0 = ax1.plot(estimatedOutcomeOnGreenLog  [totalTrialsNum*3/4 : totalTrialsNum] , linewidth = 2 , color='green' )
    S1 = ax1.plot(estimatedOutcomeOnYellowLog [totalTrialsNum*3/4 : totalTrialsNum] , linewidth = 2 , color='red' )
    
    leg=fig1.legend((S0, S1), ('green key','yellow key'), loc = (0.52,0.45))
    leg.draw_frame(False)

    ax1.axhline(0, color='0.25',ls='--', lw=1 )
    ax1.axhline(1, color='0.25',ls='--', lw=1 )

    pylab.yticks(pylab.arange(0, 1.001, 0.2))
    pylab.ylim((-0.1,1.1))
    pylab.xlim( ( -50  , totalTrialsNum/4 +50 ) )
    
    tick_lcs = []
    tick_lbs = []
    for i in range ( 0 , 9 ):
        tick_lcs.append( i*150 ) 
        tick_lbs.append(i*5)
    pylab.xticks(tick_lcs, tick_lbs)

    for i in range ( 0 , 8 ):
        if i%2==0:
            p = pylab.axvspan( i*150 + i, (i+1)*150 + i , facecolor='0.75',edgecolor='none', alpha=0.5)        

    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    ax1.set_ylabel('Subjective probability')
    ax1.set_xlabel('Time (min)')
    ax1.set_title('Fistula reward')
    fig1.savefig('etimatedWaterProbabilityFistula.eps', format='eps')

    return    

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the Lever-press per session  ------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotLeverPressPer5MinOral():

    greenKey  = numpy.zeros( [4*8] , float)
    yellowKey = numpy.zeros( [4*8] , float)
    x = numpy.arange(5, 41, 5)
        
    for i in range(0,4):
        for k in range(0,8):
            for j in range(  i*seekingTrialsNum + k*150 , i*seekingTrialsNum + k*150 + 150 ):
                greenKey [i*8 + k] = greenKey [i*8 + k] + greenLeverPress [j]
                yellowKey[i*8 + k] = yellowKey[i*8 + k] + yellowLeverPress[j]
        
    fig1 = pylab.figure( figsize=(5,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

#    S1 = ax1.plot(x[0:8]   ,greenKey[0:8]     , '-o', ms=5, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )
    S1 = ax1.plot(x  ,greenKey[8:16]    , '-o', ms=7, markeredgewidth =0, alpha=1, mfc='green',linewidth = 2 , color='green' )
    
#    S0 = ax1.plot(x[0:8]   ,yellowKey[0:8]     , '-o', ms=5, markeredgewidth =2, alpha=1, mfc='white',linewidth = 2 , color='black' )
    S0 = ax1.plot(x  ,yellowKey[8:16]    , '-o', ms=7, markeredgewidth =0, alpha=1, mfc='red',linewidth = 2 , color='red' )

    pylab.yticks(pylab.arange(10, 31, 5))
    pylab.ylim((13,32))
    pylab.xlim((0,45))
    pylab.xticks(pylab.arange(5, 41, 5))
    

    leg=fig1.legend((S1, S0), ('green key','yellow key'), loc = (0.52,0.69))
    leg.draw_frame(False)


    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    ax1.set_ylabel('Responses / 5min')
    ax1.set_xlabel('Time (min)')
    ax1.set_title('Oral reward')
    fig1.savefig('ResponsesOralReward.eps', format='eps')

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the Lever-press per session  ------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotLeverPressPer5MinFistula():

    greenKey  = numpy.zeros( [4*8] , float)
    yellowKey = numpy.zeros( [4*8] , float)
    x = numpy.arange(5, 41, 5)
        
    for i in range(0,4):
        for k in range(0,8):
            for j in range(  i*seekingTrialsNum + k*150 , i*seekingTrialsNum + k*150 + 150 ):
                greenKey [i*8 + k] = greenKey [i*8 + k] + greenLeverPress [j]
                yellowKey[i*8 + k] = yellowKey[i*8 + k] + yellowLeverPress[j]
        
    fig1 = pylab.figure( figsize=(5,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    S1 = ax1.plot(x  , greenKey[24:32]    , '-o', ms=7, markeredgewidth =0, alpha=1, mfc='green',linewidth = 2 , color='green' )
    
    S0 = ax1.plot(x  ,yellowKey[24:32]    , '-o', ms=7, markeredgewidth =0, alpha=1, mfc='red',linewidth = 2 , color='red' )

    pylab.yticks(pylab.arange(10, 31, 5))
    pylab.ylim((13,32))
    pylab.xlim((0,45))
    pylab.xticks(pylab.arange(5, 41, 5))

    leg=fig1.legend((S1, S0), ('green key','yellow key'), loc = (0.52,0.69))
    leg.draw_frame(False)


    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    ax1.set_ylabel('Responses / 5min')
    ax1.set_xlabel('Time (min)')
    ax1.set_title('Fistula reward')
    fig1.savefig('ResponsesFistulaReward.eps', format='eps')
    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the Lever-press per session  ------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotExperimentalOral():

    greenKey  = numpy.zeros( [8] , float)
    yellowKey = numpy.zeros( [8] , float)
    x = numpy.arange(5, 41, 5)
        
    greenKey =  [27,14,8.7,2.4,0.8,0.7,0.6,0.5]
    yellowKey = [17,39,37.5,10.2,6.3,3,1.5,0.5]
        
    fig1 = pylab.figure( figsize=(5,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    S1 = ax1.plot(x  ,greenKey    , '-o', ms=7, markeredgewidth =0, alpha=1, mfc='green',linewidth = 2 , color='green' )
    
    S0 = ax1.plot(x  ,yellowKey    , '-o', ms=7, markeredgewidth =0, alpha=1, mfc='red',linewidth = 2 , color='red' )

    pylab.yticks(pylab.arange(0, 41, 10))
    pylab.ylim((-5,45))
    pylab.xlim((0,45))
    pylab.xticks(pylab.arange(5, 41, 5))
    

    leg=fig1.legend((S1, S0), ('green key','yellow key'), loc = (0.52,0.69))
    leg.draw_frame(False)


    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    ax1.set_ylabel('Responses / 5min')
    ax1.set_xlabel('Time (min)')
    ax1.set_title('Oral reward')
    fig1.savefig('ExperResponsesOralReward.eps', format='eps')

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the Lever-press per session  ------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotExperimentalFistula():

    greenKey  = numpy.zeros( [8] , float)
    yellowKey = numpy.zeros( [8] , float)
    x = numpy.arange(5, 41, 5)
        
    greenKey = [22.5,15.1,11.7,6.1,2.9,0.1,0,0]
    yellowKey = [5,2.2,2.9,1.8,0.2,0,0,0]
        
    fig1 = pylab.figure( figsize=(5,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    S1 = ax1.plot(x  ,greenKey    , '-o', ms=7, markeredgewidth =0, alpha=1, mfc='green',linewidth = 2 , color='green' )
    
    S0 = ax1.plot(x  ,yellowKey    , '-o', ms=7, markeredgewidth =0, alpha=1, mfc='red',linewidth = 2 , color='red' )

    pylab.yticks(pylab.arange(0, 41, 10))
    pylab.ylim((-5,40))
    pylab.xlim((0,45))
    pylab.xticks(pylab.arange(5, 41, 5))
    

    leg=fig1.legend((S1, S0), ('green key','yellow key'), loc = (0.52,0.69))
    leg.draw_frame(False)


    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    ax1.set_ylabel('Responses / 5min')
    ax1.set_xlabel('Time (min)')
    ax1.set_title('Fistula reward')
    fig1.savefig('ExperResponsesFistulaReward.eps', format='eps')

    return


'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot all the results  ------------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotting():

    loggingFinalization()

    plotInternalState() 
#    plotLeverPressPerSession()    
#    plotLeverPressPerDose()    
    plotLeverPressPer5Min()    
#    plotEtimatedCocaineProbabilityLgA()    
#    plotEtimatedCocaineProbabilityFirstLgA()
    
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

water           = 1
waterLosePerTrial  = 0.1
leverPressCost  = 10              # Energy cost for pressing the lever

statesNum       = 1              # number of stater 
actionsNum      = 3              # number of action   action 0 = Null     action 1 = Inactive Lever Press    action 2 = Active Lever Press
initialExState  = 0

transition = numpy.zeros( [statesNum , actionsNum, statesNum] , float)
setTransition(0,0,0,1)          # From state s, and by taking a, we go to state s', with probability p
setTransition(0,1,0,1)
setTransition(0,2,0,1)

outcome = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
setOutcome(0,1,0,water)       # At state s, by doing action a and going to state s', we receive the outcome 

nonHomeostaticReward = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
setNonHomeostaticReward(0,1,0,-leverPressCost)
setNonHomeostaticReward(0,2,0,-leverPressCost)

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Definition of the Animal   --------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

#------------ Homeostatic System
initialInState          = 0
setpoint                = 50

#------------ Drive Function
m                       = 3     # Parameter of the drive function : m-th root
n                       = 4     # Parameter of the drive function : n-th pawer

#------------ Goal-directed system
updateOutcomeRate       = 0.04   # Learning rate for updating the outcome function
updateTransitionRate    = 0.04   # Learning rate for updating the transition function
updateRewardRate        = 0.04   # Learning rate for updating the non-homeostatic reward function
gamma                   = 1     # Discount factor
beta                    = 5     # Rate of exploration
searchDepth             = 1     # Depth of going into the decision tree for goal-directed valuation of choices
pruningThreshold        = 0.1   # If the probability of a transition like (s,a,s') is less than "pruningThreshold", cut it from the decision tree 

estimatedTransition             = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
estimatedOutcome                = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
estimatedNonHomeostaticReward   = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )

state                            = numpy.zeros ( [4] , float )     # a vector of the external state, internal state, setpoint, and trial

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Simulation Parameters   -----------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

animalsNum          = 100                                # Number of animals

pretrainingHours    = 0
sessionsNum         = 4                                # Number of sessions of water seeking, followed by rest in home-cage
seekingHours        = 0.66           
trialsPerHour       = 60*60/2                         # Number of trials during one hour (as each trial is supposed to be 2 seconds)
seekingTrialsNum    = 40*30  #seekingHours * trialsPerHour    # Number of trials for each water seeking session
totalTrialsNum      = sessionsNum * seekingTrialsNum

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plotting Parameters   -------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

trialsPerBlock = 10*60/4            # Each BLOCK is 10 minutes - Each minute 60 second - Each trial takes 4 seconds

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Logging Parameters   --------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

nulDoing            = numpy.zeros( [totalTrialsNum] , float)
greenLeverPress     = numpy.zeros( [totalTrialsNum] , float)
yellowLeverPress    = numpy.zeros( [totalTrialsNum] , float)
internalStateLog    = numpy.zeros( [totalTrialsNum] , float)
estimatedOutcomeOnGreenLog  = numpy.zeros( [totalTrialsNum] , float)
estimatedOutcomeOnYellowLog = numpy.zeros( [totalTrialsNum] , float)

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Simulation   ----------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

for animal in range(0,animalsNum):

    initializeAnimal          ( )
    state[3] = 0 


    state[1] = initialInState        
    setOutcome(0,1,0,water)       # At state s, by doing action a and going to state s', we receive the outcome 
    setOutcome(0,2,0,0    )       # At state s, by doing action a and going to state s', we receive the outcome 
    seeking (0) 


    setOutcome(0,1,0,0    )       # At state s, by doing action a and going to state s', we receive the outcome 
    setOutcome(0,2,0,water)       # At state s, by doing action a and going to state s', we receive the outcome 
    state[1] = initialInState        
    seeking (1) 


for animal in range(0,animalsNum):

    initializeAnimal          ( )
    state[3] = totalTrialsNum / 2 

    state[1] = initialInState        
    setOutcome(0,1,0,water)       # At state s, by doing action a and going to state s', we receive the outcome 
    setOutcome(0,2,0,0    )       # At state s, by doing action a and going to state s', we receive the outcome 
    seeking (2) 

    setOutcome(0,1,0,0    )       # At state s, by doing action a and going to state s', we receive the outcome 
    setOutcome(0,2,0,water)       # At state s, by doing action a and going to state s', we receive the outcome 
    state[1] = initialInState        
    seeking (3) 


loggingFinalization()
plotInternalStateOral() 
plotInternalStateFistula() 
plotLeverPressPer5MinOral()    
plotLeverPressPer5MinFistula()   
plotEtimatedWaterProbabilityOral() 
plotEtimatedWaterProbabilityFistula() 

plotExperimentalOral()    
plotExperimentalFistula()   

pylab.show()   