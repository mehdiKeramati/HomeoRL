import numpy
import scipy
import pylab
import cmath
import math

#################################################################
#                        Functions
#################################################################

# --------------------   Setting the probability [transitionProbability] for transiting from a state [state] to another state [nextState] by taking an action [action] 
def setTransition(state,action,nextState,transitionProbability):
    T[state][action][nextState ] = transitionProbability
    return 

# --------------------   Returns the probability [transitionProbability] of transiting from a state [state] to another state [nextState] by taking an action [action] 
def getTransition(state,action,nextState):
    return T[state][action][nextState]

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
    
    alternatives = numpy.zeros(actionsNum,float)
    for i in range(0,actionsNum):
        alternatives[i] = Q [state][i]
    
##----- Normalizing Alternatives to a range between -1 and 1
#    maxQ = 0
#    for i in range(0,actionsNum):
#        if isActionAvailable(state,i):
#            if maxQ < abs(alternatives[i]):
#                maxQ = abs(alternatives[i])
#
#    if maxQ != 0 :
#        for i in range(0,actionsNum):
#            if isActionAvailable(state,i):
#                alternatives[i] = alternatives[i]/maxQ
##----- End of normalization

    sumEQ = 0
    for i in range(0,actionsNum):
        if isActionAvailable(state,i):
            sumEQ = sumEQ + abs(cmath.exp ( alternatives[i] / beta ))
    
    index = numpy.random.uniform(0,sumEQ)
    probSum=0
    for i in range(0,actionsNum):
        if isActionAvailable(state,i):
            probSum = probSum + abs(cmath.exp ( alternatives[i] / beta ))
            if probSum >= index:
                return i

    print "Error: An unexpected (strange) problem has occured in action selection..."
    return 0

#---------------------   Transition Function: returns the next state after taking an action [action] in an state [state]
def transition(state,action):
    alternatives = T[state][action]
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
def reward(initialState,finalState):
    d1 = numpy.power(numpy.absolute(numpy.power(optimalInState-initialState,n*1.0)),(1.0/m))
    d2 = numpy.power(numpy.absolute(numpy.power(optimalInState-finalState  ,n*1.0)),(1.0/m))
    return (d1-d2)
    
#---------------------   The change in temperature (deviation from setpoint) induced by ethanol infusion, <lag> minutes after the infusion
def ethanolInducedTemp(lag):
    temperature = -1.963*numpy.exp(-0.3632*(lag/100.0)) + 1.964*numpy.exp(-4.064*(lag/100.0))
    return temperature

#---------------------   The change in temperature (deviation from setpoint) induced by shivering, <lag> minutes after shivering
def shiveringInducedTemp(lag):
    temperature = -15000*numpy.exp(-0.004705*lag) + 15000*numpy.exp(-0.004704*lag)
    return temperature

#---------------------   Update internal state
def updateInState(inState,change):
    return inState+change

#---------------------   Update : Using TD(0) algorithm
def updateQ_TD0(state,action,nextState,reward):
    counter = 0
    QNext = numpy.zeros(actionsNum,float)
    for i in range(0,actionsNum):
        if isActionAvailable(nextState,i):
            QNext[counter] = Q [nextState][i]
            counter = counter+1

    VNext               = max(QNext[0:counter])
    delta               = reward  +  gamma10 * VNext  -  Q[state][action] 
    Q[state][action]    = Q[state][action]  + alpha*delta
    
    return 

# --------------------   Initialize Q-values to zero
def initializeQ():
    Q = numpy.zeros( [ statesNum , actionsNum ] , float)
    return Q
     
#---------------------   Logging
def logging(animalN,dayN,trialN,exState,action,reward):

    stateLog    [ animalN ][ dayN ][ trialN ][ exState ] = 1
    actionLog   [ animalN ][ dayN ][ trialN ][ action ]  = 1
    QLog        [ animalN ][ dayN ][ trialN ] = Q

    return 

#---------------------   Plotting some desired variables
def plotting():
    
    plotInternalState ()
    plotShiveringAfterCSPretraining  ()
    plotQValues  ()
    plotSimulationData()
    plotExperimentalData()

    pylab.show()   

    return
    
#---------------------   Plotting some desired variables
def plotQValues():
    
    fig1 = pylab.figure( figsize=(10,4) )
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)


    Q00   = numpy.zeros( [ totalDaysNum ] , float)
    Q01   = numpy.zeros( [ totalDaysNum ] , float)
    Q11   = numpy.zeros( [ totalDaysNum ] , float)
    Q21   = numpy.zeros( [ totalDaysNum ] , float)

    for day in range(0,totalDaysNum ):            
        Q00[day] = QLog[0][day][0][0][0]
        Q01[day] = QLog[0][day][0][0][1]
        Q11[day] = QLog[0][day][0][1][1]
        Q21[day] = QLog[0][day][0][2][1]


    h00 = ax1.plot( Q00, linewidth = 2 , color = 'black')
    h01 = ax1.plot( Q01, linewidth = 2 , color = 'red')
    h20 = ax1.plot( Q11, linewidth = 2 , color = 'blue')
    h21 = ax1.plot( Q21, linewidth = 2 , color = 'green')

    leg = fig1.legend((h00, h01 , h20 , h21), ('0 : Sh','0 : NoSh','1 : NoSh','2 : NoSh'), loc = (0.20,0.20))
    leg.draw_frame(False)

    p = pylab.axvspan( pretrainingDaysNum , pretrainingDaysNum+acquisitionDaysNum , facecolor='0.75',edgecolor='none', alpha=0.5)        


    ax1.set_title('Q-values')
    fig1.savefig('Q-values.eps', format='eps')    
    
    return

#---------------------   Plotting some desired variables
def plotInternalState():

    font = {'family' : 'normal', 'size'   : 16}
    pylab.rc('font', **font)

    tempShiv = numpy.zeros([trialsNumPerDay*trialsForOneAction] , float )
    for trial in range(0,trialsNumPerDay):                   
        for miniTrial in range(0,trialsForOneAction):
            tempShiv[trial*trialsForOneAction + miniTrial] = shiveringInducedTemp( trial*trialsForOneAction + miniTrial)                


    tempEtha = numpy.zeros([trialsNumPerDay*trialsForOneAction] , float )
    for trial in range(1,trialsNumPerDay):            
        for miniTrial in range(0,trialsForOneAction):
            tempEtha[trial*trialsForOneAction + miniTrial] = ethanolInducedTemp( trial*trialsForOneAction + miniTrial - trialsForOneAction )


    tempShivEtha = numpy.zeros([trialsNumPerDay*trialsForOneAction] , float )
    injectionTrial = 1
    for trial in range(0,injectionTrial):                   
        for miniTrial in range(0,trialsForOneAction):
            tempShivEtha[trial*trialsForOneAction + miniTrial] = shiveringInducedTemp( trial*trialsForOneAction + miniTrial)                
    for trial in range(injectionTrial,trialsNumPerDay):            
        for miniTrial in range(0,trialsForOneAction):
            tempShivEtha[trial*trialsForOneAction + miniTrial] = shiveringInducedTemp( trial*trialsForOneAction + miniTrial)
            tempShivEtha[trial*trialsForOneAction + miniTrial] = tempShivEtha[trial*trialsForOneAction + miniTrial] + ethanolInducedTemp( trial*trialsForOneAction + miniTrial - injectionTrial*trialsForOneAction )

    #---------------------------- Only Shivering effect
    fig1 = pylab.figure( figsize=(6,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    ax1.axhline(0,  color='0.5',ls='--', lw=2 )
    
    ax1.axvline(0 ,  color='green', lw=2 )
    ax1.axvline(89 ,  color='0.5', lw=1.5 )
    ax1.axvline(119,  color='0.5', lw=1.5 )
    ax1.axvline(149,  color='0.5', lw=1.5 )
    ax1.axvline(179,  color='0.5', lw=1.5 )
    
    h2 = ax1.plot( tempShiv , linewidth = 2, color = 'black')

    pylab.yticks(pylab.arange(-1.5, 1.55, 0.5))
    pylab.ylim((-1.6,1.6))
    pylab.xlim((-35,60*24 + 35))

    tick_lcs = []
    tick_lbs = []
    for i in range ( 0 , 9 ):
        tick_lcs.append( i*3*60 ) 
        tick_lbs.append(i*3)
    pylab.xticks(tick_lcs, tick_lbs)

    for line in ax1.get_yticklines() + ax1.get_xticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)
    
    ax1.set_ylabel('Change of temperature')
    ax1.set_xlabel('Hour')
    ax1.set_title('Tolerance response')
    fig1.savefig('interalStateShivering.eps', format='eps')    

    #---------------------------- Only Ethanol effect
    fig1 = pylab.figure( figsize=(6,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    ax1.axhline(0,  color='0.5',ls='--', lw=2 )

    ax1.axvline(59 ,  color='red', lw=2 )
    ax1.axvline(89 ,  color='0.5', lw=1.5 )
    ax1.axvline(119,  color='0.5', lw=1.5 )
    ax1.axvline(149,  color='0.5', lw=1.5 )
    ax1.axvline(179,  color='0.5', lw=1.5 )

    h0 = ax1.plot(tempEtha , linewidth = 2, color = 'black')

    pylab.yticks(pylab.arange(-1.5, 1.55, 0.5))
    pylab.ylim((-1.6,1.6))
    pylab.xlim((-35,60*24 + 35))

    tick_lcs = []
    tick_lbs = []
    for i in range ( 0 , 9 ):
        tick_lcs.append( i*3*60 ) 
        tick_lbs.append(i*3)
    pylab.xticks(tick_lcs, tick_lbs)

    for line in ax1.get_yticklines() + ax1.get_xticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)
    
    ax1.set_ylabel('Change of temperature')
    ax1.set_xlabel('Hour')
    ax1.set_title('Ethanol injection')
    fig1.savefig('interalStateEthanol.eps', format='eps')    

    #----------------------------  Ethanol effect + Shivering effect
    fig1 = pylab.figure( figsize=(6,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    ax1.axhline(0,  color='0.5',ls='--', lw=2 )

    ax1.axvline(0 ,  color='green', lw=2 )
    ax1.axvline(59 ,  color='red', lw=2 )
    ax1.axvline(89 ,  color='0.5', lw=1.5 )
    ax1.axvline(119,  color='0.5', lw=1.5 )
    ax1.axvline(149,  color='0.5', lw=1.5 )
    ax1.axvline(179,  color='0.5', lw=1.5 )

    h1 = ax1.plot(tempShivEtha , linewidth = 2, color = 'black')

    pylab.yticks(pylab.arange(-1.5, 1.55, 0.5))
    pylab.ylim((-1.6,1.6))
    pylab.xlim((-35,60*24 + 35))

    tick_lcs = []
    tick_lbs = []
    for i in range ( 0 , 9 ):
        tick_lcs.append( i*3*60 ) 
        tick_lbs.append(i*3)
    pylab.xticks(tick_lcs, tick_lbs)

    for line in ax1.get_yticklines() + ax1.get_xticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)
    
    ax1.set_ylabel('Change of temperature')
    ax1.set_xlabel('Hour')
    ax1.set_title('Tolerance response + Ethanol injection')
    fig1.savefig('interalStateShiveringEthanol.eps', format='eps')    
    
    return

    
#---------------------   Plotting some desired variables
def plotSimulationData():

    shiver   = numpy.zeros( [ totalDaysNum ] , float)

    for animal in range(0,animalsNum):            
        for day in range(0,totalDaysNum ):            
            for trial in range(0,trialsNumPerDay):            
                if ((stateLog[animal][day][trial][0]==1) and  (actionLog[animal][day][trial][0]==1)):
                    shiver[day]   = shiver[day]   + 1

    for day in range(0,totalDaysNum):            
        shiver[day]   = shiver[day]   / animalsNum


    shiverInBlock   = numpy.zeros( [ 10 ] , float)
    #------------- Acquisition Days
    for block in range(0 , 8):            
        for day in range(0,daysNumInBlock):            
            shiverInBlock[block] = shiverInBlock[block] + shiver[pretrainingDaysNum + block*daysNumInBlock + day]
        shiverInBlock[block] = shiverInBlock[block]/daysNumInBlock

    #------------- First Day of Extinction
    block = 8
    blockWidth = daysNumInBlock
    for day in range(0,blockWidth):            
        shiverInBlock[block] = shiverInBlock[block] + shiver[pretrainingDaysNum + acquisitionDaysNum + day]
    shiverInBlock[block] = shiverInBlock[block]/daysNumInBlock

    #------------- First Day of Re-acquisition
    block = 9
    for day in range(0,blockWidth):            
        shiverInBlock[block] = shiverInBlock[block] + shiver[pretrainingDaysNum + acquisitionDaysNum + extinctionDaysNum + day]
    shiverInBlock[block] = shiverInBlock[block]/blockWidth


    #--------------------------- Plot probability of Shivering in each block
    fig1 = pylab.figure( figsize=(6,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    x = numpy.arange(1, 11 , 1)
    h0 = ax1.plot(x,shiverInBlock , '-o', ms=6, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )

    for line in ax1.get_yticklines() + ax1.get_xticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    pylab.yticks(pylab.arange(0, 1.05, 0.2))
    pylab.ylim((-0.1,1.1))
    pylab.xlim((0,11))
    
    tick_lcs = []
    tick_lbs = []
    for i in range ( 1 , 9 ):
        tick_lcs.append( i ) 
        tick_lbs.append(i)
    tick_lcs.append(9) 
    tick_lbs.append('E1')
    tick_lcs.append(10) 
    tick_lbs.append('R1')
    pylab.xticks(tick_lcs, tick_lbs)

    p = pylab.axvspan( 8.5 , 9.5 , facecolor='0.75',edgecolor='none', alpha=0.5)        
    
    ax1.set_ylabel('Response probability')
    ax1.set_xlabel('Blocks')
    fig1.savefig  ('ShiveringProbabilityPerBlock.eps', format='eps')    

    #------------------------ Computing the internal state for one day of one animal
    tempShiv = numpy.zeros([trialsNumPerDay*trialsForOneAction] , float )
    for trial in range(0,trialsNumPerDay):                   
        for miniTrial in range(0,trialsForOneAction):
            tempShiv[trial*trialsForOneAction + miniTrial] = shiveringInducedTemp( trial*trialsForOneAction + miniTrial)                


    tempEtha = numpy.zeros([trialsNumPerDay*trialsForOneAction] , float )
    for trial in range(1,trialsNumPerDay):            
        for miniTrial in range(0,trialsForOneAction):
            tempEtha[trial*trialsForOneAction + miniTrial] = ethanolInducedTemp( trial*trialsForOneAction + miniTrial - trialsForOneAction )


    #------------------------ Plot Experimental Results
    
    temperature   = numpy.zeros( [ 10 , 4 ] , float)
    x = numpy.arange(0, 40 , 1)

    for block in range(0 , 10):            
        for section in range(1,5):            
            shivTemp = tempShiv[trialsForOneAction+30*section ]
            EthaTemp = tempEtha[trialsForOneAction+30*section ]
            if block==8 : EthaTemp = 0  # Because we are in extinction 
            temperature[block][section-1] = shiverInBlock[block]*shivTemp + EthaTemp
    

    fig1 = pylab.figure( figsize=(6,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    ax1.axhline(0,  color='0.50',ls='--', lw=2 )

    block = 0
    for block in range(0 , 10):            
        h0 = ax1.plot( x[ block*4 : block*4+4 ] ,temperature [block] , '-o', ms=3, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )

    for line in ax1.get_yticklines() + ax1.get_xticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    pylab.yticks(pylab.arange(-1.5, 1.55, 0.5))
    pylab.ylim((-1.6,1.6))
    pylab.xlim((-2,41))
    
    tick_lcs = []
    tick_lbs = []
    for i in range ( 0 , 8 ):
        tick_lcs.append( 1.5 + i*4 ) 
        tick_lbs.append(i+1)
    tick_lcs.append(1.5 + 8*4) 
    tick_lbs.append('E1')
    tick_lcs.append(1.5 + 9*4) 
    tick_lbs.append('R1')
    pylab.xticks(tick_lcs, tick_lbs)

    p = pylab.axvspan( 31.5 , 35.5 , facecolor='0.75',edgecolor='none', alpha=0.5)        
    
    ax1.set_ylabel('Change of temperature')
    ax1.set_xlabel('Blocks')
    ax1.set_title('Simulation Results')
    fig1.savefig  ('temperaturePerBlockSimulation.eps', format='eps')    

    return

#---------------------   Plotting some desired variables

def plotExperimentalData():

    temperature   = numpy.zeros( [ 10 , 4 ] , float)
    x = numpy.arange(0, 40 , 1)

    temperature[0] = [-1.17  ,  -1.4375 ,  -1.3275  ,   -1.27   ]
    temperature[1] = [-0.9145,  -1.075  ,  -0.9475  ,   -0.985  ]                                                                                                                              
    temperature[2] = [-0.71  ,  -0.895  ,  -0.695   ,   -0.705  ]                                                                                                              
    temperature[3] = [-0.62  ,  -0.8475 ,  -0.74    ,   -0.64   ]                                                                                             
    temperature[4] = [-0.6285,  -0.763  ,  -0.5935  ,   -0.453  ]                                                                              
    temperature[5] = [-0.547 ,  -0.665  ,  -0.525   ,   -0.375  ]                                                              
    temperature[6] = [-0.46  ,  -0.512  ,  -0.433   ,   -0.236  ]                                              
    temperature[7] = [-0.37  ,  -0.45   ,   -0.37   ,   -0.2    ]                            
    temperature[8] = [0.29   ,  0.44    ,   0.55    ,   0.62    ]            
    temperature[9] = [-0.69  ,  -0.83   ,   -0.82   ,   -0.56   ]
                      
                      
    fig1 = pylab.figure( figsize=(6,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    ax1.axhline(0,  color='0.50',ls='--', lw=2 )

    block = 0
    for block in range(0 , 10):            
        h0 = ax1.plot( x[ block*4 : block*4+4 ] ,temperature [block] , '-o', ms=3, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )

    for line in ax1.get_yticklines() + ax1.get_xticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    pylab.yticks(pylab.arange(-1.5, 1.55, 0.5))
    pylab.ylim((-1.6,1.6))
    pylab.xlim((-2,41))
    
    tick_lcs = []
    tick_lbs = []
    for i in range ( 0 , 8 ):
        tick_lcs.append( 1.5 + i*4 ) 
        tick_lbs.append(i+1)
    tick_lcs.append(1.5 + 8*4) 
    tick_lbs.append('E1')
    tick_lcs.append(1.5 + 9*4) 
    tick_lbs.append('R1')
    pylab.xticks(tick_lcs, tick_lbs)

    p = pylab.axvspan( 31.5 , 35.5 , facecolor='0.75',edgecolor='none', alpha=0.5)        
    
    ax1.set_ylabel('Change of temperature')
    ax1.set_title('Experimental Data')
    ax1.set_xlabel('Blocks')
    fig1.savefig  ('temperaturePerBlockExperiment.eps', format='eps')    

    return


#---------------------   Plotting some desired variables
def plotShiveringAfterCSPretraining():


    shiver   = numpy.zeros( [ totalDaysNum ] , float)

    for animal in range(0,animalsNum):            
        for day in range(0,totalDaysNum ):            
            for trial in range(0,trialsNumPerDay):            
                if ((stateLog[animal][day][trial][0]==1) and  (actionLog[animal][day][trial][0]==1)):
                    shiver[day]   = shiver[day]   + 1

    for day in range(0,totalDaysNum):            
        shiver[day]   = shiver[day]   / animalsNum
        
        
    #--------------------------- Pretraining
    fig1 = pylab.figure( figsize=(6,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    x = numpy.arange(1, pretrainingDaysNum+1, 1)
    h0 = ax1.plot(x,shiver[0:pretrainingDaysNum] , '-o', ms=6, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )
    
    pylab.yticks(pylab.arange(0, 1.05, 0.2))
    pylab.ylim((-0.1,1.1))
    pylab.xlim((0,pretrainingDaysNum+1))
    tick_lcs = [1,5,10,15,20,24]
    tick_lbs = [1,5,10,15,20,24]
    pylab.xticks(tick_lcs, tick_lbs)

    for line in ax1.get_yticklines() + ax1.get_xticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)
    
    ax1.set_ylabel('Response probability')
    ax1.set_xlabel('Pre-training day')
    fig1.savefig('ShiveringUponCSPretraining.eps', format='eps')    

    #--------------------------- Acquisition
    fig1 = pylab.figure( figsize=(6,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    x = numpy.arange(1, acquisitionDaysNum+1, 1)
    h0 = ax1.plot(x,shiver[pretrainingDaysNum:pretrainingDaysNum+acquisitionDaysNum] , '-o', ms=6, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )
    
    pylab.yticks(pylab.arange(0, 1.05, 0.2))
    pylab.ylim((-0.1,1.1))
    pylab.xlim((0,acquisitionDaysNum+1))
    tick_lcs = [1,5,10,15,20,24]
    tick_lbs = [1,5,10,15,20,24]
    pylab.xticks(tick_lcs, tick_lbs)

    for line in ax1.get_yticklines() + ax1.get_xticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)
    
    ax1.set_ylabel('Response probability')
    ax1.set_xlabel('Acquisition day')
    fig1.savefig('ShiveringUponCSAcquisition.eps', format='eps')    

    #--------------------------- Extinction
    fig1 = pylab.figure( figsize=(6,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    x = numpy.arange(1, extinctionDaysNum+1, 1)
    h0 = ax1.plot(x,shiver[pretrainingDaysNum+acquisitionDaysNum:pretrainingDaysNum+acquisitionDaysNum+extinctionDaysNum] , '-o', ms=6, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )
    
    pylab.yticks(pylab.arange(0, 1.05, 0.2))
    pylab.ylim((-0.1,1.1))
    pylab.xlim((0,extinctionDaysNum+1))
    tick_lcs = [1,3,5,7,9,11]
    tick_lbs = [1,3,5,7,9,11]
    pylab.xticks(tick_lcs, tick_lbs)

    for line in ax1.get_yticklines() + ax1.get_xticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)
    
    ax1.set_ylabel('Response probability')
    ax1.set_xlabel('Extinction day')
    fig1.savefig('ShiveringUponCSExtinction.eps', format='eps')    

       
    return


#==================================================================================================================================================
#======   The code starts running from here   =====================================================================================================
#==================================================================================================================================================

#################################################################
#                        Environment's Dynamics
#################################################################

# --------------------   Number of the States
statesNum  = 5

# --------------------   Number of the Actions
actionsNum = 2

# --------------------   Initial External State
initialExState = 0

# --------------------   Final External States
#finalExStates = [6,8]

# --------------------   Transition Function : (from state s, by action a, going to state s', by probability p)
T = numpy.zeros( [ statesNum , actionsNum, statesNum ] , float)
setTransition(0,0,1,1)
setTransition(0,1,2,1)
setTransition(1,1,3,1)
setTransition(2,1,3,1)
setTransition(3,1,4,1)

# --------------------   Outcome Function : (from state s, by action a, recieving reward r)
O = numpy.zeros (statesNum * actionsNum , float )

#setOutcome(0,0,heatFromShivering)
#setOutcome(1,0,heatFromShivering)

#################################################################
#                        Agent's Dynamics
#################################################################

# --------------------    internal State
initialInState  = 0
optimalInState  = 0

# -------------------- Meta Variables
alpha           = 0.2          # Learning rate
gamma1          = 0.96         # Discount factor, for every minute
gamma10         = 0.66         # = gamma1^10 : Discount factor, for every 10 minutes

beta            = .1           # Rate of exploration

m               = 2
n               = 2

#################################################################
#                        Simulation Initializations
#################################################################

# -------------------- Simulations Variables
trialsForOneAction  = 60

trialsNumPerDay     = (60/trialsForOneAction)*24     

pretrainingDaysNum  = 24
acquisitionDaysNum  = 24
extinctionDaysNum   = 12
reacquisitionDaysNum= 3
totalDaysNum        = pretrainingDaysNum + acquisitionDaysNum + extinctionDaysNum + reacquisitionDaysNum
daysNumInBlock      = math.floor(acquisitionDaysNum/8)

animalsNum          = 1



# --------------------   Q-values
Q = numpy.zeros( [ statesNum , actionsNum ] , float)

# --------------------   Average-over-runs Variables
QLog        = numpy.zeros( [ animalsNum , totalDaysNum,  trialsNumPerDay ,  statesNum , actionsNum ] , float)
stateLog    = numpy.zeros( [ animalsNum , totalDaysNum,  trialsNumPerDay , statesNum  ] , float)
actionLog   = numpy.zeros( [ animalsNum , totalDaysNum,  trialsNumPerDay , actionsNum ] , float)
internalSLog= numpy.zeros( [ animalsNum , totalDaysNum,  trialsNumPerDay*trialsForOneAction ] , float)

#################################################################
#                        Main Code
#################################################################

for animal in range(0,animalsNum):

    Q = initializeQ()
    
#################################  Pretraining Phase  ######################################


    for day in range( 0 , pretrainingDaysNum ):

        print "Animals number: %3d  /%3d          Day %3d  /%3d       Pre-training" %( animal+1 , animalsNum , day+1 , totalDaysNum )
        
        s = initialExState
        h = initialInState
        shiveringList = []        

            

        for trial in range(0,trialsNumPerDay):            

            a = actionSelectionSoftmax ( s )
            nextS = transition         ( s , a )
            
            if a==0 :    shiveringList.append(trial)   # At what minutes of the day shivering happened?
            
            SDR = 0         # Sum of Discounted Rewards
            for miniTrial in range(0,trialsForOneAction):
                temp = 0
                for shiveringNum in range(0,len(shiveringList)):
                    temp = temp + shiveringInducedTemp( trial*trialsForOneAction + miniTrial - shiveringList[shiveringNum]*trialsForOneAction )
                internalSLog[animal][day][trial*trialsForOneAction + miniTrial] = temp
                SDR = SDR + numpy.power(gamma1,miniTrial) * reward ( h , temp )                                                     
                h = temp

            r = SDR

            logging(animal,day,trial,s,a,r)    
    
            updateQ_TD0(s,a,nextS,r)            
#            print '           Q[0,Sh]   ',Q[0][0],'           Q[0,NoSh]   ',Q[0][1],'           Q[2,Sh]   ',Q[2][0],'           Q[2,NoSh]   ',Q[2][1],'  action=' ,a
       
            s=nextS

            if (s==1) or (s==2):
                break
            
        injectionTrial = trial+1
        
        for trial in range(injectionTrial,trialsNumPerDay):            
            for miniTrial in range(0,trialsForOneAction):

                temp = 0
                for shiveringNum in range(0,len(shiveringList)):
                    temp = temp + shiveringInducedTemp( trial*trialsForOneAction + miniTrial - shiveringList[shiveringNum]*trialsForOneAction )
                internalSLog[animal][day][trial*trialsForOneAction + miniTrial] = temp

                SDR = SDR + numpy.power(gamma1, (trial-injectionTrial)*trialsForOneAction + miniTrial) * reward ( h , temp )                                                     
                h = temp

        r = SDR
        a=1
        nextS=3    
        updateQ_TD0(s,a,nextS,r)            


#################################  Acqusition Phase  ######################################
    
    for day in range( pretrainingDaysNum , pretrainingDaysNum+acquisitionDaysNum ):

        print "Animals number: %3d  /%3d          Day %3d  /%3d       Acquisition" %( animal+1 , animalsNum , day+1 , totalDaysNum )
        
        s = initialExState
        h = initialInState
        shiveringList = []        

            

        for trial in range(0,trialsNumPerDay):            

            a = actionSelectionSoftmax ( s )
            nextS = transition         ( s , a )

            if a==0 :    shiveringList.append(trial)   # At what minutes of the day shivering happened?
            
            SDR = 0         # Sum of Discounted Rewards
            for miniTrial in range(0,trialsForOneAction):
                temp = 0
                for shiveringNum in range(0,len(shiveringList)):
                    temp = temp + shiveringInducedTemp( trial*trialsForOneAction + miniTrial - shiveringList[shiveringNum]*trialsForOneAction )
                internalSLog[animal][day][trial*trialsForOneAction + miniTrial] = temp
                SDR = SDR + numpy.power(gamma1,miniTrial) * reward ( h , temp )                                                     
                h = temp

            r = SDR

            logging(animal,day,trial,s,a,r)    

            updateQ_TD0(s,a,nextS,r)            
#            print '           Q[0,Sh]   ',Q[0][0],'           Q[0,NoSh]   ',Q[0][1],'           Q[2,Sh]   ',Q[2][0],'           Q[2,NoSh]   ',Q[2][1],'  action=' ,a
       
       
            s=nextS

            if (s==1) or (s==2):
                break
            
        injectionTrial = trial+1
        
        for trial in range(injectionTrial,trialsNumPerDay):            
            for miniTrial in range(0,trialsForOneAction):

                temp = 0
                for shiveringNum in range(0,len(shiveringList)):
                    temp = temp + shiveringInducedTemp( trial*trialsForOneAction + miniTrial - shiveringList[shiveringNum]*trialsForOneAction )
                temp = temp + ethanolInducedTemp( trial*trialsForOneAction + miniTrial - injectionTrial*trialsForOneAction )
                internalSLog[animal][day][trial*trialsForOneAction + miniTrial] = temp

                SDR = SDR + numpy.power(gamma1, (trial-injectionTrial)*trialsForOneAction + miniTrial) * reward ( h , temp )                                                     
                h = temp

        r = SDR
        a=1
        nextS=3    
        updateQ_TD0(s,a,nextS,r)            


#################################  Extinction Phase  ######################################

    for day in range( pretrainingDaysNum+acquisitionDaysNum , pretrainingDaysNum+acquisitionDaysNum+extinctionDaysNum ):

        print "Animals number: %3d  /%3d          Day %3d  /%3d       Extinction" %( animal+1 , animalsNum , day+1 , totalDaysNum )
                
        
        s = initialExState
        h = initialInState
        shiveringList = []        

            

        for trial in range(0,trialsNumPerDay):            

            a = actionSelectionSoftmax ( s )
            nextS = transition         ( s , a )

            if a==0 :    shiveringList.append(trial)   # At what minutes of the day shivering happened?
            
            SDR = 0         # Sum of Discounted Rewards
            for miniTrial in range(0,trialsForOneAction):
                temp = 0
                for shiveringNum in range(0,len(shiveringList)):
                    temp = temp + shiveringInducedTemp( trial*trialsForOneAction + miniTrial - shiveringList[shiveringNum]*trialsForOneAction )
                internalSLog[animal][day][trial*trialsForOneAction + miniTrial] = temp
                SDR = SDR + numpy.power(gamma1,miniTrial) * reward ( h , temp )                                                     
                h = temp

            r = SDR

            logging(animal,day,trial,s,a,r)    

            updateQ_TD0(s,a,nextS,r)            
#            print '           Q[0,Sh]   ',Q[0][0],'           Q[0,NoSh]   ',Q[0][1],'           Q[2,Sh]   ',Q[2][0],'           Q[2,NoSh]   ',Q[2][1],'  action=' ,a
       
       
            s=nextS

            if (s==1) or (s==2):
                break
            
        injectionTrial = trial+1
        
        for trial in range(injectionTrial,trialsNumPerDay):            
            for miniTrial in range(0,trialsForOneAction):

                temp = 0
                for shiveringNum in range(0,len(shiveringList)):
                    temp = temp + shiveringInducedTemp( trial*trialsForOneAction + miniTrial - shiveringList[shiveringNum]*trialsForOneAction )
                internalSLog[animal][day][trial*trialsForOneAction + miniTrial] = temp

                SDR = SDR + numpy.power(gamma1, (trial-injectionTrial)*trialsForOneAction + miniTrial) * reward ( h , temp )                                                     
                h = temp

        r = SDR
        a=1
        nextS=3    
        updateQ_TD0(s,a,nextS,r)            
     


#################################  Re-acqusition Phase  ###################################

    for day in range( pretrainingDaysNum+acquisitionDaysNum+extinctionDaysNum , pretrainingDaysNum+acquisitionDaysNum+extinctionDaysNum+reacquisitionDaysNum ):

        print "Animals number: %3d  /%3d          Day %3d  /%3d       Re-acquisition" %( animal+1 , animalsNum , day+1 , totalDaysNum )

        s = initialExState
        h = initialInState
        shiveringList = []        

            

        for trial in range(0,trialsNumPerDay):            

            a = actionSelectionSoftmax ( s )
            nextS = transition         ( s , a )

            if a==0 :    shiveringList.append(trial)   # At what minutes of the day shivering happened?
            
            SDR = 0         # Sum of Discounted Rewards
            for miniTrial in range(0,trialsForOneAction):
                temp = 0
                for shiveringNum in range(0,len(shiveringList)):
                    temp = temp + shiveringInducedTemp( trial*trialsForOneAction + miniTrial - shiveringList[shiveringNum]*trialsForOneAction )
                internalSLog[animal][day][trial*trialsForOneAction + miniTrial] = temp
                SDR = SDR + numpy.power(gamma1,miniTrial) * reward ( h , temp )                                                     
                h = temp

            r = SDR

            logging(animal,day,trial,s,a,r)    

            updateQ_TD0(s,a,nextS,r)            
#            print '           Q[0,Sh]   ',Q[0][0],'           Q[0,NoSh]   ',Q[0][1],'           Q[2,Sh]   ',Q[2][0],'           Q[2,NoSh]   ',Q[2][1],'  action=' ,a
       
       
            s=nextS

            if (s==1) or (s==2):
                break
            
        injectionTrial = trial+1
        
        for trial in range(injectionTrial,trialsNumPerDay):            
            for miniTrial in range(0,trialsForOneAction):

                temp = 0
                for shiveringNum in range(0,len(shiveringList)):
                    temp = temp + shiveringInducedTemp( trial*trialsForOneAction + miniTrial - shiveringList[shiveringNum]*trialsForOneAction )
                temp = temp + ethanolInducedTemp( trial*trialsForOneAction + miniTrial - injectionTrial*trialsForOneAction )
                internalSLog[animal][day][trial*trialsForOneAction + miniTrial] = temp

                SDR = SDR + numpy.power(gamma1, (trial-injectionTrial)*trialsForOneAction + miniTrial) * reward ( h , temp )                                                     
                h = temp

        r = SDR
        a=1
        nextS=3    
        updateQ_TD0(s,a,nextS,r)    
        

plotting()
