import numpy
import scipy
import pylab
import cmath

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
    temperature = +7.732*numpy.exp(-0.004077*lag) - 7.732*numpy.exp(-0.005204*lag)
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
    
#    plotInternalState ()
    plotShiveringBeforeCSPretraining ()
    plotShiveringAfterCSPretraining  ()
    plotQValues  ()

    pylab.show()   

    return
    
#---------------------   Plotting some desired variables
def plotQValues():
    
    fig1 = pylab.figure( figsize=(10,4) )
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)


    Q00   = numpy.zeros( [ totalDaysNum ] , float)
    Q01   = numpy.zeros( [ totalDaysNum ] , float)
    Q20   = numpy.zeros( [ totalDaysNum ] , float)
    Q21   = numpy.zeros( [ totalDaysNum ] , float)
    Q4    = numpy.zeros( [ totalDaysNum ] , float)
    Q5   = numpy.zeros( [ totalDaysNum ] , float)

    for day in range(0,totalDaysNum ):            
        Q00[day] = QLog[0][day][0][0][0]
        Q01[day] = QLog[0][day][0][0][1]
        Q20[day] = QLog[0][day][0][2][0]
        Q21[day] = QLog[0][day][0][2][1]
        Q4 [day] = QLog[0][day][0][4][1]
        Q5 [day] = QLog[0][day][0][5][1]


    h00 = ax1.plot( Q00, linewidth = 2 , color = 'black')
    h01 = ax1.plot( Q01, linewidth = 2 , color = 'red')
    h20 = ax1.plot( Q20, linewidth = 2 , color = 'blue')
    h21 = ax1.plot( Q21, linewidth = 2 , color = 'green')
    h4  = ax1.plot( Q4 , linewidth = 2 , color = 'yellow')
    h5  = ax1.plot( Q5 , linewidth = 2 , color = 'pink')

    leg = fig1.legend((h00, h01 , h20 , h21, h4 , h5), ('0 : Sh','0 : NoSh','2 : Sh','2 : NoSh','4','5'), loc = (0.20,0.20))
    leg.draw_frame(False)

    p = pylab.axvspan( pretrainingDaysNum , pretrainingDaysNum+acquisitionDaysNum , facecolor='0.75',edgecolor='none', alpha=0.5)        


    ax1.set_title('Q-values')
    fig1.savefig('Q-values.eps', format='eps')    
    
    return

#---------------------   Plotting some desired variables
def plotInternalState():
    
    fig1 = pylab.figure( figsize=(10,4) )
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    h0 = ax1.plot(internalSLog[0][pretrainingDaysNum] , linewidth = 2 , color = 'black')
    h9 = ax1.plot(internalSLog[0][pretrainingDaysNum+acquisitionDaysNum-1] , linewidth = 2 , color = 'red')

    pylab.set_title('interalState')
    pylab.savefig('interalState.eps', format='eps')    
    
    return
    
#---------------------   Plotting some desired variables
def plotShiveringBeforeCSPretraining():
    
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
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    x = numpy.arange(1, pretrainingDaysNum+1, 1)
    h0 = ax1.plot(x,shiver[0:pretrainingDaysNum] , linewidth = 2 , color = 'black')
    
    pylab.yticks(pylab.arange(0, 1.05, 0.2))
    pylab.ylim((-0.1,1.1))
    pylab.xlim((-10,pretrainingDaysNum+10))

    for line in ax1.get_yticklines() + ax1.get_xticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)
    
    ax1.set_ylabel('Probability of tolerance-response\ninitiation before CS presentation')
    ax1.set_xlabel('Pre-training day')
    fig1.savefig('ShiveringBeforeCSPretraining.eps', format='eps')    

    #--------------------------- Acquisition
    fig1 = pylab.figure( figsize=(6,3.5) )
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    x = numpy.arange(1, acquisitionDaysNum+1, 1)
    h0 = ax1.plot(x,shiver[pretrainingDaysNum:pretrainingDaysNum+acquisitionDaysNum] , linewidth = 2 , color = 'black')
    
    pylab.yticks(pylab.arange(0, 1.05, 0.2))
    pylab.ylim((-0.1,1.1))
    pylab.xlim((-10,acquisitionDaysNum+10))

    for line in ax1.get_yticklines() + ax1.get_xticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)
    
    ax1.set_ylabel('Probability of tolerance-response\ninitiation before CS presentation')
    ax1.set_xlabel('Acquisition day')
    fig1.savefig('ShiveringBeforeCSAcquisition.eps', format='eps')    

    #--------------------------- Extinction
    fig1 = pylab.figure( figsize=(6,3.5) )
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    x = numpy.arange(1, extinctionDaysNum+1, 1)
    h0 = ax1.plot(x,shiver[pretrainingDaysNum+acquisitionDaysNum:pretrainingDaysNum+acquisitionDaysNum+extinctionDaysNum] , linewidth = 2 , color = 'black')
    
    pylab.yticks(pylab.arange(0, 1.05, 0.2))
    pylab.ylim((-0.1,1.1))
    pylab.xlim((-10,extinctionDaysNum+10))

    for line in ax1.get_yticklines() + ax1.get_xticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)
    
    ax1.set_ylabel('Probability of tolerance-response\ninitiation before CS presentation')
    ax1.set_xlabel('Extinction day')
    fig1.savefig('ShiveringBeforeCSExtinction.eps', format='eps')    
       
    return
#---------------------   Plotting some desired variables
def plotShiveringAfterCSPretraining():


    shiver   = numpy.zeros( [ totalDaysNum ] , float)

    for animal in range(0,animalsNum):            
        for day in range(0,totalDaysNum ):            
            for trial in range(0,trialsNumPerDay):            
                if ((stateLog[animal][day][trial][2]==1) and  (actionLog[animal][day][trial][0]==1)):
                    shiver[day]   = shiver[day]   + 1

    for day in range(0,totalDaysNum):            
        shiver[day]   = shiver[day]   / animalsNum
        
        
    #--------------------------- Pretraining
    fig1 = pylab.figure( figsize=(6,3.5) )
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    x = numpy.arange(1, pretrainingDaysNum+1, 1)
    h0 = ax1.plot(x,shiver[0:pretrainingDaysNum] , linewidth = 2 , color = 'black')
    
    pylab.yticks(pylab.arange(0, 1.05, 0.2))
    pylab.ylim((-0.1,1.1))
    pylab.xlim((-10,pretrainingDaysNum+10))

    for line in ax1.get_yticklines() + ax1.get_xticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)
    
    ax1.set_ylabel('Probability of tolerance-response\ninitiation upon CS presentation')
    ax1.set_xlabel('Pre-training day')
    fig1.savefig('ShiveringUponCSPretraining.eps', format='eps')    

    #--------------------------- Acquisition
    fig1 = pylab.figure( figsize=(6,3.5) )
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    x = numpy.arange(1, acquisitionDaysNum+1, 1)
    h0 = ax1.plot(x,shiver[pretrainingDaysNum:pretrainingDaysNum+acquisitionDaysNum] , linewidth = 2 , color = 'black')
    
    pylab.yticks(pylab.arange(0, 1.05, 0.2))
    pylab.ylim((-0.1,1.1))
    pylab.xlim((-10,acquisitionDaysNum+10))

    for line in ax1.get_yticklines() + ax1.get_xticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)
    
    ax1.set_ylabel('Probability of tolerance-response\ninitiation upon CS presentation')
    ax1.set_xlabel('Acquisition day')
    fig1.savefig('ShiveringUponCSAcquisition.eps', format='eps')    

    #--------------------------- Extinction
    fig1 = pylab.figure( figsize=(6,3.5) )
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    x = numpy.arange(1, extinctionDaysNum+1, 1)
    h0 = ax1.plot(x,shiver[pretrainingDaysNum+acquisitionDaysNum:pretrainingDaysNum+acquisitionDaysNum+extinctionDaysNum] , linewidth = 2 , color = 'black')
    
    pylab.yticks(pylab.arange(0, 1.05, 0.2))
    pylab.ylim((-0.1,1.1))
    pylab.xlim((-10,extinctionDaysNum+10))

    for line in ax1.get_yticklines() + ax1.get_xticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)
    
    ax1.set_ylabel('Probability of tolerance-response\ninitiation upon CS presentation')
    ax1.set_xlabel('Extinction day')
    fig1.savefig('ShiveringUponCSExtinction.eps', format='eps')    

       
    return


#==================================================================================================================================================
#======   The code starts running from here   =====================================================================================================
#==================================================================================================================================================

#################################################################
#                        Environment's Dynamics
#################################################################


coldProbability = 0.05

# --------------------   Number of the States
statesNum  = 9

# --------------------   Number of the Actions
actionsNum = 2

# --------------------   Initial External State
initialExState = 0

# --------------------   Final External States
#finalExStates = [6,8]

# --------------------   Transition Function : (from state s, by action a, going to state s', by probability p)
T = numpy.zeros( [ statesNum , actionsNum, statesNum ] , float)
setTransition(0,0,1,1-coldProbability)
setTransition(0,0,3,  coldProbability)
setTransition(0,1,0,1-coldProbability)
setTransition(0,1,2,  coldProbability)
setTransition(1,1,1,1-coldProbability)
setTransition(1,1,3,  coldProbability)
setTransition(2,0,5,1                )
setTransition(2,1,4,1                )
setTransition(3,1,6,1                )
setTransition(4,1,7,1                )
setTransition(5,1,7,1                )
setTransition(6,1,7,1                )
setTransition(7,1,8,1                )

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
alpha           = 0.04          # Learning rate
gamma1          = 0.96         # Discount factor, for every minute
gamma10         = 0.66         # = gamma1^10 : Discount factor, for every 10 minutes

beta            = 1           # Rate of exploration

m               = 2
n               = 2

#################################################################
#                        Simulation Initializations
#################################################################

# -------------------- Simulations Variables
trialsForOneAction  = 10

trialsNumPerDay     = (60/trialsForOneAction)*24     


pretrainingDaysNum  = 2000
acquisitionDaysNum  = 6000
extinctionDaysNum   = 2000
reacquisitionDaysNum= 1
totalDaysNum        = pretrainingDaysNum + acquisitionDaysNum + extinctionDaysNum + reacquisitionDaysNum

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

            if (s==4) or (s==5) or (s==6):
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
        nextS=7    
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

            if (s==4) or (s==5) or (s==6):
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
        nextS=7    
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

            if (s==4) or (s==5) or (s==6):
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
        nextS=7    
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

            if (s==4) or (s==5) or (s==6):
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
        nextS=7    
        updateQ_TD0(s,a,nextS,r)    
        

plotting()
