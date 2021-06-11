% Multiple particles are simulated, and the variance of the particles'
% position is computed...

function main()
    
    particlesNum                = 1000 ;
    trialsNum                   = 1500 ;
    window                      = 55   ;

    alpha                       = 0.4   ;
    gamma                       = 0.9   ;
    beta                        = 0.05  ;
    m                           = 3     ;
    n                           = 4     ;
    initialState                = 30     ;     


    maxX = 0 ;
    minX = 0 ;
    
    position = zeros(particlesNum,trialsNum); 
    mn = zeros(trialsNum,1);   %mean
    sd = zeros(trialsNum,1);   %standard deviations 

    
    
    for particle = 1:particlesNum

        particle
        
        x = initialState ;
        v = zeros(trialsNum,2);
        
        for trial = 1:trialsNum

            index = x +(trialsNum/2);        
            a=action(v(index,1),v(index,2),beta);
            if a==1
                actionIndex=2;
            else 
                actionIndex=1;
            end      
            v(index,actionIndex) = v(index,actionIndex) +  alpha * (reward(x,a,m,n) + gamma*max(v(index+a,1),v(index+a,2))-v(index,actionIndex));

            position(particle,trial) = x ;                        

            x = x + a ;

            
            if x>maxX
                maxX=x;
            end
            if x<minX
                minX=x;
            end
            
            
        end
    end
    

%######################## Pre-processing   

for trial = 1:trialsNum
    mn(trial) = mean (position(:,trial));
    sd(trial) = std  (position(:,trial));
end

%######################## PLOT mean    
    figure('Position', [100, 100, 450, 300]);;
    set(0,'DefaultAxesFontName', 'Calibri')
    set(0,'DefaultAxesFontSize', 24)
    set(0,'DefaultAxesFontWeight', 'normal')
    
    plot(mn,'black','linewidth', 2);
    axis([-inf,inf,-1.5,1.5]);
    xlabel('trial');
    ylabel('mean');     

%######################## PLOT standard deviation    
    figure('Position', [100, 100, 450, 300]);;
    plot(sd,'black','linewidth', 2);
    axis([-inf,inf,0,11]);
    xlabel('trial');
    ylabel('SD');     


    maxX
    minX
    
    
%################################################
%############      FUNCTIONS       ##############
%################################################    
    
%######################## drive-reduction computation    
function r=reward(x,a,m,n);
    d1 = (abs(x))^(n/m);
    d2 = (abs(x+a))^(n/m);
    r = d1-d2;
       
%######################## softmax action selection        
function a = action(v1,v2,beta);
    p1 = exp(v1*beta);
    p2 = exp(v2*beta);
    sum = p1+p2;
    p1=p1/sum;
    if rand<=p1
        a=-1;
    else
        a=1;
    end    
    