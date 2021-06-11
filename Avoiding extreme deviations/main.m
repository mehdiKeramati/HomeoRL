% Only one particle is simulated 

function main()
    
    trialsNum                   = 1000000 ;
    window                      = 55   ;

    alpha                       = 0.4   ;
    gamma                       = 0.9   ;
    beta                        = 0.05  ;
    m                           = 3     ;
    n                           = 4     ;
    x                           = 30     ;     

  
    v = zeros(trialsNum,2);
    experienced = zeros(trialsNum,1);    
        
    dynamics = zeros(500,100);
    
    for trial = 1:trialsNum

        
        l=10;
        if trial<=500*l
            if mod(trial,l)==0
                dynamics(trial/l,x+50)=1;
            end
        end
            
        index = x +(trialsNum/2);        
        experienced(index) = experienced(index) + 1;
        a=action(v(index,1),v(index,2),beta);
        if a==1
            actionIndex=2;
        else 
            actionIndex=1;
        end      
        v(index,actionIndex) = v(index,actionIndex) +  alpha * (reward(x,a,m,n) + gamma*max(v(index+a,1),v(index+a,2))-v(index,actionIndex));

        x = x + a ;
        
    end

%######################## PLOT trace    
    figure('Position', [100, 100, 450, 300]);
    imshow(dynamics);
    
%######################## PLOT visits    
    figure('Position', [100, 100, 450, 300]);;
    set(0,'DefaultAxesFontName', 'Calibri')
    set(0,'DefaultAxesFontSize', 24)
    set(0,'DefaultAxesFontWeight', 'normal')

    bar(-window:window,experienced(trialsNum/2 - window :trialsNum/2 + window));

    axis([-inf,inf,0,inf]);
    xlabel('internal states');
    ylabel('number of visits');     
    
%######################## PLOT values    
    figure('Position', [100, 100, 450, 300]);;
    p1=plot(-window:window,v(trialsNum/2 - window :trialsNum/2 + window,1),'red',  'linewidth', 2);
    hold on;
    p2=plot(-window:window,v(trialsNum/2 - window :trialsNum/2 + window,2),'blue', 'linewidth', 2);

    axis([-inf,inf,-inf,inf]);
    xlabel('internal states');
    ylabel('value');     
%    legend( 'left','right');
%    legend boxoff

%######################## PLOT action probabilities    
    figure('Position', [100, 100, 450, 300]);;
    p = zeros(trialsNum,2);
    for trial = 1:trialsNum
        p1 = exp(v(trial,1)*beta);
        p2 = exp(v(trial,2)*beta);
        sum = p1+p2;
        p1=p1/sum;    
        p2=p2/sum;    
        p(trial,1) = p1;
        p(trial,2) = p2;
        
    end
    
    plot(-window:window,p(trialsNum/2 - window :trialsNum/2 + window,1),'red', 'linewidth', 2);
    hold on;
    plot(-window:window,p(trialsNum/2 - window :trialsNum/2 + window,2),'blue', 'linewidth', 2);

    axis([-inf,inf,0,1]);
    xlabel('internal states');
    ylabel('action probability'); 
%    legend( 'left','right' );
%    legend boxoff
    
%######################## PLOT drive function    
    figure('Position', [100, 100, 450, 300]);;
    d = zeros(trialsNum,1);
    for trial = 1:trialsNum
        d(trial) = (abs(trial-trialsNum/2))^(n/m);
    end
    
    plot(-window:window,d(trialsNum/2 - window :trialsNum/2 + window),'black', 'linewidth', 2);

    axis([-inf,inf,-50,250]);
    xlabel('internal states');
    ylabel('drive'); 
    

    
function r=reward(x,a,m,n);
    d1 = (abs(x))^(n/m);
    d2 = (abs(x+a))^(n/m);
    r = d1-d2;
    
   
    
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