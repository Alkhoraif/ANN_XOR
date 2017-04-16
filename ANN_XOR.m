X = [0 0; 0 1; 1 0; 1 1]; % input
Y = [0; 1; 1; 0];  % output


patterns = 4;
epochs = 60000; 

%add button for early stopping
hstop = uicontrol('Style','PushButton','String','Stop', 'Position', [5 5 70 20],'callback','earlystop = 1;'); 
earlystop = 0;

%add button for resetting weights
hreset = uicontrol('Style','PushButton','String','Reset Wts', 'Position', get(hstop,'position')+[75 0 0 0],'callback','reset = 1;'); 
reset = 0;

%add slider to adjust the learning rate
hlr = uicontrol('Style','slider','value',.1,'Min',.01,'Max',1,'SliderStep',[0.01 0.1],'Position', get(hreset,'position')+[75 0 100 0]);


% ---------- set weights -----------------
%set initial random weights
Wh = (randn(2,4) - 0.5)/10;
Wz = (randn(1,4) - 0.5)/10;



%-----------------------------------
%--- Learning Starts Here! ---------
%-----------------------------------

%do a number of epochs
for iter = 1:epochs
    
    %get the learning rate from the slider
    alr = get(hlr,'value');
    blr = alr / 10;
    
    %loop through the patterns, selecting randomly
    for j = 1:patterns
        
      
       
        %set the current pattern
        P = X(j,:);
        act = Y(j,1);
        
        %calculate the current error for this pattern
        H = (tanh(P*Wh))';
        Z = H'*Wz';
        error = Z - act;

        % adjust weight hidden - output
        delta_HO = error.*blr .*H;
        Wz = Wz - delta_HO';

        % adjust the weights input - hidden
        delta_IH= alr.*error.*Wz'.*(1-(H.^2))*P;
        Wh = Wh - delta_IH';
        
    end
    % -- another epoch finished
    
    %plot overall network error at end of each epoch
    Z = Wz*tanh(X*Wh)';
    error = Z' - Y;
    err(iter) =  (sum(error.^2))^0.5;
    
    figure(1);
    plot(err)
    
    
    %reset weights if requested
    if reset
        Wh = (randn(2,4) - 0.5)/10;
        Wz = (randn(1,4) - 0.5)/10;
        fprintf('weights reaset after %d epochs\n',iter);
        reset = 0;
    end
    
    %stop if requested
    if earlystop
        fprintf('stopped at epoch: %d\n',iter); 
        break 
    end 

    %stop if error is small
    if err(iter) < 0.001
        fprintf('converged at epoch: %d\n',iter);
        break 
    end
       
end

fprintf('the result after training');
Z