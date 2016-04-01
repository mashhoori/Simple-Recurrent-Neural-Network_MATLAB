classdef SimpleRNN < handle
    
    properties
        w_xh    % matrix of weights between the input and hidden layers
        w_hh    % matrix of weights for recurrent connections in hidden layer
        w_ho    % matrix of weights between the hidden and output layers
        
        gw_xh   % gradient matrix of weights between the input and hidden layers
        gw_hh   % gradient matrix of weights for recurrent connections in hidden layer
        gw_ho   % gradient matrix of weights between the hidden and output layers
        
        input
        hidden
        output
        error 
    end    
    
    methods
        
        % Create a SimpleRNN object ans initialize the weights
        function rnn = SimpleRNN(inputNeuronCount, hiddenNeuronCount, outputNeuronCount)
            rnn.w_xh = randn(inputNeuronCount+1, hiddenNeuronCount)* 0.01;
            rnn.w_hh = randn(hiddenNeuronCount, hiddenNeuronCount)* 0.01;
            rnn.w_ho = randn(hiddenNeuronCount+1, outputNeuronCount)* 0.01;
        end
        
        % Calculate the output of the network for the given input.
        % If 'reset' is 1, the values for the first recurrent inputs are set
        % to zero, and if 'reset' is 0, those inputs are set to the
        % last activation of the hidden layer.
        function output = CalculateOutput(this, input, reset)
            
            numSteps = size(input, 1);  % number of steps in the input   
            
            if(reset)
                % Reset the activity of the recurrent inputs
                h_p = zeros(1, size(this.w_hh, 1));
            else
                h_p = this.hidden(end, :);
            end
            
            % Initialize the matrix for the activation of the hidden layer
            % for each step
            h = zeros(numSteps, size(this.w_hh, 1)); 
            
            % Initialize the matrix for the output of the network for each step
            o = zeros(numSteps, size(this.w_ho, 2));
            
            for i=1:numSteps
                batchInput = [1 input(i, :)]; % Augment the input by adding a '1' (bias input)
                h(i, :) = tanh(batchInput * this.w_xh + h_p * this.w_hh); 
                batchH = [1  h(i, :)];
                o(i, :) = batchH * this.w_ho;
                
                h_p = h(i, :);
            end
            
            % Calculate the final output using softmax
            maxValues = max(o, [], 2);
            netInput = bsxfun(@minus, o, maxValues);
            netInput = exp(netInput);
            o = bsxfun(@rdivide, netInput, sum(netInput, 2));            
            
            this.input = input;
            this.hidden = h;
            this.output = o;
            output = o;            
        end
        
        % Calculate the cross entropy between the target and predicted
        % values and also, compute the error
        function crossentropy = CalculateCost(this, target)
            crossentropy = sum(sum(-target .* log(this.output)));
            this.error = this.output - target;   
        end
        
        % Compute the gradient for each parameter in the network based on
        % the error field
        function ComputeGradientAndBackpropagateErrorSignal(this)
            
            % Initialize the gradient matrices
            this.gw_xh = zeros(size(this.w_xh));
            this.gw_hh = zeros(size(this.w_hh));
            this.gw_ho = zeros(size(this.w_ho));
            
            numSteps = size(this.output, 1);        
            backwardError = zeros(numSteps, size(this.w_hh, 1));
            
            for i = numSteps:-1:1
                batchH = [1 this.hidden(i, :)];                
                this.gw_ho = this.gw_ho + (batchH' * this.error(i, :));
                
                preError = this.error(i, :) * this.w_ho(2:end, :)';
                tmp = (preError + backwardError(i, :)) .* (1 - this.hidden(i, :).^2);
                
                batchInput = [1 this.input(i, :)];                                
                this.gw_xh = this.gw_xh + (batchInput' * tmp);                  
                
                if i > 1                    
                    this.gw_hh = this.gw_hh + (this.hidden(i-1, :)' * tmp);
                    backwardError(i-1, :)  = tmp * this.w_hh' ;
                end
            end             
        end
        
        % Update all the parameters
        function UpdateParameters(this)
            learningRate = 0.001;
            reg = 0.000;             
            
            % Update using gradianet descent with weight decay
            this.w_xh = this.w_xh - learningRate * (this.gw_xh - reg * this.w_xh);
            this.w_hh = this.w_hh - learningRate * (this.gw_hh - reg * this.w_hh);
            this.w_ho = this.w_ho - learningRate * (this.gw_ho - reg * this.w_ho);            
        end
        
        % Sample a text sequence from the network
        function sample = GenerateSample(this)
            
            numSteps = 100; % The length of the sequence
            
            % The first character is selected as the first character in internal input matrix
            inputChar = this.input(1, :);             
            h_p = this.hidden(1, :);      
            
            sample = zeros(1, numSteps);
            for i=1:numSteps
                batchInput = [1 inputChar]; % Augment the input by adding a '1' (bias input)
                h_p = tanh(batchInput * this.w_xh + h_p * this.w_hh); 
                batchH = [1  h_p];
                o = batchH * this.w_ho;
                
                % Convert to probablity using softmax
                netInput = exp(o - max(o));
                netInput = netInput / 0.3;
                o = netInput / sum(netInput);
                
                % Pick a character accrding to the output probability
                % distribution
                index = find(rand < cumsum(o), 1);
                inputChar = zeros(size(inputChar));
                inputChar(index) = 1;           
                
                sample(i) = index;
            end              
            
        end
                
    end




end