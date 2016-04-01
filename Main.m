
% Read the text file
text = fileread('big.txt');

% Create a list of unique characters in the text
charList = num2cell(unique(text));
numChar  = numel(charList);

% Create maps from characters to integers and vice versa
charToIndex = containers.Map(charList, 1:numel(charList));
indexToChar = containers.Map(1:numel(charList), charList);

% Convert characters to integer indices
textToIndex = values(charToIndex, num2cell(text));
textToIndex = cell2mat(textToIndex); 

% Create the one-hot code matrix 
inputMatrix = dummyvar(textToIndex);

numUnitsHidden = 100; % number of hidden units in the network

% Create a simple recurrent network having 'numChar' number of units in the
% input and output layers and 'numUnitsHidden' number of units in the
% hidden layer
sm = SimpleRNN(numChar, numUnitsHidden, numChar);

numSteps = 100; % Number of steps to look back

% Initialize cost, assuming that the network assigns equal probablity to
% all characters.
costTotal = -log(1/numChar) * numSteps; 


seqNum = 1; % Sequence number
iteration = 0; % Iteration number
position = 1; % Starting position of the next sequence
while (true)
    reset = 0;
    % Check if it is the beginning of a new iteration
    if(position+numSteps > length(inputMatrix) || iteration == 0) 
        position = 1;
        reset = 1; % To reset the activities of the first recurrent inputs
        seqNum = 1;
        iteration = iteration + 1;
    end
    
    % Select the input for the current sequence
    input = inputMatrix(position : position+numSteps-1, :);
    
    % Select the target for the current sequence
    % For each input character in the sequence, the target is the next character
    target = inputMatrix(position+1 : position+numSteps, :);
    
    % Calculate the outputs for the current sequence
    sm.CalculateOutput(input, reset);
    
    % Calculate the crossentropy between the output of the network and
    % target values
    cost = sm.CalculateCost(target);
    
    % Computer the gradient for each parameter of the network based on the
    % error
    sm.ComputeGradientAndBackpropagateErrorSignal()
    
    % Update all the parametes by gradient descent
    sm.UpdateParameters();

    % Update the smoothed cost 
    costTotal = 0.999 * costTotal + 0.001 * cost;

    if(mod(seqNum, 100) == 0)
        fprintf('iteration:%d   sample:%d   cost:%f\n', iteration, seqNum, costTotal);
        
        % Generate a sample text sequence from the network
        sample = sm.GenerateSample();
        
        % Convert integer indices to characters
        sampleChar = values(indexToChar, num2cell(sample));
        sampleText = cell2mat(sampleChar); 
        
        fprintf('Sample: \n %s \n ', sampleText);
    end
    
    % Update position and seqNum 
    position = position + numSteps;
    seqNum = seqNum + 1;
end