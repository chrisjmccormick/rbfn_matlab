function [Centers, betas, Theta] = trainFuncApproxRBFN(X_train, y_train, numRBFNeurons, normalize, beta, verbose)
% TRAINFUNCAPPROXRBFN Builds an RBF Network from the provided training set.
%   [Centers, betas, Theta] = trainFuncApproxRBFN(X_train, y_train, numRBFNeurons, normalize, beta, verbose)
%
%   When using an RBFN for function approximation, all RBF neurons will use the
%   same 'beta' value (which controls the "width" of the neurons). This beta 
%   value is supplied by the user and is usually determined through 
%   experimentation.
%
%   There are only two main steps to the training process:
%     1. Prototype selection through k-means clustering.
%     2. Training the output weights.
%
%   Parameters
%     X_train  - The training vectors, one per row
%     y_train  - Function output value for the corresponding training point in X_train.
%     numRBFNeurons - How many RBF neurons to use.
%     normalize - Whether to normalize the RBF neuron activations.
%     verbose  - Whether to print out messages about the training status.
%
%   Returns
%     Centers  - The prototype vectors stored in the RBF neurons.
%     betas    - The beta coefficient for each corresponding RBF neuron.
%     Theta    - The weights for the output layer. There is one row per 
%                neuron, and only one column.

% $Author: ChrisMcCormick $    $Date: 2015/08/24 22:00:00 $    $Revision: 1.0 $
    
    % Set 'm' to the number of data points.
    m = size(X_train, 1);
    
    % ================================================
    %       Select RBF Centers and Parameters
    % ================================================
    
    if (verbose)
        disp('1. Selecting centers through k-Means.');
    end    
    
    % ================================
    %      Find cluster centers
    % ================================
    
    % Pick random samples as the initial centroids.
    init_Centroids = kMeansInitCentroids(X_train, numRBFNeurons);
    
    % Run k-means clustering, with at most 100 iterations.
    [Centers, memberships] = kMeans(X_train, init_Centroids, 100);
        
    % ================================
    %    Compute Beta Coefficients
    % ================================
    
    % Simply use the same user-provided beta coefficient for all neurons.
    betas = ones(size(Centers, 1), 1) * beta;
    
    % ===================================
    %        Train Output Weights
    % ===================================

    % ==========================================================
    %       Compute RBF Activations Over The Training Set
    % ===========================================================
    if (verbose)
        disp('2. Calculate RBF neuron activations over full training set.');
    end

    % Get the final number of RBF neurons.
    numRBFNeurons = size(Centers, 1);
    
    % First, compute the RBF neuron activations for all training examples.

    % The X_activ matrix stores the RBF neuron activation values for each training 
    % example: one row per training example and one column per RBF neuron.
    X_activ = zeros(m, numRBFNeurons);

    % For each training example...
    for (i = 1 : m)
       
        input = X_train(i, :);
       
       % Get the activation for all RBF neurons for this input.
        p = getRBFActivations(Centers, betas, input);
        
        % Store the activation values 'z' for training example 'i'.
        X_activ(i, :) = p';
    end

    if (normalize)
        % X_activ contains 1 row per training example, and one column per RBF neuron.
        % Each row holds the neuron activations for that training example.
        %
        % Divide each neuron activation by the sum of the neuron activations for that
        % training example.
        X_activ = bsxfun(@rdivide, X_activ, sum(X_activ, 2));
    end
    
    % Add a column of 1s for the bias term.
    X_activ = [ones(m, 1), X_activ];
    
    % =============================================
    %          Learn Output Weights
    % =============================================

    if (verbose)
        disp('3. Learn output weights.');
    end

    % Calculate all of the output weights using a matrix inverse operation.
    % There is one column per category / output neuron.
    Theta = pinv(X_activ' * X_activ) * X_activ' * y_train;   
    
end

