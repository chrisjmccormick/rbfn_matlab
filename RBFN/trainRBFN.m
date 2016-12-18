function [Centers, betas, Theta] = trainRBFN(X_train, y_train, centersPerCategory, verbose)
% TRAINRBFN Builds an RBF Network from the provided training set.
%   [Centers, betas, Theta] = trainRBFN(X_train, y_train, centersPerCategory, verbose)
%    
%   There are three main steps to the training process:
%     1. Prototype selection through k-means clustering.
%     2. Calculation of beta coefficient (which controls the width of the 
%        RBF neuron activation function) for each RBF neuron.
%     3. Training of output weights for each category using gradient descent.
%
%   Parameters
%     X_train  - The training vectors, one per row
%     y_train  - The category values for the corresponding training vector.
%                Category values should be continuous starting from 1. (e.g.,
%                1, 2, 3, ...)
%     centersPerCategory - How many RBF centers to select per category. k-Means
%                          requires that you specify 'k', the number of 
%                          clusters to look for.
%     verbose  - Whether to print out messages about the training status.
%
%   Returns
%     Centers  - The prototype vectors stored in the RBF neurons.
%     betas    - The beta coefficient for each coressponding RBF neuron.
%     Theta    - The weights for the output layer. There is one row per neuron
%                and one column per output node / category.

% $Author: ChrisMcCormick $    $Date: 2014/08/18 22:00:00 $    $Revision: 1.3 $

    % Get the number of unique categories in the dataset.
    numCats = size(unique(y_train), 1);
    
    % Set 'm' to the number of data points.
    m = size(X_train, 1);
    
    % Ensure category values are non-zero and continuous.
    % This allows the index of the output node to equal its category (e.g.,
    % the first output node is category 1).
    if (any(y_train == 0) || any(y_train > numCats))
        error('Category values must be non-zero and continuous.');
    end
    
    % ================================================
    %       Select RBF Centers and Parameters
    % ================================================
    % Here I am selecting the cluster centers using k-Means clustering.
    % I've chosen to separate the data by category and cluster each 
    % category separately, though I've read that this step is often done 
    % over the full unlabeled dataset. I haven't compared the accuracy of 
    % the two approaches.
    
    if (verbose)
        disp('1. Selecting centers through k-Means.');
    end    
    
    Centers = [];
    betas = [];    
    
    % For each of the categories...
    for (c = 1 : numCats)

        if (verbose)
            fprintf('  Category %d centers...\n', c);
            if exist('OCTAVE_VERSION') fflush(stdout); end;
        end
        
        % Select the training vectors for category 'c'.
        Xc = X_train((y_train == c), :);

        % ================================
        %      Find cluster centers
        % ================================
        
        % Pick the first 'centersPerCategory' samples to use as the initial
        % centers.
        init_Centroids = Xc(1:centersPerCategory, :);
        
        % Run k-means clustering, with at most 100 iterations.
        [Centroids_c, memberships_c] = kMeans(Xc, init_Centroids, 100);    
        
        % Remove any empty clusters.
        toRemove = [];
        
        % For each of the centroids...
        for (i = 1 : size(Centroids_c, 1))
            % If this centroid has no members, mark it for removal.
            if (sum(memberships_c == i) == 0)        
                toRemove = [toRemove; i];
            end
        end
        
        % If there were empty clusters...
        if (~isempty(toRemove))
            % Remove the centroids of the empty clusters.
            Centroids_c(toRemove, :) = [];
            
            % Reassign the memberships (index values will have changed).
            memberships_c = findClosestCentroids(Xc, Centroids_c);
        end
        
        % ================================
        %    Compute Beta Coefficients
        % ================================
        if (verbose)
            fprintf('  Category %d betas...\n', c);
            if exist('OCTAVE_VERSION') fflush(stdout); end;
        end

        % Compute betas for all the clusters.
        betas_c = computeRBFBetas(Xc, Centroids_c, memberships_c);
        
        % Add the centroids and their beta values to the network.
        Centers = [Centers; Centroids_c];
        betas = [betas; betas_c];
    end

    % Get the final number of RBF neurons.
    numRBFNeurons = size(Centers, 1);
    
    % ===================================
    %        Train Output Weights
    % ===================================

    % ==========================================================
    %       Compute RBF Activations Over The Training Set
    % ===========================================================
    if (verbose)
        disp('2. Calculate RBF neuron activations over full training set.');
    end

    % First, compute the RBF neuron activations for all training examples.

    % The X_activ matrix stores the RBF neuron activation values for each 
    % training example: one row per training example and one column per RBF
    % neuron.
    X_activ = zeros(m, numRBFNeurons);

    % For each training example...
    for (i = 1 : m)
       
        input = X_train(i, :);
       
       % Get the activation for all RBF neurons for this input.
        z = getRBFActivations(Centers, betas, input);
       
        % Store the activation values 'z' for training example 'i'.
        X_activ(i, :) = z';
    end

    % Add a column of 1s for the bias term.
    X_activ = [ones(m, 1), X_activ];

    % =============================================
    %        Learn Output Weights
    % =============================================

    if (verbose)
        disp('3. Learn output weights.');
    end

    % Create a matrix to hold all of the output weights.
    % There is one column per category / output neuron.
    Theta = zeros(numRBFNeurons + 1, numCats);

    % For each category...
    for (c = 1 : numCats)

        % Make the y values binary--1 for category 'c' and 0 for all other
        % categories.
        y_c = (y_train == c);

        % Use the normal equations to solve for optimal theta.
        Theta(:, c) = pinv(X_activ' * X_activ) * X_activ' * y_c;
    end
    
end

