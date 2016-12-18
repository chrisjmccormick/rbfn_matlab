function betas = computeRBFBetas(X, centroids, memberships)
% COMPUTERBFBETAS Computes the beta coefficients for all of the specified 
% centroids.
%   betas = computeRBFBetas(X, centroids, memberships)
%   
%   This function computes the beta coefficients based on the average distance
%   between a cluster's data points and its center. The average distance is 
%   called sigma, and beta = 1 / (2*sigma^2).
%
%   Parameters:
%     X           - Matrix of all training samples, one per row.
%     centroids   - Matrix of cluster centers, one per row
%     memberships - Vector specifying the cluster membership of each data point
%                   in X. The membership is specified as the row index of the
%                   centroid in 'centroids'.
%                   
%   Returns:
%     A vector containing the beta coefficient for each centroid.

% $Author: ChrisMcCormick $    $Date: 2014/04/08 22:00:00 $    $Revision: 1.2 $

    numRBFNeurons = size(centroids, 1);

    % Compute sigma for each cluster.
    sigmas = zeros(numRBFNeurons, 1);
    
    % For each cluster...
    for (i = 1 : numRBFNeurons)
        % Select the next cluster centroid.
        center = centroids(i, :);

        % Select all of the members of this cluster.
        members = X((memberships == i), :);

        % Compute the average L2 distance to all of the members. 
    
        % Subtract the center vector from each of the member vectors.
        differences = bsxfun(@minus, members, center);
        
        % Take the sum of the squared differences.
        sqrdDiffs = sum(differences .^ 2, 2);
        
        % Take the square root to get the L2 (Euclidean) distance.
        distances = sqrt(sqrdDiffs);

        % Compute the average L2 distance, and use this as sigma.
        sigmas(i, :) = mean(distances);
    end

    % Verify no sigmas are 0.
    if (any(sigmas == 0))
        error('One of the sigma values is zero!');
    end
    
    % Compute the beta values from the sigmas.
    betas = 1 ./ (2 .* sigmas .^ 2);
    
end