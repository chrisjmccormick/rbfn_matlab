function [mean_dists] = findElbow( X, k_vals )
%FINDELBOW Tries various 'k' values and plots their mean squared distance.
% For each 'k' value, calculates the average squared L2 distance between 
% the cluster members and their assigned centroid.
% Finally, it plots the distances against the 'k' values.
%
% Parameters
%   X - Matrix of vectors to cluster, one per row.
%   k_vals - A vector of center values to try.

    % Vector to hold mean distance for each k value.
    mean_dists = zeros(length(k_vals), 1);

    fprintf('Trying k value:');
    for i = 1 : length(k_vals)

        % Select the next 'k' value to try.
        k = k_vals(i);

        fprintf(' %d,', k);
        
        % Pick random samples as the initial centroids.
        init_Centroids = kMeansInitCentroids(X, k);

        % Run k-means clustering, with at most --- iterations.
        [Centers, memberships] = kMeans(X, init_Centroids, 1000);

        % Create a vector to hold the distance between each data point 
        % and its assigned cluster centroid.
        distances = [];
        
        cluster_sizes = zeros(k, 1);
        
        for j = 1 : k
            % Select the next centroid.
            centroid = Centers(j, :);
            
            % Select all members of cluster 'j'.
            members = X(memberships == j, :);
            
            % Record how many points were assigned to this cluster.
            cluster_sizes(j) = size(members, 1);
            
            % Calculate squared L2 distance between members and the
            % centroid.
            dists = sum(bsxfun(@minus, members, centroid).^2, 2); 
            
            % Append these distance values.
            distances = [distances; dists];
        end
        
        fprintf(' -- Max cluster size: %d, Median size: %.2f, Empty clusters: %d, Singletons: %d\n', max(cluster_sizes), median(cluster_sizes), sum(cluster_sizes == 0), sum(cluster_sizes == 1));
        
        % Calculate the mean distance and store it.
        mean_dists(i) = mean(distances);
    end

    fprintf('\n');
    
    % Plot the result. The user can then inspect the plot to find the
    % elbow.
    plot(k_vals, mean_dists, 'b-');
    
end

