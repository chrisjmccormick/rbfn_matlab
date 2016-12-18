% ======== runRBFNFuncApproxExample ======== 
% This script performs Gaussian Kernel Regression on a generated 
% two-dimensional dataset.
%
% Thank you to Youngmok Yun; the 2D function is from his code here:
% http://youngmok.com/gaussian-kernel-regression-with-matlab-code/

% $Author: ChrisMcCormick $    $Date: 2015/08/24 22:00:00 $    $Revision: 1.0 $

clear;
close all;
clc;

addpath('kMeans');
addpath('RBFN');

% =================================
%             Dataset
% =================================

% Define the input range x.
x = [1:100]'; 

% Create an interesting non-linear function.
%
% The first term just creates a sine wave.
% The second term pushes the function upward after x = 50.
y_orig = sin(x/10) + (x/50).^2;

% Add some random noise to the y values. 
% randn generates datapoints with a normal distribution with mean 0 and 
% variance 1. A typical output might range from -3 to 3. The parameters
% to randn specify the matrix dimensions (a column vector with 100 rows).
y = y_orig + (0.2 * randn(100, 1));


% =================================
%       RBFN Properties
% =================================

% 1. Specify the number of RBF neurons.
numRBFNeurons = 10;

% 2. Specify whether to normalize the RBF neuron activations.
normalize = true;

% 3. Calculate the beta value to use for all neurons.
    
% Set the sigmas to a fixed value. Smaller values will fit the data
% points more tightly, while larger values will create a smoother result.
sigma = 10;

% Compute the beta value from sigma.
beta = 1 ./ (2 .* sigma.^2);

% ==================================
%            Train RBFN
% ==================================

disp('Training an RBFN on the noisy data...');

% Train the RBFN for function approximation.
[Centers, betas, Theta] = trainFuncApproxRBFN(x, y, numRBFNeurons, normalize, beta, true);

% =================================
%        Evaluate RBFN
% =================================

% Define the range of input values at which to approximate the function.
xs = [1:0.5:100]';

% Create an empty vector to hold the approximate function values.
ys = zeros(size(x));

% 2. Evaluate the trained RBFN over the query points.
% For each sample point in 'xs'...
for (i = 1:length(xs))

	% Evaluate the RBFN at the query point xs(i) and store the result in ys(i).
	ys(i) = evaluateFuncApproxRBFN(Centers, betas, Theta, true, xs(i));
	
end


% ==================================
%         Plot Result
% ==================================

figure(1);
hold on; 

% Plot the original function as a black line.
plot(x, y_orig, 'k-');

% Plot the noisy data as blue dots.
plot(x, y, '.');

% Plot the approximated function as a red line.
plot(xs, ys, 'r-');

legend('Original', 'Noisy Samples', 'Approximated');
axis([0 100 -1 5]);
title('RBFN Regression');


% ===================================
%       Plot all RBF Neurons
% ===================================

% 1. Create another plot with the noisy data and result.
figure(2);
hold on;

% Plot the noisy data as blue dots.
plot(x, y, '.');

% Plot the approximated function as a red line.
plot(xs, ys, 'r-');

legend('Noisy Samples', 'Approximated');
title('RBFN Regression');

% 2. Plot all of the RBF Neuron activations.
kernels = zeros(size(Centers, 1) + 1, length(xs));

% Evaluate all of the kernels for each sample point along the x-axis.
for (i = 1 : length(xs))
    kernels(:, i) = [1; getRBFActivations(Centers, betas, xs(i))] .* Theta;
end

% Create a color pallete to give each Gaussian a different color.
palette = jet(size(kernels, 1));

% Plot each of the Gaussians.
for (i = 1 : size(kernels, 1))
    plot(xs, kernels(i, :)', 'color', palette(i, :));
end
