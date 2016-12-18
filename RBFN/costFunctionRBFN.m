function [J, grad] = costFunctionRBFN(theta, X, y, lambda)
% COSTFUNCTIONRBFN Compute cost and gradients for gradient descent based on 
%   mean squared error over the training set.
%   [J, grad] = costFunctionRBFN(theta, X, y, lambda) computes the cost of the 
%   parameters theta for fitting the data set in X and y, and the gradients for
%   updating theta.
%   
%   To compute the mean squared error (MSE): 
%     1. Apply the weights in theta to X to get the predicted output values.
%     2. Take the difference between the predicted value and the label in y and
%        square it.
%     3. Average the squared differences over the training set.
%
%   Lambda is a regularization term which prevents the weights theta from 
%   growing too large and overfitting the data set. Cross-validation can be 
%   used to find the best value for lambda.
%
%   To perform gradient descent, we make iterative changes to the weights in 
%   theta. This function computes the changes (the gradients) for one iteration
%   of gradient descent. It does not modify theta itself.
%
%   Parameters
%     theta   - The current weights.
%     X       - The training set inputs.
%     y       - The training set labels.
%     lambda  - The regularization parameter.
%
%   Returns
%     J    - The cost of the current theta values.
%     grad - The updates to make to each theta value.

% $Author: ChrisMcCormick $    $Date: 2014/04/08 22:00:00 $    $Revision: 1.2 $
  
% ======== Compute Cost ========

m = length(y); % number of training examples

% Evaluate the hypothesis. h becomes a vector with length m.
h = zeros(m, 1);

h = X * theta;

% Compute the differences between the hypothesis and correct value.
diff = h - y;

% Take the squared difference.
sqrdDiff = diff.^2;

% Take the sum of all the squared differences.
J = sum(sqrdDiff);

% Divide by 2m to get the average squared difference.
J = J / (2 * m);

% Add the regularization term to the cost.
J = J + (lambda / (2 * m) * sum(theta(2:length(theta)).^2));

% ===== Compute Gradient ========

grad = zeros(size(theta));

% Multiply each data point by the difference between the hypothesis
% and actual value for that data point. 
grad = X' * diff;

grad = grad / m;

% Add the regularization term for theta 1 to n (but not theta_0).
grad(2 : length(theta)) = grad(2 : length(theta)) + ((lambda / m) * theta(2 : length(theta)));

end
