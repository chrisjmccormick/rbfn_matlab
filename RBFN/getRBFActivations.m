function z = getRBFActivations(centers, betas, input)
% GETRBFACTIVATIONS Compute the activation values for all RBF neurons.
%   z = getRBFActivations(centers, betas, input) Computes the activations for
%   the given input for each RBF neuron specified by 'centers' and 'betas'.
%
%   This function computes the RBF activation function for each of the RBF 
%   neurons given the supplied 'input'. Each RBF neuron is described by a 
%   prototype or "center" vector in 'centers', and a beta coefficient in
%   'betas'. 
%
%   Parameters
%     centers  - Matrix of RBF neuron center vectors, one per row.
%     betas    - Vector of beta coefficients for the corresponding RBF neuron.
%     input    - Column vector containing the input.
%
%   Returns
%     A column vector containing the activation value (between 0 and 1) for
%     each of the RBF neurons.

% $Author: ChrisMcCormick $    $Date: 2013/08/15 22:00:00 $    $Revision: 1.0 $

    % Subtract the input from all of the centers.
    % diffs becomes an k x n matrix where k is the number of centers
    % and n is the number of input dimensions.
    diffs = bsxfun(@minus, centers, input);
    
    % Take the sum of squared distances (the squared L2 distance).
    % sqrdDists becomes a k x 1 vector where k is the number of centers.
    sqrdDists = sum(diffs .^ 2, 2);

    % Apply the beta coefficient and take the negative exponent.
    z = exp(-betas .* sqrdDists);

end