function z = evaluateRBFN(Centers, betas, Theta, input)
% EVALUATERBFN Computes the outputs of an RBF Network for the provided input.
%   z = evaluateRBFN(centers, betas, weights, input) Evaluates the RBFN over
%   given input using the provided parameters.
%
%   This function computes the activation values of all of the RBFN neurons
%   in the hidden layer using the provided 'Centers' and their 'betas', then
%   computes the values for the output layer of the network using the 'Theta'
%   coefficients
%
%   Parameters
%     Centers  - The prototype vectors for the RBF neurons.
%     betas    - The beta coefficients for the corresponding prototypes.
%     Theta    - The output weights to apply to the neuron activations.
%     input    - The input vector to evaluate the RBFN over.
%
%   Returns
%     A column vector representing the network's output value for each output
%     node.
    
% $Author: ChrisMcCormick $    $Date: 2014/02/11 22:00:00 $    $Revision: 1.1 $
    
    % Compute the activations for each RBF neuron for this input.
    phis = getRBFActivations(Centers, betas, input);
    
    % Add a 1 to the beginning of the activations vector for the bias term.
    phis = [1; phis];
    
    % Multiply the activations by the weights and take the sum. Do this for
	% each category (output node). The result is a column vector with one row
	% per output node.
	%
	%   Theta = centroids x categories	  Theta' = categories x centroids
	%    phis = centroids x 1
	%       z = Theta' * phis = categories x 1
    z = Theta' * phis;
        
end