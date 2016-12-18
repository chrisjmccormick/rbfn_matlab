function z = evaluateFuncApproxRBFN(Centers, betas, Theta, normalize, input)
% EVALUATEFUNCAPPROXRBFN Computes output of RBF Network for the provided input.
%   z = evaluateFuncApproxRBFN(Centers, betas, Theta, normalize, input) 
%
%   This function computes the activation values of all of the RBFN neurons
%   in the hidden layer using the provided 'Centers' and their 'betas', then
%   computes the values for the output layer of the network using the 'Theta'
%   coefficients
%
%   Parameters
%     Centers   - The prototype vectors for the RBF neurons.
%     betas     - The beta coefficients for the corresponding prototypes.
%     Theta     - The output weights to apply to the neuron activations.
%     normalize - Whether to normalize the RBF neuron activations before 
%                 applying the output weights.
%     input     - The input vector (row vector) to evaluate the RBFN over.
%
%   Returns
%     A column vector representing the network's output value for each output
%     node.
    
% $Author: ChrisMcCormick $    $Date: 2015/08/24 22:00:00 $    $Revision: 1.0 $
    
    % Compute the activations for each RBF neuron for this input.
    phis = getRBFActivations(Centers, betas, input);
    
    % Normalize the neuron activations.  
    if (normalize)
        phis = phis ./ sum(phis);
    end
    
    % Add a 1 to the beginning of the activations vector for the bias term.
    phis = [1; phis];
    
    % Multiply the activations by the weights and take the sum. 
    z = Theta' * phis;
        
end