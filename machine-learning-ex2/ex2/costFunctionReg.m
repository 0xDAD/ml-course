function [J, grad] = costFunctionReg(theta, X, y, lambda)
    %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    %   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    %   theta as the parameter for regularized logistic regression and the
    %   gradient of the cost w.r.t. to the parameters.

    % Initialize some useful values
    m = length(y); % number of training examples

    % You need to return the following variables correctly
    J = 0;
    grad = zeros(size(theta));

    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the cost of a particular choice of theta.
    %               You should set J to the cost.
    %               Compute the partial derivatives and set grad to the partial
    %               derivatives of the cost w.r.t. each parameter in theta

    G = sigmoid(X * theta);
    LL = log(G);
    LR = log(1 - G);
    Tg = theta;
    Tg(1, 1) = 0;
    R = (Tg' * Tg) * lambda / (2 * m);
    J = -(LL' * y + LR' * (1 - y)) / m + R;

    grad = (X' * (G - y)) / m + Tg * (lambda / m);

    % =============================================================

end
