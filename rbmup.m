function y = rbmup(rbm, x)
    y = sigmoid(rbm.W * x + repmat(rbm.b, 1, size(x, 2)));
end