function y = rbmupsigmoidbin(rbm, x)
    prob = sigmoid(rbm.W * x + repmat(rbm.b, 1, size(x, 2)));
    y = prob > rand(size(prob));
end