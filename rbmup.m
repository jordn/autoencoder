function y = rbmup(rbm, x)
    if strcmp(rbm.hiddenUnits, 'linear')
        y = rbm.W * x + repmat(rbm.b, 1, size(x, 2));
    else
        y = sigmoid(rbm.W * x + repmat(rbm.b, 1, size(x, 2)));
    end
end