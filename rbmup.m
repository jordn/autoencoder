function h = rbmup(rbm, x)
    if strcmp(rbm.hiddenUnits, 'linear')
        h = rbm.W * x + repmat(rbm.b, 1, size(x, 2));
    else
        h = sigmoid(rbm.W * x + repmat(rbm.b, 1, size(x, 2)));
    end
end