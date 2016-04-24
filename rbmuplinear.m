function y = rbmuplinear(rbm, x)
    y = rbm.W * x + repmat(rbm.b, 1, size(x, 2)) + randn(size(rbm.b, 1), size(x,2));
end