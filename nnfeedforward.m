function y = nnfeedforward(nn, x)
    for layer = 2 : numel(nn.rbm)+1
        x = rbmup(nn.rbm{layer-1}, x);
    end
    y = x;
end