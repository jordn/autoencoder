function v = rbmdown(rbm, h)
    nBatchSize = size(h, 2);
    if strcmp(rbm.visibleUnits, 'linear')
        p_v1 = rbm.W' * h+ repmat(rbm.a, 1, size);
        v = p_v1 + randn(size(rbm.a,1), nBatchSize);
    else
        v = sigmoid(rbm.W'*h + repmat(rbm.a, 1, nBatchSize)); 
    end
end