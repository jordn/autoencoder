function visualiseweights(W)
    % For now assuming 100 x 784
    nNeurons = size(W,1); % This many images.
    sideLength = sqrt(size(W,2)); 
    gridLength = sqrt(nNeurons);
    gridSize = [gridLength, gridLength];
    im = nan(sideLength*gridLength);    
    for n = 1:nNeurons
        [i, j] = ind2sub(gridSize, n);
        im((i-1)*sideLength+1:i*sideLength, (j-1)*sideLength+1:j*sideLength) ...
            = reshape(W(n,:), sideLength, sideLength);
    end
    figure(2);
    imagesc(im);
    colormap bone
end

