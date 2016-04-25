function visualiseweights(W)
    % For now assuming 100 x 784
    nNeurons = size(W,1); % This many images.
    gridLength = ceil(sqrt(nNeurons));
    gridSize = [gridLength, gridLength];
    
    sideLength = ceil(sqrt(size(W,2))); % Size of each square 
    im = nan((sideLength+1)*gridLength); % squares + border
    for n = 1:nNeurons
        [i, j] = ind2sub(gridSize, n);
        weights = pad(W(n,:), sideLength*sideLength, 2);
        squareWeights = nan(sideLength+1);
        squareWeights(1:sideLength, 1:sideLength) = reshape(weights, sideLength, sideLength);
        
        im((i-1)*sideLength+i:i*sideLength+i, (j-1)*sideLength+j:j*sideLength+j) = squareWeights;
    end
    figure(2);
    imagesc(im);
    title(['Features (' num2str(nNeurons) ' units)']);
    colormap bone
end

