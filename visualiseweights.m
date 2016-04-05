function visualiseweights(w)
    % For now assuming 100 x 784
    nNeurons = size(w,1); % This many images.
    sideLength = sqrt(size(w,2));
    figure(2);
    for i = 1:nNeurons
        subplot(sqrt(nNeurons), sqrt(nNeurons), i); 
        imagesc(reshape(w(i,:), sideLength, sideLength));
        axis tight;
        axis off;
    end
    
    colormap bone
end

