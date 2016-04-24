function visualisereconstruction(data, recon)
    figure(1);
    sideLength = sqrt(length(data));
    subplot(1,2,1);
    imagesc(reshape(data, sideLength, sideLength));
    title('Data');
    subplot(1,2,2);
    imagesc(reshape(recon, sideLength, sideLength));
    colormap bone;
    title('Reconstruction');
end

