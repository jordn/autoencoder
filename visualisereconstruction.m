function visualisereconstruction(data, recon)
    figure(1);
    sideLength = ceil(sqrt(length(data)));
    subplot(1,2,1);
    imagesc(reshape(pad(data, sideLength^2, 'nan'), sideLength, sideLength));
    title('Data');
    subplot(1,2,2);
    imagesc(reshape(pad(recon, sideLength^2, 'nan'), sideLength, sideLength));
    colormap bone;
    title('Reconstruction');
end

