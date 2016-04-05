function visualiselayer(z)
    figure(1);
    sideLength = sqrt(length(z));
    imagesc(reshape(z, sideLength, sideLength));
    colormap bone
end

