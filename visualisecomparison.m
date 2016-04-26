% Compare 10
function visualisecomparison(X, labels)
    i = 1;
    lab = unique(labels);
    for i = 1:length(lab)
        index = find(labels==lab(i),1);
        data(:, i) = X{1}(:, index);
        recon(:, i) = X{end}(:, index);
        i = i + 1;
        if i > 10
            break
        end
    end

    figure(1); clf; hold off;
    imWidth = ceil(sqrt(size(data,1)));

    im = nan(imWidth+1,10*(imWidth+1));
    for i = 0:9
        im(1:imWidth, i*imWidth+1+i:imWidth*(i+1)+ i) = reshape(data(:,i+1), imWidth, imWidth);
        im(imWidth+2:imWidth*2+1, i*imWidth+1+i:imWidth*(i+1)+ i) = reshape(recon(:,i+1), imWidth, imWidth); 
    end
    imagesc(im);
    axis equal;
    axis off;
    colormap gray;
    axis off;
end