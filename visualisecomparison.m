% Compare 10
function visualisecomparison(X, labels, extra)
    i = 1;
    lab = unique(labels);
    for i = 1:length(lab)
        index = find(labels==lab(i),1);
        data(:, i) = X{1}(:, index);
        recon(:, i) = X{end}(:, index);
        if nargin > 2
            recon2(:, i) = extra(:, index);
        end
        i = i + 1;
        if i > 10
            break
        end
    end

    figure(1); clf; hold off;
    imWidth = ceil(sqrt(size(data,1)));
    if nargin < 3
        im = nan(imWidth*2+1,10*(imWidth+1));
    else
        im = nan(imWidth*3+2, 10*(imWidth+1));
    end
    
    
    for i = 0:9
        im(1:imWidth, i*imWidth+1+i:imWidth*(i+1)+ i) = reshape(data(:,i+1), imWidth, imWidth);
        im(imWidth+2:imWidth*2+1, i*imWidth+1+i:imWidth*(i+1)+ i) = reshape(recon(:,i+1), imWidth, imWidth); 
        if nargin > 2
            im(imWidth*2+3:imWidth*3+2, i*imWidth+1+i:imWidth*(i+1)+ i) = reshape(recon2(:,i+1), imWidth, imWidth); 
        end
    end
    imagesc(im);
    axis equal;
    colormap gray;
    axis off;
end