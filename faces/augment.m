%%
person = [];
for i = 1:40
    person = [person; i * ones(10,1)];
end

%% Augment
nFaces = size(faces,2);
width = sqrt(size(faces,1));
images = nan(625, nFaces*4*100);
labels = nan(1, nFaces*4*100);

j = 1;
for i = 1:size(faces,2)
    face = faces(:,i);
    label = person(i);
    
    square = reshape(face, width, width);
    
    for zoom = [0.49 0.55 0.6 0.7]
        for theta = linspace(-90, 90, 100)
            rotated = imrotate(square, theta);
            zoomed = imresize(rotated, zoom);
            w = size(zoomed,1);
            cropped = imcrop(zoomed, [w/4 w/4 25 25]);
            images(:,j) = reshape(cropped(1:25, 1:25), 625, 1);
            labels(j) = label;
            j = j+1;
            % imagesc(reshape(cropped(1:25, 1:25), 25, 25));
            % colormap bone 
            % pause(0.01)
        end
    end    
end

% Normalise to zero mean, unit variance
for i = 1:size(images,1)
    images(i,:) = images(i,:) - mean(images(i,:));
    images(i,:) = images(i,:)/std(images(i,:));
end

clear nFaces width targetWidth face square zoom theta zoomed cropped j i;