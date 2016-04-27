% Generate digits
code = [0; 0];

x1s = -120:20;
x2s = [-60:40];

layers = length(nn.sizes);
im = [];
for j = 1:length(x1s)
    x1 = [x1s(j)];
    code = [repmat([x1], 1, length(x2s))' x2s'];
    x = code';
    
    for l = (layers-1)/2:-1:1
        x = rbmdown(nn.rbm{l}, x);
    end
    imWidth = sqrt(size(x,1));
    for i = 0:length(x2s)-1
        im((j-1)*imWidth+1:j*imWidth, i*imWidth+1+i:imWidth*(i+1)+ i) = reshape(x(:,i+1), imWidth, imWidth);
    end
end

imagesc(im);
axis equal;
colormap gray;
% axis off;