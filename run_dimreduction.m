% Dimensionality reduction comparison

addpath drtoolbox;

n = 3000;
x = imagesTest(:,1:n);
labs = labelsTest(1:n);

% Set parameters
no_dims = 2;

% method = ['PCA', 'tSNE', 'Autoencoder'];
% [mappedX, mapping] = compute_mapping(x', method, no_dims);	


method = 'KernelPCA';
% [mappedX, mapping] = compute_mapping(x', method, no_dims, 'gauss', 100);
[mappedX, mapping] = compute_mapping(x', method, no_dims, 'poly', 0, 3);

figure(3);
hold off;

for i = 0:9  
    v = mappedX(labs == i, 1:2);
    s{i+1} = scatter(v(:,1), v(:,2), 'filled', 'MarkerFaceColor', colors(i+1,:));
    hold on;
%     pause(1)
    axis off 
end
 
savefig(['mnist-', method]);
% savefig(['faces-', method]);
