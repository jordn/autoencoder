%% PCA
figure(3);
hold off;
data = imagesTest;

[V, U] = pca(data', 'NumComponents', 30);
recon =  data*U*U';

for i = 0:9  
   v = U(labelsTest==i, 1:2);
    s{i+1} = scatter(v(:,1), v(:,2), 'filled', 'MarkerFaceColor', colors(i+1,:));
    hold on;
    axis off 
%     pause(1)
end
% legend('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
% savefig('mnist-2dpca')


%% SVD
figure(3);
hold off;

[V, U] = svd(imagesTest');

for i = 0:9  
   v = V(labelsTest==i, 1:2);   
    s{i+1} = scatter(v(:,1), v(:,2), 'filled', 'MarkerFaceColor', colors(i+1,:));
    axis off 
    hold on;
%     pause(1)  
end
% leg = legend('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Location', 'northwest');







