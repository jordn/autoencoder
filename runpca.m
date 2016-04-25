%% PCA
hold off;
[V, U] = pca(imagesCV');
for i = 0:9  
   v = U(labelsCV==i, 1:2);
   
    s{i+1} = scatter(v(:,1), v(:,2), 'filled', 'MarkerFaceColor', colors(i+1,:));
    hold on;
    pause(1)
    axis off 
end

legend('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')



