addpath tSNE;
figure(3);
hold off;

% Set parameters
no_dims = 2;
initial_dims = 50;
perplexity = 30;
% Run t?SNE
mapped = tsne(imagesCV', [], no_dims, initial_dims, perplexity);

%% Plot
for i = 0:9  
    v = mapped(labelsCV == i, 1:2);
    s{i+1} = scatter(v(:,1), v(:,2), 'filled', 'MarkerFaceColor', colors(i+1,:));
    hold on;
    pause(1)
    axis off 
end

legend('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
