addpath tSNE;

% Set parameters
no_dims = 2;
initial_dims = 50;
perplexity = 30;
% Run t?SNE
mapped = tsne(imagesTest', [], no_dims, initial_dims, perplexity);

%% Plot
figure(3);
hold off;

for i = 0:9  
    v = mapped(labelsTest == i, 1:2);
    s{i+1} = scatter(v(:,1), v(:,2), 'filled', 'MarkerFaceColor', colors(i+1,:));
    hold on;
%     pause(1)
    axis off 
end
 
% [hleg, hobj, hout, mout]  = legend('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Location','northwest');
% set(hleg, 'FontSize', 25);
% legend boxoff;
% set(hleg, 'Position', [-.010 0.2 hleg.Position([3 4])]);
% for i = 1:10
%     hobj(i).FontSize = 25;
%     hobj(10+i).Children.MarkerSize = 20;
% end

savefig('mnist-2dtsne');