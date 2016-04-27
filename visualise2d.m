function visualise2d(nn, images, labels)
figure(3);
hold off;
nSamples = 400;

colors = [
    0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840
    0.1       0.1       0.1
    0.32      0.12       0.6
    0.9       0.3       0.5
];

codeLayer = numel(nn.rbm)/2+1;
% Should this use RBMs with binary states etc?
for i = 0:9
    x = images(:, labels==i);
    x = x(:, 1:nSamples);
    X = nnfeedforward(nn, x);
    s{i+1} = scatter(X{codeLayer}(1,:), X{codeLayer}(2,:), 'filled', 'MarkerFaceColor', colors(i+1,:));
    axis off 
    hold on;
    %     pause(0.3)
end
[hleg, hobj, hout, mout] = legend('0', '1', '2', '3', '4', '5', '6', '7', '8', '9');
set(hleg, 'FontSize', 25, 'Location', 'northwest');
legend boxoff;
set(hleg, 'Position', [-.010 0.2 hleg.Position([3 4])]);
for i = 1:10
    hobj(i).FontSize = 25;
    hobj(10+i).Children.MarkerSize = 20;
end

end

