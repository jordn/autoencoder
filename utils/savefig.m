function savefig(name, fig, type)
%SAVEFIG Saves a figure to where I want it to go.
    figDirectory = './figs/';
    filePath = strcat(figDirectory, name, '');
    if nargin < 2
        fig = gcf;
    end;
    if nargin < 3
        type = 'eps';
    end;
    
    if type == 'eps'
        print(fig, filePath, '-depsc');
    elseif type == 'png'
        print(fig, filePath, '-dpng');
    elseif type == 'pdf'
        set(fig, 'PaperPosition', [0 0 8 8]); %Position plot at left hand corner with width 5 and height 5.
        set(fig, 'PaperSize', [8 8]); %Set the paper to have width 5 and height 5.
%         saveas(gcf, 'test', 'pdf') %Save figure
        print(fig, filePath, '-dpdf');
%         saveTightFigure(strcat(figDirectory, name, '.pdf'), fig);
    elseif type == 'pd2'
        set(fig, 'PaperPosition', [0 0 8 5]); %Position plot at left hand corner with width 5 and height 5.
        set(fig, 'PaperSize', [8 5]); %Set the paper to have width 5 and height 5.
%         saveas(gcf, 'test', 'pdf') %Save figure
        print(fig, filePath, '-dpdf');
%         saveTightFigure(strcat(figDirectory, name, '.pdf'), fig);
    end
  
end