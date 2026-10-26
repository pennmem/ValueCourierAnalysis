function publishFig(handle,cleartitle,fosi)
%PUBLISHFIG - Prepared figure for publication.
%
% Prepares a figure for publication or presentation by making all
% the fonts larger.  The default is to work on the current figure
% and to clear the title away.
%
% FUNCTION:
%   publishfig(handle,cleartitle)
%
% INPUT ARGS:
%   handle = gcf;     % handle to figure to modify
%   cleartitle = 1;   % whether to clear the title or not
%
%
if nargin < 3
    fosi = 21;
    if nargin < 2
        cleartitle = 1;
        if nargin < 1
            handle = gcf;
        end
    end
end

% set the tics on all children
h = get(gcf,'Children');
%keyboard
% careful!

for j=1%setdiff(1:length(h),2)
    set(h(j),'FontSize',fosi)
end


% set the title and labels
h = get(get(gcf,'Children'),'Title');
if length(h) > 1
    for j=1:length(h)
        set(h{j},'FontSize',28);
    end
else
    set(h,'FontSize',fosi);
end

% remove the title
if cleartitle
    title('');
end

% set the xlabel
h = get(get(gcf,'Children'),'xlabel');
if length(h) > 1
    for j=1:length(h)
        set(h{j},'FontSize',fosi);
    end
else
    set(h,'FontSize',fosi);
end

% set the ylabel
h = get(get(gcf,'Children'),'ylabel');
if length(h) > 1
    for j=1:length(h)
        set(h{j},'FontSize',fosi);
    end
else
    set(h,'FontSize',fosi);
end

% redo the legend
%legend

return

% fix the bounding box
hndl2=axes('position',[0,0,1,1],...
    'units','normalized',...
    'UserData','timestamp',...
    'visible','off');