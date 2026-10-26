function pl = plot_pres_values(pres_values, pres_types, subjects)
% plot_pres_values(pres_values, subjects)
% 
%PLOT_PRES_VALUES plot values of presented items by list type
% 
%  no outputs aside from plots

% vfr_for_subj does all the real work here
pl = apply_by_index(@pv_for_subj, ...
                           subjects, ...
			   1, ...
                           {pres_values, pres_types});
%endfunction

function plotted = pv_for_subj(pres_values, pres_types)

cols = nan(2,5,3);
cols(1,1,:) = [140 26 17];
cols(1,2,:) = [169 30 19];
cols(1,3,:) = [206 26 24];
cols(1,4,:) = [240 42 28];
cols(1,5,:) = [255 91 79];

cols(2,1,:) = [9 30 88];
cols(2,2,:) = [13 46 135];
cols(2,3,:) = [19 66 196];
cols(2,4,:) = [23 84 247];
cols(2,5,:) = [106 145 255];

cols = cols/255;

figure; hold on
plot([1 15],[25 25],'-.y','LineWidth',4)
xlines = [2.5; 7.5; 12.5];
for xi = 1:size(xlines,1)
    plot([xlines(xi) xlines(xi)],[1 50],':','Color',[.5 .5 .5],'LineWidth',4)
end

xvals = 1:size(pres_values,2);

for ti = 1:2
    these_li = find(pres_types==ti);

    for tli = 1:length(these_li)
        plot(xvals,pres_values(these_li(tli),:),'x-','LineWidth',2,...
            'Color',cols(ti,tli,:));
        plot([1 max(xvals)],mean(pres_values(these_li(tli),:)),'--','LineWidth',1,...
            'Color',cols(ti,tli,:))
    end
end

plotted = 1;
%endfunction
axis([0 16 0 51])
xlabel('Serial position')
ylabel('Value')
set(gca,'XTick',1:2:15)

% 3, 4, 9