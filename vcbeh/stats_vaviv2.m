% == DIFFS BY LIST TYPE ==

load res_050526_45sess.mat
% 
% % diff by list type: abs error based on pres
% [~,p,ci] = ttest(res.rand.abserr_pres_sub,res.temp.abserr_pres_sub);
% fprintf('abs err: rand>temp, p=%1.4f, CI=[%1.3f,%1.3f] \n',p,ci);
% 
% % p(recall): auto vs. rand. if this is different, that actually could be a
% % problem with interpreting error b/c it might be closer to pres for an
% % uninteresting reason, I think
% [~,p,ci] = ttest(mean(res.rand.spcs,2),mean(res.temp.spcs,2));
% fprintf('p(recall): rand>temp, p=%1.4f, CI=[%1.3f,%1.3f] \n',p,ci);
% 
% % temporal clustering: auto vs rand
% [~,p,ci] = ttest(res.rand.temp_clusts,res.temp.temp_clusts);
% fprintf('temp clusts: rand>temp, p=%1.4f, CI=[%1.3f,%1.3f] \n',p,ci);
% 
% % == SANITY CHECKS ON PERFORMANCE ==
% % pres value estimation
% fprintf('M=%1.2f,SD=%1.2f \n',mean(res.both.abserr_pres_sub),...
%     std(res.both.abserr_pres_sub));
% 
% % p(recall)
% prec_sub = mean(res.both.spcs,2);
% fprintf('M=%1.2f,SD=%1.2f \n',mean(prec_sub),std(prec_sub));

% == ACROSS TASKS ==

% correlation: p(recall) with value error
[r,p] = corrcoef(res.both.abserr_pres_sub,prec_sub);
fprintf('both: corr b/t abs_err and p(recall), r=%1.4f, p=%1.4f \n',r(2),p(2));

% figure
figure; hold on
xs = [0.1 6.8];
mdl = fitlm(res.both.abserr_pres_sub,prec_sub);
ys = xs*mdl.Coefficients.Estimate(2)+ones(1,2)*mdl.Coefficients.Estimate(1);
plot(xs,ys,'--','Color',[.5 .5 .5],'LineWidth',3)
plot(res.both.abserr_pres_sub,prec_sub,'ok','LineWidth',3);
%publishFig
xlabel('|Tip estimation error|')
ylabel('Recall probability')
axis([0 8 0.5 1])
set(gca,'XTick',1:2:7,'YTick',.5:.1:1,'YTickLabel',{'0.5','0.6','0.7','0.8','0.9','1.0'})
% abs(p(2)-mdl.Coefficients.pValue) = 1.06 x 10^(-16), so I think corrcoef
% and mdl are close enough to be the same

fosism = 21;
fosi = 24;
corrtext = sprintf('r=%1.2f\np=%1.3f',r(2),p(2));
text(6,.925,corrtext,'FontSize',fosism);

h = get(gcf,'Children');
set(h(1),'FontSize',fosism)

% set the title and labels
h = get(get(gcf,'Children'),'Title');
if length(h) > 1
    for j=1:length(h)
        set(h{j},'FontSize',fosi);
    end
else
    set(h,'FontSize',fosi);
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