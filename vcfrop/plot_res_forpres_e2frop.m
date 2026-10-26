% plot res for PRESENTATION
% matches res flow of analyses, but too much to have different options when calculating
% analyses

%load res_050526_45sess.mat
ll = size(res.both.spcs,2);
llvec = 1:ll;

randcol = [140 26 17]/255;
tempcol = [19 66 196]/255;

% % % == SPCs == 
figure; hold on
lmebr = LoftusMasson(res.rand.spcs);
errorbar(llvec,mean(res.rand.spcs),lmebr,':o','LineWidth',2,'Color',randcol,...
    'MarkerFaceColor',randcol)
lmebt = LoftusMasson(res.temp.spcs);
errorbar(llvec,mean(res.temp.spcs),lmebt,'--sq','LineWidth',2,'Color',tempcol, ...
    'MarkerFaceColor',tempcol)
publishFig
axis([.5 ll+.5 0 1])
xlabel('Serial Position')
ylabel('Recall Probability')
legend('Random','Autocorr','Location','Southwest')
set(gca,'XTick',3:3:15)

% % == PFRs == 
figure; hold on
%lmebr = LoftusMasson(res.rand.pfrs);
plot(llvec,mean(res.rand.pfrs),':o','LineWidth',2,'Color',randcol,...
    'MarkerFaceColor',randcol)
%lmebt = LoftusMasson(res.temp.pfrs);
plot(llvec,mean(res.temp.pfrs),'--sq','LineWidth',2,'Color',tempcol, ...
    'MarkerFaceColor',tempcol)
publishFig
axis([.5 ll+.5 0 1])
xlabel('Serial Position')
ylabel('Probability of First Recall')
legend('Random','Autocorr','Location','Northwest')
set(gca,'XTick',3:3:15)

% % == Summary: P(recall) ==
figure; hold on
valsforbarpr = [mean(res.rand.spcs,2) mean(res.temp.spcs,2)];
nsub = size(valsforbarpr,1);
xs = [1 2];
bar(xs(1),mean(valsforbarpr(:,1)),'FaceColor',randcol,'EdgeColor','k','LineWidth',2,'BarWidth',.5);
bar(xs(2),mean(valsforbarpr(:,2)),'FaceColor',tempcol,'EdgeColor','k','LineWidth',2,'BarWidth',.5);
errorbar(xs,mean(valsforbarpr),std(valsforbarpr)/sqrt(nsub-1),'k.','MarkerSize',.2,'LineWidth',2);
publishFig;
ylabel('Recall probability');
axis([0 3 0 1])
set(gca,'XTick',xs,'XTickLabel',{'Rand','Autocorr'})
% 
% % == lag-CRP ==
maxabslag = 4;
crpvecn = (-1*maxabslag):-1;
crpvecp = 1:maxabslag;

ebr = LoftusMasson(res.rand.crps(:,[crpvecn+ll crpvecp+ll]));
ebt = LoftusMasson(res.temp.crps(:,[crpvecn+ll crpvecp+ll]));

figure; hold on 
errorbar(crpvecn,mean(res.rand.crps(:,crpvecn+ll)),ebr(:,crpvecp),...
    ':o','LineWidth',2,'Color',randcol,'MarkerFaceColor',randcol)
errorbar(crpvecp,mean(res.rand.crps(:,crpvecp+ll)),ebr(:,maxabslag+crpvecp),...
    ':o','LineWidth',2,'Color',randcol,'MarkerFaceColor',randcol)
errorbar(crpvecn,mean(res.temp.crps(:,crpvecn+ll)),ebt(:,crpvecp),...
    'sq--','LineWidth',2,'Color',tempcol,'MarkerFaceColor',tempcol)
errorbar(crpvecp,mean(res.temp.crps(:,crpvecp+ll)),ebt(:,maxabslag+crpvecp),...
    'sq--','LineWidth',2,'Color',tempcol,'MarkerFaceColor',tempcol)
publishFig
axis([-.5-maxabslag maxabslag+.5 0 .6])

maxabslag = ll-1;
crpvecn = (-1*maxabslag):-1;
crpvecp = 1:maxabslag;

figure; hold on 
plot(crpvecn,mean(res.rand.crps(:,crpvecn+ll),"omitnan"),...
    ':o','LineWidth',2,'Color',randcol,'MarkerFaceColor',randcol)
plot(crpvecp,mean(res.rand.crps(:,crpvecp+ll),"omitnan"),...
    ':o','LineWidth',2,'Color',randcol,'MarkerFaceColor',randcol)
plot(crpvecn,mean(res.temp.crps(:,crpvecn+ll),"omitnan"),...
    'sq--','LineWidth',2,'Color',tempcol,'MarkerFaceColor',tempcol)
plot(crpvecp,mean(res.temp.crps(:,crpvecp+ll),"omitnan"),...
    'sq--','LineWidth',2,'Color',tempcol,'MarkerFaceColor',tempcol)
publishFig
axis([-.5-maxabslag maxabslag+.5 0 .6])
xlabel('Lag')
ylabel('Cond. Resp. Prob.')

% % == Clustering: temporal ==
figure; hold on
valsforbartc = [res.rand.temp_clusts res.temp.temp_clusts];
nsub = size(valsforbartc,1);
xs = [1 2];
bar(xs(1),mean(valsforbartc(:,1)),'FaceColor',randcol,'EdgeColor','k','LineWidth',2,'BarWidth',.5);
bar(xs(2),mean(valsforbartc(:,2)),'FaceColor',tempcol,'EdgeColor','k','LineWidth',2,'BarWidth',.5);
errorbar(xs,mean(valsforbartc),std(valsforbartc)/sqrt(nsub-1),'k.','MarkerSize',.2,'LineWidth',2);
publishFig;
ylabel('Temporal clustering score');
axis([0 3 0 1])
set(gca,'XTick',xs,'XTickLabel',{'Rand','Autocorr'})

% % == Value estimation: 4 ==
% Pres ra, Rec ra. Pres te, Recall te.
% figure; hold on
% valsforbar = [res.rand.abserr_pres_sub ...
%     res.rand.abserr_rec_sub res.temp.abserr_pres_sub ...
%     res.temp.abserr_rec_sub];
% nsub = size(valsforbar,1);
% xs = [1 2 4 5];
% bar(min(xs)-5,1,'FaceColor','k','EdgeColor','k');
% bar(min(xs)-5,1,'FaceColor','w','EdgeColor','k');
% bar(xs(1),mean(valsforbar(:,1)),'FaceColor',randcol,'EdgeColor','k','LineWidth',2);
% bar(xs(2),mean(valsforbar(:,2)),'FaceColor','w','EdgeColor',randcol,'LineWidth',2);
% bar(xs(3),mean(valsforbar(:,3)),'FaceColor',tempcol,'EdgeColor','k','LineWidth',2);
% bar(xs(4),mean(valsforbar(:,4)),'FaceColor','w','EdgeColor',tempcol,'LineWidth',2);
% errorbar(xs,mean(valsforbar),std(valsforbar)/sqrt(nsub-1),'k.','MarkerSize',.2,'LineWidth',2);
% publishFig;
% ylabel('|Value estimation error|');
% legend('From All','From Recalled','Location','Northeast');
% set(gca,'XTick',[1.5 4.5],'XTickLabel',{'Random','Autocorrelated'});
% 
% 
% % % Pres ra, Pres te. Recall ra, Recall te.
% % figure; hold on
% % valsforbar = [res.rand.abserr_pres_sub res.temp.abserr_pres_sub ...
% %     res.rand.abserr_rec_sub res.temp.abserr_rec_sub];
% % nsub = size(valsforbar,1);
% % xs = [1 2 4 5];
% % bar(min(xs)-5,1,'FaceColor',randcol,'EdgeColor',randcol);
% % bar(min(xs)-5,1,'FaceColor',tempcol,'EdgeColor',tempcol);
% % bar(xs(1),mean(valsforbar(:,1)),'FaceColor',randcol,'EdgeColor','k','LineWidth',2);
% % bar(xs(2),mean(valsforbar(:,2)),'FaceColor',tempcol,'EdgeColor','k','LineWidth',2);
% % bar(xs(3),mean(valsforbar(:,3)),'FaceColor',randcol,'EdgeColor',min(2*randcol,1),'LineWidth',2);
% % bar(xs(4),mean(valsforbar(:,4)),'FaceColor',tempcol,'EdgeColor',min(2*tempcol,1),'LineWidth',2);
% % errorbar(xs,mean(valsforbar),std(valsforbar)/sqrt(nsub-1),'k.','MarkerSize',.2,'LineWidth',2);
% % publishFig;
% % ylabel('|Value estimation error|');
% % legend('Rand','Auto','Location','Northwest');
% % set(gca,'XTick',[1.5 4.5],'XTickLabel',{'All items','Recalled items'});
% axis([min(xs)-1 max(xs)+1 0 6]);

figure
plot(res.both.abserr_pres_sub,mean(res.both.spcs,2),'ok');
publishFig
xlabel('|Tip estimation error|')
ylabel('Recall probability')

%figdir = '../../../figs/vavi/pres050726/';
%exportgraphics(figure(i),[figdir 'fig' num2str(i) '.pdf']);