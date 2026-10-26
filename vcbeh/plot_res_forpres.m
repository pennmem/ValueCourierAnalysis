% plot results for PRESentation
% matches res flow of analyses, but too much to have different options when calculating
% analyses

load res_050526_45sess.mat
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
% 
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
% 
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