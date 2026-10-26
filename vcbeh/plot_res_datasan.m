% plot res for DATA SANity, not final results (e.g., more by subject)
% matches res flow of analyses, but too much to have different options when calculating
% analyses

% datapath = 'data_04';
% load(datapath); res = ana_vavi(datapath);

llvec = 1:data.list_length;

randcol = [140 26 17]/255;
tempcol = [19 66 196]/255;

% == SPCs == 
figure; hold on
plot(llvec,res.both.spcs,'x-')
plot(llvec,mean(res.both.spcs),'k:','LineWidth',2)
axis([.5 data.list_length+.5 0 1])

figure; hold on
plot(llvec,res.rand.spcs,'x-')
plot(llvec,mean(res.rand.spcs),':','LineWidth',2,'Color',randcol)
axis([.5 data.list_length+.5 0 1])

figure; hold on
plot(llvec,res.temp.spcs,'x-')
plot(llvec,mean(res.temp.spcs),':','LineWidth',2,'Color',tempcol)
axis([.5 data.list_length+.5 0 1])

% == PFRs == 
figure; hold on
plot(llvec,res.both.pfrs,'x-')
plot(llvec,mean(res.both.pfrs),'k:','LineWidth',2)
axis([.5 data.list_length+.5 0 1])

figure; hold on
plot(llvec,res.rand.pfrs,'x-')
plot(llvec,mean(res.rand.pfrs),':','LineWidth',2,'Color',randcol)
axis([.5 data.list_length+.5 0 1])

figure; hold on
plot(llvec,res.temp.pfrs,'x-')
plot(llvec,mean(res.temp.pfrs),':','LineWidth',2,'Color',tempcol)
axis([.5 data.list_length+.5 0 1])

% == lag-CRP ==
maxabslag = data.list_length-1;
crpvecn = (-1*maxabslag):-1;
crpvecp = 1:maxabslag;

figure; hold on
plot([crpvecn crpvecp],res.rand.crps(:,[crpvecn crpvecp]+data.list_length),'x-')
plot(crpvecn,mean(res.rand.crps(:,crpvecn+data.list_length),"omitnan"),...
    ':o','LineWidth',2,'Color',randcol)
plot(crpvecp,mean(res.rand.crps(:,crpvecp+data.list_length),"omitnan"),...
    ':o','LineWidth',2,'Color',randcol)
axis([-.5-maxabslag maxabslag+.5 0 1])

figure; hold on
plot([crpvecn crpvecp],res.temp.crps(:,[crpvecn crpvecp]+data.list_length),'x-')
plot(crpvecn,mean(res.temp.crps(:,crpvecn+data.list_length),"omitnan"),...
    ':o','LineWidth',2,'Color',tempcol)
plot(crpvecp,mean(res.temp.crps(:,crpvecp+data.list_length),"omitnan"),...
    ':o','LineWidth',2,'Color',tempcol)
axis([-.5-maxabslag maxabslag+.5 0 1])


% == Clustering: temporal ==

% == Values by pres == 

% == Value by recall == 