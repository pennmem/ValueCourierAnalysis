%count_sub_sess

maxnsess = 6;

sublist = unique(data.subject);
nsub = length(sublist);

nsps = nan(nsub,1);

for si = 1:nsub
    nsps(si) = length(unique(data.session(data.subject==sublist(si))));
end

countsess = nan(1,maxnsess);
for sessi=1:maxnsess
    countsess(sessi) = sum(nsps==sessi);
end

figure
bar(1:maxnsess,countsess,'k')
publishFig
xlabel('Number of sessions')
ylabel('Number of participants')
xlim([.5 maxnsess+.5])
