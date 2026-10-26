% compare to behavioral reports 
%addpath(genpath("../../../svn/EMBAM-master/"))

% ===intrusion info
allrecmask = true(size(data.rec_itemnos));
prop_ints = intru_types(data.rec_itemnos, data.intrusions, data.subject+.1*data.session, allrecmask, ...
    data.list_length);

sslist = unique(data.subject+.1*data.session);
fprintf('subno  WLIs   PLIs   ELIs \n');
for si = 1:length(sslist)
    fprintf('%3.1f  %1.3f  %1.3f  %1.3f',[sslist(si) prop_ints(si,:)]);
    fprintf('\n');
end
fprintf('subno  WLIs   PLIs   ELIs \n');


% ===value info
plot_pres_values(data.pres_values,data.pres_types, data.subject);