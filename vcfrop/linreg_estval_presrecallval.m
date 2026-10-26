% LINear REGression with
% 1) output: signed error estimate of value based on all presented items - continuous
% 2) factor: subject - level
% 3) factor: actual value of based on mean of all recalled items - discrete
% 4) factor: random or temporal - level

load res_083026_xs0.mat
wrtab = true;
[nsub,maxntr] = size(res.both.signerr_pres_list);
% ideally I'd want this in something more like a matrix, but I guess we
% have to prep at some point, and this is what Matlab wants...
Subject = nan(nsub*maxntr,1);
ValueEst = Subject;
ValueRecall = Subject;
WeValueRecall = Subject;
ValuePres = Subject;
AsValueRecall = Subject;
ListType = cell(nsub*maxntr,1);
lts = {'rand';'temp'};

trco_sub = 0;
for si = 1:nsub
    for lti = [1 2]
        nonantr = ~isnan(res.(lts{lti}).signerr_pres_list(si,:));
        nnnt = sum(nonantr);

        ValueEst(trco_sub+(1:nnnt)) = ...
            res.(lts{lti}).est_list(si,nonantr);
        Subject(trco_sub+(1:nnnt)) = si;

        ValueRecall(trco_sub+(1:nnnt)) = ...
            res.(lts{lti}).act_rec_list(si,nonantr);

        ValuePres(trco_sub+(1:nnnt)) = ...
            res.(lts{lti}).act_pres_list(si,nonantr);

        WeValueRecall(trco_sub+(1:nnnt)) = ...
            res.(lts{lti}).act_weightrec_list(si,nonantr);

        AsValueRecall(trco_sub+(1:nnnt)) = ...
            res.(lts{lti}).act_wtascrec_list(si,nonantr);


        [ListType{trco_sub+(1:nnnt)}] = deal(lts{lti});

        trco_sub = trco_sub + nnnt;

    end
end

Subject = Subject(1:trco_sub);
ValueEst = ValueEst(1:trco_sub);
ValueRecall = ValueRecall(1:trco_sub);
ValuePres = ValuePres(1:trco_sub);
WeValueRecall = WeValueRecall(1:trco_sub);
AsValueRecall = AsValueRecall(1:trco_sub);
% ListType = ListType(1:trco_sub);
% 
% forlmtbl = table(Subject,ValueEst,ValueRecall,ListType,...
%     'VariableNames',{'Subject','ValueEst','ValueRecall','ListType'});

forlmtbl = table(Subject,ValueEst,ValueRecall,ValuePres,...
    WeValueRecall,AsValueRecall,...
    'VariableNames',{'Subject','ValueEst','ValueRecall','ValuePres',...
    'WeValueRecall','AsValueRecall'});

% I think this is unnecessary for string - could check
% no differences of presentation or recall value estimates by list type, so
% for now not including
%forlmtbl.ListType = categorical(forlmtbl.ListType);
% same results regardless of whether we do this either
%forlmtbl.Subject = categorical(forlmtbl.Subject);

%%

fitlmtbl4 = fitlme(forlmtbl,'ValueEst~1+ValueRecall*ValuePres+(1|Subject)')
%fitlmtbl5 = fitlme(forlmtbl,'ValueEst~1+WeValueRecall*ValuePres+(1|Subject)')

%%
if wrtab
    corrtabfile = 'linreg_estval.tex';
if exist(corrtabfile,'file')
    system(['rm ' corrtabfile]);
end
fidcorrtab = fopen(corrtabfile,'a');
fprintf(fidcorrtab,'\\begin{table} \n');
fprintf(fidcorrtab,'\\begin{tabular}{lll} \n');
fprintf(fidcorrtab,['Effect & Mean (SE) & $p$ ' ...
    ' \\\\ \\hline \n']);

titles = {'Intercept';'Tip, Studied Items';'Tip, Recalled Items';
    'Interaction: Tip Studied and Tip Recalled'};
inds = [1;3;2;4];

for pri = 1:length(inds)
fprintf(fidcorrtab,[titles{pri} ' & %1.3f (%1.3f) & %1.3f'],...
    fitlmtbl4.Coefficients.Estimate(inds(pri)), fitlmtbl4.Coefficients.SE(inds(pri)),...
    fitlmtbl4.Coefficients.pValue(inds(pri)));
fprintf(fidcorrtab,' \\\\ \\hline \n');
end

end

%%
% % 
% %xs = [floor(min(ValueRecall)) ceil(max(ValueRecall))];
% ys = [floor(min(ValueEst))-1 ceil(max(ValueEst))+1];
% %plotcols = [[(.1:.1:.9)' (.1:.1:.9)' (.1:.1:.9)']; [zeros(9,1) (.1:.1:.9)' zeros(9,1)]];
% plotcols = repmat([(.1:.1:.9)' (.1:.1:.9)' (.1:.1:.9)'],4,1);
% plotsyms = {'o';'sq';'^';'+'};
% avgb_fit = fitlmtbl4.Coefficients.Estimate(strcmp(fitlmtbl4.Coefficients.Name,'(Intercept)')) + ...
%     mean(fitlmtbl4.randomEffects);
% beta1_fit = fitlmtbl4.Coefficients.Estimate(strcmp(fitlmtbl4.Coefficients.Name,'ValuePres'));
% beta2_fit = fitlmtbl4.Coefficients.Estimate(strcmp(fitlmtbl4.Coefficients.Name,'ValueRecall'));
% beta12_fit = fitlmtbl4.Coefficients.Estimate(strcmp(fitlmtbl4.Coefficients.Name,'ValueRecall:ValuePres'));

% figure; hold on
% plot(xs,beta2_fit*xs+avgb_fit,'k--','LineWidth',2)
% for si = 1:nsub
%     plot(ValueRecall(Subject==si),ValueEst(Subject==si),plotsyms{ceil(si/10)},'Color',plotcols(si,:))
% end
% publishFig(18)
% ylabel('Estimated value')
% xlabel('Weighted sum of betas?')
% axis([xs(1)-1 xs(2)+1 ys]);

%%
%fitlmtbl1 = fitlme(forlmtbl,'ValueEst~1+ValuePres+(1|Subject)')
%fitlmtbl2 = fitlme(forlmtbl,'ValueEst~1+ValueRecall+(1|Subject)')
%fitlmtbl3 = fitlme(forlmtbl,'ValueEst~1+WeValueRecall+(1|Subject)')

%fitlmtbl4 = fitlme(forlmtbl,'ValueEst~1+WeValueRecall*ValuePres*ValueRecall+(1|Subject)')
%fitlmtbl5 = fitlme(forlmtbl,'ValueEst~1+WeValueRecall*ValueRecall+(1|Subject)')
%fitlmtbl6 = fitlme(forlmtbl,'ValueEst~1+WeValueRecall+(1|Subject)')
%fitlmtbl7 = fitlme(forlmtbl,'ValueEst~1+AsValueRecall*ValuePres+(1|Subject)')
%fitlmtbl9 = fitlme(forlmtbl,'ValueEst~1+WeValueRecall*ValuePres*ValueRecall+(1|Subject)')