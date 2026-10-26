% LINear REGression with
% tipv value estimates from PRESented items or RECALLed items only

load res_050526_45sess.mat
wrtab = false;
[nsub,maxntr] = size(res.both.signerr_pres_list);
% ideally I'd want this in something more like a matrix, but I guess we
% have to prep at some point, and this is what Matlab wants...
for nrec = 9:14
    Subject = nan(nsub*maxntr,1);
    ValueEst = Subject;
    ValueRecall = Subject;
    ValueNotRecall = Subject;
    ValuePres = Subject;
    AsValueRecall = Subject;
    ListType = cell(nsub*maxntr,1);
    numRecall = Subject;
    lts = {'rand';'temp'};
    ll = 15;
    Vr_coeff = nrec/ll;
    Vnr_coeff = (ll-nrec)/ll;

    trco_sub = 0;
    for si = 1:nsub
        for lti = [1 2]
            nonantr =  res.(lts{lti}).act_nrec_list(si,:)==nrec;
            % prev: ~isnan(res.(lts{lti}).signerr_pres_list(si,:))
            % checked that including isnan is redundant with just nrec_list>0
            nnnt = sum(nonantr);
            % if nnnt==0
            %     keyboard
            % end

            ValueEst(trco_sub+(1:nnnt)) = ...
                res.(lts{lti}).est_list(si,nonantr);
            Subject(trco_sub+(1:nnnt)) = si;

            ValueRecall(trco_sub+(1:nnnt)) = ...
                res.(lts{lti}).act_rec_list(si,nonantr);

            ValueNotRecall(trco_sub+(1:nnnt)) = ...
                Vnr_coeff*res.(lts{lti}).act_nonrec_list(si,nonantr);

           

            [ListType{trco_sub+(1:nnnt)}] = deal(lts{lti});

            trco_sub = trco_sub + nnnt;

        end
    end

    Subject = Subject(1:trco_sub);
    ValueEst = ValueEst(1:trco_sub);
    ValueRecall = ValueRecall(1:trco_sub);
    ValuePres = ValuePres(1:trco_sub);
    ValueNotRecall = ValueNotRecall(1:trco_sub);
    AsValueRecall = AsValueRecall(1:trco_sub);
   

    forlmtbl = table(Subject,ValueEst,ValueRecall,ValueNotRecall,...
        'VariableNames',{'Subject','ValueEst','ValueRecall','ValueNotRecall'});

    % I think this is unnecessary for string - could check
    % no differences of presentation or recall value estimates by list type, so
    % for now not including
    %forlmtbl.ListType = categorical(forlmtbl.ListType);
    % same results regardless of whether we do this either
    %forlmtbl.Subject = categorical(forlmtbl.Subject);

    %%

    fitlmtbl4 = fitlme(forlmtbl,'ValueEst~1+ValueRecall*ValueNotRecall+(1|Subject)');
    fprintf('Vr=%1.3f,p=%1.4f;Vnr=%1.3f,p=%1.4f, R^2=%1.2f \n',...
        fitlmtbl4.Coefficients.Estimate(2),fitlmtbl4.Coefficients.pValue(2),...
        fitlmtbl4.Coefficients.Estimate(3),fitlmtbl4.Coefficients.pValue(3),...
        fitlmtbl4.Rsquared.Ordinary);
   
end
%%
if wrtab
    corrtabfile = 'linreg_estval_nrr.tex';
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
% ll = data.list_length; nrec = number recalled per list
% option 1: ValueEst ~ b0 + b1*ValueRecall
% option 2: ValueEst ~ b0 + b1*nrec/ll*ValueRecall + b2*(ll-nrec)/ll*ValureNotRecall
% can't have weighting that works for both b/c nrec is different for each list
% so, run regressions with lists that have the most common recall values??
% exclude nrec=ll b/c then the two models are the same

%addpath(genpath('../../../../svn/EMBAM-master/'))
%load data_050526_45sess.mat
%spcsa = spc(data.recalls,1:length(data.subject),data.list_length);
%nreca = data.list_length*mean(spcsa,2);
% t2 = [nreca data.subject];
%for i = 1:15
%   nrecsub(i) = length(unique(t2(t2==i,2)));
%end
%x1 = histc(nreca,1:16);
% [(1:15)' x1(2:end) nrecsub']
%    1     0     0
%    2     2     2
%    3     6     4
%    4    11     7
%    5    35    11
%    6    50    13
%    7    76    17
%    8    87    16
%    9   108    19
%   10   114    18
%   11    93    20
%   12   103    20
%   13    97    19
%   14   116    15
%   15   182    13