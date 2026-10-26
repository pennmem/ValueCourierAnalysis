function res = ana_vavi(datapath,bysess)
%
% function res = ana_vavi(datapath,toplot)
%
% NOTE: lag-CRP includes OPs 1-3 because SPC looks like DFR, presumably due
% to value estimation and delay
%
% we want everything for all lists AND by condition
% function to run all of the scripts, with data and indices
% (could do fields and masks, but I think passing everything and the
% annoying code isn't worth saving time to recalculate mask subsets)


%% set up and input to inner func
%addpath(genpath('../../../../svn/EMBAM-master/'));
load(datapath,'data');
if bysess
    % analyze by session, so treat each "subject" as a subject's session
    data.subject_orig = data.subject;
    data.subject = data.subject+1000*data.session;
    % also keep track of this list for potential future use
end
res.unilist = unique(data.subject);

incs0 = ismember(0,data.session);

res.both = inner_ana_vavi(data,true(size(data.pres_types)),incs0);
res.rand = inner_ana_vavi(data,data.pres_types==1,incs0);
res.temp = inner_ana_vavi(data,data.pres_types==2,incs0);

%% inner func
function res = inner_ana_vavi(data,inds,incs0)

%% we'll use this a lot, and sometimes easier to pull from inds
recalls_inds = data.recalls(inds,:);

%% make masks
rec_mask = make_clean_recalls_mask2d(recalls_inds);
pres_mask = true(size(data.pres_itemnos(inds,:)));

%% recall probabilities
% == SPCs ==
res.spcs = spc(recalls_inds,data.subject(inds,:),data.list_length,...
    rec_mask,pres_mask);

% == PFRs ==
res.pfrs = pfr(recalls_inds,data.subject(inds,:),data.list_length,...
    rec_mask,pres_mask);

% no intrusions - pilot reports should be fine

% based on values
sublist = unique(data.subject);
uni_vals = unique(data.pres_values);
res.prec_byval = nan(length(sublist),length(uni_vals));
for vi = uni_vals'
    res.prec_byval(:,vi) = mean(spc(recalls_inds,...
        data.subject(inds,:),data.list_length,...
    rec_mask & (data.rec_values(inds,:)==vi),...
    pres_mask & (data.pres_values(inds,:)==vi)),2,"omitnan");
end

%% Recall transition analyses
% == lag-CRP ==
res.crps = crp(recalls_inds,data.subject(inds,:),data.list_length, ...
    rec_mask, rec_mask, pres_mask, pres_mask);
% recmaskxop123 =  rec_mask;
% recmaskxop123(:,1:3) = false;
% res.crps = crp(recalls_inds,data.subject(inds,:),data.list_length, ...
%     recmaskxop123, recmaskxop123, pres_mask, pres_mask);

% == Clustering: temporal ==
res.temp_clusts = temp_fact(recalls_inds,data.subject(inds,:),...
    data.list_length,rec_mask, pres_mask);

% == Clustering: value ==
% dist_fact?

%% Value estimation

if incs0
    maxnlist = 55;
else
    maxnlist = 50;
end

% == objective answer if use values from recalls only (correct only) ==
% maybe include PLI, but my thought is,
% a) no one's really expecting this, and doubtful that it'll
% make a huge difference either way, so might as well
% keep out and keep simpler to start
% b) something's messed up with PLI info if they're recalling at the right
% time, so can't guarantee how it's contributing

% estimated value but averaged for participants across lists
res.est_list = value_from_mask(data.est_values(inds,:),...
    data.subject(inds,:),true(size(data.subject(inds,:))),maxnlist);

res.act_rec_list = value_from_mask(data.rec_values(inds,:),...
    data.subject(inds,:),rec_mask,maxnlist);

% we also want to get the actual values for NONrecalled items. we'll need
% to get which items these are on a list by list basis, I think. apply to a
% pres_mask to then use value_from_mask
nonrec_mask = pres_mask;
rec_itemnos_formask = data.rec_itemnos(inds,:);
pres_itemnos_formask = data.pres_itemnos(inds,:);
for trinr = 1:size(rec_itemnos_formask,1)
    % 1) rec_mask(trinr,:): get ok OPs for this trial
    % 2) rec_itemnos_formask(trinr,rec_mask(trinr,:)): get ok rec_itemnos for
    % this trial
    % 3) whole thing: set the mask based on which items are NOT a member of
    % that
    reclist = rec_itemnos_formask(trinr,rec_mask(trinr,:)); 
    spinds = ismember(pres_itemnos_formask(trinr,:),reclist);
    if isequal(sort(pres_itemnos_formask(trinr,spinds)),sort(reclist))
        nonrec_mask(trinr,:) = ~spinds;
    else
        keyboard
    end
end

res.act_nonrec_list = value_from_mask(data.pres_values(inds,:),...
    data.subject(inds,:),nonrec_mask,maxnlist);

% I think it's easier to do here: create weighted mask of recalls based on
% OPs
dub_rm = double(rec_mask);
wtd_rec_mask_dsc = dub_rm;
wtd_rec_mask_asc = dub_rm;
nrec_rec_mask = dub_rm;
for tri = 1:size(wtd_rec_mask_dsc,1)
    % do it this way so that we always include all errors as possible OPs.
    % I don't want to have to deal with counting those differently, and I
    % do think it means something for OP anyway
    max_op = find(recalls_inds(tri,:)>0,1,'last');
    wtd_rec_mask_dsc(tri,1:max_op) = (max_op:-1:1).*wtd_rec_mask_dsc(tri,1:max_op);
    wtd_rec_mask_asc(tri,1:max_op) = (1:max_op).*wtd_rec_mask_dsc(tri,1:max_op);
    % while we're here, a cheesy way to use value_from_mask to get number
    % of correct ops per list aligned with other functions
    nrec_rec_mask(tri,1:max_op) = sum(rec_mask(tri,1:max_op))*nrec_rec_mask(tri,1:max_op);
end
res.act_weightrec_list = value_from_mask(data.rec_values(inds,:),...
    data.subject(inds,:),wtd_rec_mask_dsc,maxnlist);

res.act_wtascrec_list = value_from_mask(data.rec_values(inds,:),...
    data.subject(inds,:),wtd_rec_mask_asc,maxnlist);

res.act_wtdscrec_list = value_from_mask(data.rec_values(inds,:),...
    data.subject(inds,:),wtd_rec_mask_dsc,maxnlist);

res.act_nrec_list =  value_from_mask(nrec_rec_mask,data.subject(inds,:),...
    dub_rm,maxnlist);

% this might be Inf, but I think we just want undefined to exclude
res.act_nonrec_list(res.act_nrec_list==data.list_length) = NaN;

%[objective answer from presentation given as actualvalue, which is always
%confirmed to be the same as the mean across SPs]
res.act_pres_list = value_from_mask(data.pres_values(inds,:),...
    data.subject(inds,:),pres_mask,maxnlist);

% these should align for which lists to include
if ~isequal(isnan(res.est_list),isnan(res.act_pres_list))
    keyboard
end

% also check that it aligns with the number of sessions for each subject
for subi = sublist'
    this_sess = data.session(data.subject==subi);
    if ~isequal(incs0*5+10*length(unique(this_sess(this_sess>0))),...
            length(this_sess)) && ~isequal(subi,615)
        keyboard
    end
end

% these might not align if no recalls, which were set to Inf to distinguish
% from default NaN
if sum(sum(isinf(res.act_rec_list)))
    if isequal(isnan(res.est_list),isnan(res.act_rec_list)) 
        if isequal(isnan(res.act_rec_list),isnan(res.act_weightrec_list)) && ...
                isequal(isinf(res.act_rec_list),isinf(res.act_weightrec_list)) ...
            && isequal(isinf(res.act_rec_list),isinf(res.act_wtascrec_list)) ...
            && isequal(isinf(res.act_rec_list),isinf(res.act_wtdscrec_list)) ...
            && isequal(isinf(res.act_rec_list),isinf(res.act_nrec_list)) ...
            && isequal(isinf(res.act_rec_list),isinf(res.act_nonrec_list))
            % now that we know they align, make it easier to take the mean
            res.act_rec_list(isinf(res.act_rec_list)) = NaN;
            res.act_weightrec_list(isinf(res.act_weightrec_list)) = NaN;
            res.act_wtascrec_list(isinf(res.act_wtascrec_list)) = NaN;
            res.act_wtdscrec_list(isinf(res.act_wtdscrec_list)) = NaN;
            res.act_nrec_list(isinf(res.act_nrec_list)) = NaN;
            res.act_nonrec_list(isinf(res.act_nonrec_list)) = NaN;
        else
            keyboard
        end
        % otherwise, if no Infs then should also align
    elseif ~isequal(isnan(res.est_list),isnan(res.act_rec_list))
        keyboard
    end
end

% NaN-NaN = NaN

% to use with whether the values biased towards recall, i.e., is the signed error greater
% than actual when the recall estimate is larger?
% also keep separate by list for each participant
res.signerr_pres_list = res.est_list-res.act_pres_list;
res.abserr_rec_list = abs(res.act_rec_list-res.est_list);
res.abserr_pres_list = abs(res.act_pres_list-res.est_list);
res.abserr_wtrec_list = abs(res.act_weightrec_list-res.est_list);
res.baseerr_rec_list = abs(res.act_pres_list-25);

% also use these values to calculate more standard absolute error
res.abserr_pres_sub = mean(res.abserr_pres_list,2,"omitnan");
res.abserr_rec_sub = mean(res.abserr_rec_list,2,"omitnan");
res.abserr_wtrec_sub = mean(res.abserr_wtrec_list,2,"omitnan");
res.mm_abserr_pres_sub = [min(res.abserr_pres_list,[],2) ...
    max(res.abserr_pres_list,[],2)];
res.mm_signerr_pres_sub = [min(res.signerr_pres_list,[],2) ...
    max(res.signerr_pres_list,[],2)];

% get actual values of what we would expect
res.act_rec_sub = mean(res.act_rec_list,2,"omitnan");
res.act_pres_sub = mean(res.act_pres_list,2,"omitnan");

% get the absolute error by session, first calculating a matrix of sessions
% to include. note that we checked earlier that the number of lists aligns
sessnos = data.session(inds);
subwind = data.subject(inds);
sublist = unique(subwind);
nsub = length(sublist);
session_matrix = nan(nsub,maxnlist);
for subi = 1:nsub
    these_sess = sessnos(subwind==sublist(subi));
    session_matrix(subi,1:length(these_sess)) = these_sess;
end


% efficient code for all lists! but need to take into account that it's
% fewer lists for different conditions
% if incs0
%     % zeros(1,5)
%      session_mat = repmat([zeros(1,5) reshape(repmat(1:max(data.session),max(data.trial),1),1,[])],...
%         size(res.act_rec_list,1),1);
% else
%     session_mat = repmat(reshape(repmat(1:max(data.session),max(data.trial),1),1,[]),...
%         size(res.act_rec_list,1),1);
% end


res.abserr_rec_sess = mean_abserr_sess(res.abserr_rec_list, session_matrix, sublist,length(unique(data.session(inds,:))));
res.abserr_pres_sess = mean_abserr_sess(res.abserr_pres_list, session_matrix, sublist,length(unique(data.session(inds,:))));