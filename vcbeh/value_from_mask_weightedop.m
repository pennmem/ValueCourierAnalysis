function [total_vals] = value_from_mask_weightedop(values_matrix, ...
    subjects, mask, maxnlist, weights)
%
%
%VALUE_FROM_MASK_WEIGHTEDOP   Calculate value summed across all
% valid items for a list, WEIGHTED BY OP.
% This should work to get the mean value across all presented
% items, or recalled items only
%
%
%  [total_vals] = value_from_mask(values_matrix, subjects, mask,maxnlist)
%
%  INPUTS:
%  values_matrix:  a matrix whose elements are the value associated with each item
%                   This can work for pres or recall.
%                   For pres, rows are studied items.
%                   For recall, the rows of this matrix should represent recalls
%                   made by a single subject on a single trial.
%
%        subjects:  a column vector which indexes the rows of recalls_matrix
%                   with a subject number (or other identifier).  That is,
%                   the recall trials of subject S should be located in
%                   recalls_matrix(find(subjects==S), :)
%
%        mask:      a logical matrix of the same shape as
%                   values_matrix, which is false at positions (i, j) where
%                   the value at values_matrix(i, j) should be excluded from
%                   the value calculation
%
%       maxnlist:   maximum number of lists per subject, so that
%                   concatenating across subjects yields a matrix. fills in
%                   empty lists for a subject as NaN
%
%
%  OUTPUTS:
%        total_vals:  a matrix of values. each column is a list
%                   Rows are indexed by subject
%

% vfr_for_subj does all the real work here
total_vals = apply_by_index(@vfr_for_subj, ...
    subjects, ...
    1, ...
    {values_matrix, mask},maxnlist,weights);
%endfunction

function subj_val = vfr_for_subj(values_matrix, mask,maxnlist,weights)
% Helper for vfr:
% calculates the total value by list
subj_val = nan(1,maxnlist);

for li = 1:size(values_matrix,1)
    if sum(mask(li,:))
        subj_val(li) = sum(values_matrix(li,mask(li,:)))/sum(mask(li,:));
    else
        subj_val(li) = Inf; % to distinguish between whether we somehow skipped over this list
    end
end

%endfunction