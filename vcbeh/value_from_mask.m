function [total_vals] = value_from_mask(values_matrix, subjects, mask, maxnlist)
%
%
%VALUE_FROM_RECALLS   Calculate value averaged across all
% valid items for a list. 
% values_matrix could be used to calculate the mean value across all presented
% items, recalled items only, or the errors of indiivudal recalled items,
% with the right 
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
%        mask:      a matrix of the same shape as
%                   values_matrix, which is 0 at positions (i, j) where
%                   the value at values_matrix(i, j) should be excluded from
%                   the value calculation. could be 1's otherwise, or
%                   weighted in some way (e.g., based on output position)
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
    {values_matrix, mask},maxnlist);
%endfunction

function subj_val = vfr_for_subj(values_matrix, mask,maxnlist)
% Helper for vfr:
% calculates the total value by list
subj_val = nan(1,maxnlist);

for li = 1:size(values_matrix,1)
    if sum(mask(li,:))
        % need to omitnan for intrusions, nothing more than this in case we
        % change the setup
        subj_val(li) = sum(values_matrix(li,:).*double(mask(li,:)),"omitnan")/sum(double(mask(li,:)));
    else
        subj_val(li) = Inf; % to distinguish between whether no recalls versus NaN for another reason
    end
end

%endfunction