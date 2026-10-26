function [total_errs] = mean_abserr_sess(err_matrix, sess_matrix, subjects,maxnsess)
%
%
%COMBINE_VALUES_SESS   Calculate absolute errors averaged across all lists in a
%session
%
%
%  [total_vals] = value_from_mask(err_matrix, subjects, mask,maxnlist)
%
%  INPUTS:
%  err_matrix:  a matrix whose elements are the absolute error associated with
%  each list. rows are subjects, columns are lists
%
% sess_matrix: session number corresponding to each entry in err_matrix
%
%        subjects:  a column vector which indexes the rows of recalls_matrix
%                   with a subject number (or other identifier).  That is,
%                   the recall trials of subject S should be located in
%                   recalls_matrix(find(subjects==S), :)

%
%
%  OUTPUTS:
%        total_vals:  a matrix of values. each column is a session

% vfr_for_subj does all the real work here
total_errs = apply_by_index(@errfr_for_subj, ...
    subjects, ...
    1, ...
    {err_matrix,sess_matrix},maxnsess);
%endfunction

function subj_err = errfr_for_subj(err_vec,sess_vec,maxnsess)
% Helper for vfr:
% calculates the total value by list
subj_err = nan(1,maxnsess);

sesslist = unique(sess_vec(~isnan(sess_vec)));
for sessi = 1:length(sesslist)
    subj_err(sessi) = mean(err_vec(sess_vec==sesslist(sessi)));
end
%endfunction