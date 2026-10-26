function prop_ints = intru_types(rec_itemnos, intrusions, subjects, mask, ll)
% 
% 
%INTRU_TYPES   Calculate proportion of intrusions by type
% 
%
%  
%  prop_ints = intru_types(rec_itemnos, intrusions, subjects, mask, ll)
% 1 row for each subject. columns: WLIs, PLIs, ELIs

if ~isequal(size(rec_itemnos),size(intrusions)) || ~isequal(size(rec_itemnos),size(mask))
    error('matrix size misalignment');
end

% vfr_for_subj does all the real work here
prop_ints = apply_by_index(@int_for_subj, ...
                           subjects, ...
			   1, ...
                           {rec_itemnos, intrusions, mask}, ll);
%endfunction

function subj_ints = int_for_subj(rec_itemnos, intrusions, mask, ll)
 
% WLI: intrusion==-1 && itemno>0
% PLI: intrusion>0
% ELI: intrusion==-1 && itemno==-1
% denominator: number of words per session

subj_ints = [sum(sum(mask&rec_itemnos>0&intrusions==-1)) ...
    sum(sum(mask&intrusions>0)) ...
    sum(sum(mask&rec_itemnos==-1&intrusions==-1))] ./ (size(mask,1)*ll);

%endfunction