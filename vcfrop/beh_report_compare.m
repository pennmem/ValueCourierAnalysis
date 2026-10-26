 %addpath(genpath("../../../../svn/EMBAM-master/"))
 %addpath(genpath("../../e1_inperson/postprocess"))
% 
% % % ===intrusion info
% allrecmask = true(size(data.rec_itemnos));
% prop_ints = intru_types(data.rec_itemnos, data.intrusions, data.subject+.1*data.session, allrecmask, ...
%     data.list_length);
% 
% sslist = unique(data.subject+.1*data.session);
% fprintf('subno  WLIs   PLIs   ELIs \n');
% for si = 1:length(sslist)
%     fprintf('%3.1f  %1.3f  %1.3f  %1.3f',[sslist(si) prop_ints(si,:)]);
%     fprintf('\n');
% end
% fprintf('subno  WLIs   PLIs   ELIs \n');
% 
% % 
% % % ===value info
% % plot_pres_values(data.pres_values,data.pres_types, data.subject);
% 
% % == other beh
% %res = ana_vavi_e2frop(datapath);
% plot_res_forpres_e2frop

% % individual session info
% resbys = ana_vavi_e2frop('data_083026_ws0',true);
% unisubfromsesslist = mod(resbys.unilist,1000);
% sublist = unique(unisubfromsesslist);
% nsub = length(sublist);
% cols = nan(2,5,3);
% cols(1,1,:) = [140 26 17];
% cols(1,2,:) = [169 30 19];
% cols(1,3,:) = [206 26 24];
% cols(1,4,:) = [240 42 28];
% cols(1,5,:) = [255 91 79];
% cols(1,6,:) = [0 0 0];
% 
% cols(2,1,:) = [9 30 88];
% cols(2,2,:) = [13 46 135];
% cols(2,3,:) = [19 66 196];
% cols(2,4,:) = [23 84 247];
% cols(2,5,:) = [106 145 255];
% cols(1,6,:) = [0 0 0];
% cols = cols/255;

for subi = 1:nsub

    figure; % spcs
    spctemp = resbys.temp.spcs(unisubfromsesslist==sublist(subi),:);
    spcrand = resbys.rand.spcs(unisubfromsesslist==sublist(subi),:);
    for i = 1:size(spctemp,1)
         subplot(2,2,1); hold on
        plot(spctemp(i,:),'o-','Color',cols(2,i,:));
         subplot(2,2,2); hold on
        plot(spcrand(i,:),'o-','Color',cols(1,i,:));
    end
    ylim([0 1])

     % pfrs
    pfrtemp = resbys.temp.pfrs(unisubfromsesslist==sublist(subi),:);
    pfrrand = resbys.rand.pfrs(unisubfromsesslist==sublist(subi),:);
    for i2 = 1:size(pfrtemp,1)
        subplot(2,2,3); hold on
        plot(pfrtemp(i2,:),'o-','Color',cols(2,i2,:));
        subplot(2,2,4); hold on
        plot(pfrrand(i2,:),'o-','Color',cols(1,i2,:));
    end
    ylim([0 1])

    ll = 15;
    
    maxabslag = ll-1;
    crpvecn = (-1*maxabslag):-1;
    crpvecp = 1:maxabslag;
    crptemp = resbys.temp.crps(unisubfromsesslist==sublist(subi),:);
    crprand = resbys.rand.crps(unisubfromsesslist==sublist(subi),:);

    figure; 
    for i3 = 1:size(crptemp,1)
    subplot(2,1,1); hold on
    plot(crpvecn,crprand(i3,crpvecn+ll),...
        ':o','LineWidth',2,'Color',cols(1,i3,:),'MarkerFaceColor',cols(1,i3,:))
    plot(crpvecp,crprand(i3,crpvecp+ll),...
        ':o','LineWidth',2,'Color',cols(1,i3,:),'MarkerFaceColor',cols(1,i3,:))
    axis([-.5-maxabslag maxabslag+.5 0 .6])

    subplot(2,1,2); hold on
    plot(crpvecn,crptemp(i3,crpvecn+ll),...
        'sq--','LineWidth',2,'Color',cols(2,i3,:),'MarkerFaceColor',cols(2,i3,:))
    plot(crpvecp,crptemp(i3,crpvecp+ll),...
        'sq--','LineWidth',2,'Color',cols(2,i3,:),'MarkerFaceColor',cols(2,i3,:))
   
    axis([-.5-maxabslag maxabslag+.5 0 .6])
    end

end