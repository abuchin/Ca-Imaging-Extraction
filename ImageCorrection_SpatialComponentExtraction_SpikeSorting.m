%% IMAGE CORRECTION PART
% Rigid motion correction.

%Clean up and add paths

clear;
close all;
gcp


% SET UP WORKING DIRECTORIES

% input file name
name = './data/your_file.tif';
% output for the motion-corrected image
outputFileName = './data/your_file_output.tif';
% output for the spike-extraction
savepath='./data/Cells_7/';

addpath('./Code/Image_correction_NoRMCorre');


tic; Y = read_file(name); toc; % read the file (optional, you can also pass the path in the function instead of Y)
Y = double(Y);      % convert to double precision 
T = size(Y,ndims(Y));

% perform non-rigid motion correction
% set parameters (first try out rigid motion correction in parallel)
%%{
options_rigid = NoRMCorreSetParms('d1',size(Y,1),'d2',size(Y,2),'bin_width',50,'max_shift',15,'us_fac',50);
tic; [M1,shifts1,template1] = normcorre(Y,options_rigid); toc
M_final=M1;

% do the non-rigid motion correction (longer, but better results)
%{
options_nonrigid = NoRMCorreSetParms('d1',size(Y,1),'d2',size(Y,2),'grid_size',[32,32],'mot_uf',4,'bin_width',50,'max_shift',15,'max_dev',3,'us_fac',50);
tic; [M2,shifts2,template2] = normcorre_batch(Y,options_nonrigid); toc
M_final=M2;
%}


% Compute the metrics and show the results using video (OPTIONAL)
%{ 

nnY = quantile(Y(:),0.005);
mmY = quantile(Y(:),0.995);

[cY,mY,vY] = motion_metrics(Y,10);
[cM1,mM1,vM1] = motion_metrics(M1,10);
[cM2,mM2,vM2] = motion_metrics(M2,10);
T = length(cY);

%% plot metrics
figure;
    ax1 = subplot(2,3,1); imagesc(mY,[nnY,mmY]);  axis equal; axis tight; axis off; title('mean raw data','fontsize',14,'fontweight','bold')
    ax2 = subplot(2,3,2); imagesc(mM1,[nnY,mmY]);  axis equal; axis tight; axis off; title('mean rigid corrected','fontsize',14,'fontweight','bold')
    ax3 = subplot(2,3,3); imagesc(mM2,[nnY,mmY]); axis equal; axis tight; axis off; title('mean non-rigid corrected','fontsize',14,'fontweight','bold')
    subplot(2,3,4); plot(1:T,cY,1:T,cM1,1:T,cM2); legend('raw data','rigid','non-rigid'); title('correlation coefficients','fontsize',14,'fontweight','bold')
    subplot(2,3,5); scatter(cY,cM1); hold on; plot([0.9*min(cY),1.05*max(cM1)],[0.9*min(cY),1.05*max(cM1)],'--r'); axis square;
        xlabel('raw data','fontsize',14,'fontweight','bold'); ylabel('rigid corrected','fontsize',14,'fontweight','bold');
    subplot(2,3,6); scatter(cM1,cM2); hold on; plot([0.9*min(cY),1.05*max(cM1)],[0.9*min(cY),1.05*max(cM1)],'--r'); axis square;
        xlabel('rigid corrected','fontsize',14,'fontweight','bold'); ylabel('non-rigid corrected','fontsize',14,'fontweight','bold');
    linkaxes([ax1,ax2,ax3],'xy')

%% plot shifts

shifts_r = horzcat(shifts1(:).shifts)';
shifts_nr = cat(ndims(shifts2(1).shifts)+1,shifts2(:).shifts);
shifts_nr = reshape(shifts_nr,[],ndims(Y)-1,T);
shifts_x = squeeze(shifts_nr(:,1,:))';
shifts_y = squeeze(shifts_nr(:,2,:))';

patch_id = 1:size(shifts_x,2);
str = strtrim(cellstr(int2str(patch_id.')));
str = cellfun(@(x) ['patch # ',x],str,'un',0);

figure;
    ax1 = subplot(311); plot(1:T,cY,1:T,cM1,1:T,cM2); legend('raw data','rigid','non-rigid'); title('correlation coefficients','fontsize',14,'fontweight','bold')
            set(gca,'Xtick',[])
    ax2 = subplot(312); plot(shifts_x); hold on; plot(shifts_r(:,1),'--k','linewidth',2); title('displacements along x','fontsize',14,'fontweight','bold')
            set(gca,'Xtick',[])
    ax3 = subplot(313); plot(shifts_y); hold on; plot(shifts_r(:,2),'--k','linewidth',2); title('displacements along y','fontsize',14,'fontweight','bold')
            xlabel('timestep','fontsize',14,'fontweight','bold')
    linkaxes([ax1,ax2,ax3],'x')

%% plot a movie with the results

figure;
for t = 1:1:T
    subplot(121);imagesc(Y(:,:,t),[nnY,mmY]); xlabel('raw data','fontsize',14,'fontweight','bold'); axis equal; axis tight;
    title(sprintf('Frame %i out of %i',t,T),'fontweight','bold','fontsize',14); colormap('bone')
    subplot(122);imagesc(M2(:,:,t),[nnY,mmY]); xlabel('non-rigid corrected','fontsize',14,'fontweight','bold'); axis equal; axis tight;
    title(sprintf('Frame %i out of %i',t,T),'fontweight','bold','fontsize',14); colormap('bone')
    set(gca,'XTick',[],'YTick',[]);
    drawnow;
    pause(0.02);
end
%}
%}

% save corrected video as tiff file
% saveastiff(M_final,outputFileName);
%%


%% COMPONENT EXTRACTION PART
% 
% This part of the code takes the corrected movie and saves the components in mat file in the same folder. 
% In the saved file A - spatial components structure, C2 - temporal components saved as a matrix: rows - traces, columns - time

keep_variables M_final outputFileName savepath

addpath('./Code/ca_source_extraction')
addpath('./Code/ca_source_extraction/utilities')
addpath('./Code/cvx')

% load file
nam = outputFileName;
%nam = './data/840f_4xzoom16_1_motion_corrected.tif';          % insert path to tiff stack here
%sframe=1;                               % user input: first frame to read (optional, default 1)
%num2read=355;                           % user input: how many frames to read   (optional, default until the end)

% read tiff from the file
%Y = bigread2(nam); %,sframe,num2read);
%Y=Y(:,:,1:10:end);

% read from the previous step
Y = M_final;
% Y=Y(:,:,1:10:end); % downsampling


%Y = Y - min(Y(:)); 
if ~isa(Y,'double');    Y = double(Y);  end         % convert to single

[d1,d2,T] = size(Y);                                % dimensions of dataset
d = d1*d2;                                          % total number of pixels

% set parameters

K = 500;                                          % number of components to be found
tau = 4;                                          % std of gaussian kernel (size of neuron) 
p = 2;                                            % order of autoregressive system (p = 0 no dynamics, p=1 just decay, p = 2, both rise and decay)
merge_thr = 0.8;                                  % merging threshold

options = CNMFSetParms(...                      
    'd1',d1,'d2',d2,...                         % dimensions of datasets
    'search_method','dilate','dist',3,...       % search locations when updating spatial components
    'deconv_method','constrained_foopsi',...    % activity deconvolution method
    'temporal_iter',2,...                       % number of block-coordinate descent steps 
    'fudge_factor',0.98,...                     % bias correction for AR coefficients
    'merge_thr',merge_thr,...                    % merging threshold
    'gSig',tau...
    );
% Data pre-processing

[P,Y] = preprocess_data(Y,p);

% fast initialization of spatial components using greedyROI and HALS

[Ain,Cin,bin,fin,center] = initialize_components(Y,K,tau,options,P);  % initialize

% display centers of found components
Cn =  correlation_image(Y); %reshape(P.sn,d1,d2);  %max(Y,[],3); %std(Y,[],3); % image statistic (only for display purposes)

% do not show the initial components
% figure;imagesc(Cn);
%     axis equal; axis tight; hold all;
%     scatter(center(:,2),center(:,1),'mo');
%     title('Center of ROIs found from initialization algorithm');
%     drawnow;

% manually refine components (optional)
refine_components = false;  % flag for manual refinement
if refine_components
    [Ain,Cin,center] = manually_refine_components(Y,Ain,Cin,center,Cn,tau,options);
end
    
% update spatial components
Yr = reshape(Y,d,T);
[A,b,Cin] = update_spatial_components(Yr,Cin,fin,[Ain,bin],P,options);

% update temporal components
P.p = 0;    % set AR temporarily to zero for speed
[C,f,P,S,YrA] = update_temporal_components(Yr,A,b,Cin,fin,P,options);

% classify components
[ROIvars.rval_space,ROIvars.rval_time,ROIvars.max_pr,ROIvars.sizeA,keep] = classify_components(Y,A,C,b,f,YrA,options);

% run GUI for modifying component selection (optional, close twice to save values)
run_GUI = false;
if run_GUI
    Coor = plot_contours(A,Cn,options,1); close;
    GUIout = ROI_GUI(A,options,Cn,Coor,keep,ROIvars);   
    options = GUIout{2};
    keep = GUIout{3};    
end

% merge found components
[Am,Cm,K_m,merged_ROIs,Pm,Sm] = merge_components(Yr,A(:,keep),b,C(keep,:),f,P,S,options);

%
display_merging = 1; % flag for displaying merging example

if and(display_merging, ~isempty(merged_ROIs))
    i = 1; %randi(length(merged_ROIs));
    ln = length(merged_ROIs{i});
    figure;
        set(gcf,'Position',[300,300,(ln+2)*300,300]);
        for j = 1:ln
            subplot(1,ln+2,j); imagesc(reshape(A(:,merged_ROIs{i}(j)),d1,d2)); 
                title(sprintf('Component %i',j),'fontsize',16,'fontweight','bold'); axis equal; axis tight;
        end
        subplot(1,ln+2,ln+1); imagesc(reshape(Am(:,K_m-length(merged_ROIs)+i),d1,d2));
                title('Merged Component','fontsize',16,'fontweight','bold');axis equal; axis tight; 
        subplot(1,ln+2,ln+2);
            plot(1:T,(diag(max(C(merged_ROIs{i},:),[],2))\C(merged_ROIs{i},:))'); 
            hold all; plot(1:T,Cm(K_m-length(merged_ROIs)+i,:)/max(Cm(K_m-length(merged_ROIs)+i,:)),'--k')
            title('Temporal Components','fontsize',16,'fontweight','bold')
        drawnow;
end

% refine estimates excluding rejected components

Pm.p = p;    % restore AR value
[A2,b2,C2] = update_spatial_components(Yr,Cm,f,[Am,b],Pm,options);
[C2,f2,P2,S2,YrA2] = update_temporal_components(Yr,A2,b2,C2,f,Pm,options);


% make movie
%make_patch_video(A_or,C_or,b2,f2,Yr,Coor,options)

% do the plotting

%plot the components
[A_or,C_or,S_or,P_or] = order_ROIs(A2,C2,S2,P2); % order components

K_m = size(C_or,1);
[C_df,~] = extract_DF_F(Yr,A_or,C_or,P_or,options); % extract DF/F values (optional)

% save the calcium traces and spatial components
save(strcat(sprintf(nam),'.mat'));
save(strcat(sprintf(nam),'_temporal_components','.mat'),'C2');

temporal_comp_name=strcat(sprintf(nam),'_temporal_components','.mat');

figure;
[Coor,json_file] = plot_contours(A_or,Cn,options,1); % contour plot of spatial footprints

% display components using using gui
plot_components_GUI(Yr,A_or,C_or,b2,f2,Cn,options)
%%


%% SPIKE EXTRACTION PART
% Spikes estimation based on the recorded fluorescence components. They are called calcium here.

% Clear, but keep saved variables
keep_variables savepath temporal_comp_name

addpath('./Code/ca_spike_extraction');

nam=temporal_comp_name;
%nam = './data/840f_4xzoom16_1_motion_corrected.tif_temporal_components.mat'

load(nam);

% rename the calcium matrix
A=C2;
clear('C2');

% Loop over all cells (temporal components)

parfor j=1:1:size(A,1)

dt=0.0222;             % sampling rate, 45Hz
    
% Calibration of parameters for a given Ca trace

calcium=A(j,:);
% length(A)

calcium = calcium/mean(calcium);

% Auto-calibration only for sigma parameter

psig = spk_autosigma('par');
sigma = spk_autosigma(calcium,dt,psig)

% Estimation with MLspike with parameters set manually (except sigma)

% parameters
par = tps_mlspikes('par');

% (do not display graph summary)
par.dographsummary = false;

% time constant
par.dt = dt;

% physiological parameters for GCamp 6f non-linearity
par.a = 0.034;
par.tau = 0.76;
par.pnonlin = [0.85 -0.006];

% noise and drift
par.finetune.sigma = sigma; % not that if you ommit this line, sigma would be estimated anyway by MLspike calling spk_autosigma
par.drift.parameter = .01;

% spike estimation
[spikest fit drift] = spk_est(calcium,par);

% save estimated spike times and other parameters
parsave(strcat(savepath,sprintf('%d.mat',j)),spikest, fit, drift,calcium,dt);

end

keep_variables savepath

% load the first spike component
load(strcat(savepath,'1.mat'));

figure
%plot the first estimated trace
spk_display(dt,{spikest},{calcium fit drift})
set(gca,'Fontsize',20)
title('ML spike extraction')
%%