# Ca-Imaging-Extraction
This page collect all the links to the code that might be reused in the future studies. You are welcome to contribute your own section that might be useful for anyone else in their research.

The code was too large to put it here. The full version could be found here:
https://drive.google.com/open?id=0B4VgK-0yzdgZNFliQXJYTFpQdmc

Analysis of Ca imaging data and spike extraction

Key papers
Pnevmatikakis et al 2016
Deneux et al 2016, supplementary

Here is the code of the pipeline I created to analyse the calcium imaging movies. It contains 3 main components: 1) image stabilisation 2) component extraction and 3) spike inference. In principle there is nothing complicated in this code, it just contains the combination of different methods in one script ready to be used for your tiff stack with the calcium imaging.

Instruction:

0) Run cvx package setup. To do so go to ./Code/cvx/cvx.setup and run it in Matlab.
1) Change the directory in Matlab where ImageCorrection_SpatialComponentExtraction_SpikeSorting.m is located (it is needed to avoid hard-coding of the packages)
2) Put your data into ./data folder in the tif stack format.
3) Make a link to the data in the script

ImageCorrection_SpatialComponentExtraction_SpikeSorting.m
% SET UP WORKING DIRECTORIES
% input file name
name = './data/your_favorite_file.tif'; - path to the tif file
% output for the motion-corrected image
outputFileName = './data/your_favorite_file.tif'; - path to the output, it will safe only the results in the form of mat file
% output for the spike-extraction
savepath='./data/Cells_7/'; - folder where you want the inferred temporal components to be extracted in mat files


Tips, caveats and references:

1) Image stabilisation part is taken from Eftychios Pnevmatikakis code: https://github.com/simonsfoundation/NoRMCorre
By default in the script there is a rigid motion correction, which requires much less RAM. But if you want to perform the non-rigid image stabilisation, comment this section

% set parameters (first try out rigid motion correction in parallel)
%%{
options_rigid = NoRMCorreSetParms('d1',size(Y,1),'d2',size(Y,2),'bin_width',50,'max_shift',15,'us_fac',50);
tic; [M1,shifts1,template1] = normcorre(Y,options_rigid); toc
M_final=M1;

and uncomment this one

% do the non-rigid motion correction (longer, but better results)
%{
options_nonrigid = NoRMCorreSetParms('d1',size(Y,1),'d2',size(Y,2),'grid_size',[32,32],'mot_uf',4,'bin_width',50,'max_shift',15,'max_dev',3,'us_fac',50);
tic; [M2,shifts2,template2] = normcorre_batch(Y,options_nonrigid); toc
M_final=M2;
%}

Be aware, that non-rigid motion correction might provide better results, yet it could require the substantial amount of RAM. Matlab can get crushed if it takes too much.


2) Component extraction part is done also using Eftychios Pnevmatikakis code: https://github.com/epnev/ca_source_extraction
The short description of the method could be found in this presentation: https://docs.google.com/presentation/d/1ooyBtgxBSyzy3TKDVW3A-ONG8X1v34SsKei1Jz88ME8/edit#slide=id.p
See the original paper here: http://www.cell.com/neuron/abstract/S0896-6273(15)01084-3

Shortly speaking the algorithm aims to find the decomposition of the calcium movie into the spatial and temporal matrices. It does that by minimisation of variance of ca imaging data. Therefore image stabilisation part is very important, since motion artefacts could provide substantial distortions for the Ca signal. Another caveat is the number of the inferred components. The algorithm is non-convex, so there are multiple solutions possible. Practically speaking it seems that the algorithm works the best when you initialise larger number of the spatial components than there are neurons. There is an algorithm for merging the components determined by merge_thr parameter. If the components look too similar, try to decrease its value.


3) The last part corresponds to the spike extraction. I used the latest algorithm of inferring spikes called ML-spike developed by Thomas Deneux et al. You could find it here:
http://www.nature.com/articles/ncomms12190
Short description of this principle could be found in this movie: https://www.youtube.com/watch?v=EoWY2VeFUlM&feature=youtu.be

This algorithm analyses the calcium trace and finds the spiking sequence with the highest likelihood. The precision of spike extraction strongly depends on the calcium model being used. It determines the non-linearity of the response, when multiple spikes fired at once. In the present code, the algorithm uses non-linearity specific for GCamp-6f. If the experiments are done using the other calcium sensor, you should change these parameters

par.a = 0.034;
par.tau = 0.76;
par.pnonlin = [0.85 -0.006];

Look up the values in the supplementary of the corresponding paper, they have been calculated for GCamp-6s, OGB and some other dyes.
