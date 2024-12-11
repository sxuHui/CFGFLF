%
%
%

clear;
clc;
data_path = fullfile(pwd, filesep, "data_X", filesep);
addpath(data_path);

lib_path = fullfile(pwd, filesep, "lib", filesep);
addpath(lib_path);

dirop = dir(fullfile(data_path, '*.mat'));
datasetCandi = {dirop.name};

data_path_Hs = fullfile(pwd, filesep, "data_Hs", filesep);
exp_n = 'SKKM_Hs';
for i1 = 1:length(datasetCandi)
    data_name = datasetCandi{i1}(1:end-4);
    [X, Y] = extractXY(data_name);
    assert(size(X, 1) == size(Y, 1));
    nSmp = size(X, 1);
    nCluster = length(unique(Y));
    nKernel = 12;
    nDim = nCluster * 4;
    
    %***********************************************
    % SKKM_Hs
    %***********************************************
    fname2 = fullfile(data_path_Hs, [data_name, '_12k_SKKM_Hs.mat']);
    if ~exist(fname2, 'file')
        Hs = cell(1, nKernel);
        Evs = cell(1, nKernel);
        for iKernel = 1:nKernel
            K = X2K_rmkkm_2015_v2(X, iKernel);
            [H_normalized, H, ev, obj] = SKKM(K, nDim);
            Hs{iKernel} = H;
            Evs{iKernel} = ev;
        end
        save(fname2, 'Hs', 'Evs', 'Y');
        disp([data_name(1:end-4), ' has been completed!']);
    end
end
rmpath(data_path);
rmpath(lib_path);