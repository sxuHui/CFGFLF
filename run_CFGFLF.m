%
%
%

clear;
clc;
data_path = fullfile(pwd, filesep, "data_Hs", filesep);
addpath(data_path);
lib_path = fullfile(pwd, filesep, "lib", filesep);
addpath(lib_path);

dirop = dir(fullfile(data_path, '*.mat'));
datasetCandi = {dirop.name};

data_path_Hs = fullfile(pwd,  filesep, "data_Hs", filesep);

exp_n = 'CFGFLF';
% profile off;
% profile on;
for i1 = 1:length(datasetCandi)
    data_name = datasetCandi{i1}(1:end-4);
    dir_name = [pwd, filesep, exp_n, filesep, data_name, filesep];
    create_dir(dir_name);
    fname2 = fullfile(data_path_Hs, [data_name, '.mat']);
    load(fname2);
    nCluster = length(unique(Y));
    nView = length(Hs);
    nDim = nCluster * 4;
    
    % embeddings_s = [1, 2, 4];
    embeddings_s = [1, 2, 3, 4]; % default
    % eta_s = [1, 3, 5, 7, 9]; % default
%     eta_s = [9]; % default
    % knn_size_s = [5, 10]; % default
    diff_param_s = 0.85;%[0.05, 0.1, 0.85, 0.9, 0.95];
    knn_size_s = [5]; % default
    m_s= nCluster * [2:2:8];
    paramCell = cell(1, length(embeddings_s) * length(diff_param_s) * length(knn_size_s)*length(m_s));
    idx = 0;
    for iParam1 = 1:length(embeddings_s)
        for iParam2 = 1:length(diff_param_s)
            for iParam3 = 1:length(knn_size_s)
                for iParam4=1:length(m_s)
                    idx = idx + 1;
                    param = [];
                    param.nEmbedding = embeddings_s(iParam1);
                    param.diff_param = diff_param_s(iParam2);
                    param.knn_size = knn_size_s(iParam3);
                    param.m=m_s(iParam4);
                    paramCell{idx} = param;
                end
            end
        end
    end
    paramCell = paramCell(~cellfun(@isempty, paramCell));
    nParam = length(paramCell);
    
    nMeasure = 13;
    nRepeat = 10;
    seed = 2024;
    rng(seed);
    % Generate 50 random seeds
    random_seeds = randi([0, 1000000], 1, nRepeat);
    % Store the original state of the random number generator
    original_rng_state = rng;
    %*********************************************************************
    % CFGFLF
    %*********************************************************************
    fname2 = fullfile(dir_name, [data_name, '_CFGFLF.mat']);
    if ~exist(fname2, 'file')
        CFGFLF_result = zeros(nParam, 1, nRepeat, nMeasure);
        CFGFLF_time = zeros(nParam, 1);
        for iParam = 1:nParam
            param = paramCell{iParam};
            nEmbedding = param.nEmbedding * nCluster;
            diff_param = param.diff_param;
            knn_size = param.knn_size;
            m=param.m;
            Hs_new = cell(1, nView);
            for iKernel = 1:nView
                Hi = Hs{iKernel};
                Hs_new{iKernel} = Hi(:, 1: nEmbedding);
            end
            t1_s = tic;
            if diff_param > 0
                mu = diff_param/(1 - diff_param);
                LHs = Hs2LHs_PPR(Hs_new, mu, m, knn_size);
            else
                LHs = Hs_new;
            end
            
            for iRepeat = 1:nRepeat

                disp(['Param ', num2str(iParam), ' Repeat ', num2str(iRepeat)]);
                % Restore the original state of the random number generator
                rng(original_rng_state);
                % Set the seed for the current iteration
                rng(random_seeds(iRepeat));
                Ha = cell2mat(Hs_new);
                Ha = bsxfun(@rdivide, Ha, sqrt(sum(Ha.^2, 2)) + eps);
                label_0 = litekmeans(Ha, nCluster, 'MaxIter', 50, 'Replicates', 10);
                Y_0 = ind2vec(label_0')';
                [Iabel, Ws, alpha, beta, objHistory] = CFGFLF_fast(Hs_new, nCluster, LHs, Y_0);
                result_aio = my_eval_y(Iabel, Y);
                CFGFLF_result(iParam, 1, iRepeat, :) = result_aio';
            end
            t1 = toc(t1_s);
            CFGFLF_time(iParam) = t1/nRepeat;
        end
        a1 = sum(CFGFLF_result, 2);
        a3 = sum(a1, 3);
        a4 = reshape(a3, nParam, nMeasure);
        a4 = a4/nRepeat;
        CFGFLF_result_grid = a4;
        CFGFLF_result_summary = [max(a4, [], 1), sum(CFGFLF_time)];
        save(fname2, 'CFGFLF_result', 'CFGFLF_time', 'CFGFLF_result_summary', 'CFGFLF_result_grid');
        
        disp([data_name, ' has been completed!']);
    end
end
% profile viewer;
rmpath(data_path);
rmpath(lib_path);