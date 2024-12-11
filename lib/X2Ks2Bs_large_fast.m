function [Bs, t1s, t2s] = X2Ks2Bs_large_fast(X, nAnchor, k, blockSize)
%
% Compute Embedding from Large Kernel
%
% The memory is O(n^2)
% The time is O(n^2 k)
%
% Output
%         Bs, cell(1, 12)
%         t1s, time for kernel computation
%         t2s, time for embedding
%
[nSmp, nFea] = size(X);

nKernel = 12;
t1s = zeros(1, nKernel);
t2s = zeros(1, nKernel);
%*********************************************
% Linear Kernel
%*********************************************
t1_s = tic;
[Gnorm, t_k] = X2G_linear(X, k, blockSize);
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
Kb = X(anchor_idx, :) * X';
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
rowIdx = repmat((1:nSmp)', k, 1);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{1} = B;
t1s(1) = t_k;
t2s(1) = toc(t1_s) - t_k;


%*********************************************
% PolyPlus Kernel
%*********************************************
t1_s = tic;
degree = 2;
[Gnorm, t_k] = X2G_PolyPlus(X, k, blockSize, degree);
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
Kb = (X(anchor_idx, :) * X' + 1).^degree;
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
rowIdx = repmat((1:nSmp)', k, 1);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{2} = B;
t1s(2) = t_k;
t2s(2) = toc(t1_s) - t_k;

degree = 4;
t1_s = tic;
[Gnorm, t_k] = X2G_PolyPlus(X, k, blockSize, degree);
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
Kb = (X(anchor_idx, :) * X' + 1).^degree;
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
rowIdx = repmat((1:nSmp)', k, 1);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{3} = B;
t1s(3) = t_k;
t2s(3) = toc(t1_s) - t_k;

%*********************************************
% Polynomial Kernel
%*********************************************
t1_s = tic;
degree = 2;
[Gnorm, t_k] = X2G_Polynomial(X, k, blockSize, degree);
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
Kb = (X(anchor_idx, :) * X').^degree;
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
rowIdx = repmat((1:nSmp)', k, 1);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{4} = B;
t1s(4) = t_k;
t2s(4) = toc(t1_s) - t_k;


t1_s = tic;
degree = 4;
[Gnorm, t_k] = X2G_Polynomial(X, k, blockSize, degree);
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
Kb = (X(anchor_idx, :) * X').^degree;
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
rowIdx = repmat((1:nSmp)', k, 1);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{5} = B;
t1s(5) = t_k;
t2s(5) = toc(t1_s) - t_k;


%*********************************************
% Gaussian Kernel
%*********************************************
% Gaussian 2.^-3
t1_s = tic;
degree = 2.^-3;
[Gnorm, t_k] = X2G_Gaussian(X, k, blockSize, degree);
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
D = EuDist2(X(anchor_idx, :), X, 0);
s = mean(mean(D));
Kb = exp(-D / (2 * degree * s) );
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
rowIdx = repmat((1:nSmp)', k, 1);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{6} = B;
t1s(6) = t_k;
t2s(6) = toc(t1_s) - t_k;


% Gaussian 2.^-2
t1_s = tic;
degree = 2.^-2;
[Gnorm, t_k] = X2G_Gaussian(X, k, blockSize, degree);
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
D = EuDist2(X(anchor_idx, :), X, 0);
s = mean(mean(D));
Kb = exp(-D / (2 * degree * s) );
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
rowIdx = repmat((1:nSmp)', k, 1);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{7} = B;
t1s(7) = t_k;
t2s(7) = toc(t1_s) - t_k;


% Gaussian 2.^-1
t1_s = tic;
degree = 2.^-1;
[Gnorm, t_k] = X2G_Gaussian(X, k, blockSize, degree);
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
D = EuDist2(X(anchor_idx, :), X, 0);
s = mean(mean(D));
Kb = exp(-D / (2 * degree * s) );
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
rowIdx = repmat((1:nSmp)', k, 1);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{8} = B;
t1s(8) = t_k;
t2s(8) = toc(t1_s) - t_k;


% Gaussian 2.^0
t1_s = tic;
degree = 2.^0;
[Gnorm, t_k] = X2G_Gaussian(X, k, blockSize, degree);
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
D = EuDist2(X(anchor_idx, :), X, 0);
s = mean(mean(D));
Kb = exp(-D / (2 * degree * s) );
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
rowIdx = repmat((1:nSmp)', k, 1);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{9} = B;
t1s(9) = t_k;
t2s(9) = toc(t1_s) - t_k;


% Gaussian 2.^1
t1_s = tic;
degree = 2.^1;
[Gnorm, t_k] = X2G_Gaussian(X, k, blockSize, degree);
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
D = EuDist2(X(anchor_idx, :), X, 0);
s = mean(mean(D));
Kb = exp(-D / (2 * degree * s) );
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
rowIdx = repmat((1:nSmp)', k, 1);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{10} = B;
t1s(10) = t_k;
t2s(10) = toc(t1_s) - t_k;

% Gaussian 2.^2
t1_s = tic;
degree = 2.^2;
[Gnorm, t_k] = X2G_Gaussian(X, k, blockSize, degree);
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
D = EuDist2(X(anchor_idx, :), X, 0);
s = mean(mean(D));
Kb = exp(-D / (2 * degree * s) );
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
rowIdx = repmat((1:nSmp)', k, 1);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{11} = B;
t1s(11) = t_k;
t2s(11) = toc(t1_s) - t_k;

% Gaussian 2.^3
t1_s = tic;
degree = 2.^3;
[Gnorm, t_k] = X2G_Gaussian(X, k, blockSize, degree);
anchor_idx = AnchorSelection_ALG(Gnorm, nAnchor);
D = EuDist2(X(anchor_idx, :), X, 0);
s = mean(mean(D));
Kb = exp(-D / (2 * degree * s) );
[Kb2, Idx] = sort(Kb, 1, 'descend');
Idx = Idx(1:k, :)';
colIdx = Idx(:);
Kb2 = Kb2(1:k, :)';
val = Kb2(:);
rowIdx = repmat((1:nSmp)', k, 1);
B = sparse(rowIdx, colIdx, val, nSmp, nAnchor, nSmp * k);
B = bsxfun(@rdivide, B, max(sum(B, 2), eps));
B = bsxfun(@times, B, max(sum(B, 1), eps).^(-.5));
Bs{12} = B;
t1s(12) = t_k;
t2s(12) = toc(t1_s) - t_k;
end

function [G, t_k] = X2G_linear(X, k, blockSize)
nSmp = size(X, 1);

if ~exist('blockSize', 'var')
    blockSize = 1000;
end
blockSize = min(blockSize, nSmp);

numBlocks = ceil(nSmp / blockSize);

% Initialize Gnorm
rowIdx_cell = cell(numBlocks, 1);
colIdx_cell = cell(numBlocks, 1);
val_cell = cell(numBlocks, 1);

t_k = 0;
for iBlock = 1:numBlocks
    % Define the indices for the current block
    blockStart = (iBlock - 1) * blockSize + 1;
    blockEnd = min(iBlock * blockSize, nSmp);
    blockSizeCurr = blockEnd - blockStart + 1;

    % Compute the kernel matrix K for the current block
    t1 = tic;
    K_block = X * X(blockStart:blockEnd, :)';
    t_k = t_k + toc(t1);
    K_block(blockStart:blockEnd, :)  = K_block(blockStart:blockEnd, :) - 10^8 * eye(blockSizeCurr); % Subtract diagonal with a scalar efficiently

    % Sort the kernel matrix along columns for the current block
    [K_block_value, Idx] = sort(K_block, 1, 'descend');
    Idx = Idx(1:k, :);  % k * blockSizeCurr
    K_block_value = K_block_value(1:k, :);
    
    rowIdx_cell{iBlock} = Idx(:);
    colIdx2 = repmat((1:blockSizeCurr), k, 1);
    colIdx_cell{iBlock} = colIdx2(:) + blockStart - 1; % Adjust column indices
    val_cell{iBlock} = K_block_value(:);    
end
rowIdx = cell2mat(rowIdx_cell);
colIdx = cell2mat(colIdx_cell);
val = cell2mat(val_cell);
G = sparse(rowIdx, colIdx, val, nSmp, nSmp, nSmp * k);
G = .5 * G + .5 * G';
DSsym = 1 ./ sqrt(max(sum(G, 1), eps));
G = bsxfun(@times, G, DSsym);
G = bsxfun(@times, G, DSsym');
G = .5 * G + .5 * G';
end


function [G, t_k] = X2G_PolyPlus(X, k, blockSize, degree)
nSmp = size(X, 1);

if ~exist('blockSize', 'var')
    blockSize = 1000;
end
blockSize = min(blockSize, nSmp);

numBlocks = ceil(nSmp / blockSize);

% Initialize Gnorm
rowIdx_cell = cell(numBlocks, 1);
colIdx_cell = cell(numBlocks, 1);
val_cell = cell(numBlocks, 1);

t_k = 0;
for iBlock = 1:numBlocks
    % Define the indices for the current block
    blockStart = (iBlock - 1) * blockSize + 1;
    blockEnd = min(iBlock * blockSize, nSmp);
    blockSizeCurr = blockEnd - blockStart + 1;

    % Compute the kernel matrix K for the current block
    t1 = tic;
    D = EuDist2(X, X(blockStart:blockEnd, :), 0);
    s = mean(mean(D));
    K_block = exp(-D / (2 * degree * s) );
    t_k = t_k + toc(t1);
    K_block(blockStart:blockEnd, :)  = K_block(blockStart:blockEnd, :) - 10^8 * eye(blockSizeCurr); % Subtract diagonal with a scalar efficiently

    % Sort the kernel matrix along columns for the current block
    [K_block_value, Idx] = sort(K_block, 1, 'descend');
    Idx = Idx(1:k, :);  % k * blockSizeCurr
    K_block_value = K_block_value(1:k, :);
    
    rowIdx_cell{iBlock} = Idx(:);
    colIdx2 = repmat((1:blockSizeCurr), k, 1);
    colIdx_cell{iBlock} = colIdx2(:) + blockStart - 1; % Adjust column indices
    val_cell{iBlock} = K_block_value(:);    
end
rowIdx = cell2mat(rowIdx_cell);
colIdx = cell2mat(colIdx_cell);
val = cell2mat(val_cell);
G = sparse(rowIdx, colIdx, val, nSmp, nSmp, nSmp * k);
G = .5 * G + .5 * G';
DSsym = 1 ./ sqrt(max(sum(G, 1), eps));
G = bsxfun(@times, G, DSsym);
G = bsxfun(@times, G, DSsym');
G = .5 * G + .5 * G';
end


function [G, t_k] = X2G_Polynomial(X, k, blockSize, degree)
nSmp = size(X, 1);

if ~exist('blockSize', 'var')
    blockSize = 1000;
end
blockSize = min(blockSize, nSmp);

numBlocks = ceil(nSmp / blockSize);

% Initialize Gnorm
rowIdx_cell = cell(numBlocks, 1);
colIdx_cell = cell(numBlocks, 1);
val_cell = cell(numBlocks, 1);

t_k = 0;
for iBlock = 1:numBlocks
    % Define the indices for the current block
    blockStart = (iBlock - 1) * blockSize + 1;
    blockEnd = min(iBlock * blockSize, nSmp);
    blockSizeCurr = blockEnd - blockStart + 1;

    % Compute the kernel matrix K for the current block
    t1 = tic;
    K_block = (X * X(blockStart:blockEnd, :)').^degree;
    t_k = t_k + toc(t1);
    K_block(blockStart:blockEnd, :)  = K_block(blockStart:blockEnd, :) - 10^8 * eye(blockSizeCurr); % Subtract diagonal with a scalar efficiently

    % Sort the kernel matrix along columns for the current block
    [K_block_value, Idx] = sort(K_block, 1, 'descend');
    Idx = Idx(1:k, :);  % k * blockSizeCurr
    K_block_value = K_block_value(1:k, :);
    
    rowIdx_cell{iBlock} = Idx(:);
    colIdx2 = repmat((1:blockSizeCurr), k, 1);
    colIdx_cell{iBlock} = colIdx2(:) + blockStart - 1; % Adjust column indices
    val_cell{iBlock} = K_block_value(:);    
end
rowIdx = cell2mat(rowIdx_cell);
colIdx = cell2mat(colIdx_cell);
val = cell2mat(val_cell);
G = sparse(rowIdx, colIdx, val, nSmp, nSmp, nSmp * k);
G = .5 * G + .5 * G';
DSsym = 1 ./ sqrt(max(sum(G, 1), eps));
G = bsxfun(@times, G, DSsym);
G = bsxfun(@times, G, DSsym');
G = .5 * G + .5 * G';
end



function [G, t_k] = X2G_Gaussian(X, k, blockSize, degree)
nSmp = size(X, 1);

if ~exist('blockSize', 'var')
    blockSize = 1000;
end
blockSize = min(blockSize, nSmp);

numBlocks = ceil(nSmp / blockSize);

% Initialize Gnorm
rowIdx_cell = cell(numBlocks, 1);
colIdx_cell = cell(numBlocks, 1);
val_cell = cell(numBlocks, 1);

t_k = 0;
for iBlock = 1:numBlocks
    % Define the indices for the current block
    blockStart = (iBlock - 1) * blockSize + 1;
    blockEnd = min(iBlock * blockSize, nSmp);
    blockSizeCurr = blockEnd - blockStart + 1;

    % Compute the kernel matrix K for the current block
    t1 = tic;
    K_block = (X * X(blockStart:blockEnd, :)').^degree;
    t_k = t_k + toc(t1);
    K_block(blockStart:blockEnd, :)  = K_block(blockStart:blockEnd, :) - 10^8 * eye(blockSizeCurr); % Subtract diagonal with a scalar efficiently

    % Sort the kernel matrix along columns for the current block
    [K_block_value, Idx] = sort(K_block, 1, 'descend');
    Idx = Idx(1:k, :);  % k * blockSizeCurr
    K_block_value = K_block_value(1:k, :);
    
    rowIdx_cell{iBlock} = Idx(:);
    colIdx2 = repmat((1:blockSizeCurr), k, 1);
    colIdx_cell{iBlock} = colIdx2(:) + blockStart - 1; % Adjust column indices
    val_cell{iBlock} = K_block_value(:);    
end
rowIdx = cell2mat(rowIdx_cell);
colIdx = cell2mat(colIdx_cell);
val = cell2mat(val_cell);
G = sparse(rowIdx, colIdx, val, nSmp, nSmp, nSmp * k);
G = .5 * G + .5 * G';
DSsym = 1 ./ sqrt(max(sum(G, 1), eps));
G = bsxfun(@times, G, DSsym);
G = bsxfun(@times, G, DSsym');
G = .5 * G + .5 * G';
end