function G = X2G_linear(X, k, blockSize)
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

for iBlock = 1:numBlocks
    % Define the indices for the current block
    blockStart = (iBlock - 1) * blockSize + 1;
    blockEnd = min(iBlock * blockSize, nSmp);
    blockSizeCurr = blockEnd - blockStart + 1;

    % Compute the kernel matrix K for the current block
    K_block = X * X(blockStart:blockEnd, :)';
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