function [anchor_idx, anchor_score] = AnchorSelection_ALG(A, nAnchor)
%
% [1] Large-Scale Clustering With Structured Optimal Bipartite Graph
%
%
% Code is provided by Author Han Zhang
%
[nSmp, ~] = size(A);
anchor_score = zeros(nAnchor, nSmp);
anchor_idx = zeros(nAnchor, 1);
d = sum(A, 1);
anchor_score(1, :) = d;
[~, anchor_idx(1)] = max(anchor_score(1,:));
for iAnchor = 2:nAnchor
    anchor_score(iAnchor, :) = anchor_score(iAnchor - 1, :) .* (ones(1, nSmp) - anchor_score(iAnchor -1, :)) .* d;
    anchor_score(iAnchor, :) = anchor_score(iAnchor, :)/max(anchor_score(iAnchor, :));
    [~, anchor_idx(iAnchor)] = max(anchor_score(iAnchor, :));
end
end