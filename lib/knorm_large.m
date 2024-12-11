function K = knorm_large(K)
% knorm
DSsym = 1./sqrt(diag(K));
K = bsxfun(@times, K, DSsym);
K = bsxfun(@times, K, DSsym');
K = .5 * K + .5 * K';
end