function [ D ] = Mahalanobis( X, Y, M, p )
%UNTITLED3 Summary of this function goes here
%   X = mxd, Y = nxd,
%   D = mxn - pairwise Mahalanobis distance

% Y_temp = cell(1, size(Y, 1));
% 
% for i=1:size(Y,1)
%     Y_temp{1, i} = circshift(Y,i);
% end
% 
% D_temp = cell(size(X, 1), size(Y, 1));
% for i=1:size(Y_temp)
%     D_temp{i} = X-Y_temp{i};
% end
% 
% D_temp = cell2mat(D_temp);
% D = D_temp.*M;
% D = D*D_temp';

D = (diag(X*M*X')*ones(1,size(Y,1)) + ones(size(X,1),1)*diag(Y*M*Y')' - 2*X*M*Y').^p;
end

