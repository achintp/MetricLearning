function [ output ] = inner_func(x, E)
%INNER_FUNC Sums over dissimilar sets of points
%   Detailed explanation goes here
arr = cell2mat(E);
arr2 = exp(x.*arr);

output = sum(sum(arr2));
end

