clear;
MU1 = [2 2];
SIGMA1 = [2 0; 0 1];
MU2 = [-2 -1];
SIGMA2 = [1 0; 0 1];
MU3 = [10 4];
SIGMA3 = [3 0; 0 1];


X = [mvnrnd(MU1, SIGMA1, 1000); mvnrnd(MU2, SIGMA2, 1000); mvnrnd(MU3, SIGMA3, 1000)];
S1 = X([1:1000], :);
S2 = X([1001:2000], :);
S3 = X([2001:3000], :);


scatter(X(:,1), X(:,2), 10, '.')
hold on

S = {S1, S2, S3};
D = {{S1, S3}, {S2, S3}};
E1 = cell(length(D),1);
E2 = cell(length(S),1);
T1cell = cell(length(D),1);
T2cell = cell(length(S), 1);

for k=1:length(D)
    curCell = D{k};
    E_temp = bsxfun(@minus, diag(curCell{1}*curCell{1}'), 2*curCell{1}*curCell{2}');
    E1{k} = bsxfun(@plus, E_temp, diag(curCell{2}*curCell{2}'));
    T1cell{k}=1/(length(curCell{1})*length(curCell{2}))*ones(length(curCell{1}),1);
end

for k=1:length(S)
    curCell = S{k};
    E_temp = diag(curCell*curCell');
    E2{k} = E_temp;
    T2cell{k} = 1/(numel(curCell))*ones(size(curCell, 1), 1);
end
arr = cell2mat(E1);
arr2 = cell2mat(E2);
[~, cols] = size(arr);
t1 = cell2mat(T1cell)';
t2 = ones(cols, 1);
[~, cols] = size(arr2);
t3 = cell2mat(T2cell)';
t4 = ones(cols, 1);


cvx_begin
variables sig mew
maximize log((1/length(D))*t1*exp(sig*arr)*t2 + exp(mew))
subject to
    log((1/size(S))*t3*exp(sig*arr2)*t4 + exp(mew)) <= 0
cvx_end





