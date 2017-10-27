function [W_pca,Dsort,idx] = pca(data)

% covariance matrix
% number of samples 105
n = size(data,2);

mu = mean(data,2)'; %1*13
mean_subtract = data' - mu; %105*13
c = (mean_subtract.' * mean_subtract) / (n-1); %13*13
%cov(data')

%calculate eigenvectors of c  
% V - eigen vectors(every column is an eigenvector)
% D - eigen values(matrix where every element in the diagonal is an eigenvalue)
[V,D] = eig(c);
%sort the eigenvalues in descendent order
D=diag(D);
[Dsort,idx]=sort(D,'descend') %13*1

%sort the corresponding eigenvectors
W_pca=zeros(size(V));
for i=1:size(V,1)
    W_pca(:,i)=V(:,idx(i));
    %13*13
end

%plot the proportion of variance explained by the eigen values to decide how many principal components choose
plot(Dsort/sum(Dsort)*100)
hold on
xlabel('Number of component')
ylabel('Percentage of variance explained')
hold off

%calculate principal component scores
%y=data'*Vsort

%correlate principal components with every variable (rows are variables and
%columns are number of component)
%varcorr=corr(data',y);
end




