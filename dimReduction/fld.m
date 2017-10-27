function [W_fld,Dsort,idx] = fld(data,class)
dimension = size(data,1);
labels = unique(class); % class {0,1,2}
C = length(labels);
Sw = zeros(dimension,dimension);
Sb = zeros(dimension,dimension);
mu = mean(data,2); %13*1
 
 for i = 1:C 
    Xi = data(:,find(class == labels(i)));
    n = size(Xi,2);
    mu_i = mean(Xi,2); %13*1
    
    MiM =  mu_i - mu;
    Sb = Sb + n * MiM * MiM';
    
    XMi = bsxfun(@minus, Xi, mu_i); % minus mu-i by each column
    Sw = Sw + XMi * XMi';  %13*13  
 end
 
result = Sw;
[V, D] = eig(Sb/Sw);
  
%sort the corresponding eigenvectors
[Dsort, idx] = sort(diag(D), 'descend')

W_fld=zeros(size(V));

for i=1:size(V,1)
    W_fld(:,i)=V(:,idx(i));
end
end
