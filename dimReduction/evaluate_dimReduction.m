%% Load data
load countrydata
data = double(countrydata); % 13*105
class = countryclass; %105*1

%% covariance matrix
% number of samples 105
n = size(data,2);

mu = mean(data,2)'; %1*13
mean_subtract = data' - mu; %105*13
covD = (mean_subtract.' * mean_subtract) / (n-1) %13*13
%cov(data');
image(covD);

%% correlation matrix

% convert cov to corr
% standard deviations of each variable
var_sqrt = sqrt(diag(covD));
Inv = (1 ./ var_sqrt); %13*1
corrD = Inv' .* (Inv .* covD)
%corr(data');
figure;
image(corrD);
imagesc(corrD);

%% run PCA
% normalize each variable
%inverse_sum = 1./sum(data,2); 
%dataNorm = inverse_sum .* data;
[K,N] = size(data);
  means = sum(data,2) / N;
  centered = data - means * ones(1,N);
  vars = sum(centered.^2, 2) / (N-1);
  dataNorm = centered ./ (sqrt(vars) * ones(1,N));

[W_pca,Dsort,idx] = pca(dataNorm);

%% first two principal components
x = W_pca(:,1)' * dataNorm(:,:);
y = W_pca(:,2)' * dataNorm(:,:);

% Colourize it according to the given classification.
% find the indexs of each class {0,1,2}
x0 = x(find(class==0));
x1 = x(find(class==1));
x2 = x(find(class==2));

y0 = y(find(class==0));
y1 = y(find(class==1));
y2 = y(find(class==2));
% plot the data
figure;
plot(y0, x0, 'bo', 'markersize', 10, 'linewidth', 3);hold on;
plot(y1, x1, 'ro', 'markersize', 10, 'linewidth', 3); 
plot(y2, x2, 'go', 'markersize', 10, 'linewidth', 3);
title('PCA with first two principal components')
x41
%% run FLD
[W_fld,Dsort,idx] = fld(dataNorm, class);

%% first two principal components
a = W_fld(:,1)' * dataNorm(:,:);
b = W_fld(:,2)' * dataNorm(:,:);

% Colourize it according to the given classification.
% only consider the industrialized and the developing countries
% find the indexs of each class {0,2}
a0 = a(find(class==0));
a1 = a(find(class==1));
a2 = a(find(class==2));

b0 = b(find(class==0));
b1 = b(find(class==1));
b2 = b(find(class==2));
% plot the data
figure;
plot(b0, a0, 'bo', 'markersize', 10, 'linewidth', 3);hold on;
plot(a1,b1,  'ro', 'markersize', 10, 'linewidth', 3); 
plot(b2, a2, 'go', 'markersize', 10, 'linewidth', 3);
title('FLD with first two principal components')

