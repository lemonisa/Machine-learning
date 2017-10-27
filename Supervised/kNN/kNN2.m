function [ labelsOut, densityOut ] = kNN2(X, k, Xt, Lt)
  %% KNN Your implementation of the kNN algorithm
  %%   Inputs:
  %%               X  - Features to be classified
  %%               k  - Number of neighbors
  %%               Xt - Training features
  %%               Lt - Correct labels of each feature vector [1 2 ...]'
  %%
  %%   Output:
  %%               LabelsOut = Vector with the classified labels

  labelsOut  = zeros(size(X,2),1);
  classes = unique(Lt);
  numClasses = length(classes);

  vptree = make_vp_tree(Xt, 1:length(X));

  % NNs: matrix of nearest neighbours
  NNs = cell2mat(arrayfun( @(i) kNNwalk(vptree, Xt, X(:,i), k), 1:length(X), 'UniformOutput', false ));

  densityOut = cell2mat( arrayfun( @(i) mean(Lt(NNs) == classes(i))', 1:numClasses, 'UniformOutput', false))';
  [m, idx] = max(densityOut, [], 1);
  labelsOut = classes(idx);
end

function [d] = mat_eudist(x, y)
  d = sum(x'.^2,2)*ones(1,rows(y')) + ones(rows(x'),1)*sum( y'.^2, 2 )' - 2.*x'*y;
end

function [tree_out] = make_vp_tree(X, scope)
  len = length(scope);
  Xs = X(:,scope);

  if len == 0
    tree_out = NaN;
    return;
  end

  vantage_idx = NaN;
  if len > 40
    sample_size = ceil(0.15 * len) * 2 + 1;
    candidates = randi(len, sample_size, 1);
    references = randi(len, sample_size, 1);
    cr_dist_square = mat_eudist(X(:,scope(candidates)), X(:,scope(references)));
    cr_dist = sqrt(cr_dist_square);
    cr_med = median(cr_dist);
    spreads = sum(cr_dist_square, 2) + (sample_size .* cr_med.^2)' - 2 .* sum(cr_dist * diag(cr_med), 2);
    [m, idx] = min(spreads);
    vantage_idx = idx;
  else
    vantage_idx = randi(len, 1);
  end

  if len > 1
    vantage = scope(vantage_idx);
    non_vantage = scope(1:len ~= vantage_idx);
    our_distances = mat_eudist(X(:,vantage), X(:,non_vantage));
    med = median(our_distances);

    left = non_vantage(our_distances < med);
    right = non_vantage(our_distances >= med);

    tree_out.vantage = vantage;
    tree_out.left = make_vp_tree( X, left );
    tree_out.right = make_vp_tree( X, right );
    tree_out.median = med;
    tree_out.upper = max(our_distances);
    tree_out.lower = min(our_distances);
  else
    tree_out.vantage = scope(1);
    tree_out.left = make_vp_tree( X, [] );
    tree_out.right = make_vp_tree( X, [] );
    tree_out.median = NaN;
    tree_out.upper = NaN;
    tree_out.lower = NaN;
  end
end

bests = NaN;
taus = [];
worst_idx = NaN;

function [final_out] = kNNwalk(tree, X, query, K)
  global bests taus worst_idx;

  bests = zeros(K,1);
  taus = repmat(Inf, K, 1);
  worst_idx = 1;
  recwalk(tree, X, query, K);

  [taus, idx] = sort(taus);
  bests = bests(idx);
  final_out = bests;
  return;
end

function recwalk(node, X, query, K)
  global bests taus worst_idx;

  if ! isstruct(node)
    return;
  end

  d = mat_eudist(query, X(:,node.vantage));
  if d < taus(worst_idx)
    taus(worst_idx) = d;
    bests(worst_idx) = node.vantage;
    [m, idx] = max(taus);
    worst_idx = idx;
  end

  worst_tau = taus(worst_idx);
  if(! isnan(node.median))
    if (d < node.median)
      if (node.lower - worst_tau < d && d < node.median + worst_tau)
        recwalk(node.left, X, query, K);
      end
      if (node.median - worst_tau < d && d < node.upper + worst_tau)
        recwalk(node.right, X, query, K);
      end
    else
      if (node.median - worst_tau < d && d < node.upper + worst_tau)
        recwalk(node.right, X, query, K);
      end
      if (node.lower - worst_tau < d && d < node.median + worst_tau)
        recwalk(node.left, X, query, K);
      end
    end
  end
end
