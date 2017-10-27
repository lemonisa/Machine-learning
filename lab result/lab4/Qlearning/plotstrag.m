%% -*- mode: Octave;-*-

function plotstrag(mapno, V, st)
  subplot(2, 1, 1);
  gwdraw();

  for i = 1:size(st, 1)
    for j = 1:size(st, 2)
      gwplotarrow([i;j], st(i,j));
    end
  end
  
  subplot(2, 1, 2);
  imagesc(V); title('V function'); colorbar()
end

