%% -*- mode: Octave;-*-
function [Q, V, stragegy] = drive(mapno, alpha, gamma, maxloop)
  %% Start from current world
  gwinit(mapno);
  world = gwstate();
  Q = rand( world.xsize, world.ysize, 4 ) * (0.3) - 0.4;
  Q(1,:,2) = -Inf;
  Q(world.xsize,:,1) = -Inf;
  Q(:,1,4) = -Inf;
  Q(:,world.ysize,3) = -Inf;

  start_eps = 0.8
  eps = start_eps;
  died = 1;

  oldQ = Q;
  oldQ(isinf(Q)) = 0;
  diffQ = [];

  figure('DoubleBuffer', 'on');
  drawnow();

  allactions = [1 2 3 4];
  while died <= maxloop
    world = gwstate();
    pos = world.pos;

    while world.isterminal == 0
      act = my_choose_action(Q, pos(1), pos(2), allactions, eps);
      newworld = gwaction(act);
      newpos = newworld.pos; %% Less reference for speed
      if newworld.isvalid == 1
        Q(pos(1), pos(2), act) = (1 - alpha) * Q(pos(1), pos(2), act) +...
                                 alpha * (newworld.feedback + gamma * max(Q(newpos(1), newpos(2), :)));
      end
      world = newworld;
      pos = world.pos;
    end
    Q(pos(1), pos(2), :) = 0;

    if mod(died, 50) == 0
      P = Q;
      P(isinf(P)) = 0;
      absdiff = abs(P - oldQ);
      diffQ((died-49):died) = sum(absdiff(:));

      subplot(5,2,[1 2]);
      semilogy(diffQ);
      title(sprintf('Died: %d times; diffQ = %d; eps = %f', died, diffQ(died), eps));
      subplot(5,2, [3]); imagesc(Q(:,:,1)); title('1');
      subplot(5,2, [4]); imagesc(Q(:,:,2)); title('2');
      subplot(5,2, [5]); imagesc(Q(:,:,3)); title('3');
      subplot(5,2, [6]); imagesc(Q(:,:,4)); title('4');

      [useless, stragegy] = max(Q, [], 3);
      subplot(5,2, [7 8 9 10]); imagesc(stragegy); title('4'); colorbar();

      drawnow();
      oldQ = P;
      eps = max(0.2, start_eps + (0.2 - start_eps)/(0.8 * (maxloop - 1)) * died);
    end

    died = died +1;
    gwinit(mapno);
  end

  [V, stragegy] = max(Q, [], 3);
end
