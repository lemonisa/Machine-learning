%% -*- mode: Octave;-*-
function steps = seeoptimal(mapno, stragegy)
  gwinit(mapno);

  world = gwstate();

  gwdraw();
  steps = 1
  while world.isterminal == 0
    act = stragegy(world.pos(1), world.pos(2));
    gwplotarrow(world.pos, act);
    gwaction(act);
    world = gwstate();
    steps = steps + 1;
    if (steps > 100)
      break;
  end
end
