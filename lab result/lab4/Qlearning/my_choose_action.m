%% -*- mode: Octave;-*-

function action = my_choose_action( Q,state_x,state_y,...
                                                 actions, eps )

  %% Check the optimal action
  Q_values = Q(state_x, state_y, :);
  [Q_opt index_opt] = max(Q_values);

  prob_a(1:length(actions)) = 1/length(actions);
  rand_action = sample(actions, prob_a);
  pos_actions = [index_opt rand_action];
  action = pos_actions(sample(pos_actions, [1-eps eps]));
end

