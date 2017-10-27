%% map 1
[Q, V, st] = drive(1, 0.1, 0.9, 1500)
figure(); plotstrag(1, V, st)
%% map 2
[Q, V, st] = drive(2, 0.3, 0.9, 2500)
figure(); plotstrag(2, V, st)
%% map 3
[Q, V, st] = drive(3, 0.5, 0.9, 3000)
figure(); plotstrag(3, V, st);
%% map 4
[Q, V, st] = drive(4, 0.05, 0.95, 5000)
figure(); plotstrag(4, V, st);
