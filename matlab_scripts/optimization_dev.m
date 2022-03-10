% wn= input('frequency')
% zeta= input('damping factor')
% k= input('constant')
wn = 2;
zeta = .7;
k = 1;

num= [k*wn^2];
deno= [1 2*zeta*wn wn^2];
g= tf(num, deno);
t= feedback(g,1);
[A, B, C, D] = tf2ss(num, deno);
sys = ss(A, B, C, D);
% initial(sys, [1, 1]);
% USE LSIM
t = linspace(0, 5, 100);
x0 = [1, 1];
u = 0.2*ones(length(t), 1);

close all
lsim(sys, u, t, x0);
