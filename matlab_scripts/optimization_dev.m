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
x0 = [0, 1];
u = 0.2*ones(length(t), 1);

a = [1556 1219866; 1 0];
b = [-2864562; 0];
c = [0 1];
d = [0];

a1 = -159;
b1 = 689;
c1 = 1;
d1 = 0;

refIn = 0.23;

% tc1 = 0.855;
tc1 = 0.524;
atc1 = -1/tc1;
btc1 = 0.4/tc1;
ctc1 = 1;
dtc1 = 1;

tt = linspace(0, 3.5, 10000);
ut = refIn*ones(length(tt), 1);

x0t = [0; 2.1];
xt = [0, 0];

syst = ss(a, b, c, d);
syst1 = ss(a1, b1, c1, d1);
systc1 = ss(atc1, btc1, ctc1, dtc1);

tft = tf(syst);
tf1 = tf(syst1);
tftc1 = tf(systc1);
gt = tft/(1 - .01*tft);
g1 = tf1/(1 - .01*tf1);
gtc1 = (tftc1/(1-tftc1))/0.01;

ggtc1 = tf(0.4/tc1, [1, -1.4/tc1]);
htc1 = tf(0.4/tc1, [1, 1/tc1]);
testfeed1 = feedback(g1, 0.01);

close all


% out=lsim(syst, ut, tt, x0t);
% out1 = lsim(syst1, ut, tt, 2.45);
% outtc1 = lsim(systc1, ut, tt, 2.45);
% gout1 = lsim(g1, ut, tt,  [189; 2.707]);
% goutt = lsim(gt, ut, tt, 2.707);
% gouttc1 = lsim(gtc1, ut, tt, 2.45);
% 
% 
% for i = 0.1:0.1:0.9
%    hold on
%    lsim(ss(feedback(i*htc1, 1)), 0.5*ut, tt, 2.45);
% end

%% 

% yss = 0.19;
yss = 0.21;
% refIn = 0.23;
refIn=0.01;
% tc1 = 0.524;

% n=10000
% tc1 = 0.388;

% n=71400
% tc1 = 1.46;
tc1 = 2.59;
k1 = yss/refIn;

x01 = 2.4;

t1 = linspace(0,5, 10000);
u1 = 0.23*ones(length(t1), 1);

s = tf('s');
g1 = k1/(tc1*s + 1);

close all
figure
lsim(ss(g1), 0.01*ones(length(t1), 1) , t1, 1);
% lsim(ss(g1), u1, t1, 1);

q = 1;
r = 1;

kNew = lqr(ss(g1), q, r);
h1 = feedback(kNew*g1, 1);

figure
lsim(ss(h1), u1, t1, 2);
% lsim(ss(h1), u1, t1, x01);
% lsim(c2d(ss(feedback(kNew*g1, 1)), 0.001), 0.23*ones(length([0:0.001:1]), 1), [0:0.001:1], x01);



