%% Data generation

y0=rand(1,3);
[t,y] = ode45('Lorenz',[0.01:0.01:400],y0);
data=y(10000:end,:)';

