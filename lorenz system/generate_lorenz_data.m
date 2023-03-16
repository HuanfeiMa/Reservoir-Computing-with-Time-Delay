y0=rand(1,3);
[t,y] = ode45('Lorenz',[0.01:0.01:400],y0);
data=y(10000:end,:)';
save('lorenzdata1.mat','data')
% plot(y(:,1),y(:,3))
% box on;grid on;