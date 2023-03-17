%% ODE function for Lorenz system


function f = Lorenz(t,y )
f=[-10*y(1)+10*y(2);
    -y(1)*y(3)+28*y(1)-y(2);
    y(1)*y(2)-8/3*y(3)];
end