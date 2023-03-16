function dydt = ddefun(t,y,Z)
  dydt = [-0.2*y+50*(1/(1+(Z(1,1)/38)^6))*(1-1/(1+Z(1,2)/38))];
end