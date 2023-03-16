
N=80000;
tau=17;

t=zeros(N,1);
x=zeros(N,1);
x(1)=0.5; t(1)=0; 
a=0.2;b=0.1;h=0.5;
for k=1:N-1
  t(k+1)=t(k)+h; 
  if t(k)<tau
      k1=-b*x(k); 
      k2=-b*(x(k)+h*k1/2); 
      k3=-b*(x(k)+k2*h/2); 
      k4=-b*(x(k)+k3*h);
      x(k+1)=x(k)+(k1+2*k2+2*k3+k4)*h/6; 
  else
      n=floor((t(k)-tau)/h+1);
      k1=Df(x(n))-b*x(k); 
      k2=Df(x(n))-b*(x(k)+k1*h/2); 
      k3=Df(x(n))-b*(x(k)+k2*h/2); 
      k4=Df(x(n))-b*(x(k)+k3*h); 
      x(k+1)=x(k)+(k1+2*k2+2*k3+k4)*h/6; 
  end 
end
data=x;
save('MGdata.mat','data')