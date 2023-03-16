function B=average8(A)


% Create padded v matrix to incorporate Newman boundary conditions 
vv=[[A(end,end) A(end,:) A(end,1)];[A(:,end) A A(:,1)];[A(1,end) A(1,:) A(1,1)]];
vv1=vv(1:end-2,1:end-2);
vv2=vv(1:end-2,2:end-1);
vv3=vv(1:end-2,3:end);
vv4=vv(2:end-1,1:end-2);
vv5=vv(2:end-1,3:end);
vv6=vv(3:end,1:end-2);
vv7=vv(3:end,2:end-1);
vv8=vv(3:end,3:end);

B=(vv1+vv2+vv3+vv4+vv5+vv6+vv7+vv8)/8;

end