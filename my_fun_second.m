function yy = my_fun_second(beta,new_train)
x1=new_train(:,1);
x2=new_train(:,2);
x3=new_train(:,3);
yy=beta(1)*(x1-beta(2)).^2+beta(3)*(x2-beta(4)).^2+beta(5)*(x3-beta(6)).^2+beta(7);
end