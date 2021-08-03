function yy = my_fun_norm(beta,new_train)
x1=new_train(:,1);
x2=new_train(:,2);
x3=new_train(:,3);
yy=normpdf(x1,beta(1),1)+normpdf(x2,beta(2),1)+normpdf(x3,beta(3),1)+beta(4);
end