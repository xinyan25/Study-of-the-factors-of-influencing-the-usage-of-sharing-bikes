T = readtable('london_merged.csv');
train=T(1:8715,1:10);
test=T(8716:17414,1:10);
X = [train{:,'timestamp'} train{:,'t1'} train{:,'t2'} train{:,'hum'} train{:,'wind_speed'} train{:,'weather_code'} train{:,'is_holiday'} train{:,'is_weekend'} train{:,'season'}];
test_X = [test{:,'timestamp'} test{:,'t1'} test{:,'t2'} test{:,'hum'} test{:,'wind_speed'} test{:,'weather_code'} test{:,'is_holiday'} test{:,'is_weekend'} test{:,'season'}];
y = [train{:,'cnt'}];
%% Linear Regression
mdl = fitlm(X,y);
plot(mdl);
newMdl1 = removeTerms(mdl,"x6");
newMdl2 = step(newMdl1,'NSteps',30);
plot(newMdl2)
ypred = predict(newMdl2,test_X);
errs = ypred - test.cnt;
%histogram(errs)
%title("Histogram of residuals - test data")
%% Using nonlinfit - second order
new_train=[train{:,'timestamp'} train{:,'t1'} train{:,'hum'}];
new_test=[test{:,'timestamp'} test{:,'t1'} test{:,'hum'}];
beta0 = [10,10,10,10,10,10,10]';
[beta,r,j] = nlinfit(new_train,y,@my_fun_second,beta0);
beta
xx1=new_test(:,1);
xx2=new_test(:,2);
xx3=new_test(:,3);
yy1=beta(1)*(xx1-beta(2)).^2+beta(3)*(xx2-beta(4)).^2+beta(4)*(xx3-beta(6)).^2+beta(7);
h = histogram(yy1-test.cnt);
title("Histogram of residuals - test data - second order regression");
%% Using nonlinfit - normal distribution
new_train=[train{:,'timestamp'} train{:,'t1'} train{:,'hum'}];
new_test=[test{:,'timestamp'} test{:,'t1'} test{:,'hum'}];
beta0 = [10,10,10,10]';
[beta,r,j] = nlinfit(new_train,y,@my_fun_norm,beta0);
beta

