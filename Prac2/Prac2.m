%Prepare dataset
x = linspace(0,5,100);
y = 2*sin(1.5* x);
x_train = 5 * rand(1,20);
x_test = 5 * rand(1,20);

y_train = f(x_train) + randn(1,20);
y_test= f(x_test) + randn(1,20);

figure;
hold on;
plot(x,y);
scatter(x_train,y_train,'rs');
hold off
cftool(x_test,y_test);

figure;
hold on;
for i = 1:9
    x1 = linspace(0,5,20);
    p = polyfit(x_train,y_train,i);
    y_fit= polyval(p,x1);
    plot(x1, y_fit)
end 
hold off;


%plot(x,y);
%hold off;
%cftool(x,y);



