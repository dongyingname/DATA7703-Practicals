%Q3
myDataQ3 = importdata('prac1_q3.dat');
mean_myDataQ3 = mean(myDataQ3);
std_myDataQ3 = std(myDataQ3);

%Q4
myDataQ4 = importdata('prac1_q4.dat');
col1 = myDataQ4(:,1);
col2 = myDataQ4(:,2);
col3 = myDataQ4(:,3);
col4 = myDataQ4(:,4);
figure;
hold on
plot(col1,col2,'bo')
plot(col3,col4,'rs')
xlabel('Input')
ylabel('Output')
title('Question 4 Input vs Output Plots')
hold off

%Q5
stdQ5 = 4;
miu = 2;
myNumbers = randn(1000,1).*stdQ5+miu;
figure;
hold on
histogram(myNumbers,30)
xlabel('Random Variable');
ylabel('Frequency');
hold off

%Q6
out([1,2,3,4,5],1)
out([1,2,3,4,5],2)
out([1,2,3,4,5,6],2)
out([1,2,3,4,5,6],3)
out([1,2,3,4,5,5],4)

out([19, 34, 59, 2, 45, 83, 20],5)

