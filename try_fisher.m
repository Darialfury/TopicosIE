close all
clear all
load fisheriris


set = 3;
a = 2;
b = 3;

if(a==1)
   type1 = 'r*';
else
   type1 = 'go';
end

if(b==2)
   type2 = 'go';
else
   type2 = 'b+';
end

fraction = 0.6;

if(set==1)
class1 = [meas(((a-1)*50+1):(a*50),1) meas(((a-1)*50+1):(a*50),2)];
len1 = size(class1,1);
[ind1 ind2] = crossvalind('HoldOut',len1,fraction);
class1_train = class1(ind2,:);
class1_test = class1(ind1,:);
plot(class1(:,1),class1(:,2),type1)
labels = meshgrid(1,1:size(class1_test,1));

hold on
class2 = [meas(((b-1)*50+1):(b*50),1) meas(((b-1)*50+1):(b*50),2)];
class2_train = class2(ind2,:);
class2_test = class2(ind1,:);
plot(class2(:,1),class2(:,2),type2)
xlabel('feature1')
ylabel('feature2')
labels = [labels; meshgrid(2,1:size(class2_test,1))];

%% parameters
size_samples = size(class1_train,1);
size_test = size(class1_test,1) + size(class2_test,1);
m1 =  mean(class1);
m2 = mean(class2);
s1 = size_samples*(cov(class1_train));
s2 = size_samples*(cov(class2_train));
sw_1 = inv(s2 + s1);

w = sw_1*((m1 - m2)');

%% draw projections
data_test = [class1_test; class2_test];
vector_1d = (w')*data_test';
projections = [vector_1d' vector_1d'].*(meshgrid(w',1:size_test));

projected_test= (meshgrid(w',1:size_test)).*data_test;
plot(projections(:,1),projections(:,2),'k.')
plot([0,w(1,1)],[0,w(2,1)])

m_1 = meshgrid(((w')*(m1'))*(w),1:size_test);
m_2 = meshgrid(((w')*(m2'))*(w),1:size_test);

norms1 = sqrt(sum((m_1 - projections).^2,2));
norms2 = sqrt(sum((m_2 - projections).^2,2));

[a dec] = min([norms1 norms2],[],2);
figure()
cm = confMatrix(labels, dec, max(labels));
accuracy = ((sum(diag(cm)))/(size_test))*100;
fprintf('Experiment accuracy: %f \n', accuracy);
confMatrixShow(cm)
elseif (set==2)
class1 = [meas(((a-1)*50+1):(a*50),1) meas(((a-1)*50+1):(a*50),3)];
len1 = size(class1,1);
[ind1 ind2] = crossvalind('HoldOut',len1,fraction);
class1_train = class1(ind2,:);
class1_test = class1(ind1,:);
plot(class1(:,1),class1(:,2),type1)
labels = meshgrid(1,1:size(class1_test,1));

hold on
class2 = [meas(((b-1)*50+1):(b*50),1) meas(((b-1)*50+1):(b*50),3)];
class2_train = class2(ind2,:);
class2_test = class2(ind1,:);
plot(class2(:,1),class2(:,2),type2)
xlabel('feature1')
ylabel('feature2')
labels = [labels; meshgrid(2,1:size(class2_test,1))];

%% parameters
size_samples = size(class1_train,1);
size_test = size(class1_test,1) + size(class2_test,1);
m1 =  mean(class1);
m2 = mean(class2);
s1 = size_samples*(cov(class1_train));
s2 = size_samples*(cov(class2_train));
sw_1 = inv(s2 + s1);

w = sw_1*((m1 - m2)');

%% draw projections
data_test = [class1_test; class2_test];
vector_1d = (w')*data_test';
projections = [vector_1d' vector_1d'].*(meshgrid(w',1:size_test));

projected_test= (meshgrid(w',1:size_test)).*data_test;
plot(projections(:,1),projections(:,2),'k.')
plot([0,w(1,1)],[0,w(2,1)])

m_1 = meshgrid(((w')*(m1'))*(w),1:size_test);
m_2 = meshgrid(((w')*(m2'))*(w),1:size_test);

norms1 = sqrt(sum((m_1 - projections).^2,2));
norms2 = sqrt(sum((m_2 - projections).^2,2));

[a dec] = min([norms1 norms2],[],2);
figure()
cm = confMatrix(labels, dec, max(labels));
confMatrixShow(cm)
accuracy = ((sum(diag(cm)))/(size_test))*100;
fprintf('Experiment accuracy: %f \n', accuracy);
elseif (set==3)    
class1 = [meas(((a-1)*50+1):(a*50),2) meas(((a-1)*50+1):(a*50),4)];
len1 = size(class1,1);
[ind1 ind2] = crossvalind('HoldOut',len1,fraction);
class1_train = class1(ind2,:);
class1_test = class1(ind1,:);
plot(class1(:,1),class1(:,2),type1)
labels = meshgrid(1,1:size(class1_test,1));

hold on
class2 = [meas(((b-1)*50+1):(b*50),2) meas(((b-1)*50+1):(b*50),4)];
class2_train = class2(ind2,:);
class2_test = class2(ind1,:);
plot(class2(:,1),class2(:,2),type2)
xlabel('feature1')
ylabel('feature2')
labels = [labels; meshgrid(2,1:size(class2_test,1))];

%% parameters
size_samples = size(class1_train,1);
size_test = size(class1_test,1) + size(class2_test,1);
m1 =  mean(class1);
m2 = mean(class2);
s1 = size_samples*(cov(class1_train));
s2 = size_samples*(cov(class2_train));
sw_1 = inv(s2 + s1);

w = sw_1*((m1 - m2)');

%% draw projections
data_test = [class1_test; class2_test];
vector_1d = (w')*data_test';
projections = [vector_1d' vector_1d'].*(meshgrid(w',1:size_test));

projected_test= (meshgrid(w',1:size_test)).*data_test;
plot(projections(:,1),projections(:,2),'k.')
plot([0,w(1,1)],[0,w(2,1)])

m_1 = meshgrid(((w')*(m1'))*(w),1:size_test);
m_2 = meshgrid(((w')*(m2'))*(w),1:size_test);

norms1 = sqrt(sum((m_1 - projections).^2,2));
norms2 = sqrt(sum((m_2 - projections).^2,2));

[a dec] = min([norms1 norms2],[],2);
figure()
cm = confMatrix(labels, dec, max(labels));
confMatrixShow(cm)
accuracy = ((sum(diag(cm)))/(size_test))*100;
fprintf('Experiment accuracy: %f \n', accuracy);
end



