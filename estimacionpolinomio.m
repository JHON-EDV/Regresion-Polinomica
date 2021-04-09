%Algoritmo de regresion lineal.
load('datos_regresion.mat');

% datz=data(:,2:5);
% 
% T=ceil(0.7*length(datz));
% V=T+1;
% 
% datt=datx(1:T,:); %  70% datos de evolución
% datv=datx(T+1:end,:); %30%  %datos de valiación

figure(1)
plot(x1,y1,'*')

figure(2)
plot(x1,y1r,'*')