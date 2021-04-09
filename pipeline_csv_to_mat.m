%Pipe Line convert .csv to .mat 
%by jhon vargas
clc;close all;clear all;
M=readtable(['spotify_pro.csv']);
data = table2array(M);
save('spotify_pro.mat','data');

%%

load('datos_regresion.mat');
datos_regr = [x1';y1';y1r'];
writematrix(datos_regr,'datos_regresion.csv'); 