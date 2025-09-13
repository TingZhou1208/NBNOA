function [D_Dec,D_Obj,Fitness] = CalFitness131(PopObj,PopCon,PopDec)
% Calculate the fitness of each solution based on original objective space
% %% 方案1
% N = size(PopObj,1);
% CV = sum(max(0,PopCon),2);               %%约束度求和，，    sum(A,2)行求和
% %% 计算可行比 基于epsilon的约束度值 方案1
% CV_mean=mean(CV);
% feasible_index=PopCon==0;
% feasible_pop=PopDec(feasible_index,:);   %%可行解子种群
% NF=size(feasible_pop,1);
% rf=NF./N;                                %%可行比，可行解的数量/总的数量
% epsilon=CV_mean*rf;                      %%静态的值
% %% Detect the dominance relation between each two solutions 基于epsilon的约束支配原则
% Dominate = false(N);
% for i = 1 : N-1
%     for j = i+1 : N                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   if (CV(i)<= epsilon  && CV(j)<= epsilon ) || CV(i)==CV(j) %%两个个体都小于一个阈值为可行解 通过比较目标函数值
%             k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
%             if k == 1
%                 Dominate(i,j) = true;
%             elseif k == -1
%                 Dominate(j,i) = true;
%             end
%         else
%             if CV(i) < CV(j)
%                 Dominate(i,j) = true;
%             elseif CV(i) > CV(j)
%                 Dominate(j,i) = true;
%             end
%         end
%     end
% end
%%
N = size(PopDec,1);
CV = sum(max(0,PopCon),2);      %%约束度求和，，    sum(A,2)行求和
%% Detect the dominance relation between each two solutions 约束支配原则--优先考虑约束值，然后考虑目标函数值
Dominate = false(N);
for i = 1 : N-1
    for j = i+1 : N
        if CV(i) < CV(j) %%i的约束度比j的小 i优于j
            Dominate(i,j) = true;
        elseif CV(i) > CV(j)
            Dominate(j,i) = true; %%j的约束度比i的小 j优于i
        else
            k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
            if k == 1
                Dominate(i,j) = true;
            elseif k == -1
                Dominate(j,i) = true;
            end
        end
    end
end
%% Calculate S(i)
S = sum(Dominate,2);
%% Calculate R(i)
R = zeros(1,N);
for i = 1 : N
    R(i) = sum(S(Dominate(:,i)));
end

%% Calculate D(i)
Distance = pdist2(PopObj,PopObj);
Distance(logical(eye(length(Distance)))) = inf;
Distance = sort(Distance,2);
D_Obj = 1./(Distance(:,floor(sqrt(N)))+2);

Distance = pdist2(PopDec,PopDec);
Distance(logical(eye(length(Distance)))) = inf;
Distance = sort(Distance,2);
D_Dec = 1./(Distance(:,floor(sqrt(N)))+2);

D =  D_Obj+ D_Dec;
%% Calculate the fitnesses
Fitness = R' + D;
end
