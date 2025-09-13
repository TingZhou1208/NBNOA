function [Population,D_Dec,D_Obj ] = EnvironmentalSelection13(Population,N,n_var,n_obj)
% The environmental selection of SPEA2 based on objective space 基于约束的选择--约束优先
    %% Calculate the fitness of each solution
    PopObj=Population(:,n_var+1:n_var+n_obj);
    PopCon = Population(:,end);
    PopDec= Population(:,1:n_var);
    [D_Dec,D_Obj,Fitness] = CalFitness131(PopObj,PopCon, PopDec);
    %% Environmental selection
    Next = Fitness < 1;
    if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del  = Truncation(Population(Next,n_var+1:n_var+n_obj),Population(Next,1:n_var),sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end
    % Population for next generation
    Population = Population(Next,:);
    D_Dec      = D_Dec(Next,:);
    D_Obj     = D_Obj(Next,:);
end

function Del = Truncation(PopObj,PopDec,K)
% Select part of the solutions by truncation

    %% Truncation
    Distance_Dec = pdist2(PopDec,PopDec);
    Distance_Dec(logical(eye(length(Distance_Dec)))) = inf;
     D = Distance_Dec;
   
%     Distance_Obj = pdist2(PopObj,PopObj);
%     Distance_Obj (logical(eye(length(Distance_Obj )))) = inf;
%     D = Distance_Obj;
    
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(D(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end