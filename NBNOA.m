function [ps,pf,cons]=NBNOA(func_name,VRmin,VRmax,n_obj,Particle_Number,Max_Gen)
% MO_Ring_PSO_SCD: A multi-objective particle swarm optimization using ring topology for solving multimodal multi-objective optimization problems
% Dimension: n_var --- dimensions of decision space
%            n_obj --- dimensions of objective space
%% Input:
%                      Dimension                    Description
%      func_name       1 x length function name     the name of test function
%      VRmin           1 x n_var                    low bound of decision variable
%      VRmax           1 x n_var                    up bound of decision variable
%      n_obj           1 x 1                        dimensions of objective space
%      Particle_Number 1 x 1                        population size
%      Max_Gen         1 x 1                        maximum  generations

%% Output:
%                     Description
%      ps             Pareto set
%      pf             Pareto front
%%  Reference and Contact
% Reference: [1]Caitong Yue, Boyang Qu and Jing Liang, "A Multi-objective Particle Swarm Optimizer Using Ring Topology for Solving Multimodal Multi-objective Problems",  IEEE Transactions on Evolutionary Computation, 2017, DOI 10.1109/TEVC.2017.2754271.
%            [2]Jing Liang, Caitong Yue, and Boyang Qu, “ Multimodal multi-objective optimization: A preliminary study”, IEEE Congress on Evolutionary Computation 2016, pp. 2454-2461, 2016.
% Contact: For any questions, please feel free to send email to zzuyuecaitong@163.com.

%% Initialize parameters
n_var=size(VRmin,2);               %Obtain the dimensions of decision space
Max_FES=Max_Gen*Particle_Number;   %Maximum fitness evaluations
n_PBA=20;                           %Maximum size of PBA. The algorithm will perform better without the size limit of PBA. But it will time consuming.
iwt=0.7298;                        %Inertia weight
cc=[2.05 2.05];                    %Acceleration constants
%% Initialize particles' positions and velocities
mv=0.5*(VRmax-VRmin);
VRmin=repmat(VRmin,Particle_Number,1);
VRmax=repmat(VRmax,Particle_Number,1);
Vmin=repmat(-mv,Particle_Number,1);
Vmax=-Vmin;
pos=VRmin+(VRmax-VRmin).*rand(Particle_Number,n_var); %initialize the positions of the particles
vel=Vmin+2.*Vmax.*rand(Particle_Number,n_var);        %initialize the velocities of the particles
%% Evaluate the population
particle=pos;
for ii=1:Particle_Number
    particle(ii,n_var+1:n_var+n_obj+1)=feval(func_name,pos(ii,:));
end
fitcount=Particle_Number;            % count the number of fitness evaluations
%% Initialize personal best archive PBA and Neighborhood best archive NBA
row_of_cell=ones(1,Particle_Number); % the number of row in each cell
col_of_cell=size(particle,2);        % the number of column in each cell
PBA=mat2cell(particle,row_of_cell,col_of_cell);
EXA=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for g=1:Max_Gen
    %% 形成子种群
    [~,~,Fitness]= CalFitness14(particle(:,n_var+1:n_var+n_obj),particle(:,end),particle(:,1:n_var)); %%  方案2 基于epsilon的约束度值+SCD
    [species,guide,num,meandis,slbest,sgbest]=NBNC1(Fitness,particle,Particle_Number,g,Max_Gen,n_var);
    %% Strategy of balancing the species
    %         if g==floor(0.85*Max_Gen)
    %              [~,~,Fitness]= CalFitness14(Pbest(:,n_var+1:n_var+n_obj),Pbest(:,end),Pbest(:,1:n_var));
    %             [species,pos,vel,~,~,guide] = balance_species(Fitness, Pbest(:,1:n_var),pos,vel,VRmin(1,:),VRmax(1,:),species,num,meandis,guide);
    %         end
    %% 整个种群进化
    for i = 1:Particle_Number
        % Choose the first particle in PBA_k as pbest
        PBA_i=PBA{i,1};                    %PBA_k contains the history positions of particle_k
        pbest=PBA_i(1,:);                  %Choose the first one
        % Update velocities according to Eq.(5)
        vel(i,1:n_var)=iwt.*vel(i,1:n_var)+ cc(1).*rand(1,n_var).*(pbest(1,1:n_var)-pos(i,1:n_var))...
            + cc(2).*rand(1,n_var).*(guide(i,1:n_var)-pos(i,1:n_var));  %%更新速度
        % Make sure that velocities are in the setting bounds.
        vel(i,1:n_var)=(vel(i,1:n_var)>mv).*mv+(vel(i,1:n_var)<=mv).*vel(i,1:n_var);
        vel(i,1:n_var)=(vel(i,1:n_var)<(-mv)).*(-mv)+(vel(i,:)>=(-mv)).*vel(i,1:n_var);
        % Update positions according to Eq.(4)
        pos(i,1:n_var)=pos(i,1:n_var)+vel(i,1:n_var);
        % Make sure that positions are in the setting bounds.
        pos(i,1:n_var)=((pos(i,1:n_var)>=VRmin(1,1:n_var))&(pos(i,1:n_var)<=VRmax(1,1:n_var))).*pos(i,1:n_var)...
            +(pos(i,1:n_var)<VRmin(1,1:n_var)).*(VRmin(1,1:n_var)+0.25.*(VRmax(1,1:n_var)-VRmin(1,1:n_var)).*rand(1,n_var))+(pos(i,1:n_var)>VRmax(1,1:n_var)).*(VRmax(1,1:n_var)-0.25.*(VRmax(1,1:n_var)-VRmin(1,1:n_var)).*rand(1,n_var));
        % Evaluate the population
        particle(i,1:n_var)=pos(i,1:n_var);
        particle(i,n_var+1:n_var+n_obj+1)=feval(func_name, pos(i,1:n_var));
        fitcount=fitcount+1;
        %% Update PBA
        PBA_i=[PBA_i; particle(i,:)];
        PBA_i=unique(PBA_i,'rows','stable');%%把相同的解过滤掉
        [~,~,Fitness] = CalFitness14(PBA_i(:,n_var+1:n_var+n_obj),PBA_i(:,end),PBA_i(:,1:n_var));
        [~,index1]=sort(Fitness,'ascend');
        if size(PBA_i,1)>n_PBA
            PBA{i,1}=PBA_i(index1(1:n_PBA),:);
        else
            PBA{i,1}=PBA_i(index1,:);
        end
        Pbest(i,:)=PBA{i,1}(1,:);
    end
    pos=particle(:,1:n_var);
    %     %% 更新存档
    %     tempEXA=cell2mat(PBA);
    %     tempEXA=unique(tempEXA,'rows','stable');
    %     if size(tempEXA,1)>Particle_Number
    %         [EXA,~,~] = EnvironmentalSelection13(tempEXA,Particle_Number,n_var,n_obj);
    %     else
    %         EXA=tempEXA;
    %     end
    %     g
    %     clf;
    %     figure(g)
    %     plot(particle(:,1),particle(:,2),'r+')
    %     pause(0.01)
    if fitcount>Max_FES
        break;
    end
end
%% Output ps and pf
%% 更新存档
tempEXA=cell2mat(PBA);
tempEXA=unique(tempEXA,'rows','stable');
if size(tempEXA,1)>Particle_Number
    [EXA,~,~] = EnvironmentalSelection13(tempEXA,Particle_Number,n_var,n_obj);
else
    EXA=tempEXA;
end
ps=EXA(:,1:n_var);
pf=EXA(:,n_var+1:n_var+n_obj);
cons=EXA(:,end);
end



%         alpha= 2./(1+exp(1).^(-g*10/Max_Gen))-1;  %%文章公式11
%         [EXA,~,~] = EnvironmentalSelection1(tempEXA,Particle_Number);
%         EnvironmentalSelection2(tempEXA,Particle_Number,alpha,n_var,n_obj);
%        [PBA_k,~,~] = EnvironmentalSelection2(PBA_k,n_PBA,alpha,n_var,n_obj);
%         [PBA_k,~,~] = EnvironmentalSelection1(PBA_k,n_PBA);



%Write by Caitong Yue 2017.09.04
%Supervised by Jing Liang and Boyang Qu