% DAC_Handwriting
% Handwriting Example: 
% Laje & Buonomano (2013) Nature Neuroscience
% called from DAC_Handwriting_Demo.m
% Dean Buonomano 4/10/13

global LineHandles VoltageHandles plotXYPoint
global InputDur In1 In2 InPerturb InPerturbDur
global UpdateStep NumLineSegs numExPlot

tau   = 10;          %10 ms (time step = 1 ms)
InputDur = 50;       %Input Dur ms
InPerturbDur = 10;   %Pertubation Duration ms

InPerturb   = 0;
In1 = 0;
In2 = 0;
InAmp    = [0.3 2];  %[Pertub Amp, Input Amp];


%%% LOAD WEIGHT MATRICES %%%
load W_Handwriting;
[numEx numOut] = size(WExOut);
[numEx numIn]  = size(WInEx);

%Initialize
t=0;
Ex   = zeros(numEx,1);
ExV  = zeros(numEx,1);
Out  = zeros(numOut,1);
In   = zeros(numIn,1);
historyEx  = zeros(numExPlot,UpdateStep*NumLineSegs);
historyOut = zeros(numOut,UpdateStep);
historyOUT = zeros(numOut,UpdateStep,NumLineSegs);

%random initial state
ExV = 2*rand(numEx,1)-1;

while RUN==1
   t=t+1;
   
   %COUNT DOWN (-1) to implement the duration of the events.
   %NEGATIVE VALUES ARE NOT USED (In1>0) equals 0 or 1
   In1 = In1-1;
   In2 = In2-1;
   InPerturb = InPerturb-1;
   In = [InAmp(1)*(InPerturb>0); InAmp(2)*(In1>0); InAmp(2)*(In2>0)];

   ex_input = WExEx'*Ex + WInEx*In + NoiseValue*randn(numEx,1);   
   ExV = ExV + (-ExV + ex_input)./tau;
   Ex = tanh(ExV);
   
   Out = WExOut'*Ex;

   historyEx = [historyEx(:,2:end) Ex(1:numExPlot)];
   historyOut(:,rem(t-1,UpdateStep)+1) = Out;
   
   %% GRAPHICS UPDATE
   if rem(t,UpdateStep)==0

      for i=1:numExPlot
         set(VoltageHandles.plotLine(i),'xdata',[(t-UpdateStep*NumLineSegs+1):t],'ydata',historyEx(i,:)+(i-1)*1);
      end 
      set(handles.axVoltage,'xlim',[(t-UpdateStep*NumLineSegs+1) t]);

      for i=1:(NumLineSegs-1)
         historyOUT(:,:,i)=historyOUT(:,:,i+1);
         set(LineHandles.plotXY(i),'xdata',historyOUT(1,:,i),'ydata',historyOUT(2,:,i));
      end
      historyOUT(:,:,NumLineSegs) = historyOut;
      set(LineHandles.plotXY(NumLineSegs),'Xdata',historyOut(1,:),'Ydata',historyOut(2,:));
      set(plotXYPoint,'Xdata',historyOut(1,end),'Ydata',historyOut(2,end));
      
      drawnow;
      fprintf('t=%5d\n',t);
      assignin('base','historyEx',historyEx);
      assignin('base','historyOUT',historyOUT);
      assignin('base','t',t);
      pause(PauseValue)          %slow down graphics
            
   end

end




