%% Innate Trajectory training
% Train a nonlinear random recurrent network to reproduce its own innate trajectory.
% Calls DAC_handwriting_kernel.m
% Written by Rodrigo Laje

% "Robust Timing and Motor Patterns by Taming Chaos in Recurrent Neural Networks"
% Rodrigo Laje & Dean V. Buonomano 2013


clear;
lwidth = 2;
fsize = 10;



% parameters and hyperparameters

% network
numUnits = 800;					% number of recurrent units
numplastic_Units = 480;			% number of recurrent plastic units
p_connect = 0.1;				% sparsity parameter (probability of connection)
g = 1.5;						% synaptic strength scaling factor
scale = g/sqrt(p_connect*numUnits);	% scaling for the recurrent matrix
numInputs = 2;					% number of input units
numOut = 2;						% number of output units

% input parameters
input_pulse_value = 2.0;
start_pulse = 200;				% (ms)
reset_duration = 50;			% (ms)

% training
interval_1 = 1322;
interval_2 = 1234;
learn_every = 2;				% skip time points
start_train = start_pulse + reset_duration;
end_train_1 = start_train + interval_1;
end_train_2 = start_train + interval_2;
n_learn_loops_recu = 30;		% number of training loops (recurrent)
n_learn_loops_read = 10;		% number of training loops (readout)
n_test_loops = 10;

% numerics
dt = 1;					% numerical integration time step
tmax = max([end_train_1 end_train_2]) + 1000;%200;
n_steps = fix(tmax/dt);			% number of integration time points
time_axis = [0:dt:tmax-dt];
plot_points = 500;				% max number of points to plot
plot_skip = ceil(n_steps/plot_points);
if rem(plot_skip,2)==0
	plot_skip = plot_skip + 1;
end

% firing rate model
tau = 10.0;							% time constant (ms)
sigmoid = @(x) tanh(x);			% activation function
noise_amplitude = 0.001;



% training and testing

savefile_trained = 'DAC_handwriting_recurr800_p0.1_g1.5.mat';
loadfile_handwriting = 'DAC_handwriting_output_targets.mat';

seed = RandStream('mt19937ar','seed',0);
RandStream.setGlobalStream(seed);



% create network and get innate trajectory
TRAIN_READOUT = 0;
TRAIN_RECURR = 0;
LOAD_DATA = 0;%0;
loadfile = savefile_trained;
GET_TARGET_INNATE_X = 1;
SAVE_DATA = 1;%1;
n_loops = 5;%1;
savefile = savefile_trained;
disp('getting innate activity.');

start_train_n = start_train;
end_train_1_n = end_train_1;
end_train_2_n = end_train_2;
	target_Out = zeros(2,numOut,n_steps);
	load(loadfile_handwriting);
	target_Out(1,:,start_train_n:end_train_1_n-1) = chaos;
	target_Out(2,:,start_train_n:end_train_2_n-1) = neuron;


	%DAC_handwriting_kernel;


%% train recurrent
TRAIN_READOUT = 0;
TRAIN_RECURR = 1;
LOAD_DATA = 1;
loadfile = savefile_trained;
GET_TARGET_INNATE_X = 0;
SAVE_DATA = 1;
n_loops = n_learn_loops_recu;
savefile = savefile_trained;
disp('training recurrent:');

	DAC_handwriting_kernel;


% train readout
TRAIN_READOUT = 1;
TRAIN_RECURR = 0;
LOAD_DATA = 1;
loadfile = savefile_trained;
GET_TARGET_INNATE_X = 0;
SAVE_DATA = 1;
n_loops = n_learn_loops_read;
savefile = savefile_trained;
disp('training readout:');

	DAC_handwriting_kernel;


% load, run, plot time series
TRAIN_READOUT = 0;
TRAIN_RECURR = 0;
LOAD_DATA = 1;
loadfile = savefile_trained;
GET_TARGET_INNATE_X = 0;
SAVE_DATA = 0;%1;
n_loops = n_test_loops;
savefile = savefile_trained;
disp('testing:');

	DAC_handwriting_kernel;


disp('done.');
disp('.');



%%
