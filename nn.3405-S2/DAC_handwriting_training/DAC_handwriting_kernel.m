%% Innate Trajectory training
% Called by DAC_handwriting_script.m
% Written by Rodrigo Laje

% "Robust Timing and Motor Patterns by Taming Chaos in Recurrent Neural Networks"
% Rodrigo Laje & Dean V. Buonomano 2013



if GET_TARGET_INNATE_X == 1
	noise_amp = 0;
else
	noise_amp = noise_amplitude;
end

if LOAD_DATA == 1
	load(loadfile);
else
%% connectivity matrices

	% random sparse recurrent matrix between units.
	% indices in WXX are defined as WXX(postyn,presyn),
	% that is WXX(i,j) = connection from X(j) onto X(i)
	% then the current into the postsynaptic unit is simply
	% (post)X_current = WXX*(pre)X.

	% if p_connect is very small, you can use WXX = sprandn(numUnits,numUnits,p_connect)*scale;
	% otherwise, use the following ("sprandn will generate significantly fewer nonzeros than requested if m*n is small or density is large")
	WXX_mask = rand(numUnits,numUnits);
	WXX_mask(WXX_mask <= p_connect) = 1;
	WXX_mask(WXX_mask < 1) = 0;
	WXX = randn(numUnits,numUnits)*scale;
	WXX = sparse(WXX.*WXX_mask);
	WXX(logical(eye(size(WXX)))) = 0;	% set self-connections to zero
	WXX_ini = WXX;

	% input connections WInputX(postsyn,presyn)
	WInputX = 1*randn(numUnits,numInputs);

	% output connections WXOut(postsyn,presyn)
	WXOut = randn(numOut,numUnits)/sqrt(numUnits);
	WXOut_ini = WXOut;


%% input pulse

	start_pulse_n = round(start_pulse/dt);
	reset_duration_n = round(reset_duration/dt);
	interval_1_n = round(interval_1/dt);
	interval_2_n = round(interval_2/dt);
	start_train_n = round(start_train/dt);
	end_train_1_n = round(end_train_1/dt);
	end_train_2_n = round(end_train_2/dt);

	input_pattern = zeros(2,numInputs,n_steps);
	input_pattern(1,1,start_pulse_n:start_pulse_n+reset_duration_n - 1) = input_pulse_value*ones(1,reset_duration_n);
	input_pattern(2,2,start_pulse_n:start_pulse_n+reset_duration_n - 1) = input_pulse_value*ones(1,reset_duration_n);


%% output target

	target_Out = zeros(2,numOut,n_steps);
	load(loadfile_handwriting);
	target_Out(1,:,start_train_n:end_train_1_n-1) = chaos;
	target_Out(2,:,start_train_n:end_train_2_n-1) = neuron;


end


%% P matrix definition

if (TRAIN_RECURR == 1)
	plastic_units = [1:numplastic_Units];		% list of all recurrent units subject to plasticity
	% one P matrix for each plastic unit in the network
	delta = 1.0;				% RLS: P matrix initialization
	for i = 1:numplastic_Units;
		pre_plastic_units(i).inds = find(WXX(plastic_units(i),:));	% list of all units presynaptic to plastic_units
		num_pre_plastic_units(i) = length(pre_plastic_units(i).inds);
		P_recurr(i).P = (1.0/delta)*eye(num_pre_plastic_units(i));
	end
end
if (TRAIN_READOUT == 1)
	% one P matrix for each readout unit
	delta = 1.0;				% RLS: P matrix initialization
	P_readout = permute(repmat((1.0/delta)*eye(numUnits),[1 1 numOut]),[3 1 2]);
end



%% main loop

figure(1);
clf(1);
figure(2);
clf(2);


X_history = zeros(numUnits,n_steps,2);
Out_history = zeros(numOut,n_steps,2);

% training/testing loop
for j = 1:n_loops

	fprintf('  loop: %2d/%2d\n',j,n_loops);

	% pattern loop ('chaos' and 'neuron')
	for k = 1:2

		if TRAIN_RECURR == 1 || TRAIN_READOUT == 1
			fprintf('    pattern: ');
		end

		% auxiliary variables
		WXOut_len = zeros(1,n_steps);
		WXX_len = zeros(1,n_steps);
		dW_readout_len = zeros(1,n_steps);
		dW_recurr_len = zeros(1,n_steps);
		train_window = 0;

		% initial conditions
		Xv = 1*(2*rand(numUnits,1)-1);
		X = sigmoid(Xv);
		Out = zeros(numOut,1);


		% integration loop
		for i = 1:n_steps

			if rem(i,round(n_steps/10)) == 0 && (TRAIN_RECURR == 1 || TRAIN_READOUT == 1)
				fprintf('.');
			end

			Input = input_pattern(k,:,i)';

			% update units
			noise = noise_amp*randn(numUnits,1)*sqrt(dt);
			Xv_current = WXX*X + WInputX*Input + noise;
			Xv = Xv + ((-Xv + Xv_current)./tau)*dt;
			X = sigmoid(Xv);
			Out = WXOut*X;

			% start-end training window
			if (i == start_train_n)
				train_window = 1;
			end
			if k == 1
				if (i == end_train_1_n)
					train_window = 0;
				end
			else
				if (i == end_train_2_n)
					train_window = 0;
				end
			end

			% training
			if (train_window == 1 && rem(i,learn_every) == 0)

				if TRAIN_RECURR == 1
					% train recurrent
					error = X - Target_innate_X(:,i,k);
					for plas = 1:numplastic_Units
						X_pre_plastic = X(pre_plastic_units(plas).inds);
						P_recurr_old = P_recurr(plas).P;
						P_recurr_old_X = P_recurr_old*X_pre_plastic;
						den_recurr = 1 + X_pre_plastic'*P_recurr_old_X;
						P_recurr(plas).P = P_recurr_old - (P_recurr_old_X*P_recurr_old_X')/den_recurr;
						% update network matrix
						dW_recurr = -error(plas)*(P_recurr_old_X/den_recurr)';
						WXX(plas,pre_plastic_units(plas).inds) = WXX(plas,pre_plastic_units(plas).inds) + dW_recurr;
						% store change in weights
						dW_recurr_len(i) = dW_recurr_len(i) + sqrt(dW_recurr*dW_recurr');
					end
				end

				if TRAIN_READOUT == 1
					for out = 1:numOut
						P_readout_old = squeeze(P_readout(out,:,:));
						P_readout_old_X = P_readout_old*X;
						den_readout = 1 + X'*P_readout_old_X;
						P_readout(out,:,:) = P_readout_old - (P_readout_old_X*P_readout_old_X')/den_readout;
						% update error
						error = Out(out,:) - target_Out(k,out,i);
						% update output weights
						dW_readout = -error*(P_readout_old_X/den_readout)';
						WXOut(out,:) = WXOut(out,:) + dW_readout;
						% store change in weights
						dW_readout_len(i) = sqrt(dW_readout*dW_readout');
					end
				end

			end
			% store output
			Out_history(:,i,k) = Out;
			X_history(:,i,k) = X;
			WXOut_len(i) = sqrt(sum(reshape(WXOut.^2,numOut*numUnits,1)));
			WXX_len(i) = sqrt(sum(reshape(WXX.^2,numUnits^2,1)));
		end

		if TRAIN_RECURR == 1 || TRAIN_READOUT == 1
			fprintf(' %d/2\n',k);
		end

		% plot
		if k == 1
			% plot time series
			figure(1);
			% input, output, target
			subplot(4,2,1);
			for out = 1:numOut
				plot(time_axis(1:plot_skip:end)-start_train, squeeze(target_Out(k,out,1:plot_skip:end)),'g-','linewidth', lwidth);
				hold all;
			end
			hold on;
			for input_nbr = 1:numInputs
				plot(time_axis(1:plot_skip:end)-start_train, squeeze(input_pattern(k,input_nbr,1:plot_skip:end))/5,'b-','linewidth', lwidth);
			end
			for out = 1:numOut
				plot(time_axis(1:plot_skip:end)-start_train, squeeze(Out_history(out,1:plot_skip:end,k)), 'r-','linewidth', lwidth);
			end
			ylabel('Input/5, output, target', 'fontsize', fsize);
			xlim([time_axis([1 end]) - start_train]);
			title('Chaos');

			% recurrent activity
			subplot(4,2,[3 5]);
			for x_unit = 1:10
				plot(time_axis(1:plot_skip:end)-start_train, X_history(x_unit,1:plot_skip:end,k)+2*x_unit);
				hold all;
			end
			hold on;
			xlim([time_axis([1 end]) - start_train]);
			ylim([0 22]);
			ylabel('Recurrent units', 'fontsize', fsize);

			% training measures
			subplot(4,2,7);
			if TRAIN_RECURR == 0
				plot(time_axis(1:plot_skip:end)-start_train, WXOut_len(1:plot_skip:end), 'linewidth', lwidth);
				ylabel('|WXOut|', 'fontsize', fsize);
				legend('|WXOut|','Location','SouthEast');
			else
				plot(time_axis(1:plot_skip:end)-start_train, WXX_len(1:plot_skip:end), 'linewidth', lwidth);
				ylabel('|WXX|', 'fontsize', fsize);
				legend('|WXX|','Location','SouthEast');
			end
			subplot(4,2,7);
			xlim([time_axis([1 end]) - start_train]);
			xlabel('Time (ms)');


			% plot handwriting
			figure(2);
			subplot(2,1,k);
			plot(chaos(1,:),chaos(2,:),'k','linewidth',3);
			hold on;
			for i = 1:n_test_loops
				plot(squeeze(Out_history(1,start_train_n:end_train_1_n,k)),squeeze(Out_history(2,start_train_n:end_train_1_n,k)));
			end
			xlim([-0.6 0.6]);
			ylim([-0.2 0.4]);
			xlabel('x (readout 1)');
			ylabel('y (readout 2)');

			pause(0.1);


		else
			% plot time series
			figure(1);
			% input, output, target
			subplot(4,2,2);
			for out = 1:numOut
				plot(time_axis(1:plot_skip:end)-start_train, squeeze(target_Out(k,out,1:plot_skip:end)),'g-','linewidth', lwidth);
				hold all;
			end
			hold on;
			for input_nbr = 1:numInputs
				plot(time_axis(1:plot_skip:end)-start_train, squeeze(input_pattern(k,input_nbr,1:plot_skip:end))/5,'b-','linewidth', lwidth);
			end
			for out = 1:numOut
				plot(time_axis(1:plot_skip:end)-start_train, squeeze(Out_history(out,1:plot_skip:end,k)), 'r-','linewidth', lwidth);
			end
			ylabel('Input/5, output, target', 'fontsize', fsize);
			xlim([time_axis([1 end]) - start_train]);
			title('Neuron');

			% recurrent activity
			subplot(4,2,[4 6]);
			for x_unit = 1:10
				plot(time_axis(1:plot_skip:end)-start_train, X_history(x_unit,1:plot_skip:end,k)+2*x_unit);
				hold all;
			end
			hold on;
			set(gca,'ytick',[2:2:20],'yticklabel',[]);
			xlim([time_axis([1 end]) - start_train]);
			ylim([0 22]);
			ylabel('Recurrent units', 'fontsize', fsize);

			% training measures
			subplot(4,2,8);
			if TRAIN_RECURR == 0
				plot(time_axis(1:plot_skip:end)-start_train, WXOut_len(1:plot_skip:end), 'linewidth', lwidth);
				ylabel('|WXOut|', 'fontsize', fsize);
				legend('|WXOut|','Location','SouthEast');
			else
				plot(time_axis(1:plot_skip:end)-start_train, WXX_len(1:plot_skip:end), 'linewidth', lwidth);
				ylabel('|WXX|', 'fontsize', fsize);
				legend('|WXX|','Location','SouthEast');
			end
			subplot(4,2,8);
			xlim([time_axis([1 end]) - start_train]);
			xlabel('Time (ms)');
			
			
			% plot handwriting
			figure(2);
			subplot(2,1,k);
			plot(neuron(1,:),neuron(2,:),'k','linewidth',3);
			hold on;
			for i = 1:n_test_loops
				plot(squeeze(Out_history(1,start_train_n:end_train_2_n,k)),squeeze(Out_history(2,start_train_n:end_train_2_n,k)));
			end
			xlim([-0.6 0.6]);
			ylim([-0.2 0.4]);
			xlabel('x (readout 1)');
			ylabel('y (readout 2)');

			pause(0.1);

		end

	end
end


% get target from innate trajectory
if GET_TARGET_INNATE_X == 1
	Target_innate_X = X_history;
elseif ~exist('Target_innate_X','var')
	Target_innate_X = [];
end


if SAVE_DATA == 1
	save(savefile,'WXX','WXX_ini','WInputX','WXOut','WXOut_ini','Target_innate_X','target_Out',...
		'numUnits','numplastic_Units','p_connect','g','numInputs','numOut',...
		'plot_skip','learn_every','tau','sigmoid','scale','noise_amplitude',...
		'input_pulse_value','start_pulse','reset_duration','input_pattern',...
		'interval_1','interval_2','learn_every','start_train','end_train_1','end_train_2',...
		'n_learn_loops_recu','n_learn_loops_read','n_test_loops',...
		'dt','tmax','n_steps','time_axis','plot_points','plot_skip',...
		'tau','sigmoid','noise_amplitude');
end


%%
