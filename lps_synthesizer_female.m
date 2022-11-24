%EEEM030
%Assignment
%Linear Predictive Speech Synthesizer

close all;
clear all;
clc;


[input_sound, fs] = audioread('hod_f.wav'); %Reading the signal
ts = 1 / fs;  
input_length = 100; %Choosing a segment, 100 milisecond in length
synth_sound = 1000; %Length of systhesized sound in miliseconds
offset = 20; %In miliseconds
lpc_order = 20;  %LPC modeling order

  %Clipping the segment for processing
  
len = length(input_sound); 
segment_length_samples = min(milisecond_to_sample(input_length, fs), len);
K = max(milisecond_to_sample(offset, fs), 1);
segment_length_samples = min(segment_length_samples, len - 1);
if len < segment_length_samples + K
    K = signal_length_samples - segment_length_samples;
end
input_sound = input_sound(K : K + segment_length_samples);

  %Plotting the signal
figure(1)
plot(input_sound)
N = length(input_sound);
title('Input Sound Signal', 'FontSize', 16)
xlabel('Number of Samples', 'FontSize', 16);
ylabel('Amplitude', 'FontSize', 16);

   % Performing Fast Fourier Transform
figure (2)
Y = fft(input_sound);
   % Plotting two-sided spectrum
S2 = abs(Y); 
   % Plotting one-sided spectrum
S1 = S2(1:floor(N/2+1)); 
S1(2:end-1) = 2*S1(2:end-1);
frequency = fs*(0:(N/2))/N; 
fft_plot = plot(frequency,20*log10(abs(S1)), 'green');
fft_plot.Color(4) = 0.50;
fft_plot.LineWidth = 1.5;
xlabel('Frequency (Hz)','FontSize', 16)
ylabel('Amplitude (dB)','FontSize', 16)
hold on

  % Linear Predicive Coding Analysis

a = lpc(input_sound,lpc_order);
[lpc_amp, lpc_frequency] = freqz(1 , a , length(frequency), fs); 
amp_db = 20*log10(abs(lpc_amp));
lpc_frequency_plot = plot(lpc_frequency, amp_db, 'r');

     % Calculating  first three formants
f = islocalmax(amp_db); 
fm_frequency = lpc_frequency(f);
fm_amplitude = amp_db(f);

f_plot = plot(fm_frequency, fm_amplitude, 'b*');
text(fm_frequency(1),fm_amplitude(1),{num2str(fm_frequency(1))});
text(fm_frequency(2),fm_amplitude(2),{num2str(fm_frequency(2))});
text(fm_frequency(3),fm_amplitude(3),{num2str(fm_frequency(3))});
grid
legend('Original Spectrum', 'LPC Spectrum', 'LPC Maxima')
hold off

    %  Calculating Fundamental Frequency using Cepstrum
CEPSTRUM_FIX = 0.090;
C_LOW_PASS_COEFFICIENT = [1 -0.6];
quefrency = 30;
cepstrum = rceps(input_sound);
cepstrum_p =(0:N - 1);  
figure(3)
cepstrum_lifter = filter(1, C_LOW_PASS_COEFFICIENT, cepstrum);
ceps = cepstrum_lifter;
plot(cepstrum_p(1 : round(N / 2)), ceps(1 : round(N / 2))) 
hold on
ceps(ceps < CEPSTRUM_FIX) = 0;
cepstrum_indexes = islocalmax(ceps);
cepstrum_position = cepstrum_p(cepstrum_indexes); 
ceps = ceps(cepstrum_indexes);

cepstrum_time = quefrency < cepstrum_position;
cepstrum_position =cepstrum_position(cepstrum_time); 
ceps = ceps(cepstrum_time);

% Calculating first half portion of the signal
cepstrum_half = cepstrum_position <= round(N / 2);
cepstrum_position = cepstrum_position(cepstrum_half);
ceps = ceps(cepstrum_half);

f_plot = plot(cepstrum_position, ceps, 'b*');
f_plot.MarkerSize = 6;
f_plot.LineWidth = 1;
grid
xlabel('Quefrency','FontSize', 16)
ylabel('rceps(x[n])' , 'FontSize' , 16)
xlim([0 N / 2])
title('Cepstrum')

if  length(cepstrum_position) >= 1    
     pp = cepstrum_position(ceps == max(ceps)); %pp is pitch period
     ff = 1 / (pp / fs); %ff is fundamental frequency
     text(cepstrum_position(ceps == max(ceps)),ceps(ceps==max(ceps)),{strcat('Fundamental Frequency =',num2str(ff))});
else
    disp('No pitch period is found');
end
hold off

     %   Speech Synthesis   
figure(4)
excitation = generate_impulse(ff , fs, synth_sound);
synthesized_sound = filter(1, a , excitation);
plot(synthesized_sound);
title('Synthesized version of sound', 'FontSize' , 16)
ap = audioplayer(synthesized_sound , fs , 16); 
pause(1);
play(ap);
xlabel('Time(ms)', 'FontSize', 16);
ylabel('Frequency(KHz)', 'FontSize', 16);

audiowrite('new_hod_f.wav',synthesized_sound,fs);

    %   Converts milisecond to sample   %
function s = milisecond_to_sample(time , freq)
s = (time / 1000) * freq;
end
      %   Impulse Generation   %
function sig = generate_impulse(ff , sf, length) 

if sf < ff  
    disp('Sampling Frequency is lower than Fundamental Frequency')
    sig = [];
    return
end
desired_samples = milisecond_to_sample(length , sf);
pp = 1 / ff; 
sp = 1 / sf;  %sample period
x = round(pp / sp);
p = [1 zeros(1 , x - 1)];
desireded_cells = ceil(desired_samples / x);
sig = repmat(p, 1 , desireded_cells);
sig = sig(1 : desired_samples);
end

%Completed