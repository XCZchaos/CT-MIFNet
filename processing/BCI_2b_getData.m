% auto adjust the EEG filter，and 6 stage cheby filter

function data = getData(subject_index)

subject_index = 2; % 1-9

%% T data
session_type = 'T'; % T and E
dir_1 = ['C:\Users\24242\Desktop\AI_Reference\data_bag\BCICIV_2b_gdf\B0',num2str(subject_index),'01', session_type,'.gdf']; % set your path of the downloaded data
[s1, HDR1] = mexSLOAD(dir_1);
dir_2 = ['C:\Users\24242\Desktop\AI_Reference\data_bag\BCICIV_2b_gdf\B0',num2str(subject_index),'02', session_type,'.gdf']
[s2, HDR2] = mexSLOAD(dir_2);
dir_3 = ['C:\Users\24242\Desktop\AI_Reference\data_bag\BCICIV_2b_gdf\B0',num2str(subject_index),'03', session_type,'.gdf']
[s3, HDR3] = mexSLOAD(dir_3);



% Label 
% label = HDR.Classlabel;
labeldir_1 = ['C:\Users\24242\Desktop\AI_Reference\data_bag\2b_label\B0',num2str(subject_index),'01',session_type ,'.mat'];
labeldir_2 = ['C:\Users\24242\Desktop\AI_Reference\data_bag\2b_label\B0',num2str(subject_index),'02',session_type ,'.mat'];
labeldir_3 = ['C:\Users\24242\Desktop\AI_Reference\data_bag\2b_label\B0',num2str(subject_index),'03',session_type ,'.mat'];

load(labeldir_1);
label_1 = classlabel;
load(labeldir_2);
label_2 = classlabel;
load(labeldir_3);
label_3 = classlabel;

% construct sample - data Section 1000*22*100
Pos1 = HDR1.EVENT.POS; % use POS to get trials
% Dur = HDR.EVENT.DUR;
Typ1 = HDR1.EVENT.TYP;

k = 0;
data_1 = zeros(1000,3,120);
data_2 = zeros(1000,3,120);
data_3 = zeros(1000,3,160);
for j = 1:length(Typ1)
    if  Typ1(j) == 768
        k = k+1;
        data_1(:,:,k) = s1((Pos1(j)+750):(Pos1(j)+1749),1:3)
    end
end



% construct sample - data Section 1000*22*100
Pos2 = HDR2.EVENT.POS; % use POS to get trials
% Dur = HDR.EVENT.DUR;
Typ2 = HDR2.EVENT.TYP;

k = 0;
for j = 1:length(Typ2)
    if  Typ2(j) == 768
        k = k+1;
        data_2(:,:,k) = s2((Pos2(j)+750):(Pos2(j)+1749),1:3)
    end
end

% construct sample - data Section 1000*22*100
Pos3 = HDR3.EVENT.POS; % use POS to get trials
% Dur = HDR.EVENT.DUR;
Typ3 = HDR3.EVENT.TYP;

k = 0;
for j = 1:length(Typ3)
    if  Typ3(j) == 768
        k = k+1;
        data_3(:,:,k) = s3((Pos3(j)+750):(Pos3(j)+1749),1:3)
    end
end

% wipe off NaN
data_1(isnan(data_1)) = 0;
data_2(isnan(data_2)) = 0;
data_3(isnan(data_3)) = 0;


% E data
session_type = 'E';
dir_4 = ['C:\Users\24242\Desktop\AI_Reference\data_bag\BCICIV_2b_gdf\B0',num2str(subject_index),'04', session_type,'.gdf']
dir_5 = ['C:\Users\24242\Desktop\AI_Reference\data_bag\BCICIV_2b_gdf\B0',num2str(subject_index),'05', session_type,'.gdf']
% dir = 'D:\Lab\MI\BCICIV_2a_gdf\A01E.gdf';
[s4, HDR4] = mexSLOAD(dir_4);
[s5, HDR5] = mexSLOAD(dir_5);

% Label 
% label = HDR.Classlabel;
labeldir_4 = ['C:\Users\24242\Desktop\AI_Reference\data_bag\2b_label\B0',num2str(subject_index),'04',session_type ,'.mat'];
labeldir_5 = ['C:\Users\24242\Desktop\AI_Reference\data_bag\2b_label\B0',num2str(subject_index),'05',session_type ,'.mat'];
load(labeldir_4);
label_4 = classlabel;
load(labeldir_5)
label_5 = classlabel;


% construct sample - data Section 1000*22*120
Pos4 = HDR4.EVENT.POS;
% Dur = HDR5.EVENT.DUR;
Typ4 = HDR4.EVENT.TYP;

% Set the types of events you are interested in (e.g., 768)
target_event_type = 768;


k = 0;
data_4 = zeros(1000,3,120);
data_5 = zeros(1000,3,160);
for j = 1:length(Typ4)
    if  Typ4(j) == 768
        k = k+1;
        data_4(:,:,k) = s4((Pos4(j)+750):(Pos4(j)+1749),1:3);
     
    end
end

Pos5 = HDR5.EVENT.POS;
% Dur = HDR5.EVENT.DUR;
Typ5 = HDR5.EVENT.TYP;

k = 0;
for j = 1:length(Typ5)
    if  Typ5(j) == 768
        k = k+1;
        data_5(:,:,k) = s5((Pos5(j)+750):(Pos5(j)+1749),1:3);
     
    end
end

% wipe off NaN
data_4(isnan(data_4)) = 0;
data_5(isnan(data_5)) = 0;

%% preprocessing
% option - band-pass filter
Fs = HDR1.SampleRate; % sample rate
Wp = [4 40] / (Fs/2); 
Ws = [2 50] / (Fs/2); 
Rp = 1; 
Rs = 40; 
[n, Wn] = cheb2ord(Wp, Ws, Rp, Rs); % Calculate the filter order and cutoff frequency.
[b, a] = cheby2(n, Rs, Wn); % Calculate filter coefficients

for j = 1:120
    data_1(:,:,j) = filtfilt(b,a,data_1(:,:,j));
    data_2(:,:,j) = filtfilt(b,a,data_2(:,:,j));
    data_4(:,:,j) = filtfilt(b,a,data_4(:,:,j));
    
end


for j = 1:160
    data_3(:,:,j) = filtfilt(b,a,data_3(:,:,j));
%    data_4(:,:,j) = filtfilt(b,a,data_4(:,:,j));
    data_5(:,:,j) = filtfilt(b,a,data_5(:,:,j));
end

% option - a simple standardization
%{
eeg_mean = mean(data,3);
eeg_std = std(data,1,3); 
fb_data = (data-eeg_mean)./eeg_std;
%}

%% Save the data to a mat file 
data = cat(3, data_1, data_2, data_3);
label = cat(1, label_1, label_2, label_3);
% label = t_label + 1;
% please change to your path
saveDir = ['C:\Users\24242\Desktop\AI_Reference\data_bag\BCICIV_2b_gdf\B0',num2str(subject_index),'processing_T.mat'];
save(saveDir,'data','label');

data = cat(3, data_4, data_5);
label = cat(1,label_4, label_5);
saveDir = ['C:\Users\24242\Desktop\AI_Reference\data_bag\BCICIV_2b_gdf\B0',num2str(subject_index),'processing_E.mat'];
save(saveDir,'data','label');

end