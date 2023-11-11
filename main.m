clear,clc,close all
file_name = 'C:/Users/29594/Downloads/indy_20170124_01.mat';
file = load(file_name);
t = file.t;
fs = 1/(t(2)-t(1));     % 采样率,250Hz，一个点4ms
n_time = size(t,1);
time = 0:1/fs:(n_time-1)/fs;

finger_pos = -file.finger_pos(:,2:3);  % 手指的x,y坐标

%% 得到矩阵形式的spike
spikes = file.spikes;   % 发放时间戳
spike_array = zeros(n_time,size(spikes,1)*size(spikes,2));
n_feature=1;
for i_electrode = 1:size(spikes,1)
    for i_unit = 1:size(spikes,2)
        if ~isempty(spikes{i_electrode ,i_unit})
            ttt = round((spikes{i_electrode,i_unit}-t(1))/(t(2)-t(1)) + 1); %转换为索引值
            ttt = ttt(ttt>0&ttt<=n_time); %落在与运动学数据相同的时间范围内
            spike_array(ttt,n_feature)=1;
            n_feature = n_feature+1;
        end
    end
end
spike_array = spike_array(:,sum(spike_array,1)~=0);

%% 分窗
bin_time = 0.1; %100ms
bin_len = round(bin_time*fs);  %窗长
n_bin = round(n_time/bin_len); %窗数目
kin = zeros(n_bin,3*size(finger_pos,2));
fr = zeros(n_bin,size(spike_array,2));
for i_bin =1:n_bin
    fr(i_bin,:) = sum(spike_array((i_bin-1)*bin_len+1:i_bin*bin_len,:),1)/bin_time;
    kin(i_bin,1) = finger_pos(i_bin*bin_len,1); %P_x
    kin(i_bin,2) = finger_pos(i_bin*bin_len,2); %P_y
    kin(i_bin,3) = (finger_pos(i_bin*bin_len,1) - finger_pos((i_bin-1)*bin_len+1,1))/bin_time;
    kin(i_bin,4) = (finger_pos(i_bin*bin_len,2) - finger_pos((i_bin-1)*bin_len+1,2))/bin_time;
end
kin(2:end,5) = diff(kin(:,3))/bin_time; %A_x
kin(2:end,6) = diff(kin(:,4))/bin_time; %A_x

fs = 1/bin_time;
time = 0:1/fs:(n_bin-1)/fs;


figure()
idx1=2000;
idx2=2100;
subplot(1,3,1)
plot(time(idx1:idx2),kin(idx1:idx2,1:2))
title('位置')
xlabel('时间(s)')

subplot(1,3,2)
plot(time(idx1:idx2),kin(idx1:idx2,3:4))
title('速度')
xlabel('时间(s)')

subplot(1,3,3)
plot(time(idx1:idx2),kin(idx1:idx2,5:6))
title('加速度')
legend('x方向','y方向')
xlabel('时间(s)')


figure()
stem(time(idx1:idx2),fr(idx1:idx2,23), 'Marker', 'none')
title('发放率')
xlabel('时间(s)')
ylabel('发放率(Hz)')





    
    
    

       




