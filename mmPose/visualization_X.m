close all
% 讀取 JSON 檔案
filename = 'Matched_Data/kenny/kenny_5.json';
jsonText = fileread(filename);
data = jsondecode(jsonText);

% 創建一個 figure
figure;
hold on;
grid on;
xlabel('X');
ylabel('Y');
zlabel('Z');
title('3D Visualization of X points');

%axis([-1.26 1.26 -1.26 1.26 -1.08 1.08]);
axis([-1.5 1.5 1 4 0 3]);
% **建立時間標籤**
h_time_text = annotation('textbox', [0.15, 0.85, 0.1, 0.05], 'String', '', ...
                         'FontSize', 12, 'Color', 'r', 'EdgeColor', 'none', 'FontWeight', 'bold');
% 遍歷每個 frame
for i = 1:length(data)
    cla
    frame = data(i);  % 當前 frame
    X_points = frame.X;
    
    % 解析 X, Y, Z 座標
    x_coords = [];
    y_coords = [];
    z_coords = [];
    
    for j = 1:length(X_points)
        x_coords = [x_coords, X_points(j).x];
        y_coords = [y_coords, X_points(j).y];
        z_coords = [z_coords, X_points(j).z];
    end
    
    % 繪製 3D 點
    scatter3(x_coords, z_coords, y_coords, 'filled');
    % **更新時間標籤**
    timestamp = frame.time;
    set(h_time_text, 'String', sprintf('Time: %.2f sec', timestamp));
    drawnow limitrate;
    pause(0.01); % 每個 frame 停 0.5 秒，模擬動畫效果
end

hold off;
close all
