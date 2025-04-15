function visualization_Y_v2(json_filename)
    % 讀取 JSON 文件
    fid = fopen(json_filename, 'r');
    if fid == -1
        error("Error: Could not open file '%s'. Check if the file exists and path is correct.", json_filename);
    end
    raw = fread(fid, inf);
    fclose(fid);
    json_str = char(raw');
    json_data = jsondecode(json_str);

    % **Kinect v2 (25 個關節) 的連接關係**
    skeleton_connections = [4 3; 3 21; 21 2; 2 1;  % 頭 → 頸 → 肩膀 → 胸口 → 胯部
                            21 5; 5 6; 6 7; 7 8; 8 23; 8 22; % 左臂 (肩膀 → 肘 → 手腕 → 手)
                            21 9; 9 10; 10 11; 11 12; 12 24; 12 25;  % 右臂 (肩膀 → 肘 → 手腕 → 手)
                            1 13; 13 14; 14 15; 15 16; % 左腿 (臀部 → 膝蓋 → 腳踝 → 腳)
                            1 17; 17 18; 18 19; 19 20; % 右腿 (臀部 → 膝蓋 → 腳踝 → 腳)
                            ];

    % 創建 figure
    figure;
    hold on;
    grid on;
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title('Kinect v2 Skeleton Visualization');

    % **固定 XYZ 軸範圍**

    axis([-1.1 1.1 0 7 -2.5 1.5]);
    view([0,0]) % 俯視圖
    
    % **預先建立點雲和骨架線條物件**
    h_scatter = scatter3(nan, nan, nan, 50, 'filled'); % 預設為空
    h_lines = gobjects(size(skeleton_connections, 1), 1); % 預先建立連接線物件

    % 初始化所有連接線
    for k = 1:size(skeleton_connections, 1)
        h_lines(k) = plot3(nan, nan, nan, 'b', 'LineWidth', 2);
    end

    % **建立時間標籤**
    h_time_text = annotation('textbox', [0.15, 0.85, 0.1, 0.05], 'String', '', ...
                             'FontSize', 12, 'Color', 'r', 'EdgeColor', 'none', 'FontWeight', 'bold');

    % **迴圈更新動畫**
    for t = 1:length(json_data)
        % 取得關節點座標
        coordinates_struct = json_data(t).coordinates;
        coordinates = [[coordinates_struct.x]; [coordinates_struct.y]; [coordinates_struct.z]]';

        % **檢查是否有 25 個關節點**
        if size(coordinates, 1) ~= 25
            warning('Frame %d 的關節數量錯誤 (預期: 25, 實際: %d)', t, size(coordinates, 1));
            continue; % 跳過這一幀
        end

        % **更新點雲數據**
        set(h_scatter, 'XData', coordinates(:,1), 'YData', coordinates(:,3), 'ZData', coordinates(:,2));

        % **更新骨架線條**
        for k = 1:size(skeleton_connections, 1)
            joint1 = skeleton_connections(k, 1);
            joint2 = skeleton_connections(k, 2);
            set(h_lines(k), 'XData', [coordinates(joint1,1), coordinates(joint2,1)], ...
                            'YData', [coordinates(joint1,3), coordinates(joint2,3)], ...
                            'ZData', [coordinates(joint1,2), coordinates(joint2,2)]);
        end

        % **更新時間標籤**
        timestamp = json_data(t).time;
        set(h_time_text, 'String', sprintf('Time: %.2f sec', timestamp));

        % 讓畫面更流暢
        drawnow limitrate;
        pause(0.005); % 減少 pause 讓動畫更快
    end

    % 動畫結束後關閉 figure
    close all
    json_data
end
