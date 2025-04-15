function collect_v2(varargin)
    %% === 解析輸入參數 ===
    if nargin == 0
        name = strtrim(input('請輸入名字: ', 's'));
    else
        name = varargin{1};
    end

    %% === 初始化 Kinect ===
    imaqreset;
    imaqmex('feature','-limitPhysicalMemoryUsage',false);
    depthVid = videoinput('kinect', 2, 'Depth_512x424');
    set(depthVid, 'FramesPerTrigger', inf);
    %triggerconfig(depthVid, 'immediate');  % 立即錄影
    depthSrc = getselectedsource(depthVid);
    depthSrc.EnableBodyTracking = "on";
    preview(depthVid);
    pause(3);
    start(depthVid);

    %% === 設定儲存路徑與編號 ===
    output_folder = sprintf('Y(v4)/%s', name);
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    % 掃描已存在的 Y_*.json 檔案以決定從哪個編號開始
    json_pattern = fullfile(output_folder, 'Y_*.json');
    existing_files = dir(json_pattern);
    start_index = 0;
    if ~isempty(existing_files)
        indices = zeros(length(existing_files),1);
        for i = 1:length(existing_files)
            tokens = regexp(existing_files(i).name, 'Y_(\d+)\.json', 'tokens');
            if ~isempty(tokens)
                indices(i) = str2double(tokens{1});
            end
        end
        start_index = max(indices) + 1;
    end

    %% === 錄製設定 ===
    max_duration = 180;     % 總共錄多久（秒）
    chunk_duration = 60;   % 每隔幾秒存一段
    num_chunk = floor(max_duration / chunk_duration);
    fprintf('開始錄製總共 %d 秒，每 %d 秒儲存一次...\n', max_duration, chunk_duration);

    %% === 錄製主迴圈 ===
    for chunk = 1:num_chunk
        pause(chunk_duration);  % 錄一段資料
        availableFrames = depthVid.FramesAvailable;
        depthVid

        if availableFrames == 0
            fprintf('⚠ 第 %d 段無 frames 可用，略過。\n', chunk);
            continue;
        end

        try
            [~, ~, metaDataDepth] = getdata(depthVid, availableFrames);
            flushdata(depthVid);  % 清除 buffer
        catch ME
            warning('🚨 getdata 發生錯誤: %s', E.message);
            continue;
        end

        %% === 處理 frames，轉為 JSON 格式 ===
        jsonData = struct([]);
        validFrames = 1;

        for i = 1:length(metaDataDepth)
            trackedBodies = find(metaDataDepth(i).IsBodyTracked);
            if isempty(trackedBodies)
                continue;
            end
            coords = metaDataDepth(i).JointPositions(:, :, trackedBodies);
            if size(coords,1) ~= 25
                continue;
            end
            timestamp = metaDataDepth(i).AbsTime;
            dateTime = datetime(timestamp(1),timestamp(2),timestamp(3),timestamp(4),timestamp(5),timestamp(6),'TimeZone','UTC');
            unixTime = round(posixtime(dateTime), 3);

            jsonData(validFrames).time = unixTime;
            for j = 1:25
                jsonData(validFrames).coordinates(j).x = coords(j,1);
                jsonData(validFrames).coordinates(j).y = coords(j,2);
                jsonData(validFrames).coordinates(j).z = coords(j,3);
            end
            validFrames = validFrames + 1;
        end

        %% === 儲存 JSON ===
        if isempty(jsonData)
            fprintf('⚠ 第 %d 段無有效骨架，略過。\n', chunk);
            continue;
        end

        chunk_index = start_index + chunk - 1;
        jsonPath = sprintf('%s/Y_%d.json', output_folder, chunk_index);
        jsonString = jsonencode(jsonData, 'PrettyPrint', true);
        fid = fopen(jsonPath, 'w');
        fprintf(fid, '%s', jsonString);
        fclose(fid);

        fprintf('✔ 儲存 %s（%d 幀）\n', jsonPath, length(jsonData));
    end

    %% === 清除與結束 ===
    stop(depthVid);
    closepreview(depthVid);
    fprintf('✅ 所有資料儲存完成。\n');
end
