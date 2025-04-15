function collect_v2(varargin)
    if nargin == 0
        user_input = input('請輸入名字: ', 's');
        name = strtrim(user_input);
    else
        name = varargin{1};
    end

    imaqreset;
    imaqmex('feature','-limitPhysicalMemoryUsage',false);

    depthVid   = videoinput('kinect', 2, 'Depth_512x424');
    set(depthVid, 'FramesPerTrigger', inf);
    depthSrc = getselectedsource(depthVid);
    depthSrc.EnableBodyTracking = "on";
    preview(depthVid);
    start(depthVid);
    pause(2); % 等待啟動穩定

    output_folder = sprintf('Y(v4)/%s', name);
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    max_duration = 180; % 最長錄製秒數
    chunk_duration = 60; % 每隔幾秒儲存一次
    num_chunk = floor(max_duration / chunk_duration);
    fprintf('開始錄製總共 %d 秒，每 %d 秒儲存一次...\n', max_duration, chunk_duration);

    tic;
    for chunk = 1:num_chunk
        pause(chunk_duration);  % 錄一段資料
        depthVid
        
        availableFrames = depthVid.FramesAvailable;
        if availableFrames == 0
            fprintf('⚠ 第 %d 段無 frames 可用，略過。\n', chunk);
            continue;
        end
    
        try
            [~, ~, metaDataDepth] = getdata(depthVid, availableFrames);
        catch ME
            warning('🚨 getdata 發生錯誤: %s', E.message);
            continue;
        end
        
        flushdata(depthVid);  % 清除 buffer，避免記憶體堆積
    
        % 後續 JSON 儲存處理略...

        jsonData = struct([]);
        validFrames = 1;

        for i = 1:length(metaDataDepth)
            trackedBodies = find(metaDataDepth(i).IsBodyTracked);
            if ~isempty(trackedBodies)
                timestamp = metaDataDepth(i).AbsTime;
                coords = metaDataDepth(i).JointPositions(:, :, trackedBodies);
                if size(coords,1) ~= 25
                    continue;
                end
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
        end

        if isempty(jsonData)
            fprintf('第 %d 段未偵測到骨架，略過。\n', chunk);
            continue;
        end

        jsonString = jsonencode(jsonData, 'PrettyPrint', true);
        jsonPath = sprintf('%s/Y_%d.json', output_folder, chunk);
        fid = fopen(jsonPath, 'w');
        fprintf(fid, '%s', jsonString);
        fclose(fid);
        fprintf('✔ 儲存 %s\n', jsonPath);
    end

    stop(depthVid);
    closepreview(depthVid);
    fprintf('✅ 所有資料儲存完成。\n');
end
