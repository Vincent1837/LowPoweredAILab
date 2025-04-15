function collect_v2(varargin)
    % 如果沒有輸入參數，則從命令列輸入
    if nargin == 0
        user_input = input('請輸入名字和數字（以空格分隔）: ', 's');
        tokens = strsplit(user_input);  % 以空格分割輸入
    else
        tokens = varargin;  % 使用命令列參數
    end
    % 確保輸入包含兩個參數
    if length(tokens) ~= 2
        error("請輸入正確的格式，例如: collect('kenny', '1')");
    end
    name = tokens{1};  % 第一個值為 name
    number = str2double(tokens{2});  % 第二個值為 number，轉換為數字
    if isnan(number)
        error('Number must be a valid integer.');
    end
    jsonFiles = sprintf('Y(v4)/%s/Y_%d.json', name, number);
  

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 初始化 Kinect 的 RGB 和深度數據輸入
    imaqmex('feature','-limitPhysicalMemoryUsage',false);

    depthVid   = videoinput('kinect', 2, 'Depth_512x424');
    set(depthVid, 'FramesPerTrigger', inf);
    % depthVid.PreviewFullBitDepth = 'on';
    preview(depthVid); % 在影像視窗內顯示預覽
    % Get the VIDEOSOURCE object from the depth device's VIDEOINPUT object.
    depthSrc = getselectedsource(depthVid);
    depthSrc.EnableBodyTracking = "on";
    depthSrc

    % start logging of acquired data.

    start(depthVid);
    
    %input('欲暫停時，任意輸入(without ^C): ','s')
    
    %%
     % Retrieve the frames
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    pause(180);  % 要錄製幾秒的資料
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [~, ~, metaDataDepth] = getdata(depthVid, depthVid.FramesAvailable);
    metaDataDepth(10)
    
    % 停止捕獲
    stop(depthVid);
    closepreview(depthVid);
    close all
    
    % 從metaDataDepth取出有效的frames及其對應之AbsTime
    frame = {};
    validFrames = 1;
    
    for i = 1:size(metaDataDepth)
        % See which skeletons were tracked.
        trackedBodies = find(metaDataDepth(i).IsBodyTracked);
        
        % Skeleton's joint indices with respect to the color image
        if ~isempty(trackedBodies)  % 確保有骨架數據被追蹤
            frame{validFrames} = {metaDataDepth(validFrames).AbsTime, metaDataDepth(i).JointPositions(:, :, trackedBodies)};
            validFrames = validFrames + 1;
        end
    end
    
    if isempty(frame)
        error('kinect 無法識別pcl，故無法建立 %s\n',jsonFiles);
    end
    
    % 初始化一個空結構數組來存放 JSON 結構
    jsonData = struct([]);
    
    % 遍歷 frame 中的每個幀數據，提取 time 值並轉換為 datetime
    for i = 1:length(frame)
    
        timestamp = frame{i}{1};
    
        coordinates = frame{i}{2};  % 這是一個 25x3 的矩陣
    
        if ~isequal(size(coordinates) , [25,3])
            error('存在frame不到25個key-point')
        end
    
        dateTime = datetime(timestamp(1), timestamp(2), timestamp(3), timestamp(4), timestamp(5), timestamp(6), 'TimeZone', 'UTC');
    
        % 將 datetime 轉換為 Unix 時間戳，並四捨五入到小數點後三位
        unixTime = round(posixtime(dateTime), 3);
        jsonData(i).time = unixTime;
        
        for j = 1:length(coordinates)
            jsonData(i).coordinates(j).x = coordinates(j,1);
            jsonData(i).coordinates(j).y = coordinates(j,2);
            jsonData(i).coordinates(j).z = coordinates(j,3);
        end
    end
    
    % 將結構數據編碼為 JSON 字符串
    jsonString = jsonencode(jsonData,"PrettyPrint",true);
    output_folder = sprintf('Y(v4)/%s', name);  % 目標資料夾
    if ~exist(output_folder, 'dir')  % 如果資料夾不存在
        mkdir(output_folder);  % 創建資料夾
    end
    
    % 保存到 JSON 文件
    fileID = fopen(jsonFiles, 'w');
    fprintf(fileID, '%s', jsonString);
    fclose(fileID);
    fprintf('File %s 已建立完成。\n', jsonFiles);