imgArr = cell(10, 1);
labelsArr = cell(10, 1);
labelNumsArr = cell(10, 1);
fsuperPixels = zeros(10, 1);
supPixels = zeros(460, 700, 10);

%PART 1
for i=1:10
    fname = strcat("hw3_data/0", string(i), ".png");
    if i == 10
        fname = strcat("hw3_data/", string(i), ".png");
    end
    image = imread(fname);
    image = imresize(image, [460, 700]);
    %[x, y, c] = size(image);
    %fprintf("x: %d", x);
    imgArr{i} = image;
end

for i = 1:10
    image = imgArr{i};
    [labels, numoflabels] = superpixels(image, 250);
    figure;
    bMask = boundarymask(labels);
    supPixels(:, :, i) = labels;
    imshow(imoverlay(image, bMask, 'red'), "InitialMagnification", 67);
    fsuperPixels(i) = numoflabels;
    labelsArr{i} = labels;
    labelNumsArr{i} = numoflabels;
end

%PART 2
wavelength = [10 15 20 25];
orientation = [0 45 90 135];
gaborBank = gabor(wavelength, orientation);
gaborRepr = [];

for i= 1:10
    image = imgArr{i};
    gray = rgb2gray(image);
    [mag, phase] = imgaborfilt(gray, gaborBank);
    gaborRes = imgaborfilt(gray, gaborBank);
    numoflabels = labelNumsArr{i};
    labels = labelsArr{i};
    figure
    subplot(4,4,1);
    x = 1;
    y = 1;
    for k = 1:16
        subplot(4,4, k)
        imshow(gaborRes(:,:,k),[]);
        wave = wavelength(x);
        ori = orientation(y);
        title(sprintf('%d, %d',wave,ori));
        y = y + 1;
        if y > 4
            y = 1;
            x = x + 1; 
        end
        
    end
    avgGabors = zeros(numoflabels, 16);
    for j = 1:numoflabels
        [x, y] = find(labels == j);
        pts = horzcat(x,y);
        for k=1:length(x)
            avgGabors(j, :) = avgGabors(j, :) + reshape(mag(pts(k, 1), pts(k, 2), :), [1, 16]);
        end
        avgGabors(j, :) = avgGabors(j, :) ./ length(x); %Gabor Average
    end
    gaborRepr = [gaborRepr; avgGabors];
end
gaborRepr = normalize(gaborRepr, "range");% Ni x 16
fprintf("num of superpix: %d\n", labelNumsArr{1})
fprintf("num of superpix: %s\n", mat2str(size(gaborRepr)))

%PART3 
clusterCount = 20;
threshold = 50;
clusterlbls = kmeans(gaborRepr, clusterCount);
finalreg = [];
for i=1:10
    if i == 1
        lbls = clusterlbls(1:fsuperPixels(1));
        repr_i = gaborRepr(1:fsuperPixels(1), :);
    elseif i == 10
        lbls = clusterlbls(sum(fsuperPixels(1:10 - 1)) : sum(fsuperPixels(1:10 - 1)) + fsuperPixels(10));
        repr_i = gaborRepr(sum(fsuperPixels(1:10 - 1)) : sum(fsuperPixels(1:10 - 1)) + fsuperPixels(10), :);  
    else
        lbls = clusterlbls(sum(fsuperPixels(1:i)) : sum(fsuperPixels(1:i)) + fsuperPixels(i));
        repr_i = gaborRepr(sum(fsuperPixels(1:i)) : sum(fsuperPixels(1:i)) + fsuperPixels(i), :);
    end
    pColorMat = zeros(460, 700);
    pixels = transpose(struct2cell(regionprops(supPixels(:, :, i), "PixelList")));
    for j = 1:length(pixels)
        [x, y] = size(pixels{j});
        for k = 1:x
            pColorMat(pixels{j}(k, 1), pixels{j}(k, 2)) = lbls(j);
        end
    end
    
    figure
    image = imgArr{i};
    imagePseudo = ind2rgb(rgb2gray(image),jet);
    pColorPseudo = label2rgb(pColorMat);
    pColorPseudo = ind2rgb(rgb2gray(pColorPseudo),jet);
    rot = imrotate(pColorPseudo, 270);
    crop = imcrop(rot, [0 0 700 460]);
    fuse = imfuse(imagePseudo, crop, 'blend');
    imshow(crop);
    imshow(fuse);
    
    figure
    rot = imrotate(label2rgb(pColorMat), 270);
    crop = imcrop(rot, [0 0 700 460]);
    fuse = imfuse(image, crop, 'blend');
    imshow(fuse);
  
    %PART 4
    st = transpose(struct2cell(regionprops(supPixels(:, :, i), "PixelList")));
    for j = 1:fsuperPixels(i)
        rPxls = st{j};
        xMin = min(rPxls(:, 1));
        xMax = max(rPxls(:, 1));
        yMin = min(rPxls(:, 2));
        yMax = max(rPxls(:, 2));
        r = pdist([xMin, yMin; xMax, yMax], "euclidean") / 2;
        center = [(xMin + xMax) / 2 (yMin + yMax) / 2];
        firstNeig= r * 1.5;
        secondNeig = r * 2.5;
        
        firstNeig_j = [];
        secondNeig_j = []; 
        
        for k = 1:fsuperPixels(i)
            if k == j
                continue
            else
                rPxls_k = st{k};
                dist= sqrt((rPxls_k(:, 1) - center(1)).^2 + (rPxls_k(:, 2) - center(2)).^2);
                firstNeigC = find(dist <= firstNeig);
                secondNeigC = find(firstNeig < dist & dist < secondNeig);
                
                if length(firstNeigC) / length(rPxls_k) >= threshold
                    firstNeig_j = [firstNeig_j; repr_i(k, :)];
                elseif length(secondNeigC) / length(rPxls_k) >= threshold
                    secondNeig_j = [secondNeig_j; repr_i(k, :)];
                end
            end
        end
        
        firstNeig_j = mean(firstNeig_j, 1);
        secondNeig_j = mean(secondNeig_j, 1);
        if isempty(firstNeig_j) == 1
            firstNeig_j = zeros(1, 16);
        end
        if isempty(secondNeig_j) == 1
            secondNeig_j = zeros(1, 16);
        end
        finalreg = [finalreg; repr_i(j, :) firstNeig_j secondNeig_j];    
        %fprintf("size: %s\n", mat2str(size(finalreg)))
    end  
end

clusLabels = kmeans(finalreg, clusterCount);
for i = 1:10
    if i == 1
        lbls = clusLabels(1:fsuperPixels(1));
    elseif i == 10
        lbls = clusLabels(sum(fsuperPixels(1:10 - 1)) : sum(fsuperPixels(1:10 - 1)) + fsuperPixels(10));
    else
        lbls = clusLabels(sum(fsuperPixels(1:i)) : sum(fsuperPixels(1:i)) + fsuperPixels(i));
    end
    
    pColorMat = zeros(700, 460);
	pixels = transpose(struct2cell(regionprops(supPixels(:, :, i), "PixelList")));
    for j = 1:length(pixels)
        [x, y] = size(pixels{j});
        for k = 1:x
            pColorMat(pixels{j}(k, 1), pixels{j}(k, 2)) = lbls(j);
        end
    end
    
    figure
    image = imgArr{i};
    imagePseudo = ind2rgb(rgb2gray(image),jet);
    pColorPseudo = label2rgb(pColorMat);
    pColorPseudo = ind2rgb(rgb2gray(pColorPseudo),jet);
    rot = imrotate(pColorPseudo, 270);
    crop = imcrop(rot, [0 0 700 460]);
    fuse = imfuse(imagePseudo, crop, 'blend');
    imshow(fuse);
    
    figure
    rot = imrotate(label2rgb(pColorMat), 270);
    crop = imcrop(rot, [0 0 700 460]);
    fuse = imfuse(image, crop, 'blend');
    imshow(fuse);
end