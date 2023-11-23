#include "linK3DExtractor.h"

using namespace cv;

namespace BoW3D
{
    LinK3D_Extractor::LinK3D_Extractor(
            int nScans_, 
            float scanPeriod_, 
            float minimumRange_, 
            float distanceTh_,           
            int matchTh_):
            nScans(nScans_), 
            scanPeriod(scanPeriod_), 
            minimumRange(minimumRange_),   
            distanceTh(distanceTh_),          
            matchTh(matchTh_)
            {
                scanNumTh = ceil(nScans / 6);
                ptNumTh = ceil(1.5 * scanNumTh);                
            }

    void LinK3D_Extractor::removeClosedPointCloud(
            const pcl::PointCloud<pcl::PointXYZ> &cloud_in,
            pcl::PointCloud<pcl::PointXYZ> &cloud_out)
    {
        if (&cloud_in != &cloud_out)
        {
            cloud_out.header = cloud_in.header;
            cloud_out.points.resize(cloud_in.points.size());
        }

        size_t j = 0;

        for (size_t i = 0; i < cloud_in.points.size(); ++i)
        {
            if (cloud_in.points[i].x * cloud_in.points[i].x 
                + cloud_in.points[i].y * cloud_in.points[i].y 
                + cloud_in.points[i].z * cloud_in.points[i].z 
                < minimumRange * minimumRange)
            {
                continue;
            }
                
            cloud_out.points[j] = cloud_in.points[i];
            j++;
        }

        if (j != cloud_in.points.size())
        {
            cloud_out.points.resize(j);
        }

        cloud_out.height = 1;
        cloud_out.width = static_cast<uint32_t>(j);
        cloud_out.is_dense = true;
    }

    void LinK3D_Extractor::extractEdgePoint(
            pcl::PointCloud<pcl::PointXYZ>::Ptr pLaserCloudIn, 
            ScanEdgePoints &edgePoints)
    {
        vector<int> scanStartInd(nScans, 0);
        vector<int> scanEndInd(nScans, 0);

        pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
        laserCloudIn = *pLaserCloudIn;
        vector<int> indices;

        pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
        removeClosedPointCloud(laserCloudIn, laserCloudIn);

        int cloudSize = laserCloudIn.points.size();
        float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
        float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y, laserCloudIn.points[cloudSize - 1].x) + 2 * M_PI;
    
        if (endOri - startOri > 3 * M_PI)
        {
            endOri -= 2 * M_PI;
        }
        else if (endOri - startOri < M_PI)
        {
            endOri += 2 * M_PI;
        }
        
        bool halfPassed = false;
        int count = cloudSize;
        pcl::PointXYZI point;
        vector<pcl::PointCloud<pcl::PointXYZI>> laserCloudScans(nScans);
        
        for (int i = 0; i < cloudSize; i++)
        {
            point.x = laserCloudIn.points[i].x;
            point.y = laserCloudIn.points[i].y;
            point.z = laserCloudIn.points[i].z;
            
            float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
            int scanID = 0;

            if (nScans == 16)
            {
                scanID = int((angle + 15) / 2 + 0.5);
                if (scanID > (nScans - 1) || scanID < 0)
                {
                    count--;
                    continue;
                }
            }
            else if (nScans == 32)
            {
                scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
                if (scanID > (nScans - 1) || scanID < 0)
                {
                    count--;
                    continue;
                }
            }
            else if (nScans == 64)
            {   
                if (angle >= -8.83)
                    scanID = int((2 - angle) * 3.0 + 0.5);
                else
                    scanID = nScans / 2 + int((-8.83 - angle) * 2.0 + 0.5);

                if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
                {
                    count--;
                    continue;
                }
            }
            else
            {
                printf("wrong scan number\n");
            }
            
            float ori = -atan2(point.y, point.x);
            if (!halfPassed)
            { 
                if (ori < startOri - M_PI / 2)
                {
                    ori += 2 * M_PI;
                }
                else if (ori > startOri + M_PI * 3 / 2)
                {
                    ori -= 2 * M_PI;
                }

                if (ori - startOri > M_PI)
                {
                    halfPassed = true;
                }
            }
            else
            {
                ori += 2 * M_PI;
                if (ori < endOri - M_PI * 3 / 2)
                {
                    ori += 2 * M_PI;
                }
                else if (ori > endOri + M_PI / 2)
                {
                    ori -= 2 * M_PI;
                }
            }

            point.intensity = ori;
            laserCloudScans[scanID].points.push_back(point);            
        }

        size_t scanSize = laserCloudScans.size();
        edgePoints.resize(scanSize);
        cloudSize = count;
                
        for(int i = 0; i < nScans; i++)
        {
            int laserCloudScansSize = laserCloudScans[i].size();
            if(laserCloudScansSize >= 15)
            {
                for(int j = 5; j < laserCloudScansSize - 5; j++)
                {
                    float diffX = laserCloudScans[i].points[j - 5].x + laserCloudScans[i].points[j - 4].x
                                + laserCloudScans[i].points[j - 3].x + laserCloudScans[i].points[j - 2].x
                                + laserCloudScans[i].points[j - 1].x - 10 * laserCloudScans[i].points[j].x
                                + laserCloudScans[i].points[j + 1].x + laserCloudScans[i].points[j + 2].x
                                + laserCloudScans[i].points[j + 3].x + laserCloudScans[i].points[j + 4].x
                                + laserCloudScans[i].points[j + 5].x;
                    float diffY = laserCloudScans[i].points[j - 5].y + laserCloudScans[i].points[j - 4].y
                                + laserCloudScans[i].points[j - 3].y + laserCloudScans[i].points[j - 2].y
                                + laserCloudScans[i].points[j - 1].y - 10 * laserCloudScans[i].points[j].y
                                + laserCloudScans[i].points[j + 1].y + laserCloudScans[i].points[j + 2].y
                                + laserCloudScans[i].points[j + 3].y + laserCloudScans[i].points[j + 4].y
                                + laserCloudScans[i].points[j + 5].y;
                    float diffZ = laserCloudScans[i].points[j - 5].z + laserCloudScans[i].points[j - 4].z
                                + laserCloudScans[i].points[j - 3].z + laserCloudScans[i].points[j - 2].z
                                + laserCloudScans[i].points[j - 1].z - 10 * laserCloudScans[i].points[j].z
                                + laserCloudScans[i].points[j + 1].z + laserCloudScans[i].points[j + 2].z
                                + laserCloudScans[i].points[j + 3].z + laserCloudScans[i].points[j + 4].z
                                + laserCloudScans[i].points[j + 5].z;

                    float curv = diffX * diffX + diffY * diffY + diffZ * diffZ;
                    if(curv > 10 && curv < 20000)
                    {
                        float ori = laserCloudScans[i].points[j].intensity;
                        float relTime = (ori - startOri) / (endOri - startOri);

                        PointXYZSCA tmpPt;
                        tmpPt.x = laserCloudScans[i].points[j].x;
                        tmpPt.y = laserCloudScans[i].points[j].y;
                        tmpPt.z = laserCloudScans[i].points[j].z;
                        tmpPt.scan_position = i + scanPeriod * relTime;
                        tmpPt.curvature = curv;
                        tmpPt.angle = ori; 
                        edgePoints[i].emplace_back(tmpPt);
                    }
                }
            }
        }            
    }    

    //Roughly divide the areas to save time for clustering.
    void LinK3D_Extractor::divideArea(ScanEdgePoints &scanCloud, ScanEdgePoints &sectorAreaCloud)
    {
        sectorAreaCloud.resize(120); //The horizontal plane is divided into 120 sector area centered on LiDAR coordinate.
        int numScansPt = scanCloud.size();
        if(numScansPt == 0)
        {
            return;
        }
        // 遍历所有线束
        for(int i = 0; i < numScansPt; i++) 
        {
            int numAScanPt = scanCloud[i].size(); // 当前一根线束上的所有边缘点
            for(int j = 0; j < numAScanPt; j++)
            {                
                int areaID = 0;
                float angle = scanCloud[i][j].angle; // 当前点的角度[单位弧度]
                
                if(angle > 0 && angle < 2 * M_PI)
                {
                    areaID = std::floor((angle / (2 * M_PI)) * 120);
                }   
                else if(angle > 2 * M_PI)
                {
                    areaID = std::floor(((angle - 2 * M_PI) / (2 * M_PI)) * 120);
                }
                else if(angle < 0)
                {
                    areaID = std::floor(((angle + 2 * M_PI) / (2 * M_PI)) * 120);
                }
                // 角度归一化到[0-1]，乘以120变成int整数来存放每个扇区的点
                sectorAreaCloud[areaID].push_back(scanCloud[i][j]);
            }
        }
    }

    // 计算一个扇区内的各个点到原点的距离之和的均值
    float LinK3D_Extractor::computeClusterMean(vector<PointXYZSCA> &cluster)
    {        
        float distSum = 0;
        int numPt = cluster.size();

        for(int i = 0; i < numPt; i++)
        {
            distSum += distXY(cluster[i]);
        }

        return (distSum/numPt);
    }
    
    // 计算一个扇区内的各个点的 x 和 y 坐标的均值
    void LinK3D_Extractor::computeXYMean(vector<PointXYZSCA> &cluster, std::pair<float, float> &xyMeans)
    {         
        int numPt = cluster.size();
        float xSum = 0;
        float ySum = 0;

        for(int i = 0; i < numPt; i++)
        {
            xSum += cluster[i].x;
            ySum += cluster[i].y;
        }

        float xMean = xSum/numPt;
        float yMean = ySum/numPt;
        xyMeans = std::make_pair(xMean, yMean);
    }

    void LinK3D_Extractor::getCluster(const ScanEdgePoints &sectorAreaCloud, ScanEdgePoints &clusters)
    {    
        ScanEdgePoints tmpclusters;
        PointXYZSCA curvPt;
        vector<PointXYZSCA> dummy(1, curvPt); 

        int numArea = sectorAreaCloud.size(); // 获取多少个扇区

        //Cluster for each sector area.
        for(int i = 0; i < numArea; i++)
        {
            if(sectorAreaCloud[i].size() < 6) // 扇区内点数太少就跳过
                continue;

            int numPt = sectorAreaCloud[i].size();  // 一个扇区内点的个数      
            ScanEdgePoints curAreaCluster(1, dummy);
            curAreaCluster[0][0] = sectorAreaCloud[i][0];

            for(int j = 1; j < numPt; j++) // 遍历扇区内的点
            {
                int numCluster = curAreaCluster.size(); // 当前计算到的扇区的个数

                for(int k = 0; k < numCluster; k++)
                {
                    float mean = computeClusterMean(curAreaCluster[k]); // 计算一个扇区内的各个点到原点的距离之和的均值
                    std::pair<float, float> xyMean;
                    computeXYMean(curAreaCluster[k], xyMean); // 计算一个扇区内的各个点的 x 和 y 坐标的均值
                    
                    PointXYZSCA tmpPt = sectorAreaCloud[i][j];
                                        
                    if(abs(distXY(tmpPt) - mean) < distanceTh  // 如果扇区内的一个点离之前算的距离均值以及x y 的均值相差都不超过阈值，那么就放入当前簇 curAreaCluster
                        && abs(xyMean.first - tmpPt.x) < distanceTh 
                        && abs(xyMean.second - tmpPt.y) < distanceTh)
                    {
                        curAreaCluster[k].emplace_back(tmpPt);
                        break;
                    }
                    else if(abs(distXY(tmpPt) - mean) >= distanceTh && k == numCluster-1) // 没看懂，当前簇增加一簇
                    {
                        curAreaCluster.emplace_back(dummy);
                        curAreaCluster[numCluster][0] = tmpPt;
                    }
                    else
                    { 
                        continue; 
                    }                    
                }
            }

            int numCluster = curAreaCluster.size();
            for(int j = 0; j < numCluster; j++)
            {
                int numPt = curAreaCluster[j].size();

                if(numPt < ptNumTh)
                {
                    continue;
                }
                tmpclusters.emplace_back(curAreaCluster[j]);
            }
        }

        int numCluster = tmpclusters.size();
        
        vector<bool> toBeMerge(numCluster, false);
        multimap<int, int> mToBeMergeInd;
        set<int> sNeedMergeInd;

        //Merge the neighbor clusters.
        for(int i = 0; i < numCluster; i++)
        {
            if(toBeMerge[i]){
                continue;
            }
            float means1 = computeClusterMean(tmpclusters[i]);
            std::pair<float, float> xyMeans1;
            computeXYMean(tmpclusters[i], xyMeans1);

            for(int j = 1; j < numCluster; j++)
            {
                if(toBeMerge[j])
                {
                    continue;
                }

                float means2 = computeClusterMean(tmpclusters[j]);
                std::pair<float, float> xyMeans2;
                computeXYMean(tmpclusters[j], xyMeans2);

                if(abs(means1 - means2) < 2*distanceTh 
                    && abs(xyMeans1.first - xyMeans2.first) < 2*distanceTh 
                    && abs(xyMeans1.second - xyMeans2.second) < 2*distanceTh)
                {
                    mToBeMergeInd.insert(std::make_pair(i, j));
                    sNeedMergeInd.insert(i);
                    toBeMerge[i] = true;
                    toBeMerge[j] = true;
                }
            }

        }

        if(sNeedMergeInd.empty())
        {
            for(int i = 0; i < numCluster; i++)
            {
                clusters.emplace_back(tmpclusters[i]);
            }
        }
        else
        {
            for(int i = 0; i < numCluster; i++)
            {
                if(toBeMerge[i] == false)
                {
                    clusters.emplace_back(tmpclusters[i]);
                }
            }
            
            for(auto setIt = sNeedMergeInd.begin(); setIt != sNeedMergeInd.end(); ++setIt)
            {
                int needMergeInd = *setIt;
                auto entries = mToBeMergeInd.count(needMergeInd);
                auto iter = mToBeMergeInd.find(needMergeInd);
                vector<int> vInd;

                while(entries)
                {
                    int ind = iter->second;
                    vInd.emplace_back(ind);
                    ++iter;
                    --entries;
                }

                clusters.emplace_back(tmpclusters[needMergeInd]);
                size_t numCluster = clusters.size();

                for(size_t j = 0; j < vInd.size(); j++)
                {
                    for(size_t ptNum = 0; ptNum < tmpclusters[vInd[j]].size(); ptNum++)
                    {
                        clusters[numCluster - 1].emplace_back(tmpclusters[vInd[j]][ptNum]);
                    }
                }
            }
        }       
    }

    void LinK3D_Extractor::computeDirection(pcl::PointXYZI ptFrom, pcl::PointXYZI ptTo, Eigen::Vector2f &direction)
    {
        direction(0, 0) = ptTo.x - ptFrom.x;
        direction(1, 0) = ptTo.y - ptFrom.y;
    }
    // 计算所有簇的质心
    vector<pcl::PointXYZI> LinK3D_Extractor::getMeanKeyPoint(const ScanEdgePoints &clusters, ScanEdgePoints &validCluster)
    {        
        int count = 0;
        int numCluster = clusters.size();
        vector<pcl::PointXYZI> keyPoints;
        vector<pcl::PointXYZI> tmpKeyPoints;
        ScanEdgePoints tmpEdgePoints;
        map<float, int> distanceOrder;

        for(int i = 0; i < numCluster; i++)
        {
            int ptCnt = clusters[i].size();      
            if(ptCnt < ptNumTh) // 一簇里面的点太少即跳过
            {
                continue;
            }

            vector<PointXYZSCA> tmpCluster;
            set<int> scans;
            float x = 0, y = 0, z = 0, intensity = 0;
            for(int ptNum = 0; ptNum < ptCnt; ptNum++)
            {
                PointXYZSCA pt = clusters[i][ptNum];          
                int scan = int(pt.scan_position); // 取整就等同于处于哪个线束
                scans.insert(scan);

                x += pt.x;
                y += pt.y;
                z += pt.z;
                intensity += pt.scan_position;
            }

            if(scans.size() < (size_t)scanNumTh) // 一簇里面分属不同线束的太少了？
            {
                continue;
            }

            pcl::PointXYZI pt; // 求均值
            pt.x = x/ptCnt;
            pt.y = y/ptCnt;
            pt.z = z/ptCnt;
            pt.intensity = intensity/ptCnt;

            float distance = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z; // 质心到原点的距离

            auto iter = distanceOrder.find(distance);
            if(iter != distanceOrder.end()) // 找到就跳过，因为是 float类型的，找到一样的概率基本没有
            {
                continue;
            }

            distanceOrder[distance] = count; 
            count++;
            
            tmpKeyPoints.emplace_back(pt);
            tmpEdgePoints.emplace_back(clusters[i]);            
        }

        for(auto iter = distanceOrder.begin(); iter != distanceOrder.end(); iter++)
        {
            int index = (*iter).second;
            pcl::PointXYZI tmpPt = tmpKeyPoints[index];
            
            keyPoints.emplace_back(tmpPt);
            validCluster.emplace_back(tmpEdgePoints[index]);
        }
                
        return keyPoints;
    }

    float LinK3D_Extractor::fRound(float in)
    {
        float f;
        int temp = std::round(in * 10);
        f = temp/10.0;
        
        return f;
    }

    void LinK3D_Extractor::getDescriptors(const vector<pcl::PointXYZI> &keyPoints, 
                                          cv::Mat &descriptors)
    {
        if(keyPoints.empty())
        {
            return;
        }

        int ptSize = keyPoints.size();

        descriptors = cv::Mat::zeros(ptSize, 180, CV_32FC1); 

        vector<vector<float>> distanceTab;
        vector<float> oneRowDis(ptSize, 0);
        distanceTab.resize(ptSize, oneRowDis);

        vector<vector<Eigen::Vector2f>> directionTab;
        Eigen::Vector2f direct(0, 0);
        vector<Eigen::Vector2f> oneRowDirect(ptSize, direct);
        directionTab.resize(ptSize, oneRowDirect);

        //Build distance and direction tables for fast descriptor generation.
        for(size_t i = 0; i < keyPoints.size(); i++)
        {
            for(size_t j = i+1; j < keyPoints.size(); j++)
            {
                float dist = distPt2Pt(keyPoints[i], keyPoints[j]);
                distanceTab[i][j] = fRound(dist);
                distanceTab[j][i] = distanceTab[i][j];

                Eigen::Vector2f tmpDirection;
                                
                tmpDirection(0, 0) = keyPoints[j].x - keyPoints[i].x;
                tmpDirection(1, 0) = keyPoints[j].y - keyPoints[i].y;

                directionTab[i][j] = tmpDirection;
                directionTab[j][i] = -tmpDirection;
            }
        }

        for(size_t i = 0; i < keyPoints.size(); i++)
        {
            vector<float> tempRow(distanceTab[i]);
            std::sort(tempRow.begin(), tempRow.end());
            int Index[3];
           
            //Get the closest three keypoints of current keypoint.
            for(int k = 0; k < 3; k++)
            {                
                vector<float>::iterator it1 = find(distanceTab[i].begin(), distanceTab[i].end(), tempRow[k+1]); 
                if(it1 == distanceTab[i].end())
                {
                    continue;
                }
                else
                {
                    Index[k] = std::distance(distanceTab[i].begin(), it1);
                }
            }

            //Generate the descriptor for each closest keypoint. 
            //The final descriptor is based on the priority of the three closest keypoint.
            for(int indNum = 0; indNum < 3; indNum++)
            {
                int index = Index[indNum]; // 获取3个最近的距离在表中的索引
                Eigen::Vector2f mainDirection;
                mainDirection = directionTab[i][index]; // 主方向，即 k0与k1距离最近的方向
                
                vector<vector<float>> areaDis(180);  
                areaDis[0].emplace_back(distanceTab[i][index]); // 主方向对应的距离
                          
                for(size_t j = 0; j < keyPoints.size(); j++)
                {
                    if(j == i || (int)j == index) // 跳过自己与3个最近点
                    {
                        continue;
                    }
                    
                    Eigen::Vector2f otherDirection = directionTab[i][j]; // 三个最近点之外的其他方向
                
                    Eigen::Matrix2f matrixDirect;
                    matrixDirect << mainDirection(0, 0), mainDirection(1, 0), otherDirection(0, 0), otherDirection(1, 0);
                    float deter = matrixDirect.determinant(); // 主方向与其他方向求行列式

                    int areaNum = 0;
                    // k0 与 k2 的夹角，弧度表示
                    double cosAng = (double)mainDirection.dot(otherDirection) / (double)(mainDirection.norm() * otherDirection.norm());                                 
                    if(abs(cosAng) - 1 > 0)
                    {   
                        continue;
                    }
                                       
                    float angle = acos(cosAng) * 180 / M_PI;
                    
                    if(angle < 0 || angle > 180)
                    {
                        continue;
                    }
                    
                    if(deter > 0) // 行列式大于0
                    {
                        areaNum = ceil((angle - 1) / 2);                         
                    }
                    else // 行列式小于0
                    {
                        if(angle - 2 < 0)
                        { 
                            areaNum = 0;
                        }
                        else
                        {
                            angle = 360 - angle;
                            areaNum = ceil((angle - 1) / 2); 
                        }   
                    }

                    if(areaNum != 0)
                    {
                        areaDis[areaNum].emplace_back(distanceTab[i][j]); // 放入每个区域的距离，第一个维度是扇区
                    }
                }
                
                float *descriptor = descriptors.ptr<float>(i); // 行指针，指向当前点描述子向量                               

                for(int areaNum = 0; areaNum < 180; areaNum++) // 遍历 180个扇区
                {
                    if(areaDis[areaNum].size() == 0) // 如果该扇区没有点
                    {
                        continue;
                    }
                    else
                    {
                        std::sort(areaDis[areaNum].begin(), areaDis[areaNum].end()); // 根据距离对该扇区进行排序

                        if(descriptor[areaNum] == 0)
                        {
                            descriptor[areaNum] = areaDis[areaNum][0]; // 该扇区的最近距离作为描述值
                        }                        
                    }
                }                
            }            
        }
    }

    void LinK3D_Extractor::match(
            vector<pcl::PointXYZI> &curAggregationKeyPt, 
            vector<pcl::PointXYZI> &toBeMatchedKeyPt,
            cv::Mat &curDescriptors, 
            cv::Mat &toBeMatchedDescriptors, 
            vector<pair<int, int>> &vMatchedIndex)
    {        
        int curKeypointNum = curAggregationKeyPt.size();
        int toBeMatchedKeyPtNum = toBeMatchedKeyPt.size();
        
        multimap<int, int> matchedIndexScore;      
        multimap<int, int> mMatchedIndex;
        set<int> sIndex;
       
        for(int i = 0; i < curKeypointNum; i++)
        {
            std::pair<int, int> highestIndexScore(0, 0);
            float* pDes1 = curDescriptors.ptr<float>(i);
            
            for(int j = 0; j < toBeMatchedKeyPtNum; j++)
            {
                int sameDimScore = 0;
                float* pDes2 = toBeMatchedDescriptors.ptr<float>(j); 
                
                for(int bitNum = 0; bitNum < 180; bitNum++)
                {                    
                    if(pDes1[bitNum] != 0 && pDes2[bitNum] != 0 && abs(pDes1[bitNum] - pDes2[bitNum]) <= 0.2){
                        sameDimScore += 1;
                    }
                    
                    if(bitNum > 90 && sameDimScore < 3){
                        break;                        
                    }                    
                }
               
                if(sameDimScore > highestIndexScore.second)
                {
                    highestIndexScore.first = j;
                    highestIndexScore.second = sameDimScore;
                }
            }
            
            //Used for removing the repeated matches.
            matchedIndexScore.insert(std::make_pair(i, highestIndexScore.second)); //Record i and its corresponding score.
            mMatchedIndex.insert(std::make_pair(highestIndexScore.first, i)); //Record the corresponding match between j and i.
            sIndex.insert(highestIndexScore.first); //Record the index that may be repeated matches.
        }

        //Remove the repeated matches.
        for(set<int>::iterator setIt = sIndex.begin(); setIt != sIndex.end(); ++setIt)
        {
            int indexJ = *setIt;
            auto entries = mMatchedIndex.count(indexJ);
            if(entries == 1)
            {
                auto iterI = mMatchedIndex.find(indexJ);
                auto iterScore = matchedIndexScore.find(iterI->second);
                if(iterScore->second >= matchTh)
                {                    
                    vMatchedIndex.emplace_back(std::make_pair(iterI->second, indexJ));
                }           
            }
            else
            { 
                auto iter1 = mMatchedIndex.find(indexJ);
                int highestScore = 0;
                int highestScoreIndex = -1;

                while(entries)
                {
                    int indexI = iter1->second;
                    auto iterScore = matchedIndexScore.find(indexI);
                    if(iterScore->second > highestScore){
                        highestScore = iterScore->second;
                        highestScoreIndex = indexI;
                    }                
                    ++iter1;
                    --entries;
                }

                if(highestScore >= matchTh)
                {                                       
                    vMatchedIndex.emplace_back(std::make_pair(highestScoreIndex, indexJ));                    
                }            
            }
        }
    }

    //Remove the edge keypoints with low curvature for further edge keypoints matching.
    void LinK3D_Extractor::filterLowCurv(ScanEdgePoints &clusters, ScanEdgePoints &filtered)
    {
        int numCluster = clusters.size();
        filtered.resize(numCluster);
        for(int i = 0; i < numCluster; i++)
        {
            int numPt = clusters[i].size();
            ScanEdgePoints tmpCluster;
            vector<int> vScanID;

            for(int j = 0; j < numPt; j++)
            {
                PointXYZSCA pt = clusters[i][j];
                int scan = int(pt.scan_position);
                auto it = std::find(vScanID.begin(), vScanID.end(), scan);

                if(it == vScanID.end())
                {
                    vScanID.emplace_back(scan);
                    vector<PointXYZSCA> vPt(1, pt);
                    tmpCluster.emplace_back(vPt);
                }
                else
                {
                    int filteredInd = std::distance(vScanID.begin(), it);
                    tmpCluster[filteredInd].emplace_back(pt);
                }
            }

            for(size_t scanID = 0; scanID < tmpCluster.size(); scanID++)
            {
                if(tmpCluster[scanID].size() == 1)
                {
                    filtered[i].emplace_back(tmpCluster[scanID][0]);
                }
                else
                {
                    float maxCurv = 0;
                    PointXYZSCA maxCurvPt;
                    for(size_t j = 0; j < tmpCluster[scanID].size(); j++)
                    {
                        if(tmpCluster[scanID][j].curvature > maxCurv)
                        {
                            maxCurv = tmpCluster[scanID][j].curvature;
                            maxCurvPt = tmpCluster[scanID][j];
                        }
                    }

                    filtered[i].emplace_back(maxCurvPt);
                }
            }  
        }
    }

    //Get the edge keypoint matches based on the matching results of aggregation keypoints.
    void LinK3D_Extractor::findEdgeKeypointMatch(
            ScanEdgePoints &filtered1, 
            ScanEdgePoints &filtered2, 
            vector<std::pair<int, int>> &vMatched, 
            vector<std::pair<PointXYZSCA, PointXYZSCA>> &matchPoints)
    {
        int numMatched = vMatched.size();
        for(int i = 0; i < numMatched; i++)
        {
            pair<int, int> matchedInd = vMatched[i];
                        
            int numPt1 = filtered1[matchedInd.first].size();
            int numPt2 = filtered2[matchedInd.second].size();

            map<int, int> mScanID_Index1;
            map<int, int> mScanID_Index2;

            for(int i = 0; i < numPt1; i++)
            {
                int scanID1 = int(filtered1[matchedInd.first][i].scan_position);
                pair<int, int> scanID_Ind(scanID1, i);
                mScanID_Index1.insert(scanID_Ind);
            }

            for(int i = 0; i < numPt2; i++)
            {
                int scanID2 = int(filtered2[matchedInd.second][i].scan_position);
                pair<int, int> scanID_Ind(scanID2, i);
                mScanID_Index2.insert(scanID_Ind);
            }

            for(auto it1 = mScanID_Index1.begin(); it1 != mScanID_Index1.end(); it1++)
            {
                int scanID1 = (*it1).first;
                auto it2 = mScanID_Index2.find(scanID1);
                if(it2 == mScanID_Index2.end()){
                    continue;
                }
                else
                {
                    vector<PointXYZSCA> tmpMatchPt;
                    PointXYZSCA pt1 = filtered1[matchedInd.first][(*it1).second];
                    PointXYZSCA pt2 = filtered2[matchedInd.second][(*it2).second];
                    
                    pair<PointXYZSCA, PointXYZSCA> matchPt(pt1, pt2);
                    matchPoints.emplace_back(matchPt);
                }
            }
        }
    }

    void LinK3D_Extractor::CeresICP(vector<pcl::PointXYZI> &keyPoints, vector<pcl::PointXYZI> &keyPoints_last, 
                                    vector<pair<int, int>> &index_match, ros::Time &timestamp_ros,
                                    Eigen::Quaterniond &q_w_curr, Eigen::Vector3d &t_w_curr) 
    {
        vector<cv::Point3f> points_cur, points_last;
        for (size_t i = 0; i < index_match.size(); i++)
        {
            cv::Point3f point_cur, point_last;
            point_cur.x = keyPoints[index_match[i].first].x;
            point_cur.y = keyPoints[index_match[i].first].y;
            point_cur.z = keyPoints[index_match[i].first].z;
            point_last.x = keyPoints[index_match[i].second].x;
            point_last.y = keyPoints[index_match[i].second].y;
            point_last.z = keyPoints[index_match[i].second].z;
            points_cur.emplace_back(point_cur);
            points_last.emplace_back(point_last);
        }

        static double T_currToLast[6] = {0,0,0,0,0,0};

        // static Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
        // static Eigen::Vector3d t_w_curr(0, 0, 0);

        Eigen::Quaterniond q_last_curr;
        Eigen::Vector3d t_last_curr;

        ceres::Problem problem;
        for (size_t i = 0; i < points_cur.size(); i++) {
            // ceres::CostFunction *cost_function = ICPCeres::Create(points_last[i],points_cur[i]);
            ceres::CostFunction *cost_function = ICPCeres::Create(points_cur[i],points_last[i]);
            ceres::LossFunction *lost_function = new ceres::HuberLoss(0.1);
            problem.AddResidualBlock(cost_function,lost_function,T_currToLast);
        }
        
        ceres::Solver::Options options;
        options.max_num_iterations = 4;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);


        Mat R_vec = (Mat_<double>(3,1) << T_currToLast[0], T_currToLast[1], T_currToLast[2]);
        Mat R_cvest;
        // 罗德里格斯公式，旋转向量转旋转矩阵
        cv::Rodrigues(R_vec, R_cvest);
        Eigen::Matrix<double,3,3> R_est;
        cv::cv2eigen(R_cvest, R_est);
        Eigen::Quaterniond q(R_est.inverse());
        q.normalize();
        q_last_curr = q;
        //cout << "q = \n" << q.x() << " " << q.y() << " " << q.z() << " " << q.w()<< endl;
        //cout << -T_currToLast[3] << " " <<  -T_currToLast[4] << " " << -T_currToLast[5] << endl;
        //cout<<"R_est="<<R_est<<endl;
        Eigen::Vector3d t_est(T_currToLast[3], T_currToLast[4], T_currToLast[5]);
        t_last_curr = -t_est;
        cout<<" t_est ======================= " << t_est <<endl;
        Eigen::Isometry3d T(R_est);//构造变换矩阵与输出
        T.pretranslate(t_est);
        //cout << "T = \n" << T.matrix().inverse()<<endl;
        t_w_curr = t_w_curr + q_w_curr * t_last_curr;
        q_w_curr = q_w_curr * q_last_curr;
    
    }

    void LinK3D_Extractor::process(pcl::PointCloud<pcl::PointXYZ>::Ptr pLaserCloudIn, 
                                   vector<pcl::PointXYZI> &keyPoints, cv::Mat &descriptors, 
                                   ScanEdgePoints &validCluster, ros::Time &timestamp_ros,
                                   Eigen::Quaterniond &q_w_curr, Eigen::Vector3d &t_w_curr)
    {
        ScanEdgePoints edgePoints;
        // 1. 提取当前帧的边缘点。根据线束存储边缘点
        extractEdgePoint(pLaserCloudIn, edgePoints); // edgePoints 的第一维表示的是一个Scan，第二维表示的是同一个Scan上的边缘点
        // 2.1 输入边缘点，输出3D扇形区域点，根据扇区存储边缘点
        ScanEdgePoints sectorAreaCloud;
        divideArea(edgePoints, sectorAreaCloud); // sectorAreaCloud 第一维表示的是哪个扇区，第二维是扇区内的数据
        // 2.2 输入扇形区域点，输出聚合点，大容器：所有簇，小容器：一簇的所有点
        ScanEdgePoints clusters;
        getCluster(sectorAreaCloud, clusters);  
        // 2.3 计算所有簇的质心
        vector<int> index;
        keyPoints = getMeanKeyPoint(clusters, validCluster);
        // 3. 创建描述子
        getDescriptors(keyPoints, descriptors);

        if(keyPoints_last_.size()) {
            vector<pair<int, int>> index_match;
            index_match.clear();
            match(keyPoints, keyPoints_last_, descriptors, descriptors_last_, index_match);

            CeresICP(keyPoints, keyPoints_last_, index_match, timestamp_ros, q_w_curr, t_w_curr);
            cout<<" index_match.size ======================= " << index_match.size() <<endl;
        }

        keyPoints_last_.assign(keyPoints.begin(),keyPoints.end());
        descriptors_last_ = descriptors;
       
    }

}
