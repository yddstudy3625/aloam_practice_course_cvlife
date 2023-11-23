#pragma once

#include <vector>
#include <map>
#include <set>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>

#include <eigen3/Eigen/Dense>
#include <Eigen/Core>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#include <math.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace std;
using namespace cv;

struct ICPCeres
{
    ICPCeres(Point3f uvw, Point3f xyz) : _uvw(uvw), _xyz(xyz) {}
    // 残差的计算
    template <typename T>
    bool operator()(
        const T *const camera, // 模型参数，有4维
        T *residual) const     // 残差
    {
        T p[3];
        T point[3];
        point[0] = T(_xyz.x);
        point[1] = T(_xyz.y);
        point[2] = T(_xyz.z);
        ceres::AngleAxisRotatePoint(camera, point, p); //计算RP
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5]; //相机坐标2
        residual[0] = T(_uvw.x) - p[0];
        residual[1] = T(_uvw.y) - p[1];
        residual[2] = T(_uvw.z) - p[2];
        return true;
    }
    static ceres::CostFunction *Create(const Point3f uvw, const Point3f xyz)
    {
        return (new ceres::AutoDiffCostFunction<ICPCeres, 3, 6>(
            new ICPCeres(uvw, xyz)));
    }
    const Point3f _uvw;
    const Point3f _xyz;
};

struct PointXYZSCA
{
    PCL_ADD_POINT4D;
    float scan_position;
    float curvature;
    float angle;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZSCA,
                                  (float, x, x)(float, y, y)(float, z, z)(float, scan_position, scan_position)(float, curvature, curvature)(float, angle, angle))

typedef vector<vector<PointXYZSCA>> ScanEdgePoints;

namespace BoW3D
{
#define EdgePointCloud pcl::PointCloud<PointXYZSCA>
#define distXY(a) sqrt(a.x *a.x + a.y * a.y)
#define distOri2Pt(a) sqrt(a.x *a.x + a.y * a.y + a.z * a.z)
#define distPt2Pt(a, b) sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z))

    using std::atan2;
    using std::cos;
    using std::sin;

    class Frame;

    class LinK3D_Extractor
    {
    public:
        LinK3D_Extractor(int nScans_, float scanPeriod_, float minimumRange_, float distanceTh_, int matchTh_);

        ~LinK3D_Extractor() {}

        bool comp(int i, int j)
        {
            return cloudCurvature[i] < cloudCurvature[j];
        }

        void removeClosedPointCloud(const pcl::PointCloud<pcl::PointXYZ> &cloud_in,
                                    pcl::PointCloud<pcl::PointXYZ> &cloud_out);

        void extractEdgePoint(pcl::PointCloud<pcl::PointXYZ>::Ptr pLaserCloudIn, ScanEdgePoints &edgePoints);

        void divideArea(ScanEdgePoints &scanEdgePoints, ScanEdgePoints &sectorAreaCloud);

        float computeClusterMean(vector<PointXYZSCA> &cluster);

        void computeXYMean(vector<PointXYZSCA> &cluster, pair<float, float> &xyMeans);

        void getCluster(const ScanEdgePoints &sectorAreaCloud, ScanEdgePoints &clusters);

        void computeDirection(pcl::PointXYZI ptFrom,
                              pcl::PointXYZI ptTo,
                              Eigen::Vector2f &direction);

        vector<pcl::PointXYZI> getMeanKeyPoint(const ScanEdgePoints &clusters,
                                               ScanEdgePoints &validCluster);

        float fRound(float in);

        void getDescriptors(const vector<pcl::PointXYZI> &keyPoints, cv::Mat &descriptors);

        void match(vector<pcl::PointXYZI> &curAggregationKeyPt,
                   vector<pcl::PointXYZI> &toBeMatchedKeyPt,
                   cv::Mat &curDescriptors,
                   cv::Mat &toBeMatchedDescriptors,
                   vector<pair<int, int>> &vMatchedIndex);

        void filterLowCurv(ScanEdgePoints &clusters, ScanEdgePoints &filtered);

        void findEdgeKeypointMatch(ScanEdgePoints &filtered1,
                                   ScanEdgePoints &filtered2,
                                   vector<pair<int, int>> &vMatched,
                                   vector<pair<PointXYZSCA, PointXYZSCA>> &matchPoints);

        void process(pcl::PointCloud<pcl::PointXYZ>::Ptr pLaserCloudIn,
                     vector<pcl::PointXYZI> &keyPoints,
                     cv::Mat &descriptors,
                     ScanEdgePoints &validCluster, ros::Time &timestamp_ros,
                     Eigen::Quaterniond &q_w_curr, Eigen::Vector3d &t_w_curr);

        void CeresICP(vector<pcl::PointXYZI> &keyPoints, vector<pcl::PointXYZI> &keyPoints_last,
                      vector<pair<int, int>> &index_match, ros::Time &timestamp_ros,
                      Eigen::Quaterniond &q_w_curr, Eigen::Vector3d &t_w_curr);

    private:
        int nScans;
        float scanPeriod;
        float minimumRange;

        float distanceTh;
        int matchTh;
        int scanNumTh;
        int ptNumTh;

        vector<pcl::PointXYZI> keyPoints_last_;
        cv::Mat descriptors_last_;

        float cloudCurvature[400000];
    };
}
