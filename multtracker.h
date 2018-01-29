#ifndef MULTTRACKER_H
#define MULTTRACKER_H
#include <opencv2/opencv.hpp>
#include <time.h>
#include <iostream>
#include<fstream>
#include <array>

#define ALPHA_COEFFICIENT      0.2     /* 目标模型更新权重*/
//cv1
#define B(image,x,y) ((uchar*)(image->imageData + image->widthStep*(y)))[(x)*3]		//B
#define G(image,x,y) ((uchar*)(image->imageData + image->widthStep*(y)))[(x)*3+1]	//G
#define R(image,x,y) ((uchar*)(image->imageData + image->widthStep*(y)))[(x)*3+2]	//R
//cv2
#define MB(image,i,j) image.at<cv::Vec3b>(j,i)[0]
#define MG(image,i,j) image.at<cv::Vec3b>(j,i)[1]
#define MR(image,i,j) image.at<cv::Vec3b>(j,i)[2]
# define R_BIN      8  /* 红色分量的直方图条数 */
# define G_BIN      8  /* 绿色分量的直方图条数 */
# define B_BIN      8  /* 兰色分量的直方图条数 */

# define R_SHIFT    5  /* 与上述直方图条数对应 */
# define G_SHIFT    5  /* 的R、G、B分量左移位数 */
# define B_SHIFT    5  /* log2( 256/8 )为移动位数 */

#define MAX_WEIGHT 0.001
static float Pi_Thres = (float)0.90; /* 权重阈值   */
typedef struct __SpaceState {  /* 状态空间变量 */
    int xt;               /* x坐标位置 */
    int yt;               /* y坐标位置 */
    float v_xt;           /* x方向运动速度 */
    float v_yt;           /* y方向运动速度 */
    int Hxt;              /* x方向半窗宽 */
    int Hyt;              /* y方向半窗宽 */
    float at_dot;         /* 尺度变换速度，粒子所代表的那一片区域的尺度变化速度 */
} SPACESTATE;

typedef struct __UpdataLocation{
    int xc;
    int yc;
    int Wx_h;
    int Hy_h;
    float max_weight;
}UpdataLocation;

typedef struct __SrcFace{
    cv::Mat src;
    cv::Mat face;
    cv::Rect rect;//脸的位置
    bool is_disappear;//跟踪丢失标志
}SrcFace;
//int xin,yin;//跟踪时输入的中心点
//int xout,yout;//跟踪时得到的输出中心点
//int Wid,Hei;//图像的大小
//int WidIn,HeiIn;//输入的半宽与半高
//int WidOut,HeiOut;//输出的半宽与半高

static float DELTA_T = (float)0.05;    /* 帧频，可以为30，25，15，10等 */
static float VELOCITY_DISTURB = 40.0;  /* 速度扰动幅值   */
static float SCALE_DISTURB = 0.0;      /* 窗宽高扰动幅度 */
static float SCALE_CHANGE_D = (float)0.001;   /* 尺度变换速度扰动幅度 */

static const int NParticle = 100;
# define SIGMA2       0.02
class MultTracker
{
public:
    MultTracker();
    void ClearAll()
    {
        if(this->faceModelHist_.size()){
            for(int i = 0;i < this->faceModelHist_.size();i++){
                this->faceModelHist_[i].clear();
            }
            this->faceModelHist_.clear();
        }
        if(this->faceSpaceStates_.size()){
            for(int i = 0;i < this->faceSpaceStates_.size();i++){
                this->faceSpaceStates_[i].clear();
            }
            this->faceSpaceStates_.clear();
        }
        if(this->faceWeights_.size()){
            for(int i = 0;i < this->faceWeights_.size();i++){
                this->faceWeights_[i].clear();
            }
            this->faceWeights_.clear();
        }
        if(this->img_ != NULL) delete [] img_;
        return;
    }
    float computeOverLap(cv::Rect& rect1, cv::Rect& rect2);
    void CalcuColorHistogram( int x0, int y0, int Wx, int Hy,
                             unsigned char * image, int W, int H,
                             float * ColorHist, int bins );
    /*
    计算Bhattacharyya系数
    输入参数：
    float * p, * q：      两个彩色直方图密度估计
    int bins：            直方图条数
    返回值：
    Bhattacharyya系数
    */
    float CalcuBhattacharyya( float * p, float * q, int bins )
    {
        int i;
        float rho;

        rho = 0.0;
        for ( i = 0; i < bins; i++ )
            rho = (float)(rho + sqrt( p[i]*q[i] ));
        return( rho );
    }

    float CalcuWeightedPi( float rho )
    {
        float pi_n, d2;
        d2 = 1 - rho;
        pi_n = (float)(exp( - d2/SIGMA2 ));
        return( pi_n );
    }
    /*
    获得一个[0,1]之间的随机数
    */
    inline float rand0_1()
    {
        return(rand()/float(RAND_MAX));
    }
    /*
    获得一个x - N(u,sigma)Gaussian分布的随机数
    */
    float randGaussian( float u, float sigma );
    /*
    计算归一化累计概率c'_i
    输入参数：
    float * weight：    为一个有N个权重（概率）的数组
    int N：             数组元素个数
    输出参数：
    float * cumulateWeight： 为一个有N+1个累计权重的数组，
    cumulateWeight[0] = 0;
    */
    void NormalizeCumulatedWeight( float * weight, float * cumulateWeight, int N )
    {
        int i;

        for ( i = 0; i < N+1; i++ )
            cumulateWeight[i] = 0;
        for ( i = 0; i < N; i++ )
            cumulateWeight[i+1] = cumulateWeight[i] + weight[i];
        for ( i = 0; i < N+1; i++ )
            cumulateWeight[i] = cumulateWeight[i]/ cumulateWeight[N];

        return;
    }

    /*
    折半查找，在数组NCumuWeight[N]中寻找一个最小的j，使得
    NCumuWeight[j] <=v
    float v：              一个给定的随机数
    float * NCumuWeight：  权重数组
    int N：                数组维数
    返回值：
    数组下标序号
    */
    int BinearySearch( float v, float * NCumuWeight, int N )
    {
        int l, r, m;

        l = 0; 	r = N-1;   /* extreme left and extreme right components' indexes */
        while ( r >= l)
        {
            m = (l+r)/2;
            if ( v >= NCumuWeight[m] && v < NCumuWeight[m+1] ) return( m );
            if ( v < NCumuWeight[m] ) r = m - 1;
            else l = m + 1;
        }
        return( 0 );
    }

    /*
    重新进行重要性采样
    输入参数：
    float * c：          对应样本权重数组pi(n)
    int N：              权重数组、重采样索引数组元素个数
    输出参数：
    int * ResampleIndex：重采样索引数组
    */
    void ImportanceSampling( float * c, int * ResampleIndex, int N );

    /*
    样本选择，从N个输入样本中根据权重重新挑选出N个
    输入参数：
    SPACESTATE * state：     原始样本集合（共N个）
    float * weight：         N个原始样本对应的权重
    int N：                  样本个数
    输出参数：
    SPACESTATE * state：     更新过的样本集
    */
    void ReSelect( SPACESTATE * state, float * weight, int N );

    /*
    传播：根据系统状态方程求取状态预测量
    状态方程为： S(t) = A S(t-1) + W(t-1)
    W(t-1)为高斯噪声
    输入参数：
    SPACESTATE * state：      待求的状态量数组
    int N：                   待求状态个数
    输出参数：
    SPACESTATE * state：      更新后的预测状态量数组
    */
    void Propagate( SPACESTATE * state, int N,cv::Mat& trackImg );

    /*
    观测，根据状态集合St中的每一个采样，观测直方图，然后
    更新估计量，获得新的权重概率
    输入参数：
    SPACESTATE * state：      状态量数组
    int N：                   状态量数组维数
    unsigned char * image：   图像数据，按从左至右，从上至下的顺序扫描，
    颜色排列次序：RGB, RGB, ...
    int W, H：                图像的宽和高
    float * ObjectHist：      目标直方图
    int hbins：               目标直方图条数
    输出参数：
    float * weight：          更新后的权重
    */
    void Observe( SPACESTATE * state, float * weight, int N,
                 unsigned char * image, int W, int H,
                 float * ObjectHist, int hbins );

    /*
    估计，根据权重，估计一个状态量作为跟踪输出
    输入参数：
    SPACESTATE * state：      状态量数组
    float * weight：          对应权重
    int N：                   状态量数组维数
    输出参数：
    SPACESTATE * EstState：   估计出的状态量
    */
    void Estimation( SPACESTATE * state, float * weight, int N,
                    SPACESTATE & EstState );
    /*
    模型更新
    输入参数：
    SPACESTATE EstState：   状态量的估计值
    float * TargetHist：    目标直方图
    int bins：              直方图条数
    float PiT：             阈值（权重阈值）
    unsigned char * img：   图像数据，RGB形式
    int W, H：              图像宽高
    输出：
    float * TargetHist：    更新的目标直方图
    ************************************************************/
    void ModelUpdate( SPACESTATE EstState, float * TargetHist, int bins, float PiT,
                     unsigned char * img, int W, int H );
    int ColorParticleTracking( unsigned char * image, int W, int H,
                              std::vector<UpdataLocation>&faceloca ,cv::Mat& trackImg);

    void MatToImge(cv::Mat & mat,int w,int h){
        int i,j;
        for(j = 0;j < h;j++)
            for(i = 0;i < w; i++){
                img_[(j*w + i)*3] = MR(mat,i,j);
                img_[(j*w + i)*3 + 1] = MG(mat,i,j);
                img_[(j*w + i)*3 + 2] = MB(mat,i,j);
            }
    }

/*
 * 输入：当前帧图像
 * 输入：当前帧检测到的目标列表
 * 输出：目标位置 大小
 * 输出：更新需要保存的人脸列表
*/
    void tracking(cv::Mat& img,std::vector<cv::Rect>&DetectedFaces);
    //如果判断是新脸，则加入到跟踪列表中，如果判断是正在跟踪的脸，则将当前跟踪的脸位置矫正一下，根据脸大小确定是否替换当前脸。
    bool isNewFace(cv::Rect& rect,int &index);

    /*
    初始化系统
    int x0, y0：        初始给定的图像目标区域坐标
    int Wx, Hy：        目标的半宽高
    unsigned char * img：图像数据，RGB形式
    int W, H：          图像宽高
    */
    int Initialize( int x0, int y0, int Wx, int Hy,
                   unsigned char * img, int W, int H,SPACESTATE*states,float* weights,float*ModelHist );
    int Initialize( cv::Rect &rect,
                   unsigned char * img, int W, int H ,SPACESTATE*states,float* weights,float*ModelHist);
    void newfacePrepare();

    /*
     * 对正在跟踪的疑似重复的目标去重操作。
    */
    void removeTwiceTrackingTarget();

    /*
     * 清除跟丢的跟踪目标
    */
    void ClearDisapperedTarget();
public:
    std::vector<SrcFace> trackedFaces_;
    int face_nums;//当前跟踪的人脸数量
    //每一张脸的spaceState都是一个数组，可能有N个脸
    std::vector<std::vector<SPACESTATE> > faceSpaceStates_;
    std::vector<std::vector<float> > faceWeights_;
    std::vector<std::vector<float> > faceModelHist_;//人脸区域直方图列表
    std::vector<UpdataLocation> faceLocations_;

    unsigned char * img_;
    bool start_ = false;
    int Wid;
    int Hei;

};

#endif // MULTTRACKER_H


















