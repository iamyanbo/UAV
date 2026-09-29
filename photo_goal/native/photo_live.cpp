// Live RGB bridge for the pinned Photo-SLAM source. No directory replay/viewer.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include "ORB-SLAM3/include/System.h"
#include "ORB-SLAM3/include/MapPoint.h"
#include "ORB-SLAM3/include/KeyFrame.h"
#include "include/gaussian_mapper.h"
#include <algorithm>
#include <mutex>
namespace py = pybind11;

class LiveSession {
    std::shared_ptr<ORB_SLAM3::System> slam;
    std::shared_ptr<GaussianMapper> mapper;
    std::string config;
    std::mutex mutex;
    unsigned long revision=0;
    bool capped=false;
    bool mapper_dirty=false;
public:
    LiveSession(std::string vocabulary,std::string tracking,std::string mapping): config(mapping) {
        slam=std::make_shared<ORB_SLAM3::System>(vocabulary,tracking,ORB_SLAM3::System::MONOCULAR);
        mapper=std::make_shared<GaussianMapper>(slam,config,"",0,torch::kCUDA);
        mapper->setKeepTraining(false);
        mapper->setDoInactiveGeoDensify(false);
        if(slam->GetImageScale()!=1.f) throw std::runtime_error("Live camera requires image scale 1");
    }
    py::dict track(py::array_t<uint8_t,py::array::c_style> input,double stamp,long frame,bool mapping_enabled) {
        if(input.ndim()!=3 || input.shape(0)!=480 || input.shape(1)!=640 || input.shape(2)!=3)
            throw std::runtime_error("Expected 480x640 RGB");
        std::lock_guard<std::mutex> lock(mutex);
        cv::Mat image(480,640,CV_8UC3,const_cast<uint8_t*>(input.data()));
        Sophus::SE3f pose;
        {
            py::gil_scoped_release release;
            // A bounded local atlas. Explicit reset invalidates every old target
            // and scale fit, while Python retains the last immutable snapshot.
            auto atlas=slam->getAtlas();
            if(!mapping_enabled)atlas->clearMappingOperation();
            if(atlas->GetAllKeyFrames().size()>=128 || atlas->GetAllMapPoints().size()>=50000 || capped) {
                slam->Reset(); ++revision;
                mapper_dirty=true;capped=false;
            }
            pose=slam->TrackMonocular(image,stamp,{},std::to_string(frame));
            if(!mapping_enabled)atlas->clearMappingOperation();
        }
        if(slam->MapChanged()) ++revision;
        py::dict result;result["revision"]=revision;
        result["tracking"]=slam->GetTrackingState()==2;
        std::vector<float> transform;auto matrix=pose.inverse().matrix();
        for(int i=0;i<4;++i)for(int j=0;j<4;++j)transform.push_back(matrix(i,j));
        result["pose"]=transform;
        py::list points,keyframes,depth_samples;
        auto map=slam->getAtlas()->GetCurrentMap();
        {
            std::unique_lock<std::mutex> map_lock(map->mMutexMapUpdate);
            auto keys=map->GetAllKeyFrames();
            std::sort(keys.begin(),keys.end(),[](auto a,auto b){return a->mTimeStamp>b->mTimeStamp;});
            for(auto key:keys) {
                if(key->isBad())continue;
                keyframes.append(py::make_tuple("kf-"+std::to_string(key->mnId),std::stol(key->mNameFile),key->mTimeStamp));
                if(py::len(keyframes)==3)break;
            }
            auto tracked=slam->GetTrackedMapPoints();auto pixels=slam->GetTrackedKeyPointsUn();
            for(size_t i=0;i<tracked.size() && i<pixels.size();++i) {
                auto point=tracked[i];if(!point || point->isBad() || point->Observations()<2)continue;
                auto p=point->GetWorldPos();auto local=pose*p;
                if(local.z()<=0)continue;
                if(py::len(points)<256)points.append(py::make_tuple("mp-"+std::to_string(point->mnId),p.x(),p.y(),p.z(),point->Observations()));
                if(py::len(depth_samples)<512)depth_samples.append(py::make_tuple(pixels[i].pt.x,pixels[i].pt.y,local.z()));
            }
        }
        result["keyframes"]=keyframes;result["geometry"]=points;result["depth_samples"]=depth_samples;
        if(slam->MapChanged()) {
            result["revision"]=++revision;result["tracking"]=false;
            result["geometry"]=py::list();result["depth_samples"]=py::list();
        }
        return result;
    }
    bool map_tick() {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex);
        if(capped)return false;
        if(mapper_dirty) {
            mapper=std::make_shared<GaussianMapper>(slam,config,"",0,torch::kCUDA);
            mapper->setKeepTraining(false);mapper->setDoInactiveGeoDensify(false);mapper_dirty=false;
        }
        bool ran=mapper->liveStep();
        if(ran && (mapper->gaussians_->getXYZ().size(0)>=100000 || mapper->scene_->keyframes().size()>=64))capped=true;
        return ran;
    }
    void close() {py::gil_scoped_release release;std::lock_guard<std::mutex> lock(mutex);mapper->signalStop();slam->Shutdown();}
};
PYBIND11_MODULE(photo_slam_live,m) {
    m.attr("upstream_commit")="f8bfb2f0809c003ccc3fd577dc43c576fcafa4ac";
    m.attr("bridge_schema")="photo-slam-live/v1";
    py::class_<LiveSession>(m,"Session").def(py::init<std::string,std::string,std::string>())
        .def("track",&LiveSession::track).def("map_tick",&LiveSession::map_tick).def("close",&LiveSession::close);
}
