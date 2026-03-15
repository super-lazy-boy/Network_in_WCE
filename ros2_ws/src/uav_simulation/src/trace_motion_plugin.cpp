#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Events.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/common/common.hh>

#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>
#include <ignition/math/Quaternion.hh>

#include <nlohmann/json.hpp>

#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <memory>

namespace gazebo
{
class TraceMotionPlugin : public WorldPlugin
{
public:
  struct UavState
  {
    int id = -1;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double yaw = 0.0;
  };

  using FrameMap = std::unordered_map<int, UavState>;

public:
  TraceMotionPlugin() = default;
  ~TraceMotionPlugin() override = default;

  void Load(physics::WorldPtr world, sdf::ElementPtr sdf) override
  {
    this->world_ = world;

    // -----------------------------
    // 1. 先读 SDF 参数，再用环境变量覆盖
    // -----------------------------
    if (sdf->HasElement("trace_file"))
      this->trace_file_ = sdf->Get<std::string>("trace_file");

    if (sdf->HasElement("position_scale"))
      this->position_scale_ = sdf->Get<double>("position_scale");

    if (sdf->HasElement("z_offset"))
      this->z_offset_ = sdf->Get<double>("z_offset");

    if (sdf->HasElement("playback_rate"))
      this->playback_rate_ = sdf->Get<double>("playback_rate");

    if (sdf->HasElement("loop"))
      this->loop_ = sdf->Get<bool>("loop");

    if (sdf->HasElement("start_delay"))
      this->start_delay_ = sdf->Get<double>("start_delay");

    // 用环境变量覆盖，方便 launch 动态传参
    this->ReadEnvOverride();

    if (this->trace_file_.empty())
    {
      gzerr << "[TraceMotionPlugin] 未提供 trace_file。\n";
      return;
    }

    if (!this->LoadTrace(this->trace_file_))
    {
      gzerr << "[TraceMotionPlugin] 读取轨迹失败: " << this->trace_file_ << "\n";
      return;
    }

    gzmsg << "[TraceMotionPlugin] 轨迹加载成功: "
          << "frames=" << this->frames_.size()
          << ", step_dt=" << this->step_dt_
          << ", position_scale=" << this->position_scale_
          << ", z_offset=" << this->z_offset_
          << ", playback_rate=" << this->playback_rate_
          << ", loop=" << (this->loop_ ? "true" : "false")
          << ", start_delay=" << this->start_delay_
          << "\n";

    this->update_connection_ = event::Events::ConnectWorldUpdateBegin(
      std::bind(&TraceMotionPlugin::OnUpdate, this, std::placeholders::_1));
  }

private:
  void ReadEnvOverride()
  {
    if (const char* v = std::getenv("UAV_TRACE_FILE"))
      this->trace_file_ = std::string(v);

    if (const char* v = std::getenv("UAV_POSITION_SCALE"))
      this->position_scale_ = std::stod(v);

    if (const char* v = std::getenv("UAV_Z_OFFSET"))
      this->z_offset_ = std::stod(v);

    if (const char* v = std::getenv("UAV_PLAYBACK_RATE"))
      this->playback_rate_ = std::stod(v);

    if (const char* v = std::getenv("UAV_START_DELAY"))
      this->start_delay_ = std::stod(v);

    if (const char* v = std::getenv("UAV_LOOP"))
    {
      std::string s(v);
      std::transform(s.begin(), s.end(), s.begin(), ::tolower);
      this->loop_ = (s == "1" || s == "true" || s == "yes");
    }
  }

  bool LoadTrace(const std::string& path)
  {
    std::ifstream ifs(path);
    if (!ifs.is_open())
    {
      gzerr << "[TraceMotionPlugin] 无法打开轨迹文件: " << path << "\n";
      return false;
    }

    nlohmann::json j;
    ifs >> j;

    if (!j.contains("frames") || !j["frames"].is_array())
    {
      gzerr << "[TraceMotionPlugin] JSON 中缺少 frames 数组\n";
      return false;
    }

    this->step_dt_ = j.value("step_dt", 1.0);

    const auto& frames_json = j["frames"];
    if (frames_json.empty())
    {
      gzerr << "[TraceMotionPlugin] frames 为空\n";
      return false;
    }

    this->frames_.clear();
    this->expected_ids_.clear();

    bool first_frame = true;

    for (const auto& frame_j : frames_json)
    {
      FrameMap frame_map;

      if (!frame_j.contains("uavs") || !frame_j["uavs"].is_array())
        continue;

      for (const auto& u : frame_j["uavs"])
      {
        UavState s;
        s.id = u.value("id", -1);
        s.x = u.value("x", 0.0);
        s.y = u.value("y", 0.0);
        s.z = u.value("z", 0.0);
        s.yaw = u.value("yaw", 0.0);

        frame_map[s.id] = s;

        if (first_frame)
          this->expected_ids_.push_back(s.id);
      }

      first_frame = false;
      this->frames_.push_back(frame_map);
    }

    if (this->frames_.size() < 2)
    {
      gzerr << "[TraceMotionPlugin] 轨迹帧数不足，至少需要 2 帧\n";
      return false;
    }

    this->total_duration_ = this->step_dt_ * static_cast<double>(this->frames_.size() - 1);
    return true;
  }

  static double NormalizeAngle(double angle)
  {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
  }

  static double ShortestAngularDistance(double from, double to)
  {
    return NormalizeAngle(to - from);
  }

  bool ResolveModels()
  {
    bool all_found = true;

    for (int id : this->expected_ids_)
    {
      if (this->models_.count(id) > 0)
        continue;

      const std::string model_name = "uav_" + std::to_string(id);
      auto model = this->world_->ModelByName(model_name);

      if (!model)
      {
        all_found = false;
        continue;
      }

      this->models_[id] = model;

      // 额外保险：把所有 link 的重力关掉，减少抖动/下落
      for (auto& link : model->GetLinks())
      {
        if (link)
          link->SetGravityMode(false);
      }

      gzmsg << "[TraceMotionPlugin] 已发现模型: " << model_name << "\n";
    }

    for (int id : this->expected_ids_)
    {
      if (this->models_.count(id) == 0)
      {
        all_found = false;
        break;
      }
    }

    if (all_found && !this->models_ready_)
    {
      this->models_ready_ = true;
      this->start_sim_time_ = this->world_->SimTime().Double() + this->start_delay_;
      gzmsg << "[TraceMotionPlugin] 所有 UAV 已就绪，开始播放时间 = "
            << this->start_sim_time_ << " s\n";
    }

    return this->models_ready_;
  }

  void ApplyInterpolatedPose(std::size_t idx0, std::size_t idx1, double alpha)
  {
    const auto& f0 = this->frames_[idx0];
    const auto& f1 = this->frames_[idx1];

    for (int id : this->expected_ids_)
    {
      auto it_model = this->models_.find(id);
      if (it_model == this->models_.end())
        continue;

      auto it0 = f0.find(id);
      if (it0 == f0.end())
        continue;

      UavState s0 = it0->second;
      UavState s1 = s0;

      auto it1 = f1.find(id);
      if (it1 != f1.end())
        s1 = it1->second;

      double x = (1.0 - alpha) * s0.x + alpha * s1.x;
      double y = (1.0 - alpha) * s0.y + alpha * s1.y;
      double z = (1.0 - alpha) * s0.z + alpha * s1.z;

      double dyaw = ShortestAngularDistance(s0.yaw, s1.yaw);
      double yaw = NormalizeAngle(s0.yaw + alpha * dyaw);

      x = x * this->position_scale_;
      y = y * this->position_scale_;
      z = z * this->position_scale_ + this->z_offset_;

      ignition::math::Pose3d pose(
        ignition::math::Vector3d(x, y, z),
        ignition::math::Quaterniond(0.0, 0.0, yaw)
      );

      auto model = it_model->second;
      model->SetLinearVel(ignition::math::Vector3d(0, 0, 0));
      model->SetAngularVel(ignition::math::Vector3d(0, 0, 0));
      model->SetWorldPose(pose);
    }
  }

  void HoldFirstFrame()
  {
    this->ApplyInterpolatedPose(0, 1, 0.0);
  }

  void OnUpdate(const common::UpdateInfo& info)
  {
    if (!this->world_ || this->frames_.empty())
      return;

    if (!this->ResolveModels())
    {
      // 模型还没全部 spawn 完成，先不播放
      return;
    }

    const double sim_time = info.simTime.Double();

    if (sim_time < this->start_sim_time_)
    {
      // 开始播放前，把 UAV 固定在第一帧，防止抖动
      this->HoldFirstFrame();
      return;
    }

    double play_time = (sim_time - this->start_sim_time_) * this->playback_rate_;

    if (this->loop_)
    {
      if (this->total_duration_ > 1e-9)
        play_time = std::fmod(play_time, this->total_duration_);
    }
    else
    {
      play_time = std::min(play_time, this->total_duration_);
    }

    std::size_t idx0 = static_cast<std::size_t>(std::floor(play_time / this->step_dt_));
    if (idx0 >= this->frames_.size() - 1)
      idx0 = this->frames_.size() - 2;

    std::size_t idx1 = idx0 + 1;

    double local_t = play_time - static_cast<double>(idx0) * this->step_dt_;
    double alpha = this->step_dt_ > 1e-9 ? (local_t / this->step_dt_) : 0.0;
    alpha = std::max(0.0, std::min(1.0, alpha));

    this->ApplyInterpolatedPose(idx0, idx1, alpha);
  }

private:
  physics::WorldPtr world_;
  event::ConnectionPtr update_connection_;

  std::string trace_file_;
  double step_dt_ = 1.0;
  double total_duration_ = 0.0;

  double position_scale_ = 0.05;
  double z_offset_ = 2.0;
  double playback_rate_ = 1.0;
  double start_delay_ = 0.5;
  bool loop_ = true;

  bool models_ready_ = false;
  double start_sim_time_ = 0.0;

  std::vector<int> expected_ids_;
  std::vector<FrameMap> frames_;
  std::unordered_map<int, physics::ModelPtr> models_;
};

GZ_REGISTER_WORLD_PLUGIN(TraceMotionPlugin)
}  // namespace gazebo