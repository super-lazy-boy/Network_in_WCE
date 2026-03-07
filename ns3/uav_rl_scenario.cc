#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/olsr-helper.h"
#include "ns3/aodv-helper.h"
#include "ns3/dsdv-helper.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace ns3;
using json = nlohmann::json;

NS_LOG_COMPONENT_DEFINE("UavRlScenarioImproved");

struct NodeState
{
  uint32_t id = 0;
  Vector position{0.0, 0.0, 0.0};
};

struct Frame
{
  uint32_t t = 0;
  std::vector<NodeState> nodes;
  std::vector<double> txPowerDbmPerNode;
  std::vector<double> bandwidthHzPerNode;
  int32_t topologyPolicy = -1;
};

struct IntervalFlowState
{
  uint64_t txPackets = 0;
  uint64_t rxPackets = 0;
  uint64_t rxBytes = 0;
  double delaySumSec = 0.0;
  double jitterSumSec = 0.0;
};

static std::unique_ptr<std::ofstream> g_topologyCsv;
static std::unique_ptr<std::ofstream> g_flowCsv;
static std::unordered_map<uint32_t, IntervalFlowState> g_prevFlowState;

static std::string
ToUpper(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
  return s;
}

static std::vector<Frame>
LoadTrace(const std::string& tracePath)
{
  std::ifstream in(tracePath);
  if (!in.is_open())
  {
    NS_FATAL_ERROR("Cannot open trace: " << tracePath);
  }

  json root;
  in >> root;

  if (!root.contains("frames") || !root["frames"].is_array())
  {
    NS_FATAL_ERROR("Invalid trace format: missing frames[]");
  }

  std::vector<Frame> frames;
  for (const auto& jf : root["frames"])
  {
    Frame f;
    f.t = jf.value("t", 0u);

    if (jf.contains("topology") && jf["topology"].contains("nodes"))
    {
      for (const auto& jn : jf["topology"]["nodes"])
      {
        NodeState s;
        s.id = jn.at("id").get<uint32_t>();
        const auto& p = jn.at("position");
        s.position = Vector(p[0].get<double>(), p[1].get<double>(), p[2].get<double>());
        f.nodes.push_back(s);
      }
      if (jf["topology"].contains("policy"))
      {
        f.topologyPolicy = jf["topology"]["policy"].get<int32_t>();
      }
    }

    if (jf.contains("power") && jf["power"].contains("tx_power_dbm_per_node"))
    {
      for (const auto& v : jf["power"]["tx_power_dbm_per_node"])
      {
        f.txPowerDbmPerNode.push_back(v.get<double>());
      }
    }

    if (jf.contains("bandwidth") && jf["bandwidth"].contains("hz_per_node"))
    {
      for (const auto& v : jf["bandwidth"]["hz_per_node"])
      {
        f.bandwidthHzPerNode.push_back(v.get<double>());
      }
    }

    frames.push_back(std::move(f));
  }

  return frames;
}

static std::vector<Vector>
GetNodePositions(NodeContainer nodes)
{
  std::vector<Vector> positions(nodes.GetN());
  for (uint32_t i = 0; i < nodes.GetN(); ++i)
  {
    Ptr<MobilityModel> mob = nodes.Get(i)->GetObject<MobilityModel>();
    if (mob == nullptr)
    {
      NS_FATAL_ERROR("Node has no MobilityModel");
    }
    positions[i] = mob->GetPosition();
  }
  return positions;
}

static std::vector<std::vector<uint32_t>>
BuildAdjacencyByRange(const std::vector<Vector>& pos, double rangeMeters)
{
  const uint32_t n = static_cast<uint32_t>(pos.size());
  std::vector<std::vector<uint32_t>> adj(n);

  const double r2 = rangeMeters * rangeMeters;
  for (uint32_t i = 0; i < n; ++i)
  {
    for (uint32_t j = i + 1; j < n; ++j)
    {
      const double dx = pos[i].x - pos[j].x;
      const double dy = pos[i].y - pos[j].y;
      const double dz = pos[i].z - pos[j].z;
      const double d2 = dx * dx + dy * dy + dz * dz;
      if (d2 <= r2)
      {
        adj[i].push_back(j);
        adj[j].push_back(i);
      }
    }
  }
  return adj;
}

static uint32_t
LargestConnectedComponentSize(const std::vector<std::vector<uint32_t>>& adj)
{
  const uint32_t n = static_cast<uint32_t>(adj.size());
  std::vector<bool> visited(n, false);
  uint32_t best = 0;

  for (uint32_t i = 0; i < n; ++i)
  {
    if (visited[i])
    {
      continue;
    }

    uint32_t size = 0;
    std::queue<uint32_t> q;
    q.push(i);
    visited[i] = true;

    while (!q.empty())
    {
      const uint32_t u = q.front();
      q.pop();
      ++size;
      for (uint32_t v : adj[u])
      {
        if (!visited[v])
        {
          visited[v] = true;
          q.push(v);
        }
      }
    }
    best = std::max(best, size);
  }

  return best;
}

static void
ConfigureRouting(InternetStackHelper& stack, const std::string& routingProtocol)
{
  const std::string routing = ToUpper(routingProtocol);

  if (routing == "OLSR")
  {
    OlsrHelper helper;
    stack.SetRoutingHelper(helper);
  }
  else if (routing == "AODV")
  {
    AodvHelper helper;
    stack.SetRoutingHelper(helper);
  }
  else if (routing == "DSDV")
  {
    DsdvHelper helper;
    stack.SetRoutingHelper(helper);
  }
  else
  {
    NS_FATAL_ERROR("Unsupported routing protocol: " << routingProtocol);
  }
}

static void
ConfigureWifiChannel(YansWifiChannelHelper& channel,
                     const std::string& lossModel,
                     double logDistanceExponent,
                     double referenceLoss,
                     double maxRangeMeters)
{
  channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");

  const std::string model = ToUpper(lossModel);
  if (model == "FRIIS")
  {
    channel.AddPropagationLoss("ns3::FriisPropagationLossModel");
  }
  else if (model == "LOGDISTANCE")
  {
    channel.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
                               "Exponent",
                               DoubleValue(logDistanceExponent),
                               "ReferenceLoss",
                               DoubleValue(referenceLoss));
  }
  else if (model == "NAKAGAMI")
  {
    channel.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
                               "Exponent",
                               DoubleValue(logDistanceExponent),
                               "ReferenceLoss",
                               DoubleValue(referenceLoss));
    channel.AddPropagationLoss("ns3::NakagamiPropagationLossModel");
  }
  else if (model == "RANGE")
  {
    channel.AddPropagationLoss("ns3::RangePropagationLossModel",
                               "MaxRange",
                               DoubleValue(maxRangeMeters));
  }
  else
  {
    NS_FATAL_ERROR("Unsupported loss model: " << lossModel);
  }
}

static std::vector<uint32_t>
BuildTrafficDestinations(const std::string& trafficMode,
                         uint32_t nNodes,
                         uint32_t sinkNode,
                         uint64_t trafficSeed)
{
  std::vector<uint32_t> dst(nNodes, sinkNode);
  const std::string mode = ToUpper(trafficMode);

  if (mode == "MANY-TO-ONE")
  {
    for (uint32_t i = 0; i < nNodes; ++i)
    {
      dst[i] = sinkNode;
    }
    return dst;
  }

  if (mode == "RANDOM-PAIRS")
  {
    Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
    uv->SetStream(static_cast<int64_t>(trafficSeed));
    for (uint32_t i = 0; i < nNodes; ++i)
    {
      uint32_t d = i;
      while (d == i)
      {
        d = uv->GetInteger(0, nNodes - 1);
      }
      dst[i] = d;
    }
    return dst;
  }

  if (mode == "FIXED-PAIRS")
  {
    for (uint32_t i = 0; i < nNodes; ++i)
    {
      dst[i] = (i + nNodes / 2) % nNodes;
      if (dst[i] == i)
      {
        dst[i] = (i + 1) % nNodes;
      }
    }
    return dst;
  }

  NS_FATAL_ERROR("Unsupported trafficMode: " << trafficMode);
  return dst;
}

static void
ApplyFrameContinuous(const Frame& current,
                     const Frame* next,
                     NodeContainer nodes,
                     const std::vector<Ptr<WifiNetDevice>>& wifiDevices,
                     bool applyTxPower,
                     double stepSeconds)
{
  std::unordered_map<uint32_t, Vector> posCurr;
  std::unordered_map<uint32_t, Vector> posNext;

  for (const auto& n : current.nodes)
  {
    posCurr[n.id] = n.position;
  }
  if (next != nullptr)
  {
    for (const auto& n : next->nodes)
    {
      posNext[n.id] = n.position;
    }
  }

  for (uint32_t i = 0; i < nodes.GetN(); ++i)
  {
    Ptr<ConstantVelocityMobilityModel> mob = nodes.Get(i)->GetObject<ConstantVelocityMobilityModel>();
    if (mob == nullptr)
    {
      NS_FATAL_ERROR("Node has no ConstantVelocityMobilityModel");
    }

    auto itCurr = posCurr.find(i);
    if (itCurr != posCurr.end())
    {
      mob->SetPosition(itCurr->second);
    }

    Vector v(0.0, 0.0, 0.0);
    if (next != nullptr)
    {
      auto itNext = posNext.find(i);
      if (itCurr != posCurr.end() && itNext != posNext.end())
      {
        v = Vector((itNext->second.x - itCurr->second.x) / stepSeconds,
                   (itNext->second.y - itCurr->second.y) / stepSeconds,
                   (itNext->second.z - itCurr->second.z) / stepSeconds);
      }
    }
    mob->SetVelocity(v);

    if (applyTxPower && i < current.txPowerDbmPerNode.size())
    {
      Ptr<WifiPhy> phy = wifiDevices[i]->GetPhy();
      phy->SetTxPowerStart(current.txPowerDbmPerNode[i]);
      phy->SetTxPowerEnd(current.txPowerDbmPerNode[i]);
    }
  }
}

static void
LogTopologySnapshot(NodeContainer nodes,
                    double commRangeMeters,
                    double timeNowSec)
{
  if (!g_topologyCsv)
  {
    return;
  }

  const std::vector<Vector> pos = GetNodePositions(nodes);
  const auto adj = BuildAdjacencyByRange(pos, commRangeMeters);

  double avgDegree = 0.0;
  uint32_t isolated = 0;
  for (const auto& nbrs : adj)
  {
    avgDegree += static_cast<double>(nbrs.size());
    if (nbrs.empty())
    {
      ++isolated;
    }
  }
  avgDegree /= std::max<uint32_t>(1, static_cast<uint32_t>(adj.size()));

  const uint32_t lcc = LargestConnectedComponentSize(adj);
  const double lccRatio = static_cast<double>(lcc) / std::max<uint32_t>(1, nodes.GetN());

  (*g_topologyCsv) << std::fixed << std::setprecision(6) << timeNowSec << "," << avgDegree << ","
                   << lccRatio << "," << isolated << "\n";
}

static void
LogFlowSnapshot(Ptr<FlowMonitor> monitor,
                Ptr<Ipv4FlowClassifier> classifier,
                uint16_t basePort,
                uint32_t nNodes,
                double intervalSec,
                double timeNowSec)
{
  if (!g_flowCsv)
  {
    return;
  }

  monitor->CheckForLostPackets();
  const auto stats = monitor->GetFlowStats();

  uint64_t deltaTxPackets = 0;
  uint64_t deltaRxPackets = 0;
  uint64_t deltaRxBytes = 0;
  double deltaDelaySum = 0.0;
  double deltaJitterSum = 0.0;

  for (const auto& kv : stats)
  {
    const auto tuple = classifier->FindFlow(kv.first);
    if (tuple.destinationPort < basePort || tuple.destinationPort >= basePort + nNodes)
    {
      continue;
    }

    const auto& st = kv.second;
    IntervalFlowState curr;
    curr.txPackets = st.txPackets;
    curr.rxPackets = st.rxPackets;
    curr.rxBytes = st.rxBytes;
    curr.delaySumSec = st.delaySum.GetSeconds();
    curr.jitterSumSec = st.jitterSum.GetSeconds();

    IntervalFlowState prev;
    auto itPrev = g_prevFlowState.find(kv.first);
    if (itPrev != g_prevFlowState.end())
    {
      prev = itPrev->second;
    }

    deltaTxPackets += (curr.txPackets - prev.txPackets);
    deltaRxPackets += (curr.rxPackets - prev.rxPackets);
    deltaRxBytes += (curr.rxBytes - prev.rxBytes);
    deltaDelaySum += (curr.delaySumSec - prev.delaySumSec);
    deltaJitterSum += (curr.jitterSumSec - prev.jitterSumSec);

    g_prevFlowState[kv.first] = curr;
  }

  const double throughputMbps = (intervalSec > 0.0) ? (deltaRxBytes * 8.0) / intervalSec / 1e6 : 0.0;
  const double pdr = (deltaTxPackets > 0) ? static_cast<double>(deltaRxPackets) / deltaTxPackets : 0.0;
  const double avgDelay = (deltaRxPackets > 0) ? deltaDelaySum / deltaRxPackets : 0.0;
  const double avgJitter = (deltaRxPackets > 0) ? deltaJitterSum / deltaRxPackets : 0.0;
  const uint64_t lostPackets = (deltaTxPackets >= deltaRxPackets) ? (deltaTxPackets - deltaRxPackets) : 0;

  (*g_flowCsv) << std::fixed << std::setprecision(6) << timeNowSec << "," << throughputMbps << "," << pdr
               << "," << avgDelay << "," << avgJitter << "," << deltaTxPackets << "," << deltaRxPackets
               << "," << lostPackets << "\n";
}

static void
ScheduleSnapshots(const std::vector<Frame>& frames,
                  NodeContainer nodes,
                  Ptr<FlowMonitor> monitor,
                  Ptr<Ipv4FlowClassifier> classifier,
                  double stepSeconds,
                  double topologyRangeMeters,
                  uint16_t basePort,
                  uint32_t nNodes)
{
  for (size_t i = 0; i < frames.size(); ++i)
  {
    const double t = stepSeconds * static_cast<double>(i);
    Simulator::Schedule(Seconds(t), &LogTopologySnapshot, nodes, topologyRangeMeters, t);
    if (i > 0)
    {
      Simulator::Schedule(Seconds(t),
                          &LogFlowSnapshot,
                          monitor,
                          classifier,
                          basePort,
                          nNodes,
                          stepSeconds,
                          t);
    }
  }
}

int
main(int argc, char* argv[])
{
  std::string trace = "ns3_trace.json";
  double stepSeconds = 1.0;
  double appStart = 2.0;
  double appStopPad = 2.0;
  bool applyTxPower = true;

  std::string routing = "OLSR";
  std::string lossModel = "LogDistance";
  std::string trafficMode = "many-to-one";

  double topologyRangeMeters = 350.0;
  double maxRangeMeters = 350.0;
  double logDistanceExponent = 2.7;
  double referenceLoss = 46.6777;

  uint32_t seed = 1;
  uint32_t run = 1;
  uint64_t trafficSeed = 7;
  uint16_t basePort = 9000;

  double txPowerStartDbm = 15.0;
  double txPowerEndDbm = 15.0;
  bool enablePcap = false;

  CommandLine cmd(__FILE__);
  cmd.AddValue("trace", "Path to ns3_trace.json exported by Python side", trace);
  cmd.AddValue("step", "Simulation seconds represented by each frame", stepSeconds);
  cmd.AddValue("applyTxPower", "Apply tx power schedule from trace", applyTxPower);
  cmd.AddValue("routing", "Routing protocol: OLSR/AODV/DSDV", routing);
  cmd.AddValue("lossModel", "Propagation loss: Friis/LogDistance/Nakagami/Range", lossModel);
  cmd.AddValue("trafficMode", "Traffic pattern: many-to-one/random-pairs/fixed-pairs", trafficMode);
  cmd.AddValue("topologyRange", "Range used for topology statistics in meters", topologyRangeMeters);
  cmd.AddValue("maxRange", "Maximum range for RangePropagationLossModel in meters", maxRangeMeters);
  cmd.AddValue("logExponent", "Exponent for LogDistancePropagationLossModel", logDistanceExponent);
  cmd.AddValue("referenceLoss", "Reference loss for LogDistancePropagationLossModel", referenceLoss);
  cmd.AddValue("seed", "Global RNG seed", seed);
  cmd.AddValue("run", "RNG run index", run);
  cmd.AddValue("trafficSeed", "Random stream for random-pairs traffic generation", trafficSeed);
  cmd.AddValue("basePort", "Base UDP port used by per-node servers", basePort);
  cmd.AddValue("txPowerStart", "Default PHY TxPowerStart in dBm", txPowerStartDbm);
  cmd.AddValue("txPowerEnd", "Default PHY TxPowerEnd in dBm", txPowerEndDbm);
  cmd.AddValue("enablePcap", "Enable PCAP tracing", enablePcap);
  cmd.Parse(argc, argv);

  RngSeedManager::SetSeed(seed);
  RngSeedManager::SetRun(run);

  const auto frames = LoadTrace(trace);
  if (frames.empty())
  {
    NS_FATAL_ERROR("Trace contains zero frames");
  }

  const uint32_t nNodes = static_cast<uint32_t>(frames.front().nodes.size());
  if (nNodes < 2)
  {
    NS_FATAL_ERROR("Need at least 2 nodes");
  }

  NodeContainer nodes;
  nodes.Create(nNodes);

  MobilityHelper mobility;
  mobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
  mobility.Install(nodes);

  // 先把初始位置/速度置零，第一帧再覆盖。
  for (uint32_t i = 0; i < nNodes; ++i)
  {
    Ptr<ConstantVelocityMobilityModel> mob = nodes.Get(i)->GetObject<ConstantVelocityMobilityModel>();
    mob->SetPosition(Vector(0.0, 0.0, 0.0));
    mob->SetVelocity(Vector(0.0, 0.0, 0.0));
  }

  YansWifiChannelHelper channel;
  ConfigureWifiChannel(channel, lossModel, logDistanceExponent, referenceLoss, maxRangeMeters);

  YansWifiPhyHelper phy;
  phy.SetChannel(channel.Create());
  phy.Set("TxPowerStart", DoubleValue(txPowerStartDbm));
  phy.Set("TxPowerEnd", DoubleValue(txPowerEndDbm));
  phy.Set("RxGain", DoubleValue(0.0));
  phy.Set("TxGain", DoubleValue(0.0));

  WifiHelper wifi;
  wifi.SetStandard(WIFI_STANDARD_80211a);
  wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                               "DataMode",
                               StringValue("OfdmRate24Mbps"),
                               "ControlMode",
                               StringValue("OfdmRate6Mbps"));

  WifiMacHelper mac;
  mac.SetType("ns3::AdhocWifiMac");

  NetDeviceContainer devs = wifi.Install(phy, mac, nodes);
  std::vector<Ptr<WifiNetDevice>> wifiDevices;
  wifiDevices.reserve(devs.GetN());
  for (uint32_t i = 0; i < devs.GetN(); ++i)
  {
    Ptr<WifiNetDevice> d = DynamicCast<WifiNetDevice>(devs.Get(i));
    if (d == nullptr)
    {
      NS_FATAL_ERROR("Device is not WifiNetDevice");
    }
    wifiDevices.push_back(d);
  }

  if (enablePcap)
  {
    phy.EnablePcapAll("uav_rl_scenario");
  }

  InternetStackHelper stack;
  ConfigureRouting(stack, routing);
  stack.Install(nodes);

  Ipv4AddressHelper address;
  address.SetBase("10.0.0.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces = address.Assign(devs);

  // 在所有节点上安装接收端，便于多种 trafficMode 复用。
  ApplicationContainer serverApps;
  for (uint32_t i = 0; i < nNodes; ++i)
  {
    UdpServerHelper server(basePort + i);
    serverApps.Add(server.Install(nodes.Get(i)));
  }
  serverApps.Start(Seconds(1.0));

  const uint32_t sinkNode = 0;
  const auto destinations = BuildTrafficDestinations(trafficMode, nNodes, sinkNode, trafficSeed);

  ApplicationContainer clientApps;
  for (uint32_t src = 0; src < nNodes; ++src)
  {
    const uint32_t dst = destinations[src];
    if (dst == src)
    {
      continue;
    }

    UdpClientHelper client(interfaces.GetAddress(dst), basePort + dst);
    client.SetAttribute("MaxPackets", UintegerValue(0));
    client.SetAttribute("Interval", TimeValue(MilliSeconds(20)));
    client.SetAttribute("PacketSize", UintegerValue(1024));
    clientApps.Add(client.Install(nodes.Get(src)));
  }
  clientApps.Start(Seconds(appStart));

  FlowMonitorHelper flowmon;
  Ptr<FlowMonitor> monitor = flowmon.InstallAll();

  for (size_t i = 0; i < frames.size(); ++i)
  {
    const Frame* next = (i + 1 < frames.size()) ? &frames[i + 1] : nullptr;
    Simulator::Schedule(Seconds(stepSeconds * static_cast<double>(i)),
                        &ApplyFrameContinuous,
                        frames[i],
                        next,
                        nodes,
                        wifiDevices,
                        applyTxPower,
                        stepSeconds);
  }

  Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());

  g_topologyCsv = std::make_unique<std::ofstream>("topology_stats.csv");
  g_flowCsv = std::make_unique<std::ofstream>("flow_stats.csv");
  if (!g_topologyCsv->is_open() || !g_flowCsv->is_open())
  {
    NS_FATAL_ERROR("Failed to open CSV output files");
  }
  (*g_topologyCsv) << "time_s,avg_degree,largest_component_ratio,isolated_nodes\n";
  (*g_flowCsv) << "time_s,throughput_mbps,pdr,avg_delay_s,avg_jitter_s,tx_packets,rx_packets,lost_packets\n";

  ScheduleSnapshots(frames, nodes, monitor, classifier, stepSeconds, topologyRangeMeters, basePort, nNodes);

  const double stopTime = stepSeconds * static_cast<double>(frames.size()) + appStopPad;
  clientApps.Stop(Seconds(stopTime - 0.5));
  serverApps.Stop(Seconds(stopTime));

  Simulator::Stop(Seconds(stopTime));
  Simulator::Run();

  monitor->CheckForLostPackets();
  const auto stats = monitor->GetFlowStats();

  double totalRxMbps = 0.0;
  uint64_t totalTxPackets = 0;
  uint64_t totalRxPackets = 0;
  uint64_t totalLostPackets = 0;
  double sumDelaySec = 0.0;
  double sumJitterSec = 0.0;
  uint64_t delaySamples = 0;

  for (const auto& kv : stats)
  {
    const auto tuple = classifier->FindFlow(kv.first);
    if (tuple.destinationPort < basePort || tuple.destinationPort >= basePort + nNodes)
    {
      continue;
    }

    const auto& st = kv.second;
    const double duration = (st.timeLastRxPacket - st.timeFirstTxPacket).GetSeconds();
    if (duration > 1e-9)
    {
      totalRxMbps += (st.rxBytes * 8.0) / duration / 1e6;
    }

    totalTxPackets += st.txPackets;
    totalRxPackets += st.rxPackets;
    totalLostPackets += st.lostPackets;
    sumDelaySec += st.delaySum.GetSeconds();
    sumJitterSec += st.jitterSum.GetSeconds();
    delaySamples += st.rxPackets;
  }

  const double pdr = (totalTxPackets > 0) ? static_cast<double>(totalRxPackets) / totalTxPackets : 0.0;
  const double avgDelay = (delaySamples > 0) ? sumDelaySec / delaySamples : 0.0;
  const double avgJitter = (delaySamples > 0) ? sumJitterSec / delaySamples : 0.0;

  NS_LOG_UNCOND("Frames=" << frames.size() << ", Nodes=" << nNodes << ", Routing=" << routing
                           << ", TrafficMode=" << trafficMode << ", LossModel=" << lossModel);
  NS_LOG_UNCOND("Aggregate Throughput (Mbps): " << totalRxMbps);
  NS_LOG_UNCOND("PDR: " << pdr);
  NS_LOG_UNCOND("Avg Delay (s): " << avgDelay);
  NS_LOG_UNCOND("Avg Jitter (s): " << avgJitter);
  NS_LOG_UNCOND("Lost Packets: " << totalLostPackets);

  monitor->SerializeToXmlFile("flowmon_uav_rl.xml", true, true);

  if (g_topologyCsv)
  {
    g_topologyCsv->close();
  }
  if (g_flowCsv)
  {
    g_flowCsv->close();
  }

  Simulator::Destroy();
  return 0;
}