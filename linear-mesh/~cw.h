#include "ns3/core-module.h"
#include "ns3/applications-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/opengym-module.h"
//#include "ns3/csma-module.h"
#include "ns3/propagation-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/ipv4-flow-classifier.h"
#include "ns3/yans-wifi-channel.h"

#include <fstream>
#include <string>
#include <math.h>
#include <ctime>   //timestampi
#include <iomanip> // put_time
#include <deque>

using namespace std;
using namespace ns3;

NS_LOG_COMPONENT_DEFINE("OpenGym");

class Simulation;
class AIGymInterface;
void tickHandler(AIGymInterface inter);

class Simulation
{
  private:
    void installTrafficGenerator(Ptr<ns3::Node> fromNode, Ptr<ns3::Node> toNode, int port, string offeredLoad, AIGymInterface inter);

    void PopulateARPcache()
    {
        Ptr<ArpCache> arp = CreateObject<ArpCache>();
        arp->SetAliveTimeout(Seconds(3600 * 24 * 365));

        for (NodeList::Iterator i = NodeList::Begin(); i != NodeList::End(); ++i)
        {
            Ptr<Ipv4L3Protocol> ip = (*i)->GetObject<Ipv4L3Protocol>();
            NS_ASSERT(ip != 0);
            ObjectVectorValue interfaces;
            ip->GetAttribute("InterfaceList", interfaces);

            for (ObjectVectorValue::Iterator j = interfaces.Begin(); j != interfaces.End(); j++)
            {
                Ptr<Ipv4Interface> ipIface = (*j).second->GetObject<Ipv4Interface>();
                NS_ASSERT(ipIface != 0);
                Ptr<NetDevice> device = ipIface->GetDevice();
                NS_ASSERT(device != 0);
                Mac48Address addr = Mac48Address::ConvertFrom(device->GetAddress());

                for (uint32_t k = 0; k < ipIface->GetNAddresses(); k++)
                {
                    Ipv4Address ipAddr = ipIface->GetAddress(k).GetLocal();
                    if (ipAddr == Ipv4Address::GetLoopback())
                        continue;

                    ArpCache::Entry *entry = arp->Add(ipAddr);
                    Ipv4Header ipv4Hdr;
                    ipv4Hdr.SetDestination(ipAddr);
                    Ptr<Packet> p = Create<Packet>(100);
                    entry->MarkWaitReply(ArpCache::Ipv4PayloadHeaderPair(p, ipv4Hdr));
                    entry->MarkAlive(addr);
                }
            }
        }

        for (NodeList::Iterator i = NodeList::Begin(); i != NodeList::End(); ++i)
        {
            Ptr<Ipv4L3Protocol> ip = (*i)->GetObject<Ipv4L3Protocol>();
            NS_ASSERT(ip != 0);
            ObjectVectorValue interfaces;
            ip->GetAttribute("InterfaceList", interfaces);

            for (ObjectVectorValue::Iterator j = interfaces.Begin(); j != interfaces.End(); j++)
            {
                Ptr<Ipv4Interface> ipIface = (*j).second->GetObject<Ipv4Interface>();
                ipIface->SetAttribute("ArpCache", PointerValue(arp));
            }
        }
    }

  public:
    uint32_t CW = 0;
    double envStepTime = 1;
    double simulationTime = 11; //seconds
    Ptr<FlowMonitor> monitor;
    FlowMonitorHelper flowmon;

    bool verbose = false;
    int nWifi = 5;
    bool tracing = false;
    bool useRts = false;
    int mcs = 11;
    int channelWidth = 20;
    int guardInterval = 800;
    string offeredLoad = "150";
    int port = 1025;
    string outputCsv = "cw.csv";
    int rng = 1;
    int warmup = 1;

    void setUp(AIGymInterface inter);

    bool SetCw(Ptr<Node> node, uint32_t cwMinValue = 0, uint32_t cwMaxValue = 0)
    {
        CW = cwMinValue;
        Ptr<NetDevice> dev = node->GetDevice(0);
        Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice>(dev);
        Ptr<WifiMac> wifi_mac = wifi_dev->GetMac();
        Ptr<RegularWifiMac> rmac = DynamicCast<RegularWifiMac>(wifi_mac);
        PointerValue ptr;
        rmac->GetAttribute("Txop", ptr);
        Ptr<Txop> txop = ptr.Get<Txop>();

        // if both set to the same value then we have uniform backoff?
        if (cwMinValue != 0)
        {
            NS_LOG_DEBUG("Set CW min: " << cwMinValue);
            txop->SetMinCw(cwMinValue);
        }

        if (cwMaxValue != 0)
        {
            NS_LOG_DEBUG("Set CW max: " << cwMaxValue);
            txop->SetMaxCw(cwMaxValue);
        }
        return true;
    }

    void printInfo(void)
    {
        double flowThr;

        ofstream myfile;
        myfile.open(outputCsv, ios::app);

        Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(this->flowmon.GetClassifier());
        std::map<FlowId, FlowMonitor::FlowStats> stats = this->monitor->GetFlowStats();
        for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin(); i != stats.end(); ++i)
        {
            auto time = std::time(nullptr); //Get timestamp
            auto tm = *std::localtime(&time);
            Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(i->first);
            flowThr = i->second.rxBytes * 8.0 / this->simulationTime / 1000 / 1000;
            NS_LOG_UNCOND("Flow " << i->first << " (" << t.sourceAddress << " -> " << t.destinationAddress << ")\tThroughput: " << flowThr << " Mbps\tTime: " << i->second.timeLastRxPacket.GetSeconds() - i->second.timeFirstTxPacket.GetSeconds() << " s\tRx packets " << i->second.rxPackets);
            myfile << std::put_time(&tm, "%Y-%m-%d %H:%M") << "," << CW << "," << nWifi << "," << RngSeedManager::GetRun() << "," << t.sourceAddress << "," << t.destinationAddress << "," << flowThr;
            myfile << std::endl;
        }
        myfile.close();
    }
};

class AIGymInterface
{
  private:
    float ticks = 0;
    uint32_t last_packets = 0;
    std::deque<float> *history;
    uint64_t g_rxPktNum = 0;
    Simulation *sim;

  public:
    uint32_t history_length = 10;

    Ptr<OpenGymDataContainer> MyGetObservation(void)
    {
        recordHistory();

        std::vector<uint32_t> shape = {
            this->history_length,
        };
        Ptr<OpenGymBoxContainer<float>> box = CreateObject<OpenGymBoxContainer<float>>(shape);

        NS_LOG_UNCOND("History length: " << this->history_length);

        for (uint32_t i = 0; i < this->history_length; i++)
            box->AddValue((*history)[i]);

        NS_LOG_UNCOND("MyGetObservation: " << box);
        return box;
    }

    float MyGetReward(void)
    {
        this->ticks += this->sim->envStepTime;

        float res = g_rxPktNum - last_packets;
        last_packets = g_rxPktNum;
        if (this->ticks <= 3 * this->sim->envStepTime)
            return 0.0;

        return res;
    }

    void recordHistory(void)
    {
        static float lastValue = 0.0;
        static float calls = 0;
        calls++;
        NS_LOG_UNCOND("g_rxPktNum: " << this->g_rxPktNum);

        float obs = this->g_rxPktNum - lastValue;
        lastValue = this->g_rxPktNum;

        this->history->push_front(obs * (1500 - 20 - 8 - 8) * 8.0 / 1000 / 1000);
        this->history->pop_back();

        if (calls < history_length)
            Simulator::Schedule(Seconds(sim->envStepTime), &tickHandler, *this);
    }

    bool MyExecuteActions(Ptr<OpenGymDataContainer> action)
    {
        NS_LOG_UNCOND("MyExecuteActions: " << action);

        Ptr<OpenGymBoxContainer<float>> box = DynamicCast<OpenGymBoxContainer<float>>(action);
        std::vector<float> actionVector = box->GetData();

        uint32_t new_cw = pow(2, actionVector.at(0) * 5 + 5);

        Config::Set("/$ns3::NodeListPriv/NodeList/*/$ns3::Node/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::RegularWifiMac/BE_Txop/$ns3::QosTxop/MinCw", UintegerValue(new_cw));
        Config::Set("/$ns3::NodeListPriv/NodeList/*/$ns3::Node/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::RegularWifiMac/BE_Txop/$ns3::QosTxop/MaxCw", UintegerValue(new_cw));
        return true;
    }

    std::string MyGetExtraInfo(void)
    {
        static float lastValue = 0.0;
        float obs = g_rxPktNum - lastValue;
        lastValue = g_rxPktNum;

        std::string myInfo = std::to_string(obs * (1500 - 20 - 8 - 8) * 8.0 / 1024 / 1024);
        NS_LOG_UNCOND("MyGetExtraInfo: " << myInfo);

        return myInfo;
    }

    bool MyGetGameOver(void)
    {
        bool isGameOver = false;
        NS_LOG_UNCOND("MyGetGameOver: " << isGameOver);
        return isGameOver;
    }

    Ptr<OpenGymSpace> MyGetActionSpace(void)
    {
        float low = 0.0;
        float high = 10.0;
        std::vector<uint32_t> shape = {
            1,
        };
        std::string dtype = TypeNameGet<float>();
        Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace>(low, high, shape, dtype);
        NS_LOG_UNCOND("MyGetActionSpace: " << space);
        return space;
    }

    Ptr<OpenGymSpace> MyGetObservationSpace(void)
    {
        float low = 0.0;
        float high = 10.0;
        std::vector<uint32_t> shape = {
            history_length,
        };
        std::string dtype = TypeNameGet<float>();
        Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace>(low, high, shape, dtype);
        NS_LOG_UNCOND("MyGetObservationSpace: " << space);
        return space;
    }

    Ptr<OpenGymInterface> build(uint32_t openGymPort)
    {
        Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface>(openGymPort);
        openGymInterface->SetGetActionSpaceCb(MakeCallback(&AIGymInterface::MyGetActionSpace, this));
        openGymInterface->SetGetObservationSpaceCb(MakeCallback(&AIGymInterface::MyGetObservationSpace, this));
        openGymInterface->SetGetGameOverCb(MakeCallback(&AIGymInterface::MyGetGameOver, this));
        openGymInterface->SetGetObservationCb(MakeCallback(&AIGymInterface::MyGetObservation, this));
        openGymInterface->SetGetRewardCb(MakeCallback(&AIGymInterface::MyGetReward, this));
        openGymInterface->SetGetExtraInfoCb(MakeCallback(&AIGymInterface::MyGetExtraInfo, this));
        openGymInterface->SetExecuteActionsCb(MakeCallback(&AIGymInterface::MyExecuteActions, this));

        return openGymInterface;
    }

    AIGymInterface(Simulation *sim)
    {
        this->sim = sim;
        this->history = new std::deque<float>(this->history_length, 0.0);
    }

    void DestRxPkt(Ptr<const Packet> packet)
    {
        NS_LOG_DEBUG("Client received a packet of " << packet->GetSize() << " bytes");
        g_rxPktNum++;
    }
};

void Simulation::setUp(AIGymInterface inter)
{
    Ptr<MatrixPropagationLossModel> lossModel = CreateObject<MatrixPropagationLossModel>();
    lossModel->SetDefaultLoss(50);

    NodeContainer wifiStaNode;
    wifiStaNode.Create(nWifi);
    NodeContainer wifiApNode;
    wifiApNode.Create(1);

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    Ptr<YansWifiChannel> chan = channel.Create();
    chan->SetPropagationLossModel(lossModel);
    chan->SetPropagationDelayModel(CreateObject<ConstantSpeedPropagationDelayModel>());

    YansWifiPhyHelper phy = YansWifiPhyHelper::Default();
    if (tracing)
    {
        phy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
    }
    phy.SetChannel(chan);

    // Set guard interval
    phy.Set("GuardInterval", TimeValue(NanoSeconds(guardInterval)));

    WifiMacHelper mac;
    WifiHelper wifi;

    wifi.SetStandard(WIFI_PHY_STANDARD_80211ax_5GHZ);

    std::ostringstream oss;
    oss << "HeMcs" << mcs;
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager", "DataMode", StringValue(oss.str()),
                                 "ControlMode", StringValue(oss.str()));

    //802.11ac PHY
    /*
  phy.Set ("ShortGuardEnabled", BooleanValue (0));
  wifi.SetStandard (WIFI_PHY_STANDARD_80211ac);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
  "DataMode", StringValue ("VhtMcs8"),
  "ControlMode", StringValue ("VhtMcs8"));
 */
    //802.11n PHY
    //phy.Set ("ShortGuardEnabled", BooleanValue (1));
    //wifi.SetStandard (WIFI_PHY_STANDARD_80211n_5GHZ);
    //wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
    //                              "DataMode", StringValue ("HtMcs7"),
    //                              "ControlMode", StringValue ("HtMcs7"));

    Ssid ssid = Ssid("ns3-80211ax");

    mac.SetType("ns3::StaWifiMac",
                "Ssid", SsidValue(ssid),
                "ActiveProbing", BooleanValue(false),
                "BE_MaxAmpduSize", UintegerValue(0));

    NetDeviceContainer staDevice;
    staDevice = wifi.Install(phy, mac, wifiStaNode);

    mac.SetType("ns3::ApWifiMac",
                "EnableBeaconJitter", BooleanValue(false),
                "Ssid", SsidValue(ssid));

    NetDeviceContainer apDevice;
    apDevice = wifi.Install(phy, mac, wifiApNode);

    // Set channel width
    Config::Set("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/ChannelWidth", UintegerValue(channelWidth));

    // mobility.
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();

    positionAlloc->Add(Vector(0.0, 0.0, 0.0));
    positionAlloc->Add(Vector(1.0, 0.0, 0.0));
    mobility.SetPositionAllocator(positionAlloc);

    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

    mobility.Install(wifiApNode);
    mobility.Install(wifiStaNode);
    /* Internet stack*/
    InternetStackHelper stack;
    stack.Install(wifiApNode);
    stack.Install(wifiStaNode);
    //Random

    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(rng);

    Ipv4AddressHelper address;
    address.SetBase("192.168.1.0", "255.255.255.0");
    Ipv4InterfaceContainer staNodeInterface;
    Ipv4InterfaceContainer apNodeInterface;

    staNodeInterface = address.Assign(staDevice);
    apNodeInterface = address.Assign(apDevice);

    if (CW)
    {
        Config::Set("/$ns3::NodeListPriv/NodeList/*/$ns3::Node/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::RegularWifiMac/BE_Txop/$ns3::QosTxop/MinCw", UintegerValue(CW));
        Config::Set("/$ns3::NodeListPriv/NodeList/*/$ns3::Node/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::RegularWifiMac/BE_Txop/$ns3::QosTxop/MaxCw", UintegerValue(CW));
    }

    for (int i = 0; i < nWifi; ++i)
    {
        installTrafficGenerator(wifiStaNode.Get(i), wifiApNode.Get(0), port++, offeredLoad, inter);
    }

    PopulateARPcache();
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    this->monitor = this->flowmon.InstallAll();
    this->monitor->SetAttribute("StartTime", TimeValue(Seconds(warmup)));
}

void Simulation::installTrafficGenerator(Ptr<ns3::Node> fromNode, Ptr<ns3::Node> toNode, int port, string offeredLoad, AIGymInterface inter)
{

    Ptr<Ipv4> ipv4 = toNode->GetObject<Ipv4>();           // Get Ipv4 instance of the node
    Ipv4Address addr = ipv4->GetAddress(1, 0).GetLocal(); // Get Ipv4InterfaceAddress of xth interface.

    ApplicationContainer sourceApplications, sinkApplications;

    uint8_t tosValue = 0x70; //AC_BE
    double simulationTime = 20;
    //Add random fuzz to app start time
    double min = 0.0;
    double max = 1.0;
    Ptr<UniformRandomVariable> fuzz = CreateObject<UniformRandomVariable>();
    fuzz->SetAttribute("Min", DoubleValue(min));
    fuzz->SetAttribute("Max", DoubleValue(max));

    InetSocketAddress sinkSocket(addr, port);
    sinkSocket.SetTos(tosValue);

    OnOffHelper onOffHelper("ns3::UdpSocketFactory", sinkSocket);
    onOffHelper.SetConstantRate(DataRate(offeredLoad + "Mbps"), 1500 - 20 - 8 - 8);
    sourceApplications.Add(onOffHelper.Install(fromNode)); //fromNode

    UdpServerHelper sink(port);
    sinkApplications = sink.Install(toNode);

    sinkApplications.Start(Seconds(0.0));
    sinkApplications.Stop(Seconds(simulationTime + 1));

    Ptr<UdpServer> udpServer = DynamicCast<UdpServer>(sinkApplications.Get(0));
    udpServer->TraceConnectWithoutContext("Rx", MakeCallback(&AIGymInterface::DestRxPkt, &inter));

    sourceApplications.Start(Seconds(1.0));
    sourceApplications.Stop(Seconds(simulationTime + 1));
}

void ScheduleNextStateRead(double envStepTime, Ptr<OpenGymInterface> openGymInterface)
{
    Simulator::Schedule(Seconds(envStepTime), &ScheduleNextStateRead, envStepTime, openGymInterface);
    openGymInterface->NotifyCurrentState();
}

void tickHandler(AIGymInterface inter)
{
    (*inter)->recordHistory();
}

int main(int argc, char *argv[])
{
    Simulation sim;
    AIGymInterface inter(&sim);

    uint32_t openGymPort = 5555;
    uint32_t simSeed = 1;

    CommandLine cmd;
    cmd.AddValue("CW", "Value of Contention Window", sim.CW);
    cmd.AddValue("historyLength", "Length of history window", inter.history_length);
    cmd.AddValue("nWifi", "Number of wifi 802.11ax STA devices", sim.nWifi);
    cmd.AddValue("verbose", "Tell echo applications to log if true", sim.verbose);
    cmd.AddValue("tracing", "Enable pcap tracing", sim.tracing);
    cmd.AddValue("rng", "Number of RngRun", sim.rng);
    cmd.AddValue("simTime", "Simulation time in seconds. Default: 10s", sim.simulationTime);
    cmd.AddValue("envStepTime", "Step time in seconds. Default: 0.1s", sim.envStepTime);

    cmd.Parse(argc, argv);

    NS_LOG_UNCOND("Ns3Env parameters:");
    NS_LOG_UNCOND("--simulationTime: " << sim.simulationTime);
    NS_LOG_UNCOND("--openGymPort: " << openGymPort);
    NS_LOG_UNCOND("--envStepTime: " << sim.envStepTime);
    NS_LOG_UNCOND("--seed: " << simSeed);

    if (sim.verbose)
    {
        LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
        LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);
    }

    if (sim.useRts)
    {
        Config::SetDefault("ns3::WifiRemoteStationManager::RtsCtsThreshold", StringValue("0"));
    }

    sim.setUp(inter);
    Simulator::Schedule(Seconds(1.0), &tickHandler, inter);
    // Simulator::Schedule(Seconds(2.0), &ScheduleNextStateRead, sim.envStepTime, inter.build(openGymPort));

    Simulator::Stop(Seconds(sim.simulationTime + 1));
    Simulator::Run();

    Simulator::Destroy();
    sim.printInfo();

    return 0;
}

// Traffic generator declaration from lte-wifi.cc
