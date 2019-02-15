#include "ns3/core-module.h"
#include "ns3/applications-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
//#include "ns3/csma-module.h"
#include "ns3/propagation-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/ipv4-flow-classifier.h"
#include "ns3/yans-wifi-channel.h"

#include <fstream>
#include <string>
#include <ctime> //timestampi
#include <iomanip> // put_time

using namespace std;
using namespace ns3;

void installTrafficGenerator(Ptr<ns3::Node> fromNode, Ptr<ns3::Node> toNode, int port, string offeredLoad);
void PopulateARPcache ();

int
main (int argc, char *argv[])
{
  bool verbose = false;
  int nWifi = 1;
  bool tracing = false;
  bool useRts = false;
  double simulationTime = 20; //seconds
  int mcs = 11; 
  int CW = 0;
  int channelWidth = 20;
  int guardInterval = 800;
  string offeredLoad = "150";
  int port = 1025;
  string outputCsv = "cw.csv";
  int rng = 1;
  int warmup = 2;

  CommandLine cmd;
  cmd.AddValue ("CW", "Value of Contention Window", CW);
  cmd.AddValue ("nWifi", "Number of wifi 802.11ax STA devices", nWifi);
  cmd.AddValue ("verbose", "Tell echo applications to log if true", verbose);
  cmd.AddValue ("tracing", "Enable pcap tracing", tracing);
  cmd.AddValue ("rng", "Number of RngRun", rng);
  cmd.Parse (argc,argv);

  if (verbose)
    {
     LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
     LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_INFO);
    }

  if (useRts)
    {
      Config::SetDefault ("ns3::WifiRemoteStationManager::RtsCtsThreshold", StringValue ("0"));
    }

  Ptr<MatrixPropagationLossModel> lossModel = CreateObject<MatrixPropagationLossModel> ();
  lossModel->SetDefaultLoss (50);

  NodeContainer wifiStaNode;
  wifiStaNode.Create (nWifi);
  NodeContainer wifiApNode;
  wifiApNode.Create (1);

  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
  Ptr<YansWifiChannel> chan = channel.Create();
  chan->SetPropagationLossModel(lossModel);
  chan->SetPropagationDelayModel (CreateObject <ConstantSpeedPropagationDelayModel> ());

  YansWifiPhyHelper phy = YansWifiPhyHelper::Default ();
  if (tracing){
      phy.SetPcapDataLinkType (WifiPhyHelper::DLT_IEEE802_11_RADIO);
  }
  phy.SetChannel (chan);

  // Set guard interval
  phy.Set ("GuardInterval", TimeValue (NanoSeconds (guardInterval)));

  WifiMacHelper mac;
  WifiHelper wifi;
  
  wifi.SetStandard (WIFI_PHY_STANDARD_80211ax_5GHZ);

  std::ostringstream oss;
  oss << "HeMcs" << mcs;
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager","DataMode", StringValue (oss.str ()),
                               "ControlMode", StringValue (oss.str ()));
  
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

  Ssid ssid = Ssid ("ns3-80211ax");

  mac.SetType ("ns3::StaWifiMac",
               "Ssid", SsidValue (ssid),
                "ActiveProbing", BooleanValue (false),
                "BE_MaxAmpduSize", UintegerValue (0));

  NetDeviceContainer staDevice;
  staDevice = wifi.Install (phy, mac, wifiStaNode);

  mac.SetType ("ns3::ApWifiMac",
               "EnableBeaconJitter", BooleanValue (false),
               "Ssid", SsidValue (ssid));

  NetDeviceContainer apDevice;
  apDevice = wifi.Install (phy, mac, wifiApNode);

  // Set channel width
  Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/ChannelWidth", UintegerValue (channelWidth));

  // mobility.
  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();

  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (1.0, 0.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);

  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");

  mobility.Install (wifiApNode);
  mobility.Install (wifiStaNode);
  /* Internet stack*/
  InternetStackHelper stack;
  stack.Install (wifiApNode);
  stack.Install (wifiStaNode);
  //Random

  RngSeedManager::SetSeed(1);
  RngSeedManager::SetRun(rng);

  Ipv4AddressHelper address;
  address.SetBase ("192.168.1.0", "255.255.255.0");
  Ipv4InterfaceContainer staNodeInterface;
  Ipv4InterfaceContainer apNodeInterface;

  staNodeInterface = address.Assign (staDevice);
  apNodeInterface = address.Assign (apDevice);
 
  if (CW) {
    Config::Set("/$ns3::NodeListPriv/NodeList/*/$ns3::Node/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::RegularWifiMac/BE_Txop/$ns3::QosTxop/MinCw",UintegerValue(CW));
    Config::Set("/$ns3::NodeListPriv/NodeList/*/$ns3::Node/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::RegularWifiMac/BE_Txop/$ns3::QosTxop/MaxCw",UintegerValue(CW));
  }

  for(int i = 0; i < nWifi; ++i) {
    installTrafficGenerator(wifiStaNode.Get(i),wifiApNode.Get(0), port++, offeredLoad);
  }

  PopulateARPcache();
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  Ptr<FlowMonitor> monitor;
  FlowMonitorHelper flowmon;
  monitor = flowmon.InstallAll();
  monitor->SetAttribute ("StartTime", TimeValue (Seconds (warmup)));

  Simulator::Stop (Seconds (simulationTime + 1));

  Simulator::Run ();

  double flowThr;

  ofstream myfile;
  myfile.open (outputCsv, ios::app);

  /* Contents of CSV output file
  Timestamp, CW, nWifi, RngRun, SourceIP, DestinationIP, Throughput
  */

  Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier> (flowmon.GetClassifier ());
  std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats ();
  for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin (); i != stats.end (); ++i) {
      auto time = std::time(nullptr); //Get timestamp
      auto tm = *std::localtime(&time);
      Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow (i->first);
      //flowThr=i->second.rxBytes * 8.0 / (i->second.timeLastRxPacket.GetSeconds () - i->second.timeFirstTxPacket.GetSeconds ()) / 1024 / 1024;
      flowThr = i->second.rxBytes * 8.0 / simulationTime / 1000 / 1000;
      NS_LOG_UNCOND ("Flow " << i->first  << " (" << t.sourceAddress << " -> " << t.destinationAddress << ")\tThroughput: " <<  flowThr  << " Mbps\tTime: " << i->second.timeLastRxPacket.GetSeconds () - i->second.timeFirstTxPacket.GetSeconds () << " s\tRx packets " << i->second.rxPackets << "\n");
                myfile << std::put_time(&tm, "%Y-%m-%d %H:%M") << "," << CW << "," << nWifi << "," << RngSeedManager::GetRun() << "," << t.sourceAddress << "," << t.destinationAddress << "," << flowThr;
                myfile << std::endl;
        }
        myfile.close();
        
  Simulator::Destroy ();

  return 0;
}

// Traffic generator declaration from lte-wifi.cc
void installTrafficGenerator(Ptr<ns3::Node> fromNode, Ptr<ns3::Node> toNode, int port, string offeredLoad) {

        Ptr<Ipv4> ipv4 = toNode->GetObject<Ipv4> (); // Get Ipv4 instance of the node
        Ipv4Address addr = ipv4->GetAddress (1, 0).GetLocal (); // Get Ipv4InterfaceAddress of xth interface.

        ApplicationContainer sourceApplications, sinkApplications;

        uint8_t tosValue = 0x70; //AC_BE
        double simulationTime = 20;
        //Add random fuzz to app start time
        double min = 0.0;
        double max = 1.0;
        Ptr<UniformRandomVariable> fuzz = CreateObject<UniformRandomVariable> ();
        fuzz->SetAttribute ("Min", DoubleValue (min));
        fuzz->SetAttribute ("Max", DoubleValue (max));

        InetSocketAddress sinkSocket (addr, port);
        sinkSocket.SetTos (tosValue);
        //OnOffHelper onOffHelper ("ns3::TcpSocketFactory", sinkSocket);
        OnOffHelper onOffHelper ("ns3::UdpSocketFactory", sinkSocket);
        onOffHelper.SetConstantRate (DataRate (offeredLoad + "Mbps"), 1500-20-8-8);
        sourceApplications.Add (onOffHelper.Install (fromNode)); //fromNode


        //PacketSinkHelper packetSinkHelper ("ns3::TcpSocketFactory", sinkSocket);
        PacketSinkHelper packetSinkHelper ("ns3::UdpSocketFactory", sinkSocket);
        sinkApplications.Add (packetSinkHelper.Install (toNode)); //toNode

        sinkApplications.Start (Seconds (0.0));
        sinkApplications.Stop (Seconds (simulationTime + 1));
        sourceApplications.Start (Seconds (1.0+fuzz->GetValue ()));
        sourceApplications.Stop (Seconds (simulationTime + 1));

}

void PopulateARPcache () {
	Ptr<ArpCache> arp = CreateObject<ArpCache> ();
	arp->SetAliveTimeout (Seconds (3600 * 24 * 365) );

	for (NodeList::Iterator i = NodeList::Begin (); i != NodeList::End (); ++i)
	{
		Ptr<Ipv4L3Protocol> ip = (*i)->GetObject<Ipv4L3Protocol> ();
		NS_ASSERT (ip !=0);
		ObjectVectorValue interfaces;
		ip->GetAttribute ("InterfaceList", interfaces);

		for (ObjectVectorValue::Iterator j = interfaces.Begin (); j != interfaces.End (); j++)
		{
			Ptr<Ipv4Interface> ipIface = (*j).second->GetObject<Ipv4Interface> ();
			NS_ASSERT (ipIface != 0);
			Ptr<NetDevice> device = ipIface->GetDevice ();
			NS_ASSERT (device != 0);
			Mac48Address addr = Mac48Address::ConvertFrom (device->GetAddress () );

			for (uint32_t k = 0; k < ipIface->GetNAddresses (); k++)
			{
				Ipv4Address ipAddr = ipIface->GetAddress (k).GetLocal();
				if (ipAddr == Ipv4Address::GetLoopback ())
					continue;

				ArpCache::Entry *entry = arp->Add (ipAddr);
				Ipv4Header ipv4Hdr;
				ipv4Hdr.SetDestination (ipAddr);
				Ptr<Packet> p = Create<Packet> (100);
				entry->MarkWaitReply (ArpCache::Ipv4PayloadHeaderPair (p, ipv4Hdr));
				entry->MarkAlive (addr);
			}
		}
	}

	for (NodeList::Iterator i = NodeList::Begin (); i != NodeList::End (); ++i)
	{
		Ptr<Ipv4L3Protocol> ip = (*i)->GetObject<Ipv4L3Protocol> ();
		NS_ASSERT (ip !=0);
		ObjectVectorValue interfaces;
		ip->GetAttribute ("InterfaceList", interfaces);

		for (ObjectVectorValue::Iterator j = interfaces.Begin (); j != interfaces.End (); j ++)
		{
			Ptr<Ipv4Interface> ipIface = (*j).second->GetObject<Ipv4Interface> ();
			ipIface->SetAttribute ("ArpCache", PointerValue (arp) );
		}
	}
}



