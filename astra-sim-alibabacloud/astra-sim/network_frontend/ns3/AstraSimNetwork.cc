/* 
*Copyright (c) 2024, Alibaba Group;
*Licensed under the Apache License, Version 2.0 (the "License");
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at

*   http://www.apache.org/licenses/LICENSE-2.0

*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an "AS IS" BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
*/

#include "astra-sim/system/AstraNetworkAPI.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/RecvPacketEventHadndlerData.hh"
#include "astra-sim/system/Common.hh"
#include "SimCCL/mock/MockNcclLog.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "entry.h"
#include "MultiJobScheduler.h"
#include <execinfo.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <stdio.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <sys/stat.h>
#ifdef NS3_MTP
#include "ns3/mtp-interface.h"
#endif
#ifdef NS3_MPI
#include "ns3/mpi-interface.h"
#include <mpi.h>
#endif

#define RESULT_PATH "./ncclFlowModel_"

using namespace std;
using namespace ns3;

extern std::map<FlowKey, AstraSim::ncclFlowTag> receiver_pending_queue;
extern uint32_t node_num, switch_num, link_num, trace_num, nvswitch_num, gpus_per_server;
extern GPUType gpu_type;
extern std::vector<int>NVswitchs;

struct sim_event {
  void *buffer;
  uint64_t count;
  int type;
  int dst;
  int tag;
  string fnType;
};

class ASTRASimNetwork;
static std::vector<AstraSim::Sys*> g_job_systems;
static std::vector<ASTRASimNetwork*> g_job_networks;
static std::unique_ptr<SimAiMultiJob::Scheduler> g_job_scheduler;
static std::map<uint32_t, uint32_t> g_job_finished_ranks;
static std::map<uint32_t, uint32_t> g_job_expected_ranks;
static uint32_t g_next_runtime_offset = 0;
static std::string BuildTraceWorkload(const SimAiMultiJob::JobSpec& job) {
  if (job.communication_bytes == 0) return job.workload;
  const std::string directory = "./results/trace_workloads";
  const std::string path = directory + "/job_" +
                           std::to_string(job.job_id) + ".txt";
  ::mkdir("./results", 0755);
  ::mkdir(directory.c_str(), 0755);
  std::ofstream output(path.c_str(), std::ios::out | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create trace workload: " + path);
  }
  output << "HYBRID_TRANSFORMER_FWD_IN_BCKWD "
         << "model_parallel_NPU_group: " << job.num_gpus
         << " ep: 1 pp: 1 vpp: 1 ga: 1 all_gpus: " << job.num_gpus
         << " checkpoints: 0 checkpoint_initiates: 0 pp_comm: 0\n";
  output << "1\n";
  output << "trace_job -1 1 ALLREDUCE " << job.communication_bytes
         << " 1 NONE 0 1 NONE 0 1\n";
  return path;
}
static std::string JobResultPath(const SimAiMultiJob::JobSpec& job) {
  const std::string marker = "/workloads/";
  const std::string workload = job.workload;
  const std::size_t marker_pos = workload.find(marker);
  std::string base = marker_pos == std::string::npos
                         ? "./results/trace_jobs"
                         : workload.substr(0, marker_pos);
  const std::string path = base + "/logs/job_" +
                           std::to_string(job.job_id) + "/";
  std::string current;
  for (char ch : path) {
    current.push_back(ch);
    if (ch == '/') {
      ::mkdir(current.c_str(), 0755);
    }
  }
  return path;
}
static void FinishScheduledJob(uint32_t job_id) {
  if (g_job_scheduler) {
    g_job_scheduler->Finish(job_id, Simulator::Now().GetNanoSeconds());
    if (g_job_scheduler->Empty()) {
      Simulator::Stop();
    }
  }
}
static void SubmitScheduledJob(SimAiMultiJob::JobSpec job) {
  if (g_job_scheduler) {
    g_job_scheduler->Submit(job, Simulator::Now().GetNanoSeconds());
  }
}

class ASTRASimNetwork : public AstraSim::AstraNetworkAPI {
private:
  int runtime_offset;
  int local_gpu_count;
  std::vector<int> gpu_map;
  std::vector<int> switch_map;
  int map_node(int node) const {
    if (node >= 0 && node < static_cast<int>(gpu_map.size())) {
      return gpu_map[node];
    }
    const int switch_index = node - local_gpu_count;
    if (switch_index >= 0 && switch_index < static_cast<int>(switch_map.size())) {
      return switch_map[switch_index];
    }
    throw std::runtime_error("job-local network node is outside its placement");
  }

public:
  queue<sim_event> sim_event_queue;
  ASTRASimNetwork(int rank, int runtime_offset,
                  const std::vector<int>& gpu_map,
                  const std::vector<int>& switch_map)
      : AstraNetworkAPI(rank),
        runtime_offset(runtime_offset),
        local_gpu_count(static_cast<int>(gpu_map.size())),
        gpu_map(gpu_map),
        switch_map(switch_map) {
  }
  ASTRASimNetwork(int rank, int runtime_offset)
      : ASTRASimNetwork(
            rank, runtime_offset,
            [&]() {
              std::vector<int> ids(node_num - switch_num - nvswitch_num);
              for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int>(i);
              return ids;
            }(),
            [&]() {
              std::vector<int> ids;
              for (uint32_t i = 0; i < nvswitch_num; ++i) {
                ids.push_back(static_cast<int>(node_num - switch_num - nvswitch_num + i));
              }
              return ids;
            }()) {}
  ~ASTRASimNetwork() {}
  int sim_comm_size(AstraSim::sim_comm comm, int *size) { return 0; }
  int sim_finish() {
    if (g_job_scheduler && !g_job_scheduler->Empty()) {
      return 0;
    }
    for (auto it = nodeHash.begin(); it != nodeHash.end(); it++) {
      pair<int, int> p = it->first;
      if (p.second == 0) {
        std::cout << "sim_finish on sent, " << " Thread id: " << pthread_self() << std::endl;
        cout << "All data sent from node " << p.first << " is " << it->second
             << "\n";
      } else {
        std::cout << "sim_finish on received, " << " Thread id: " << pthread_self() << std::endl;
        cout << "All data received by node " << p.first << " is " << it->second
             << "\n";
      }
    }
    exit(0);
    return 0;
  }
  double sim_time_resolution() { return 0; }
  int sim_init(AstraSim::AstraMemoryAPI *MEM) { return 0; }
  AstraSim::timespec_t sim_get_time() {
    AstraSim::timespec_t timeSpec;
    timeSpec.time_val = Simulator::Now().GetNanoSeconds();
    return timeSpec;
  }
  virtual void sim_schedule(AstraSim::timespec_t delta,
                            void (*fun_ptr)(void *fun_arg), void *fun_arg) {
    task1 t;
    t.type = 2;
    t.fun_arg = fun_arg;
    t.msg_handler = fun_ptr;
    t.schTime = delta.time_val;
    Simulator::Schedule(NanoSeconds(t.schTime), t.msg_handler, t.fun_arg);
    return;
  }
  virtual int sim_send(void *buffer,   
                       uint64_t count, 
                       int type,       
                       int dst,
                       int tag,                       
                       AstraSim::sim_request *request, 
                       void (*msg_handler)(void *fun_arg), void *fun_arg) {
    const int physical_src = map_node(rank);
    dst = map_node(dst);
    task1 t;
    t.src = physical_src;
    t.dest = dst;
    t.count = count;
    t.type = 0;
    t.fun_arg = fun_arg;
    t.msg_handler = msg_handler;
    {
      #ifdef NS3_MTP
      MtpInterface::explicitCriticalSection cs;
      #endif
      sentHash[make_flow_key(request->flowTag, t.src, t.dest)] = t;
      #ifdef NS3_MTP
      cs.ExitSection();
      #endif
    }
    SendFlow(physical_src, dst, count, msg_handler, fun_arg, tag, request);
    return 0;
  }
  virtual int sim_recv(void *buffer, uint64_t count, int type, int src, int tag,
                       AstraSim::sim_request *request,
                       void (*msg_handler)(void *fun_arg), void *fun_arg) {
    #ifdef NS3_MTP
    MtpInterface::explicitCriticalSection cs;
    #endif
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    AstraSim::ncclFlowTag flowTag = request->flowTag;
    src = map_node(src);
    task1 t;
    t.src = src;
    t.dest = map_node(rank);
    t.count = count;
    t.type = 1;
    t.fun_arg = fun_arg;
    t.msg_handler = msg_handler;
    AstraSim::RecvPacketEventHadndlerData* ehd = (AstraSim::RecvPacketEventHadndlerData*) t.fun_arg;
    AstraSim::EventType event = ehd->event;
    ehd->flowTag.job_id = flowTag.job_id;
    tag = ehd->flowTag.tag_id;
    FlowKey key = make_flow_key(ehd->flowTag, t.src, t.dest);
    FlowKey pending_key = make_flow_key(ehd->flowTag, t.dest, t.src);
    NcclLog->writeLog(NcclLogLevel::DEBUG,"[Receive event registration] src %d sim_recv on rank %d tag_id %d channdl id %d",src,rank,tag,ehd->flowTag.channel_id);
    
    if (recvHash.find(key) != recvHash.end()) {
      uint64_t count = recvHash[key];
      if (count == t.count) {
        recvHash.erase(key);
        assert(ehd->flowTag.child_flow_id == -1 && ehd->flowTag.current_flow_id == -1);
        if(receiver_pending_queue.count(pending_key)!= 0) {
          AstraSim::ncclFlowTag pending_tag = receiver_pending_queue[pending_key];
          receiver_pending_queue.erase(pending_key);
          ehd->flowTag = pending_tag;
        } 
        #ifdef NS3_MTP
        cs.ExitSection();
        #endif
        t.msg_handler(t.fun_arg);
        goto sim_recv_end_section;
      } else if (count > t.count) {
        recvHash[key] = count - t.count;
        assert(ehd->flowTag.child_flow_id == -1 && ehd->flowTag.current_flow_id == -1);
        if(receiver_pending_queue.count(pending_key)!= 0) {
          AstraSim::ncclFlowTag pending_tag = receiver_pending_queue[pending_key];
          receiver_pending_queue.erase(pending_key);
          ehd->flowTag = pending_tag;
        } 
        #ifdef NS3_MTP
        cs.ExitSection();
        #endif
        t.msg_handler(t.fun_arg);
        goto sim_recv_end_section;
      } else {
        recvHash.erase(key);
        t.count -= count;
        expeRecvHash[key] = t;
      }
    } else {
      if (expeRecvHash.find(key) == expeRecvHash.end()) {
        expeRecvHash[key] = t;
          NcclLog->writeLog(NcclLogLevel::DEBUG," [Packet arrived late, registering first] recvHash do not find expeRecvHash.new make src  %d dest  %d t.count:  %llu channel_id  %d current_flow_id  %d",t.src,t.dest,t.count,tag,flowTag.current_flow_id);
          
      } else {
        uint64_t expecount =
            expeRecvHash[key].count;
          NcclLog->writeLog(NcclLogLevel::DEBUG," [Packet arrived late, re-registering] recvHash do not find expeRecvHash.add make src  %d dest  %d expecount:  %d t.count:  %d tag_id  %d current_flow_id  %d",t.src,t.dest,expecount,t.count,tag,flowTag.current_flow_id);
          
      }
    }
    #ifdef NS3_MTP
    cs.ExitSection();
    #endif

sim_recv_end_section:    
    return 0;
  }
  void handleEvent(int dst, int cnt) {
  }
};

static void StartJob(const SimAiMultiJob::JobSpec& job,
                     const SimAiMultiJob::Placement& placement) {
  const uint32_t gpu_num = node_num - switch_num - nvswitch_num;
  const uint32_t gpus_per_job_node = std::max(1u, gpus_per_server);
  if (placement.gpu_ids.size() != placement.num_gpus ||
      placement.num_gpus != job.num_gpus) {
    throw std::runtime_error("scheduler returned an invalid GPU placement");
  }
  if (NVswitchs.empty()) {
    for (uint32_t i = 0; i < nvswitch_num; ++i) {
      NVswitchs.push_back(static_cast<int>(gpu_num + i));
    }
  }
  for (uint32_t gpu_id : placement.gpu_ids) {
    if (gpu_id >= gpu_num) {
      throw std::runtime_error("job placement exceeds GPU nodes");
    }
  }
  const bool full_server_layout =
      placement.num_gpus % gpus_per_job_node == 0;

  // SimCCL keeps ranks local to a job.  The network frontend is responsible
  // for translating those local ranks to the selected physical nodes.
  std::vector<int> physical_gpus;
  physical_gpus.reserve(placement.gpu_ids.size());
  for (uint32_t gpu_id : placement.gpu_ids) {
    physical_gpus.push_back(static_cast<int>(gpu_id));
  }
  std::vector<int> physical_switches;
  std::vector<int> logical_switches;
  std::vector<uint32_t> selected_servers;
  const uint32_t logical_gpus_per_node = full_server_layout
                                             ? gpus_per_job_node
                                             : placement.num_gpus;
  for (uint32_t local = 0; local < placement.num_gpus;
       local += logical_gpus_per_node) {
    const uint32_t server = placement.gpu_ids[local] / gpus_per_job_node;
    if (full_server_layout) {
      for (uint32_t rank = local; rank < local + logical_gpus_per_node; ++rank) {
        if (placement.gpu_ids[rank] / gpus_per_job_node != server) {
          throw std::runtime_error("placement splits a physical GPU server");
        }
      }
    }
    if (server >= NVswitchs.size()) {
      throw std::runtime_error("job placement has no NVSwitch mapping");
    }
    selected_servers.push_back(server);
    physical_switches.push_back(NVswitchs[server]);
    logical_switches.push_back(static_cast<int>(placement.num_gpus +
                                                selected_servers.size() - 1));
  }

  g_job_expected_ranks[job.job_id] = placement.num_gpus;
  g_job_finished_ranks[job.job_id] = 0;
  const uint32_t runtime_offset = g_next_runtime_offset;
  g_next_runtime_offset += placement.num_gpus;
  if (g_job_systems.size() < runtime_offset + placement.num_gpus) {
    g_job_systems.resize(runtime_offset + placement.num_gpus, nullptr);
    g_job_networks.resize(runtime_offset + placement.num_gpus, nullptr);
  }
  for (uint32_t local = 0; local < placement.num_gpus; ++local) {
    const uint32_t runtime_rank = runtime_offset + local;
    std::vector<int> local_gpu_map = physical_gpus;
    std::vector<int> local_switch_map = physical_switches;
    g_job_networks[runtime_rank] = new ASTRASimNetwork(
        local, runtime_offset, local_gpu_map, local_switch_map);
    const std::string workload_path = BuildTraceWorkload(job);
    const bool trace_job = job.name.compare(0, 6, "trace-") == 0 ||
                           job.workload.find("cluster-trace") != std::string::npos;
    const std::string result_path = trace_job ? JobResultPath(job) : RESULT_PATH;
    g_job_systems[runtime_rank] = new AstraSim::Sys(
        g_job_networks[runtime_rank], nullptr, local, runtime_offset, 1,
        {static_cast<int>(placement.num_gpus)}, {1}, "", workload_path, 1, 1,
        1, 1, 0, result_path, job.name, true, false, gpu_type,
        {static_cast<int>(placement.num_gpus)}, logical_switches,
        logical_gpus_per_node, job.job_id);
    if (g_job_systems[runtime_rank]->workload == nullptr ||
        g_job_systems[runtime_rank]->workload->all_gpus !=
            static_cast<int>(placement.num_gpus)) {
      throw std::runtime_error("workload all_gpus does not match its placement");
    }
    g_job_systems[runtime_rank]->job_id = job.job_id;
    g_job_systems[runtime_rank]->num_gpus = placement.num_gpus;
    g_job_systems[runtime_rank]->nvswitch_id =
        logical_switches[local / logical_gpus_per_node];
    g_job_systems[runtime_rank]->workload_finished_callback =
        [](uint32_t finished_job_id) {
          uint32_t& finished = g_job_finished_ranks[finished_job_id];
          if (++finished == g_job_expected_ranks[finished_job_id] &&
              g_job_scheduler) {
            Simulator::Schedule(NanoSeconds(1), &FinishScheduledJob,
                                finished_job_id);
          }
        };
    g_job_systems[runtime_rank]->workload->fire();
  }
}

struct user_param {
  int thread;
  string workload;
  string network_topo;
  string network_conf;
  string jobs_file;
  string scheduler_policy;
  string gpu_resource_file;
  string scheduler_log_file;
  uint64_t gpu_memory_capacity;
  uint64_t host_memory_capacity;
  user_param() {
    thread = 1;
    workload = "";
    network_topo = "";
    network_conf = "";
    jobs_file = "";
    scheduler_policy = "first-fit";
    gpu_resource_file = "";
    scheduler_log_file = "SimAI_scheduler.csv";
    gpu_memory_capacity = 0;
    host_memory_capacity = 0;
  };
  ~user_param(){};
};

static int user_param_prase(int argc,char * argv[],struct user_param* user_param){
  int opt;
  while ((opt = getopt(argc,argv,"ht:w:g:s:n:c:j:p:m:H:R:L:"))!=-1){
    switch (opt)
    {
    case 'h':
      /* code */
      std::cout<<"-t    number of threads,default 1"<<std::endl;
      std::cout<<"-w    workloads default none "<<std::endl;
      std::cout<<"-n    network topo"<<std::endl;
      std::cout<<"-c    network_conf"<<std::endl;
      std::cout<<"-j    CSV job file: job_id,workload,gpus,submit_time_ns[,name[,gpu_memory_bytes[,host_memory_bytes[,communication_bytes]]]]"<<std::endl;
      std::cout<<"-p    multi-job policy: first-fit or best-fit"<<std::endl;
      std::cout<<"-m    per-GPU memory capacity in bytes (0 means unspecified)"<<std::endl;
      std::cout<<"-H    per-GPU host memory capacity in bytes (0 means unspecified)"<<std::endl;
      std::cout<<"-R    per-GPU resource CSV: gpu_id,gpu_memory_bytes,host_memory_bytes"<<std::endl;
      std::cout<<"-L    scheduler CSV log path"<<std::endl;
      return 1;
      break;
    case 't':
      user_param->thread = stoi(optarg);
      break;
    case 'w':
      user_param->workload = optarg;
      break;
    case 'n':
      user_param->network_topo = optarg;
      break;
    case 'c':
      user_param->network_conf = optarg;
      break;
    case 'j':
      user_param->jobs_file = optarg;
      break;
    case 'p':
      user_param->scheduler_policy = optarg;
      break;
    case 'm':
      user_param->gpu_memory_capacity = stoull(optarg);
      break;
    case 'H':
      user_param->host_memory_capacity = stoull(optarg);
      break;
    case 'R':
      user_param->gpu_resource_file = optarg;
      break;
    case 'L':
      user_param->scheduler_log_file = optarg;
      break;
    default:
      std::cerr<<"-h    help message"<<std::endl;
      return 1;
    }
  }
  return 0 ;
}

int main(int argc, char *argv[]) {
  struct user_param user_param;
  MockNcclLog::set_log_name("SimAI.log");
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(NcclLogLevel::INFO," init SimAI.log ");
  if(user_param_prase(argc,argv,&user_param)){
    return 0;
  }
  #ifdef NS3_MTP
  MtpInterface::Enable(user_param.thread);
  #endif
  
  main1(user_param.network_topo,user_param.network_conf);
  int nodes_num = node_num - switch_num;
  int gpu_num = node_num - nvswitch_num - switch_num;

  if (!user_param.jobs_file.empty()) {
    try {
      std::vector<SimAiMultiJob::JobSpec> jobs =
          SimAiMultiJob::LoadJobs(user_param.jobs_file);
      if (jobs.empty()) {
        throw std::runtime_error("job file contains no jobs");
      }
      for (const auto& job : jobs) {
        if (job.num_gpus == 0 || job.num_gpus > static_cast<uint32_t>(gpu_num)) {
          throw std::runtime_error("job " + std::to_string(job.job_id) +
                                   " GPU count exceeds the simulated GPU capacity");
        }
      }
      g_job_systems.assign(gpu_num, nullptr);
      g_job_networks.assign(gpu_num, nullptr);
      std::vector<SimAiMultiJob::GpuResource> resources(gpu_num);
      for (auto& resource : resources) {
        resource.gpu_memory_capacity_bytes = user_param.gpu_memory_capacity;
        resource.host_memory_capacity_bytes = user_param.host_memory_capacity;
      }
      if (!user_param.gpu_resource_file.empty()) {
        resources = SimAiMultiJob::LoadGpuResources(
            user_param.gpu_resource_file, gpu_num,
            user_param.gpu_memory_capacity, user_param.host_memory_capacity);
      }
      g_job_scheduler.reset(new SimAiMultiJob::Scheduler(
          resources, gpus_per_server,
          SimAiMultiJob::ParsePolicy(user_param.scheduler_policy)));
      g_job_scheduler->SetLogFile(user_param.scheduler_log_file);
      AstraSim::Sys::SetMultiJobMode(true);
      g_job_scheduler->SetStartCallback(StartJob);
      for (const auto& job : jobs) {
        Simulator::Schedule(NanoSeconds(job.submit_time_ns),
                            &SubmitScheduledJob, job);
      }
      Simulator::Stop(Seconds(simulator_stop_time));
      std::cout << "Running " << jobs.size() << " scheduled SimAI workloads.\n";
      Simulator::Run();
      Simulator::Destroy();
      #ifdef NS3_MPI
      MpiInterface::Disable();
      #endif
      return 0;
    } catch (const std::exception& error) {
      std::cerr << "multi-job setup failed: " << error.what() << std::endl;
      return 1;
    }
  }

  std::map<int, int> node2nvswitch; 
  for(int i = 0; i < gpu_num; ++ i) {
    node2nvswitch[i] = gpu_num + i / gpus_per_server;
  }
  for(int i = gpu_num; i < gpu_num + nvswitch_num; ++ i){
    node2nvswitch[i] = i;
    NVswitchs.push_back(i);
  } 

  LogComponentEnable("OnOffApplication", LOG_LEVEL_INFO);
  LogComponentEnable("PacketSink", LOG_LEVEL_INFO);
  LogComponentEnable("GENERIC_SIMULATION", LOG_LEVEL_INFO);

  std::vector<ASTRASimNetwork *> networks(nodes_num, nullptr);
  std::vector<AstraSim::Sys *> systems(nodes_num, nullptr);

  for (int j = 0; j < nodes_num; j++) {
    networks[j] =
        new ASTRASimNetwork(j ,0);
    systems[j ] = new AstraSim::Sys(
        networks[j], 
        nullptr,                  
        j,                        
        0,               
        1,                        
        {nodes_num},        
        {1},          
        "", 
        user_param.workload, 
        1, 
        1,          
        1,          
        1,
        0,                 
        RESULT_PATH, 
        "test1",            
        true,               
        false,               
        gpu_type,
        {gpu_num},
        NVswitchs,
        gpus_per_server
    );
    systems[j ]->nvswitch_id = node2nvswitch[j];
    systems[j ]->num_gpus = nodes_num - nvswitch_num;
  }
  for (int i = 0; i < nodes_num; i++) {
    systems[i]->workload->fire();
  }
  std::cout << "simulator run " << std::endl;

  Simulator::Run();
  Simulator::Stop(Seconds(2000000000));
  Simulator::Destroy();
  
  #ifdef NS3_MPI
  MpiInterface::Disable ();
  #endif
  return 0;
}
