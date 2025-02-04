#include "memorysim.h"
#include "cmd_handler.h"
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>

class CmdLineOptions {
public:
    // 解析命令行参数
    void parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.substr(0, 2) == "--") {  // 长选项
                std::string key = arg.substr(2);
                std::string value = (i + 1 < argc && argv[i + 1][0] != '-') ? argv[++i] : "";
                options[key] = value;
            } else if (arg[0] == '-') {  // 短选项
                std::string key = arg.substr(1);
                std::string value = (i + 1 < argc && argv[i + 1][0] != '-') ? argv[++i] : "";
                options[key] = value;
            } else {  // 位置参数
                positionalArgs.push_back(arg);
            }
        }
    }

    // 获取选项值
    std::string getOption(const std::string& key, const std::string& defaultValue = "") const {
        auto it = options.find(key);
        return (it != options.end()) ? it->second : defaultValue;
    }

    // 检查选项是否存在
    bool hasOption(const std::string& key) const {
        return options.find(key) != options.end();
    }

    // 获取位置参数
    const std::vector<std::string>& getPositionalArgs() const {
        return positionalArgs;
    }

private:
    std::unordered_map<std::string, std::string> options;
    std::vector<std::string> positionalArgs;
};

int idx, idy;

int is_stop = 0;

int main(int argc,char * argv[])
{
    spdlog::set_pattern("[DRAMSIM3] %v");

    CmdLineOptions cmd;
    cmd.parse(argc, argv);

    // 获取命令行选项
    if (cmd.hasOption("help")||cmd.hasOption("h")) {
        std::cout << "Usage: program [options] <src_x> <src_y>\n"
                  << "Options:\n"
                  << "  -h, --help         Show this help message\n"
                  << "  -c <config_file>           Set dramsim3 config file\n"
                  << "  -o <output_dir>    Set the memory info output_dir\n"
                  << "  -n <filename>         Set the memory storage file name\n"
                  << "  -s <value>  Set the memory storage size /MB\n";
        return 0;
    }

    //获取命令行选项
    std::string config_file = cmd.getOption("c", "configs/DDR4_8Gb_x8_2400.ini");
    std::string output_dir = cmd.getOption("o", "results");
    std::string storage_filename = cmd.getOption("n", "storage.bin");
    uint64_t storage_size = std::stoull(cmd.getOption("s", "8192")) * 1024 * 1024;

    // 初始化DRAMSim3
    DRAMSim3Wrapper dramsim3_wrapper(config_file, output_dir, storage_filename, storage_size);

    // 获取位置参数
    const auto& args = cmd.getPositionalArgs();
    
    if(args.size() != 2)
    {
        {
            std::lock_guard<std::mutex> lock(DRAMSim3Wrapper::getLogMutex());
            spdlog::error("Usage: {} <src_x> <src_y>", args[0]);
        }
        
        return -1;
    }
    else {
        idx = std::stoi(args[0]);
        idy = std::stoi(args[1]);
    }


    // 向框架发送启动命令 登记DDR内存
    InterChiplet::sendStartMemCmd(idx,idy);
    InterChiplet::SyncCommand __cmd = InterChiplet::parseCmd();
    if(__cmd.m_res_list[0] == "0")
    {
        {
            std::lock_guard<std::mutex> lock(DRAMSim3Wrapper::getLogMutex());
            spdlog::error("Memory [{},{}] has been duplicately registered", idx, idy);
        }
        return -1;
    }
    else
    {
        {
            std::lock_guard<std::mutex> lock(DRAMSim3Wrapper::getLogMutex());
            spdlog::info("Memory [{},{}] has been registered.", idx, idy);
        }
    }
    

    while(!is_stop)
    {
        // 读取命令
        InterChiplet::SyncCommand __cmd = InterChiplet::parseCmd();
        switch (__cmd.m_type)
        {
        case InterChiplet::SC_READMEM:
        {
            std::string file_name="../"+InterChiplet::pipeName(__cmd.m_src,__cmd.m_dst);
            uint64_t  addr = __cmd.m_addr;
            uint64_t nbytes = __cmd.m_nbytes;

            char * buffer = new char[nbytes];
            memset(buffer,0,nbytes);

            dramsim3_wrapper.SendRequest(addr,true,buffer,nbytes);
            dramsim3_wrapper.ReadFromStorage(addr,buffer,nbytes);
            writeMemData(file_name, buffer, nbytes);

            delete[] buffer;
            
            {
                std::lock_guard<std::mutex> lock(DRAMSim3Wrapper::getLogMutex());
                InterChiplet::sendResultMemCmd(__cmd.m_src[0],__cmd.m_src[1],__cmd.m_dst[0],__cmd.m_dst[1],addr,nbytes);
            }

            break;
        }
        case InterChiplet::SC_WRITEMEM:
        {
            std::string file_name="../"+InterChiplet::pipeName(__cmd.m_src,__cmd.m_dst);
            uint64_t  addr = __cmd.m_addr;
            uint64_t nbytes = __cmd.m_nbytes;

            char * buffer = new char[nbytes];
            memset(buffer,0,nbytes);

            readMemData(file_name, buffer, nbytes);
            dramsim3_wrapper.SendRequest(addr,false,buffer,nbytes);
            dramsim3_wrapper.WriteToStorage(addr,buffer,nbytes);

            delete[] buffer;

            {
                std::lock_guard<std::mutex> lock(DRAMSim3Wrapper::getLogMutex());
                InterChiplet::sendResultMemCmd(__cmd.m_src[0],__cmd.m_src[1],__cmd.m_dst[0],__cmd.m_dst[1],addr,nbytes);
            }

            break;
        }
        case InterChiplet::SC_STOPMEM:
            is_stop = 1;
            break;
        default:
            break;
        }
    }

    
}