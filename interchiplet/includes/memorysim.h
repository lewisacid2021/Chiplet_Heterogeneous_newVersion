#pragma once

#include "dramsim3.h"
#include <iostream>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <cstring>
#include <stdexcept>
#include "spdlog/spdlog.h"

class DRAMSim3Wrapper {
public:
    DRAMSim3Wrapper(const std::string &config_file, const std::string &output_dir, const std::string &storage_file, uint64_t memory_size);
    ~DRAMSim3Wrapper();

    void SendRequest(uint64_t address, bool is_read, const void *data, int64_t nbyte);
    void Start();
    void Stop();
    void ReadFromStorage(uint64_t addr, void *data, int64_t nbyte);
    void WriteToStorage(uint64_t addr, const void *data, int64_t nbyte);

    // 提供公共接口访问日志锁
    static std::mutex& getLogMutex() {
        return log_mutex;
    }

private:
    void ClockTickLoop();
    void ReadCallback(uint64_t address);
    void WriteCallback(uint64_t address);

    std::unique_ptr<dramsim3::MemorySystem> memory_system;
    std::thread clock_thread;
    bool exit_flag;

    std::fstream storage_file;
    std::string file_name;
    size_t memory_size;

    // 日志锁
    static std::mutex log_mutex;
};

// 初始化静态成员变量
std::mutex DRAMSim3Wrapper::log_mutex;

DRAMSim3Wrapper::DRAMSim3Wrapper(const std::string &config_file, const std::string &output_dir, const std::string &storage_filename, uint64_t memory_size)
    : exit_flag(false), memory_size(memory_size), file_name(storage_filename) {

    spdlog::set_pattern("[DRAMSIM3] %v");

    // 创建 DRAMSim3 内存系统实例并设置回调
    memory_system = std::make_unique<dramsim3::MemorySystem>(
        config_file, output_dir,
        std::bind(&DRAMSim3Wrapper::ReadCallback, this, std::placeholders::_1),
        std::bind(&DRAMSim3Wrapper::WriteCallback, this, std::placeholders::_1)
    );

    // 打开文件并初始化
    storage_file.open(storage_filename, std::ios::in | std::ios::out | std::ios::binary);
    if (!storage_file.is_open()) {
        storage_file.open(storage_filename, std::ios::out | std::ios::binary);
        storage_file.seekp(memory_size - 1);
        storage_file.write("", 1); // 扩展文件到指定大小
        storage_file.close();
        storage_file.open(storage_filename, std::ios::in | std::ios::out | std::ios::binary);
    }

    Start();
}

DRAMSim3Wrapper::~DRAMSim3Wrapper() {
    Stop();
    storage_file.close();
    if (remove(file_name.c_str()) != 0) {
        spdlog::error("Error deleting file: {}", file_name);
    } else {
        spdlog::info("File deleted successfully: {}", file_name);
    }

    // 显式打印统计信息
    memory_system->PrintStats();
}

void DRAMSim3Wrapper::Start() {
    clock_thread = std::thread(&DRAMSim3Wrapper::ClockTickLoop, this);
}

void DRAMSim3Wrapper::Stop() {
    {
        std::lock_guard<std::mutex> lock(log_mutex);
        exit_flag = true;
    }
    if (clock_thread.joinable()) clock_thread.join();
}

void DRAMSim3Wrapper::SendRequest(uint64_t address, bool is_read, const void *data, int64_t nbyte) {
    if (nbyte <= 0 || !data) {
        throw std::invalid_argument("Invalid request parameters");
    }

    // 加锁后记录日志
    {
        std::lock_guard<std::mutex> log_lock(log_mutex);
        spdlog::debug("Request added: address=0x{:x}, type={}, size={} bytes", address, is_read ? "read" : "write", nbyte);
    }

    // 直接添加事务到 DRAMsim3
    memory_system->AddTransaction(address, !is_read);
}

void DRAMSim3Wrapper::ClockTickLoop() {
    while (true) {
        {
            std::lock_guard<std::mutex> log_lock(log_mutex);
            if (exit_flag) break;
        }

        memory_system->ClockTick();
        std::this_thread::sleep_for(std::chrono::microseconds(10));  // 控制时钟频率
    }
}

void DRAMSim3Wrapper::ReadCallback(uint64_t address) {
    // 加锁后记录日志
    {
        std::lock_guard<std::mutex> log_lock(log_mutex);
        spdlog::info("Read complete for address: 0x{:x}", address);
    }
}

void DRAMSim3Wrapper::WriteCallback(uint64_t address) {
    // 加锁后记录日志
    {
        std::lock_guard<std::mutex> log_lock(log_mutex);
        spdlog::info("Write complete for address: 0x{:x}", address);
    }
}

void DRAMSim3Wrapper::ReadFromStorage(uint64_t addr, void *data, int64_t nbyte) {
    storage_file.seekg(addr);
    if (!storage_file.read(reinterpret_cast<char*>(data), nbyte)) {
        // 加锁后记录日志
        {
            std::lock_guard<std::mutex> log_lock(log_mutex);
            spdlog::error("Failed to read from storage at address: 0x{:x}", addr);
        }
        throw std::runtime_error("File read error");
    }
}

void DRAMSim3Wrapper::WriteToStorage(uint64_t addr, const void *data, int64_t nbyte) {
    storage_file.seekp(addr);
    if (!storage_file.write(reinterpret_cast<const char*>(data), nbyte)) {
        // 加锁后记录日志
        {
            std::lock_guard<std::mutex> log_lock(log_mutex);
            spdlog::error("Failed to write to storage at address: 0x{:x}", addr);
        }
        throw std::runtime_error("File write error");
    }
    storage_file.flush();
}