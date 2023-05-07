#pragma once
#include <webgpu.hpp>
#include <wgpu.h>
#include <vector>
#include <filesystem>
#include <fstream>

// Martin's abstraction layer
// Give up a lot of flexibility for some convenience.
namespace MAL {

class Buffer {
public:
    Buffer() = default;
    Buffer(wgpu::Device &_device,
            WGPUBufferDescriptor _desc)
        : desc_(_desc)
        , buffer_{_device.createBuffer(_desc)}
    {
    }
    wgpu::Buffer &get() {
        return buffer_;
    }
    wgpu::BufferDescriptor desc() const {
        return desc_;
    }
    size_t size() const {
        return desc_.size;
    }
private:
    wgpu::BufferDescriptor desc_;
    wgpu::Buffer buffer_ = nullptr;
};

template<typename T>
class StaticVectorBuffer : public Buffer {
public:
    StaticVectorBuffer() = default;
    StaticVectorBuffer(wgpu::Device &_device,
            std::vector<T> const &_vec,
            WGPUBufferUsageFlags _usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Storage)
        : Buffer{_device,
            WGPUBufferDescriptor{
                .nextInChain = nullptr,
                .label = "StaticVectorBuffer",
                .size = _vec.size() * sizeof(_vec.front()),
                        .usage=_usage}}
    {
        auto queue = _device.getQueue();
        queue.writeBuffer(get(), 0, _vec.data(), desc().size);
    }
};

struct BindGroupWithLayout {
    wgpu::BindGroupLayout layout = nullptr;
    wgpu::BindGroup group = nullptr;
};

class BindGroupBuilder {
public:
    BindGroupBuilder() = default;
    BindGroupBuilder &add_buffer(Buffer _buffer,
            WGPUShaderStageFlags _visibility,
            WGPUBufferBindingType _type)
    {
        auto idx = static_cast<uint32_t>(layout_entries_.size());
        layout_entries_.emplace_back(WGPUBindGroupLayoutEntry{
                .binding = idx,
                .visibility = _visibility,
                .buffer.type = _type,
                });
        binding_entries_.emplace_back(WGPUBindGroupEntry{
                .binding = idx,
                .buffer = _buffer.get(),
                .offset = 0,
                .size = _buffer.desc().size});

        return *this;
    }
    BindGroupWithLayout build(wgpu::Device &_device)
    {
        BindGroupWithLayout bg;
        bg.layout = _device.createBindGroupLayout(WGPUBindGroupLayoutDescriptor{
            .entryCount = static_cast<uint32_t>(layout_entries_.size()),
            .entries = layout_entries_.data()});
        bg.group = _device.createBindGroup(WGPUBindGroupDescriptor{
                .layout = bg.layout,
                .entryCount=static_cast<uint32_t>(binding_entries_.size()),
                .entries = binding_entries_.data()});
        return bg;
    }
private:
    std::vector<wgpu::BindGroupLayoutEntry> layout_entries_;
    std::vector<wgpu::BindGroupEntry> binding_entries_;
};


class ShaderLoader
{
    using path = std::filesystem::path;
public:
    ShaderLoader() = default;
    ShaderLoader(wgpu::Device &_device, const path& _common_path)
        : device_(_device)
        , common_src_{slurp(_common_path)}
        , desc_common_{WGPUShaderModuleWGSLDescriptor{
            .chain = {.next = nullptr,
                .sType = wgpu::SType::ShaderModuleWGSLDescriptor,},
            .code=common_src_.c_str()
        }}
    {
    }

    wgpu::ShaderModule load(const path& _path)
    {
        auto source = common_src_ + slurp(_path);
        auto wgsl_desc = WGPUShaderModuleWGSLDescriptor{
            .chain = {
                .next = nullptr,//&desc_common_.chain,
                .sType = wgpu::SType::ShaderModuleWGSLDescriptor,
            },
            .code = source.c_str(),
        };
        auto desc = WGPUShaderModuleDescriptor{
            .nextInChain = &wgsl_desc.chain,
            //.label=_path.c_str(),
            .hintCount = 0,
            .hints = nullptr,
        };
        return wgpuDeviceCreateShaderModule(device_, &desc);
    }
private:
    static std::string slurp(const path &_path) {
        std::ifstream file(_path);
        if (!file.is_open()) {
            throw std::runtime_error("cant open shader file " + _path.string());
        }
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        std::string contents(size, ' ');
        file.seekg(0);
        file.read(contents.data(), size);
        return contents;
    }
    wgpu::Device device_ = nullptr;
    std::string common_src_;
    wgpu::ShaderModuleWGSLDescriptor desc_common_;
};

} // namespace MAL
