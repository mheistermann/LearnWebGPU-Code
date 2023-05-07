#pragma once
#include <webgpu.hpp>
#include <wgpu.h>
#include <vector>

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

} // namespace MAL
