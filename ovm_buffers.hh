#pragma once
#include "mal.hh"
#include <array>
#include <webgpu.hpp>

using Vec3 = std::array<float, 3>;
using Vec4 = std::array<float, 4>;
using Tet = std::array<uint32_t, 4>;

struct TetMeshData {
    std::vector<Vec3> vertices;
    std::vector<Tet> tets;
};

class TetMeshBuffer {
public:
    TetMeshBuffer() = default;
    TetMeshBuffer(wgpu::Device &_device,
            TetMeshData const&data,
            WGPUShaderStageFlags _visibility = wgpu::ShaderStage::Compute)
        : vertices_{_device, data.vertices}
        , tets_{_device, data.tets}
        , bind_group_{MAL::BindGroupBuilder()
            .add_buffer(vertices_, _visibility, wgpu::BufferBindingType::ReadOnlyStorage)
            .add_buffer(tets_, _visibility, wgpu::BufferBindingType::ReadOnlyStorage)
            .build(_device)}
        , n_tets_(data.tets.size())
    {
    }
    MAL::BindGroupWithLayout const &bind_group() const {
        return bind_group_;
    }
    size_t n_tets() const {
        return n_tets_;
    }
    private:
        MAL::StaticVectorBuffer<Vec3> vertices_;
        MAL::StaticVectorBuffer<Tet> tets_;
        MAL::BindGroupWithLayout bind_group_;
        size_t n_tets_;
};


inline TetMeshBuffer make_tet_mesh_buffer(wgpu::Device &_device,
            WGPUShaderStageFlags _visibility = wgpu::ShaderStage::Compute)
{
    TetMeshData data;
    data.vertices.push_back({.5,.5,.5});
    data.vertices.push_back({0,0,1});
    data.vertices.push_back({0,1,0});
    data.vertices.push_back({0,1,0});
    data.tets.push_back({0,1,2,3});
    return TetMeshBuffer(_device, data, _visibility);
}

// computed by compute pipeline, used in render pipeline
class TetVertsBuffer {
    struct Entry {
        std::array<Vec4, 4> positions;
        // TODO: 4 vectors of 3 gets padded to 4 of 4 in shader size, why?
    };
public:
    TetVertsBuffer() = default;
    TetVertsBuffer(wgpu::Device &_device, size_t n_tets)
        : buffer_(_device, {
                .nextInChain = nullptr,
                .label = "TetVertsBuffer",
                .size = n_tets * sizeof(Entry),
                .usage = wgpu::BufferUsage::Storage,
                })
        , bind_group_read_{MAL::BindGroupBuilder()
            .add_buffer(buffer_,
                    wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
                    wgpu::BufferBindingType::ReadOnlyStorage)
            .build(_device)}
        , bind_group_write_{MAL::BindGroupBuilder()
            .add_buffer(buffer_,
                    wgpu::ShaderStage::Compute,
                    wgpu::BufferBindingType::Storage)
            .build(_device)}
    {}

    MAL::BindGroupWithLayout const& bind_group_read() const {
        return bind_group_read_;
    }
    MAL::BindGroupWithLayout const& bind_group_write() const {
        return bind_group_write_;
    }

private:
    MAL::Buffer buffer_;
    MAL::BindGroupWithLayout bind_group_read_;
    MAL::BindGroupWithLayout bind_group_write_;
};

