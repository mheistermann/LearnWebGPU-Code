#pragma once
#include "mal.hh"
#include <array>
#include <webgpu.hpp>

using Vec3 = std::array<float, 3>;
using Tet = std::array<uint32_t, 4>;

struct TetMeshData {
    std::vector<Vec3> vertices;
    std::vector<Tet> tets;
};

class TetMeshBuffer {
public:
    TetMeshBuffer(wgpu::Device &_device,
            TetMeshData const&data,
            WGPUShaderStageFlags _visibility = wgpu::ShaderStage::Compute)
        : vertices_{_device, data.vertices}
        , tets_{_device, data.tets}
        , bind_group_{MAL::BindGroupBuilder()
            .add_buffer(vertices_, _visibility, wgpu::BufferBindingType::ReadOnlyStorage)
            .add_buffer(tets_, _visibility, wgpu::BufferBindingType::ReadOnlyStorage)
            .build(_device)}
    {
    }
    wgpu::BindGroup const &bind_group() const {
        return bind_group_;
    }
    private:
        MAL::StaticVectorBuffer<Vec3> vertices_;
        MAL::StaticVectorBuffer<Tet> tets_;
        wgpu::BindGroup bind_group_;
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
