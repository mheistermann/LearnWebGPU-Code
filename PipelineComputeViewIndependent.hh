#pragma once


#include <webgpu.hpp>
#include "mal.hh"
#include "ovm_buffers.hh"


class PipelineComputeViewIndependent
{
public:
    PipelineComputeViewIndependent() = default;
    PipelineComputeViewIndependent(std::shared_ptr<MAL::RenderContext> _context,
            std::shared_ptr<TetMeshBuffer> _tet_mesh_buffer,
            std::shared_ptr<TetVertsBuffer> _tet_verts_buffer)
        : context_(std::move(_context))
        , tet_mesh_buffer_(std::move(_tet_mesh_buffer))
        , tet_verts_buffer_(std::move(_tet_verts_buffer))
        , m_compute_shader(context_->shader_loader().load(RESOURCE_DIR "/shaders/compute_view_indep.wsl"))
    {
        auto device = context_->device();
        std::array<WGPUBindGroupLayout, 2> layouts = {
            (WGPUBindGroupLayout&)tet_mesh_buffer_->bind_group().layout,
            (WGPUBindGroupLayout&)tet_verts_buffer_->bind_group_write().layout};

        wgpu::PipelineLayoutDescriptor pipelineLayoutDesc;
        pipelineLayoutDesc.bindGroupLayoutCount = 2;
        pipelineLayoutDesc.bindGroupLayouts = &layouts.front();


        wgpu::ComputePipelineDescriptor pipelineDesc;
        pipelineDesc.compute.entryPoint = "computeTetVerts";
        pipelineDesc.compute.module = m_compute_shader;
        pipelineDesc.layout = device.createPipelineLayout(pipelineLayoutDesc);
        pipeline_ = device.createComputePipeline(pipelineDesc);
    }
    void run() {
        auto device = context_->device();
        std::cout << "onCompute()" << std::endl;
        // Initialize a command encoder
        wgpu::Queue queue = device.getQueue();
        wgpu::CommandEncoderDescriptor encoderDesc {wgpu::Default};
        wgpu::CommandEncoder encoder = device.createCommandEncoder(encoderDesc);

        // Create and use compute pass here!
        //
        wgpu::ComputePassDescriptor computePassDesc;
        computePassDesc.timestampWriteCount = 0;
        computePassDesc.timestampWrites = nullptr;
        wgpu::ComputePassEncoder computePass = encoder.beginComputePass(computePassDesc);

        computePass.setPipeline(pipeline_);
        computePass.setBindGroup(0, tet_mesh_buffer_->bind_group().group, 0, nullptr);
        computePass.setBindGroup(1, tet_verts_buffer_->bind_group_write().group, 0, nullptr);
        const auto n_tets = tet_mesh_buffer_->n_tets();
        const uint32_t wg_size = 32;
        const auto wg_count = static_cast<uint32_t>((n_tets + wg_size-1)/ wg_size);
        computePass.dispatchWorkgroups(wg_count,1,1);
        computePass.end();

    // Clean up
#if !defined(WEBGPU_BACKEND_WGPU)
        wgpuComputePassEncoderRelease(computePass);
#endif

        // Encode and submit the GPU commands
        wgpu::CommandBuffer commands = encoder.finish(wgpu::CommandBufferDescriptor{});
        queue.submit(commands);

        // Clean up
#if !defined(WEBGPU_BACKEND_WGPU)
        wgpuCommandBufferRelease(commands);
        wgpuCommandEncoderRelease(encoder);
        wgpuQueueRelease(queue);
#endif
        std::cout << "onCompute submitted" << std::endl;
        }
private:

    std::shared_ptr<MAL::RenderContext> context_;
    std::shared_ptr<TetMeshBuffer> tet_mesh_buffer_;
    std::shared_ptr<TetVertsBuffer> tet_verts_buffer_;
    wgpu::ShaderModule m_compute_shader = nullptr;
    wgpu::ComputePipeline pipeline_ = nullptr;
};
