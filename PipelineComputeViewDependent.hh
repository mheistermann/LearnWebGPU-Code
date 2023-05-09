#pragma once


#include <webgpu.hpp>
#include "mal.hh"
#include "ovm_buffers.hh"


class PipelineComputeViewDependent
{
    struct Uniforms {
        MAL::Vec3f camera_in_object_space {0.f};
    };
public:
    PipelineComputeViewDependent() = default;
    PipelineComputeViewDependent(std::shared_ptr<MAL::RenderContext> _context,
            std::shared_ptr<TetVertsBuffer> _tet_verts_buffer,
            std::shared_ptr<TetPrecomputeViewDepBuffer> _viewdep_buffer)
        : context_(std::move(_context))
        , tet_verts_buffer_(std::move(_tet_verts_buffer))
        , viewdep_buffer_(std::move(_viewdep_buffer))
        , shader_(context_->shader_loader().load(RESOURCE_DIR "/shaders/compute_viewdep.wsl"))
        , uniforms_{MAL::UniformBuffer<Uniforms>(context_->device(), wgpu::ShaderStage::Compute)}
    {
        auto device = context_->device();
        std::array<WGPUBindGroupLayout, 3> layouts = {
            (WGPUBindGroupLayout&)tet_verts_buffer_->bind_group_read().layout,
            (WGPUBindGroupLayout&)viewdep_buffer_->bind_group_write().layout,
            (WGPUBindGroupLayout&)uniforms_.bind_group().layout,
        };

        wgpu::PipelineLayoutDescriptor pipelineLayoutDesc;
        pipelineLayoutDesc.bindGroupLayoutCount = layouts.size();
        pipelineLayoutDesc.bindGroupLayouts = &layouts.front();


        wgpu::ComputePipelineDescriptor pipelineDesc;
        pipelineDesc.compute.entryPoint = "computeViewDependent";
        pipelineDesc.compute.module = shader_;
        pipelineDesc.layout = device.createPipelineLayout(pipelineLayoutDesc);
        pipeline_ = device.createComputePipeline(pipelineDesc);
    }
    void set_camera_in_object_space(glm::vec3 const &pos) {
        uniforms_.u().camera_in_object_space.vec = pos;
        uniforms_dirty_ = true;
    }
    void run() {
        maybe_upload_uniforms();
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
        computePass.setBindGroup(0, tet_verts_buffer_->bind_group_read().group, 0, nullptr);
        computePass.setBindGroup(1, viewdep_buffer_->bind_group_write().group, 0, nullptr);
        const auto n_tets = tet_verts_buffer_->n_tets();
        const uint32_t wg_size = 32;
        const uint32_t wg_count = (n_tets + wg_size-1)/ wg_size;
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
    void maybe_upload_uniforms() {
        if (!uniforms_dirty_) return;
        uniforms_.upload();
        uniforms_dirty_ = false;
    }

    std::shared_ptr<MAL::RenderContext> context_;
    std::shared_ptr<TetVertsBuffer> tet_verts_buffer_;
    std::shared_ptr<TetPrecomputeViewDepBuffer> viewdep_buffer_;
    wgpu::ShaderModule shader_ = nullptr;
    MAL::UniformBuffer<Uniforms> uniforms_;
    wgpu::ComputePipeline pipeline_ = nullptr;
    bool uniforms_dirty_ = true;
};
