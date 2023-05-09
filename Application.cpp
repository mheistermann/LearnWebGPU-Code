/**
 * This file is part of the "Learn WebGPU for C++" book.
 *   https://github.com/eliemichel/LearnWebGPU
 *
 * MIT License
 * Copyright (c) 2022-2023 Elie Michel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "Application.h"
#include "ResourceManager.h"

#include <GLFW/glfw3.h>
#include "glfw3webgpu.h"

#define GLM_FORCE_LEFT_HANDED
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <glm/gtx/polar_coordinates.hpp>

#include <imgui.h>
#include <backends/imgui_impl_wgpu.h>
#include <backends/imgui_impl_glfw.h>

#include <webgpu.hpp>
#include <wgpu.h> // wgpuTextureViewDrop

#include <iostream>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <array>

constexpr float PI = 3.14159265358979323846f;
static constexpr auto tet_strip = std::array<uint16_t, 6> {0, 1, 2, 3, 0, 1};





using namespace wgpu;
using VertexAttributes = ResourceManager::VertexAttributes;
using glm::mat4x4;
using glm::vec4;
using glm::vec3;
using glm::vec2;

// GLFW callbacks
void onWindowResize(GLFWwindow* window, int width, int height) {
	(void)width; (void)height;
	auto pApp = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
	if (pApp != nullptr) pApp->onResize();
}
void onWindowMouseMove(GLFWwindow* window, double xpos, double ypos) {
	auto pApp = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
	if (pApp != nullptr) pApp->onMouseMove(xpos, ypos);
}
void onWindowMouseButton(GLFWwindow* window, int button, int action, int mods) {
	auto pApp = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
	if (pApp != nullptr) pApp->onMouseButton(button, action, mods);
}
void onWindowScroll(GLFWwindow* window, double xoffset, double yoffset) {
	auto pApp = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
	if (pApp != nullptr) pApp->onScroll(xoffset, yoffset);
}

bool Application::onInit() {
	// Create instance
	m_instance = createInstance(InstanceDescriptor{});
	if (!m_instance) {
		std::cerr << "Could not initialize WebGPU!" << std::endl;
		return false;
	}

	if (!glfwInit()) {
		std::cerr << "Could not initialize GLFW!" << std::endl;
		return false;
	}

	// Create window
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
	m_window = glfwCreateWindow(640, 480, "Learn WebGPU", NULL, NULL);
	if (!m_window) {
		std::cerr << "Could not open window!" << std::endl;
		return false;
	}

	// Add window callbacks
	glfwSetWindowUserPointer(m_window, this);
	glfwSetFramebufferSizeCallback(m_window, onWindowResize);
	glfwSetCursorPosCallback(m_window, onWindowMouseMove);
	glfwSetMouseButtonCallback(m_window, onWindowMouseButton);
	glfwSetScrollCallback(m_window, onWindowScroll);

	// Create surface and adapter
	std::cout << "Requesting adapter..." << std::endl;
	m_surface = glfwGetWGPUSurface(m_instance, m_window);
	RequestAdapterOptions adapterOpts{};
	adapterOpts.compatibleSurface = m_surface;
	Adapter adapter = m_instance.requestAdapter(adapterOpts);
	std::cout << "Got adapter: " << adapter << std::endl;

	std::cout << "Requesting device..." << std::endl;
	SupportedLimits supportedLimits;
	adapter.getLimits(&supportedLimits);
	RequiredLimits requiredLimits = Default;
	requiredLimits.limits.maxVertexAttributes = 4;
	requiredLimits.limits.maxVertexBuffers = 2;
	requiredLimits.limits.maxBindGroups = 3;
	requiredLimits.limits.maxUniformBuffersPerShaderStage = 3;
	requiredLimits.limits.maxUniformBufferBindingSize = 16 * 4 * sizeof(float);
	requiredLimits.limits.maxStorageBufferBindingSize =1234; // TODO: set properly
	requiredLimits.limits.maxStorageBuffersPerShaderStage = 3;
	requiredLimits.limits.maxComputeWorkgroupStorageSize = 1234; // TODO
	requiredLimits.limits.maxComputeWorkgroupSizeX = 32;
	requiredLimits.limits.maxComputeWorkgroupSizeY = 1;
	requiredLimits.limits.maxComputeWorkgroupSizeZ = 1;
	requiredLimits.limits.maxComputeInvocationsPerWorkgroup = 32;
	requiredLimits.limits.maxComputeWorkgroupsPerDimension = 1234; // TODO

	requiredLimits.limits.minUniformBufferOffsetAlignment = supportedLimits.limits.minUniformBufferOffsetAlignment;
	requiredLimits.limits.minStorageBufferOffsetAlignment = supportedLimits.limits.minStorageBufferOffsetAlignment;

	// Create device
	DeviceDescriptor deviceDesc{};
	deviceDesc.label = "My Device";
	deviceDesc.requiredFeaturesCount = 0;
	deviceDesc.requiredLimits = &requiredLimits;
	deviceDesc.defaultQueue.label = "The default queue";
	m_device = adapter.requestDevice(deviceDesc);
	std::cout << "Got device: " << m_device << std::endl;

	// Add an error callback for more debug info
	m_uncapturedErrorCallback = m_device.setUncapturedErrorCallback([](ErrorType type, char const* message) {
		std::cout << "Device error: type " << type;
		if (message) std::cout << " (message: " << message << ")";
		std::cout << std::endl;
	});

	// Create swapchain
	m_swapChainFormat = m_surface.getPreferredFormat(adapter);
	buildSwapChain();


    m_context = std::make_shared<MAL::RenderContext>(m_device);
	std::cout << "Creating shader modules..." << std::endl;
	//m_shaderModule = ResourceManager::loadShaderModule(RESOURCE_DIR "/shader.wsl", m_device);
	m_vertex_shader= m_context->shader_loader().load(RESOURCE_DIR "/shaders/vertex.wsl");
	m_fragment_shader = m_context->shader_loader().load(RESOURCE_DIR "/shaders/fragment.wsl");
	//std::cout << "Shader module: " << m_compute_shaderModule << std::endl;


	std::cout << "Creating buffers..." << std::endl;
	createBuffers();

	std::cout << "Creating compute pipeline: view indep..." << std::endl;
    m_pipeline_compute_view_indep = PipelineComputeViewIndependent(m_context,
            m_tet_mesh_buffer,
            m_tet_verts_buffer);
	std::cout << "Creating compute pipeline: view dependent..." << std::endl;
    m_pipeline_compute_view_dep = PipelineComputeViewDependent(m_context,
            m_tet_verts_buffer,
            m_tet_precompute_viewdep_buffer);


	std::cout << "Creating render pipeline..." << std::endl;
	buildRenderPipeline();

	initGui();

	return true;
}
void Application::createBuffers()
{
	TetMeshBuffer buf;
	Queue queue = m_device.getQueue();
	m_tet_mesh_buffer = make_tet_mesh_buffer(m_device);
	m_tet_verts_buffer = std::make_shared<TetVertsBuffer>(m_device, m_tet_mesh_buffer->n_tets());
	m_tet_precompute_viewdep_buffer = std::make_shared<TetPrecomputeViewDepBuffer>(m_device, m_tet_mesh_buffer->n_tets());


	BufferDescriptor bufferDesc;

	// Create uniform buffer
	bufferDesc.size = sizeof(MyUniforms);
	bufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
	bufferDesc.mappedAtCreation = false;
	m_uniformBuffer = m_device.createBuffer(bufferDesc);

	bufferDesc.size = sizeof(tet_strip);
	bufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Index;
	bufferDesc.mappedAtCreation = false;
	m_indexBuffer = m_device.createBuffer(bufferDesc);
	queue.writeBuffer(m_indexBuffer, 0, tet_strip.data(), tet_strip.size() * sizeof(tet_strip.front()));

	// Upload the initial value of the uniforms
	m_uniforms.time = 1.0f;
	m_uniforms.color = { 0.0f, 1.0f, 0.4f, 1.0f };

	// Matrices
	m_uniforms.modelMatrix = mat4x4(1.0);
	m_uniforms.viewMatrix = glm::lookAt(vec3(-2.0f, -3.0f, 2.0f), vec3(0.0f), vec3(0, 0, 1));
	m_uniforms.projectionMatrix = glm::perspective(45 * PI / 180, 640.0f / 480.0f, 0.01f, 100.0f);

	queue.writeBuffer(m_uniformBuffer, 0, &m_uniforms, sizeof(MyUniforms));
	updateViewMatrix();

	buildDepthBuffer();


}
void Application::buildRenderPipeline() {

	// Create binding layout
	m_bindingLayoutEntries.resize(1, Default);

	BindGroupLayoutEntry& bindingLayout = m_bindingLayoutEntries[0];
	bindingLayout.binding = 0;
	bindingLayout.visibility = ShaderStage::Vertex | ShaderStage::Fragment;
	bindingLayout.buffer.type = BufferBindingType::Uniform;
	bindingLayout.buffer.minBindingSize = sizeof(MyUniforms);

	// Create bindings
	m_bindings.resize(1);

	m_bindings[0].binding = 0;
	m_bindings[0].buffer = m_uniformBuffer;
	m_bindings[0].offset = 0;
	m_bindings[0].size = sizeof(MyUniforms);

	std::cout << "Creating render pipeline..." << std::endl;
	RenderPipelineDescriptor pipelineDesc{};

	pipelineDesc.vertex.bufferCount = 0;
	pipelineDesc.vertex.buffers = nullptr;

	pipelineDesc.vertex.module = m_vertex_shader;
	pipelineDesc.vertex.entryPoint = "vs_main";
	pipelineDesc.vertex.constantCount = 0;
	pipelineDesc.vertex.constants = nullptr;

	pipelineDesc.primitive.topology = PrimitiveTopology::TriangleStrip;
	pipelineDesc.primitive.stripIndexFormat = IndexFormat::Uint16;
	pipelineDesc.primitive.frontFace = FrontFace::CCW;
	pipelineDesc.primitive.cullMode = CullMode::None;

	FragmentState fragmentState{};
	pipelineDesc.fragment = &fragmentState;
	fragmentState.module = m_fragment_shader;
	fragmentState.entryPoint = "fs_main";
	fragmentState.constantCount = 0;
	fragmentState.constants = nullptr;

	BlendState blendState{};
	//blendState.color.srcFactor = BlendFactor::SrcAlpha; // TODO additive blending
	//blendState.color.dstFactor = BlendFactor::OneMinusSrcAlpha;
	blendState.color.srcFactor = BlendFactor::One; // TODO additive blending
	blendState.color.dstFactor = BlendFactor::One;
	blendState.color.operation = BlendOperation::Add;

	blendState.alpha.srcFactor = BlendFactor::Zero;
	blendState.alpha.dstFactor = BlendFactor::One;
	blendState.alpha.operation = BlendOperation::Add;

	ColorTargetState colorTarget{};
	colorTarget.format = m_swapChainFormat;
	colorTarget.blend = &blendState;
	colorTarget.writeMask = ColorWriteMask::All;

	fragmentState.targetCount = 1;
	fragmentState.targets = &colorTarget;

	DepthStencilState depthStencilState = Default;
	depthStencilState.depthCompare = CompareFunction::Less;
	depthStencilState.depthWriteEnabled = false; // only solid renderingg will write
	depthStencilState.format = m_depthTextureFormat;
	depthStencilState.stencilReadMask = 0;
	depthStencilState.stencilWriteMask = 0;
	pipelineDesc.depthStencil = &depthStencilState;

	pipelineDesc.multisample.count = 1;
	pipelineDesc.multisample.mask = ~0u;
	pipelineDesc.multisample.alphaToCoverageEnabled = false;

	// Create a bind group layout
	BindGroupLayoutDescriptor bindGroupLayoutDesc{};
	bindGroupLayoutDesc.entryCount = (uint32_t)m_bindingLayoutEntries.size();
	bindGroupLayoutDesc.entries = m_bindingLayoutEntries.data();
	BindGroupLayout bindGroupLayout = m_device.createBindGroupLayout(bindGroupLayoutDesc);

	std::array<WGPUBindGroupLayout, 3> layouts = {
		bindGroupLayout,
		(WGPUBindGroupLayout&)m_tet_verts_buffer->bind_group_read().layout,
		(WGPUBindGroupLayout&)m_tet_precompute_viewdep_buffer->bind_group_read().layout,
    };

	// Create the pipeline layout
	PipelineLayoutDescriptor layoutDesc{};
	layoutDesc.bindGroupLayoutCount = layouts.size();
	layoutDesc.bindGroupLayouts = &layouts.front();
	PipelineLayout layout = m_device.createPipelineLayout(layoutDesc);
	pipelineDesc.layout = layout;

	m_pipeline = m_device.createRenderPipeline(pipelineDesc);
	std::cout << "Render pipeline: " << m_pipeline << std::endl;

	// Create bind group
	BindGroupDescriptor bindGroupDesc{};
	bindGroupDesc.layout = bindGroupLayout;
	bindGroupDesc.entryCount = (uint32_t)m_bindings.size();
	bindGroupDesc.entries = m_bindings.data();
	m_renderBindGroup = m_device.createBindGroup(bindGroupDesc);
}

void Application::buildSwapChain() {
	int width, height;
	glfwGetFramebufferSize(m_window, &width, &height);

	std::cout << "Creating swapchain..." << std::endl;
	m_swapChainDesc = SwapChainDescriptor{};
	m_swapChainDesc.width = (uint32_t)width;
	m_swapChainDesc.height = (uint32_t)height;
	m_swapChainDesc.usage = TextureUsage::RenderAttachment;
	m_swapChainDesc.format = m_swapChainFormat;
	m_swapChainDesc.presentMode = PresentMode::Fifo;
	m_swapChain = m_device.createSwapChain(m_surface, m_swapChainDesc);
	std::cout << "Swapchain: " << m_swapChain << std::endl;
}

void Application::buildDepthBuffer() {
	// Destroy previously allocated texture
	if (m_depthTexture != nullptr) m_depthTexture.destroy();

	std::cout << "Creating depth texture..." << std::endl;
	// Create the depth texture
	TextureDescriptor depthTextureDesc;
	depthTextureDesc.dimension = TextureDimension::_2D;
	depthTextureDesc.format = m_depthTextureFormat;
	depthTextureDesc.mipLevelCount = 1;
	depthTextureDesc.sampleCount = 1;
	depthTextureDesc.size = { m_swapChainDesc.width, m_swapChainDesc.height, 1 };
	depthTextureDesc.usage = TextureUsage::RenderAttachment;
	depthTextureDesc.viewFormatCount = 1;
	depthTextureDesc.viewFormats = (WGPUTextureFormat*)&m_depthTextureFormat;
	m_depthTexture = m_device.createTexture(depthTextureDesc);
	std::cout << "Depth texture: " << m_depthTexture << std::endl;

	// Create the view of the depth texture manipulated by the rasterizer
	TextureViewDescriptor depthTextureViewDesc;
	depthTextureViewDesc.aspect = TextureAspect::DepthOnly;
	depthTextureViewDesc.baseArrayLayer = 0;
	depthTextureViewDesc.arrayLayerCount = 1;
	depthTextureViewDesc.baseMipLevel = 0;
	depthTextureViewDesc.mipLevelCount = 1;
	depthTextureViewDesc.dimension = TextureViewDimension::_2D;
	depthTextureViewDesc.format = m_depthTextureFormat;
	m_depthTextureView = m_depthTexture.createView(depthTextureViewDesc);
}

void Application::onCompute() {
    m_pipeline_compute_view_indep.run();
}
void Application::onFrame() {
	//std::cout << "onFrame" << std::endl;
	glfwPollEvents();
	Queue queue = m_device.getQueue();

	updateDragInertia();

	// Update uniform buffer
	m_uniforms.time = static_cast<float>(glfwGetTime());
	queue.writeBuffer(m_uniformBuffer, offsetof(MyUniforms, time), &m_uniforms.time, sizeof(MyUniforms::time));

	TextureView nextTexture = m_swapChain.getCurrentTextureView();
	if (!nextTexture) {
		std::cerr << "Cannot acquire next swap chain texture" << std::endl;
		return;
	}

	CommandEncoderDescriptor commandEncoderDesc{};
	commandEncoderDesc.label = "Command Encoder";
	CommandEncoder encoder = m_device.createCommandEncoder(commandEncoderDesc);

	RenderPassDescriptor renderPassDesc{};

	RenderPassColorAttachment colorAttachment;
	colorAttachment.view = nextTexture;
	colorAttachment.resolveTarget = nullptr;
	colorAttachment.loadOp = LoadOp::Clear;
	colorAttachment.storeOp = StoreOp::Store;
	colorAttachment.clearValue = Color{ 0.05, 0.05, 0.05, 1.0 };
	renderPassDesc.colorAttachmentCount = 1;
	renderPassDesc.colorAttachments = &colorAttachment;

	RenderPassDepthStencilAttachment depthStencilAttachment;
	depthStencilAttachment.view = m_depthTextureView;
	depthStencilAttachment.depthClearValue = 100.0f;
	depthStencilAttachment.depthLoadOp = LoadOp::Clear;
	depthStencilAttachment.depthStoreOp = StoreOp::Store;
	depthStencilAttachment.depthReadOnly = false;
	depthStencilAttachment.stencilClearValue = 0;
	depthStencilAttachment.stencilLoadOp = LoadOp::Clear;
	depthStencilAttachment.stencilStoreOp = StoreOp::Store;
	depthStencilAttachment.stencilReadOnly = true;

	renderPassDesc.depthStencilAttachment = &depthStencilAttachment;
	renderPassDesc.timestampWriteCount = 0;
	renderPassDesc.timestampWrites = nullptr;
	RenderPassEncoder renderPass = encoder.beginRenderPass(renderPassDesc);

	renderPass.setPipeline(m_pipeline);

	renderPass.setIndexBuffer(m_indexBuffer, IndexFormat::Uint16, 0, tet_strip.size() * sizeof(tet_strip.front()));
	//renderPass.setVertexBuffer(0, m_tetVerts, 0, m_tetVerts.size() * sizeof(m_tetVerts.begin()));
	renderPass.setBindGroup(0, m_renderBindGroup, 0, nullptr);
	renderPass.setBindGroup(1, m_tet_verts_buffer->bind_group_read().group, 0, nullptr);
	renderPass.setBindGroup(2, m_tet_precompute_viewdep_buffer->bind_group_read().group, 0, nullptr);

	//void drawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t baseVertex, uint32_t firstInstance);
	renderPass.drawIndexed(tet_strip.size(), m_tet_mesh_buffer->n_tets(), 0, 0, 0);

	updateGui(renderPass);

	renderPass.end();

	CommandBuffer command = encoder.finish(CommandBufferDescriptor{});
	queue.submit(command);

	wgpuTextureViewDrop(nextTexture);
	m_swapChain.present();
}

void Application::onFinish() {
	for (auto texture : m_textures) {
		texture.destroy();
	}
	m_depthTexture.destroy();

	glfwDestroyWindow(m_window);
	glfwTerminate();
}

void Application::onResize() {
	buildSwapChain();
	buildDepthBuffer();

	float ratio = m_swapChainDesc.width / (float)m_swapChainDesc.height;
	m_uniforms.projectionMatrix = glm::perspective(45 * PI / 180, ratio, 0.01f, 100.0f);
	m_device.getQueue().writeBuffer(m_uniformBuffer, offsetof(MyUniforms, projectionMatrix), &m_uniforms.projectionMatrix, sizeof(MyUniforms::projectionMatrix));
}

void Application::onMouseMove(double xpos, double ypos) {
	if (m_drag.active) {
		vec2 currentMouse = vec2(-(float)xpos, (float)ypos);
		vec2 delta = (currentMouse - m_drag.startMouse) * m_drag.sensitivity;
		m_cameraState.angles = m_drag.startCameraState.angles + delta;
		m_cameraState.angles.y = glm::clamp(m_cameraState.angles.y, -PI / 2 + 1e-5f, PI / 2 - 1e-5f);
		updateViewMatrix();

		// Inertia
		m_drag.velocity = delta - m_drag.previousDelta;
		m_drag.previousDelta = delta;
	}
}

void Application::onMouseButton(int button, int action, int mods) {
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	if (io.WantCaptureMouse) {
		return;
	}

	(void)mods;
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		switch(action) {
		case GLFW_PRESS:
			m_drag.active = true;
			double xpos, ypos;
			glfwGetCursorPos(m_window, &xpos, &ypos);
			m_drag.startMouse = vec2(-(float)xpos, (float)ypos);
			m_drag.startCameraState = m_cameraState;
			break;
		case GLFW_RELEASE:
			m_drag.active = false;
			break;
		}
	}
}

void Application::onScroll(double xoffset, double yoffset) {
	(void)xoffset;
	m_cameraState.zoom += m_drag.scrollSensitivity * (float)yoffset;
	m_cameraState.zoom = glm::clamp(m_cameraState.zoom, -2.0f, 2.0f);
	updateViewMatrix();
}

bool Application::isRunning() {
	return !glfwWindowShouldClose(m_window);
}

void Application::updateViewMatrix() {
	float cx = cos(m_cameraState.angles.x);
	float sx = sin(m_cameraState.angles.x);
	float cy = cos(m_cameraState.angles.y);
	float sy = sin(m_cameraState.angles.y);
	vec3 position = vec3(cx * cy, sx * cy, sy) * std::exp(-m_cameraState.zoom);
	m_uniforms.viewMatrix = glm::lookAt(position, vec3(0.0f), vec3(0, 0, 1));
	m_device.getQueue().writeBuffer(m_uniformBuffer, offsetof(MyUniforms, viewMatrix), &m_uniforms.viewMatrix, sizeof(MyUniforms::viewMatrix));
}

void Application::updateDragInertia() {
	constexpr float eps = 1e-4f;
	if (!m_drag.active) {
		if (std::abs(m_drag.velocity.x) < eps && std::abs(m_drag.velocity.y) < eps) {
			return;
		}
		m_cameraState.angles += m_drag.velocity;
		m_cameraState.angles.y = glm::clamp(m_cameraState.angles.y, -PI / 2 + 1e-5f, PI / 2 - 1e-5f);
		m_drag.velocity *= m_drag.intertia;
		updateViewMatrix();
	}
}

void Application::initGui() {
	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOther(m_window, true);
	ImGui_ImplWGPU_Init(m_device, 3, m_swapChainFormat, m_depthTextureFormat);
}

namespace ImGui {
bool DragDirection(const char *label, vec4& direction) {
	vec2 angles = glm::degrees(glm::polar(vec3(direction)));
	bool changed = ImGui::DragFloat2(label, glm::value_ptr(angles));
	direction = vec4(glm::euclidean(glm::radians(angles)), direction.w);
	return changed;
}
} // namespace ImGui

void Application::updateGui(RenderPassEncoder renderPass) {
	// Start the Dear ImGui frame
	ImGui_ImplWGPU_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	bool changed = false;
	ImGui::Begin("Lighting");
	changed = ImGui::ColorEdit3("Color #0", glm::value_ptr(m_lightingUniforms.colors[0])) || changed;
	changed = ImGui::DragDirection("Direction #0", m_lightingUniforms.directions[0]) || changed;
	changed = ImGui::ColorEdit3("Color #1", glm::value_ptr(m_lightingUniforms.colors[1])) || changed;
	changed = ImGui::DragDirection("Direction #1", m_lightingUniforms.directions[1]) || changed;
	ImGui::End();
	m_lightingUniformsChanged = changed;

	ImGui::Render();
	ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), renderPass);
}


