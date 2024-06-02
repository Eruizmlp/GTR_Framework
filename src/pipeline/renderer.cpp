#include "renderer.h"

#include <algorithm> //sort

#include "camera.h"
#include "../gfx/gfx.h"
#include "../gfx/shader.h"
#include "../gfx/mesh.h"
#include "../gfx/texture.h"
#include "../gfx/fbo.h"
#include "../pipeline/prefab.h"
#include "../pipeline/material.h"
#include "../pipeline/animation.h"
#include "../utils/utils.h"
#include "../extra/hdre.h"
#include "../core/ui.h"

#include "scene.h"


using namespace SCN;

//some globals
GFX::Mesh sphere;

GFX::Mesh cube;
GFX::FBO* gbuffers = nullptr;
GFX::FBO* ssao_fbo = nullptr;
GFX::FBO* blur_fbo1 = nullptr;
GFX::FBO* blur_fbo2 = nullptr;
GFX::FBO* illumination_fbo = nullptr;
GFX::FBO* irradiance_fbo = nullptr;
GFX::FBO* tonemapping_fbo = nullptr;
std::vector<Vector3f> random_points;


Renderer::Renderer(const char* shader_atlas_filename)
{
	render_wireframe = false;
	render_boundaries = false;

	irradiance_scene = true;

	scene = nullptr;
	skybox_cubemap = nullptr;
	sunLight = nullptr;

	shadowmap_size = 1024;
	ssao_radius = 10.0f;
	ssao_max_distance = 1.0f;

	//Flags
	multipass = true;
	alpha_sorting = false;
	disable_dithering = false;
	showSSAO = false;

	pipeline_mode = ePipeLineMode::DEFERRED;
	view_gbuffers = eShowGbuffers::NONE;

	if (!GFX::Shader::LoadAtlas(shader_atlas_filename))
		exit(1);
	GFX::checkGLErrors();

	random_points = generateSpherePoints(64, 1.0f, true);

	sphere.createSphere(1.0f);
	sphere.uploadToVRAM();
	cube.createCube(1.0f);
	cube.uploadToVRAM();	
}

void Renderer::setupScene()
{
	if (scene->skybox_filename.size())
		skybox_cubemap = GFX::Texture::Get(std::string(scene->base_folder + "/" + scene->skybox_filename).c_str());
	else
		skybox_cubemap = nullptr;
}

void Renderer::renderScene(SCN::Scene* scene, Camera* camera)
{
	this->scene = scene;
	setupScene();
	extractSceneInfo(scene, camera);
	generateShadowMaps(camera);

	if (pipeline_mode == ePipeLineMode::DEFERRED)
		renderSceneDeferred(scene, camera);

	else if (pipeline_mode == ePipeLineMode::FORWARD)
		renderSceneForward(scene, camera);

}

void Renderer::renderSceneDeferred(SCN::Scene* scene, Camera* camera)
{
	vec2 size = CORE::getWindowSize();
	GFX::Mesh* quad = GFX::Mesh::getQuad();

	if (!gbuffers)
	{
		gbuffers = new GFX::FBO();
		gbuffers->create(size.x, size.y, 3, GL_RGBA, GL_UNSIGNED_BYTE, true);
	}
	gbuffers->bind();

	camera->enable();
	glEnable(GL_DEPTH_TEST);
	glClearColor(0, 0, 0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	for (auto& re : opaqueRenderables)
		if(camera->testBoxInFrustum(re.boundingbox.center,re.boundingbox.halfsize))
			renderMeshWithMaterialGBuffers(re.model, re.mesh, re.material);

	gbuffers->unbind();

	//SSAO

	if (!ssao_fbo)
	{
		ssao_fbo = new GFX::FBO();
		ssao_fbo->create(size.x/2, size.y/2, 1, GL_RGB, GL_UNSIGNED_BYTE, false);
		ssao_fbo->color_textures[0]->setName("SSAO");
	}

	ssao_fbo->bind();
	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	GFX::Shader* ssao_shader = GFX::Shader::Get("ssao");
	assert(ssao_shader);
	ssao_shader->enable();
	ssao_shader->setUniform("u_depth_texture", gbuffers->depth_texture, 0);
	ssao_shader->setUniform("u_normal_texture", gbuffers->color_textures[1], 1);
	ssao_shader->setUniform("u_iRes", vec2(1.0 / size.x*2, 1.0 / size.y*2));
	ssao_shader->setUniform("u_inverse_viewprojection", camera->inverse_viewprojection_matrix);
	ssao_shader->setUniform("u_viewprojection", camera->viewprojection_matrix);
	ssao_shader->setUniform("u_radius", ssao_radius);
	ssao_shader->setUniform("u_ssao_max_distance", ssao_max_distance);
	ssao_shader->setUniform3Array("u_random_points", (float*)&random_points[0], random_points.size());
	ssao_shader->setUniform("near", camera->near_plane); 
	ssao_shader->setUniform("far", camera->far_plane);  

	quad->render(GL_TRIANGLES);

	ssao_fbo->unbind();

	// First Blur Pass (Horizontal)
	if (!blur_fbo1)
	{
		blur_fbo1 = new GFX::FBO();
		blur_fbo1->create(size.x, size.y, 1, GL_RGB, GL_UNSIGNED_BYTE, false);
	}

	blur_fbo1->bind();
	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	GFX::Shader* blur_h_shader = GFX::Shader::Get("blur_h");
	assert(blur_h_shader);
	blur_h_shader->enable();
	blur_h_shader->setUniform("u_ssao_texture", ssao_fbo->color_textures[0], 0);
	blur_h_shader->setUniform("u_iRes", vec2(1.0 / size.x*2, 1.0 / size.y*2));

	quad->render(GL_TRIANGLES);

	blur_fbo1->unbind();

	// Second Blur Pass (Vertical)
	if (!blur_fbo2)
	{
		blur_fbo2 = new GFX::FBO();
		blur_fbo2->create(size.x, size.y, 1, GL_RGB, GL_UNSIGNED_BYTE, false);
	}

	blur_fbo2->bind();
	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	GFX::Shader* blur_v_shader = GFX::Shader::Get("blur_v");
	assert(blur_v_shader);
	blur_v_shader->enable();
	blur_v_shader->setUniform("u_ssao_texture", blur_fbo1->color_textures[0], 0);
	blur_v_shader->setUniform("u_iRes", vec2(1.0 / size.x*2, 1.0 / size.y*2));

	quad->render(GL_TRIANGLES);

	blur_fbo2->unbind();

	// Illumination pass

	if (!illumination_fbo)
	{
		illumination_fbo = new GFX::FBO();
		illumination_fbo->create(size.x, size.y, 1, GL_RGBA, GL_UNSIGNED_BYTE, false);
	}

	illumination_fbo->bind();

	glClearColor(scene->background_color.x, scene->background_color.y, scene->background_color.z, 1.0);
	glClearColor(0, 0, 0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (skybox_cubemap)
		renderSkybox(skybox_cubemap);

	//Pintar luces
	GFX::Shader* deferred_global = GFX::Shader::Get("deferred_global");
	assert(deferred_global);
	deferred_global->enable();
	deferred_global->setUniform("u_color_texture", gbuffers->color_textures[0], 0);
	deferred_global->setUniform("u_normal_texture", gbuffers->color_textures[1], 1);
	deferred_global->setUniform("u_emissive_texture", gbuffers->color_textures[2], 2);
	deferred_global->setUniform("u_depth_texture", gbuffers->depth_texture, 3);
	deferred_global->setUniform("u_ssao_texture", blur_fbo2->color_textures[0], 4);  

	deferred_global->setUniform("u_ambient_light", scene->ambient_light);
	deferred_global->setUniform("u_iRes", vec2( 1.0 / size.x, 1.0 / size.y));
	deferred_global->setUniform("u_inverse_viewprojection", camera->inverse_viewprojection_matrix);
	deferred_global->setUniform("u_camera_position", camera->eye);


	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_ALWAYS);
	glDisable(GL_BLEND);


	//TODO: SPHERE RENDERING FOR POINT AND SPOT LIGHTS.
	for (auto light : lights)
	{
		LightToShader(light, deferred_global);
		quad->render(GL_TRIANGLES);
		deferred_global->setUniform("u_ambient_light", vec3(0.0));
		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE);
		glDisable(GL_DEPTH_TEST);
	}

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);

	//Forward for alpha
	for (auto re : transparentRenderables)
		if (camera->testBoxInFrustum(re.boundingbox.center, re.boundingbox.halfsize))
			renderMeshWithMaterialLight(re.model, re.mesh, re.material);


	glDisable(GL_BLEND);
	glDepthFunc(GL_LESS);
	glDisable(GL_DEPTH_TEST);


	illumination_fbo->unbind();

	// Tonemapping Pass and gamma correction 
	if (!tonemapping_fbo)
	{
		tonemapping_fbo = new GFX::FBO();
		tonemapping_fbo->create(size.x, size.y, 1, GL_RGBA, GL_UNSIGNED_BYTE, false);
	}

	tonemapping_fbo->bind();
	glClear(GL_COLOR_BUFFER_BIT);

	GFX::Shader* tonemapping_shader = GFX::Shader::Get("tonemapping");
	assert(tonemapping_shader);
	tonemapping_shader->enable();
	tonemapping_shader->setUniform("u_hdr_texture", illumination_fbo->color_textures[0], 0);

	quad->render(GL_TRIANGLES);

	tonemapping_fbo->unbind();

	//illumination_fbo->color_textures[0]->toViewport();
	tonemapping_fbo->color_textures[0]->toViewport();
	
	if (view_gbuffers != eShowGbuffers::NONE)
	{
		if (view_gbuffers == eShowGbuffers::COLOR)
			gbuffers->color_textures[0]->toViewport();

		if (view_gbuffers == eShowGbuffers::NORMAL)
			gbuffers->color_textures[1]->toViewport();

		if (view_gbuffers == eShowGbuffers::EXTRA)
			gbuffers->color_textures[2]->toViewport();

		if (view_gbuffers == eShowGbuffers::DEPTH)
			gbuffers->depth_texture->toViewport();

		if (view_gbuffers == eShowGbuffers::EMISSIVE)
			gbuffers->color_textures[3]->toViewport();
	}

	if (showSSAO)
		//ssao_fbo->color_textures[0]->toViewport();
		blur_fbo2->color_textures[0]->toViewport();
}


void Renderer::renderSceneForward(SCN::Scene* scene, Camera* camera)
{

	camera->enable();

	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);

	//set the clear color (the background color)
	glClearColor(scene->background_color.x, scene->background_color.y, scene->background_color.z, 1.0);

	// Clear the color and the depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	GFX::checkGLErrors();

	//render skybox
	if (skybox_cubemap)
		renderSkybox(skybox_cubemap);

	//Ordeno los renderables (transparentes) en funcion de su distancia a camara 
	if (!alpha_sorting)
		sortTransparentRenderables(transparentRenderables);

	//combine renderables 
	renderables.reserve(opaqueRenderables.size() + transparentRenderables.size());
	renderables.insert(renderables.end(), opaqueRenderables.begin(), opaqueRenderables.end());
	renderables.insert(renderables.end(), transparentRenderables.begin(), transparentRenderables.end());

	for (auto& re : renderables)
		if(camera->testBoxInFrustum(re.boundingbox.center,re.boundingbox.halfsize))
			renderMeshWithMaterialLight(re.model, re.mesh, re.material);
	
}

void Renderer::extractSceneInfo(SCN::Scene* scene, Camera* camera)
{
	//Clear temp containers
	opaqueRenderables.clear();
	transparentRenderables.clear();
	renderables.clear();

	lights.clear();

	//render entities
	for (int i = 0; i < scene->entities.size(); ++i)
	{
		BaseEntity* ent = scene->entities[i];
		if (!ent->visible)
			continue;

		//is a prefab!
		if (ent->getType() == eEntityType::PREFAB)
		{
			PrefabEntity* pent = (SCN::PrefabEntity*)ent;
			if (pent->prefab) {
				//extract prefab info
				extractRenderables(&pent->root, camera);
			}
		}

		//is a light!
		else if (ent->getType() == eEntityType::LIGHT)
		{
			LightEntity* light = (SCN::LightEntity*)ent;
			mat4 model = light->root.getGlobalMatrix();
			//Añado la luz directional siempre y si el area de la luz esta en el frustrum de la camara
			if (light->light_type == eLightType::DIRECTIONAL || light->light_type == eLightType::SPOT || camera->testSphereInFrustum(model.getTranslation(), light->max_distance))
				lights.push_back(light);

			if (!sunLight && light->light_type == eLightType::DIRECTIONAL)
				sunLight = light;
		}
	}
}


void Renderer::renderSkybox(GFX::Texture* cubemap)
{
	Camera* camera = Camera::current;

	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	if (render_wireframe)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glFrontFace(GL_CW);

	GFX::Shader* shader = GFX::Shader::Get("skybox");
	if (!shader)
		return;
	shader->enable();

	Matrix44 m;
	m.setTranslation(camera->eye.x, camera->eye.y, camera->eye.z);
	m.scale(1000, 1000, 1000);
	shader->setUniform("u_model", m);
	cameraToShader(camera, shader);
	shader->setUniform("u_texture", cubemap, 0);
	cube.render(GL_TRIANGLES);
	shader->disable();
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glEnable(GL_DEPTH_TEST);
	glFrontFace(GL_CCW);
}

//renders a node of the prefab and its children
void Renderer::renderNode(SCN::Node* node, Camera* camera)
{
	if (!node->visible)
		return;

	//compute global matrix
	Matrix44 node_model = node->getGlobalMatrix(true);

	//does this node have a mesh? then we must render it
	if (node->mesh && node->material)
	{
		//compute the bounding box of the object in world space (by using the mesh bounding box transformed to world space)
		BoundingBox world_bounding = transformBoundingBox(node_model, node->mesh->box);

		//if bounding box is inside the camera frustum then the object is probably visible
		if (camera->testBoxInFrustum(world_bounding.center, world_bounding.halfsize))
		{
			if (render_boundaries)
				node->mesh->renderBounding(node_model, true);
			renderMeshWithMaterialLight(node_model, node->mesh, node->material);
		}
	}

	//iterate recursively with children
	for (int i = 0; i < node->children.size(); ++i)
		renderNode(node->children[i], camera);
}

void Renderer::extractRenderables(SCN::Node* node, Camera* camera)
{
	if (!node->visible)
		return;

	//compute global matrix
	Matrix44 node_model = node->getGlobalMatrix(true);

	//does this node have a mesh? then we must render it
	if (node->mesh && node->material)
	{
		//compute the bounding box of the object in world space (by using the mesh bounding box transformed to world space)
		BoundingBox world_bounding = transformBoundingBox(node_model, node->mesh->box);

		//if bounding box is inside the camera frustum then the object is probably visible
		if (camera->testBoxInFrustum(world_bounding.center, world_bounding.halfsize))
		{
			Renderable re;
			re.model = node_model;
			re.mesh = node->mesh;
			re.material = node->material;
			re.boundingbox = transformBoundingBox(re.model, re.mesh->box);
			re.dist_to_cam = camera->eye.distance(world_bounding.center);


			if (isOpaque(*re.material)) {
				opaqueRenderables.push_back(re);
			}
			else {
				transparentRenderables.push_back(re);
			}
		}
	}

	//iterate recursively with children
	for (int i = 0; i < node->children.size(); ++i)
		extractRenderables(node->children[i], camera);
}


void Renderer::sortTransparentRenderables(std::vector<Renderable>& transparentRenderables) {
	// Encuentra el inicio de los renderables transparentes
	std::sort(transparentRenderables.begin(), transparentRenderables.end(),
		[](const Renderable& a, const Renderable& b) {
			return a.dist_to_cam > b.dist_to_cam;  // de más lejano a más cercano
		});
}


bool Renderer::isOpaque(SCN::Material& material)
{
	return material.alpha_mode == SCN::NO_ALPHA;
}

void Renderer::generateShadowMaps(Camera* main_camera)
{
	if (!sunLight)
		return;

	LightEntity* light = sunLight;

	if (!light->cast_shadows)
		return;

	if (light->shadowmap_fbo == nullptr || light->shadowmap_fbo->width != shadowmap_size)
	{
		//Si he cambiado el size del shadowmap.
		if (light->shadowmap_fbo)
			delete light->shadowmap_fbo;

		light->shadowmap_fbo = new GFX::FBO();
		light->shadowmap_fbo->setDepthOnly(shadowmap_size, shadowmap_size);

	}

	light->shadowmap_fbo->bind();
	glClear(GL_DEPTH_BUFFER_BIT);

	Camera camera;

	vec3 pos = main_camera->eye;

	camera.lookAt(pos, pos + light->root.global_model.frontVector() * -1.0f, vec3(0, 1, 0));

	//Solo para la direccional
	camera.setOrthographic(light->area * -0.5, light->area * 0.5, light->area * -0.5, light->area * 0.5, light->near_distance, light->max_distance);
	//compute texel size in world units, where frustum size is the distance from left to right in the camera
	float grid = light->area / (float)shadowmap_size;
	camera.enable();
	//snap camera X,Y to that size in camera space assuming
	camera.view_matrix.M[3][0] = round(camera.view_matrix.M[3][0] / grid) * grid;
	camera.view_matrix.M[3][1] = round(camera.view_matrix.M[3][1] / grid) * grid;
	//update viewproj matrix (be sure no one changes it)
	camera.viewprojection_matrix = camera.view_matrix * camera.projection_matrix;

	for (auto& re : opaqueRenderables)
		if (camera.testBoxInFrustum(re.boundingbox.center, re.boundingbox.halfsize))
			renderMeshWithMaterialFlat(re.model, re.mesh, re.material);

	light->shadowmap_viewprojection = camera.viewprojection_matrix;
	light->shadowmap_fbo->unbind();
	/*
	for (int i = 0; i < lights.size(); i++)
	{
		LightEntity* light = lights[i];
		if (light->light_type == eLightType::POINT)
			continue;

		if (light->light_type == eLightType::SPOT)
			continue;
	}
	*/
}


//Renders a mesh given its transform and material
void Renderer::renderMeshWithMaterial(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material)
{
	//in case there is nothing to do
	if (!mesh || !mesh->getNumVertices() || !material)
		return;
	assert(glGetError() == GL_NO_ERROR);

	//define locals to simplify coding
	GFX::Shader* shader = NULL;
	GFX::Texture* texture = NULL;
	Camera* camera = Camera::current;

	texture = material->textures[SCN::eTextureChannel::ALBEDO].texture;
	//texture = material->emissive_texture;
	//texture = material->metallic_roughness_texture;
	//texture = material->normal_texture;
	//texture = material->occlusion_texture;
	if (texture == NULL)
		texture = GFX::Texture::getWhiteTexture(); //a 1x1 white texture

	//select the blending
	if (material->alpha_mode == SCN::eAlphaMode::BLEND)
	{
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
	else
		glDisable(GL_BLEND);

	//select if render both sides of the triangles
	if (material->two_sided)
		glDisable(GL_CULL_FACE);
	else
		glEnable(GL_CULL_FACE);
	assert(glGetError() == GL_NO_ERROR);

	glEnable(GL_DEPTH_TEST);

	//chose a shader
	shader = GFX::Shader::Get("texture");

	assert(glGetError() == GL_NO_ERROR);

	//no shader? then nothing to render
	if (!shader)
		return;
	shader->enable();

	//upload uniforms
	shader->setUniform("u_model", model);
	cameraToShader(camera, shader);
	float t = getTime();
	shader->setUniform("u_time", t);

	shader->setUniform("u_color", material->color);
	if (texture)
		shader->setUniform("u_texture", texture, 0);

	//this is used to say which is the alpha threshold to what we should not paint a pixel on the screen (to cut polygons according to texture alpha)
	shader->setUniform("u_alpha_cutoff", material->alpha_mode == SCN::eAlphaMode::MASK ? material->alpha_cutoff : 0.001f);

	if (render_wireframe)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	//do the draw call that renders the mesh into the screen
	mesh->render(GL_TRIANGLES);

	//disable shader
	shader->disable();

	//set the render state as it was before to avoid problems with future renders
	glDisable(GL_BLEND);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void Renderer::renderMeshWithMaterialGBuffers(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material)
{
	//in case there is nothing to do
	if (!mesh || !mesh->getNumVertices() || !material)
		return;
	assert(glGetError() == GL_NO_ERROR);

	//define locals to simplify coding
	GFX::Shader* shader = NULL;
	GFX::Texture* texture = NULL;
	GFX::Texture* textureEmissive = NULL;
	GFX::Texture* textureMetallicRoughness = NULL;
	GFX::Texture* textureNormal = NULL;

	Camera* camera = Camera::current;

	texture = material->textures[SCN::eTextureChannel::ALBEDO].texture;
	textureMetallicRoughness = material->textures[SCN::eTextureChannel::METALLIC_ROUGHNESS].texture;
	textureNormal = material->textures[SCN::eTextureChannel::NORMALMAP].texture;
	textureEmissive = material->textures[SCN::eTextureChannel::EMISSIVE].texture;


	if (texture == NULL)
		texture = GFX::Texture::getWhiteTexture(); //a 1x1 white texture

	glDisable(GL_BLEND);

	//select if render both sides of the triangles
	if (material->two_sided)
		glDisable(GL_CULL_FACE);
	else
		glEnable(GL_CULL_FACE);
	assert(glGetError() == GL_NO_ERROR);

	glEnable(GL_DEPTH_TEST);

	//chose a shader
	shader = GFX::Shader::Get("gbuffers");
	assert(shader);

	assert(glGetError() == GL_NO_ERROR);

	//no shader? then nothing to render
	if (!shader)
		return;
	shader->enable();

	//upload uniforms
	shader->setUniform("u_model", model);
	cameraToShader(camera, shader);

	shader->setUniform("u_irradiance_scene_active", irradiance_scene);

	shader->setUniform("u_color", material->color);
	if (texture)
		shader->setUniform("u_texture", texture, 0);

	if (textureMetallicRoughness)
		shader->setUniform("u_roughness_metallic_texture", textureMetallicRoughness, 1);
		shader->setUniform("u_metallic_factor", material->metallic_factor);
		shader->setUniform("u_roughness_factor", material->roughness_factor);

	if (textureNormal)
		shader->setUniform("u_normal_texture", textureNormal, 2);
		

	//Si el material tiene factor emisivo y su textura es emisiva tambien, envio la flag al shader y su correspondiente textura
	if (material->emissive_factor.length() > 0 && textureEmissive) {
		shader->setUniform("u_emissive_texture_enabled", true);
		shader->setUniform("u_emissive_texture", textureEmissive, 3);
		shader->setUniform("u_emissive_factor", material->emissive_factor);

	}

	//Si solo tengo factor emisivo
	else if (material->emissive_factor.length() > 0) {
		shader->setUniform("u_emissive_texture_enabled", false);
		shader->setUniform("u_emissive_factor", material->emissive_factor);
	}
	//Si no tengo factor emisivo, lo pongo a 0.
	else {
		shader->setUniform("u_emissive_texture_enabled", false);
		//Establecer el factor emisivo a cero para materiales no emisivos
		shader->setUniform("u_emissive_factor", Vector3f(0, 0, 0));
	}


	//this is used to say which is the alpha threshold to what we should not paint a pixel on the screen (to cut polygons according to texture alpha)
	shader->setUniform("u_alpha_cutoff", material->alpha_mode == SCN::eAlphaMode::MASK ? material->alpha_cutoff : 0.001f);

	//do the draw call that renders the mesh into the screen
	mesh->render(GL_TRIANGLES);

	//disable shader
	shader->disable();

}

//Renders a mesh given its transform and material
void Renderer::renderMeshWithMaterialFlat(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material)
{
	//in case there is nothing to do
	if (!mesh || !mesh->getNumVertices() || !material)
		return;
	assert(glGetError() == GL_NO_ERROR);

	//define locals to simplify coding
	GFX::Shader* shader = NULL;
	GFX::Texture* texture = NULL;
	Camera* camera = Camera::current;

	texture = material->textures[SCN::eTextureChannel::ALBEDO].texture;

	if (texture == NULL)
		texture = GFX::Texture::getWhiteTexture(); //a 1x1 white texture

	glDisable(GL_BLEND);

	//select if render both sides of the triangles
	if (material->two_sided)
		glDisable(GL_CULL_FACE);
	else
		glEnable(GL_CULL_FACE);
	assert(glGetError() == GL_NO_ERROR);

	glEnable(GL_DEPTH_TEST);

	//chose a shader
	shader = GFX::Shader::Get("texture");

	assert(glGetError() == GL_NO_ERROR);

	//no shader? then nothing to render
	if (!shader)
		return;
	shader->enable();

	//upload uniforms
	shader->setUniform("u_model", model);
	cameraToShader(camera, shader);

	shader->setUniform("u_color", material->color);
	if (texture)
		shader->setUniform("u_texture", texture, 0);

	//this is used to say which is the alpha threshold to what we should not paint a pixel on the screen (to cut polygons according to texture alpha)
	shader->setUniform("u_alpha_cutoff", material->alpha_mode == SCN::eAlphaMode::MASK ? material->alpha_cutoff : 0.001f);

	//do the draw call that renders the mesh into the screen
	mesh->render(GL_TRIANGLES);

	//disable shader
	shader->disable();

}


void Renderer::renderMeshWithMaterialLight(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material)
{
	//in case there is nothing to do
	if (!mesh || !mesh->getNumVertices() || !material)
		return;
	assert(glGetError() == GL_NO_ERROR);

	//define locals to simplify coding
	GFX::Shader* shader = NULL;
	GFX::Texture* texture = NULL;
	GFX::Texture* textureEmissive = NULL;
	GFX::Texture* textureAO = NULL;
	GFX::Texture* textureNormal = NULL;

	Camera* camera = Camera::current;


	//Cargo las distintas texturas

	texture = material->textures[SCN::eTextureChannel::ALBEDO].texture;
	textureAO = material->textures[SCN::eTextureChannel::OCCLUSION].texture;
	textureNormal = material->textures[SCN::eTextureChannel::NORMALMAP].texture;
	
	if (texture == NULL)
		texture = GFX::Texture::getWhiteTexture(); //a 1x1 white texture

	// Si el material tiene textura emisiva, la guardo en textureEmissive
	if (material->textures[SCN::eTextureChannel::EMISSIVE].texture) {
		textureEmissive = material->textures[SCN::eTextureChannel::EMISSIVE].texture;
	}

	//Si la textura emissive no está definida o es NULL
	if (!textureEmissive) {
		textureEmissive = GFX::Texture::getBlackTexture(); // Fallback a una textura negra
	}

	if (textureNormal == NULL)
		textureNormal = GFX::Texture::getWhiteTexture(); //a 1x1 black texture

	if (textureAO == NULL)
		textureAO = GFX::Texture::getBlackTexture(); //a 1x1 black texture



	//select the blending
	if (material->alpha_mode == SCN::eAlphaMode::BLEND)
	{
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
	else
		glDisable(GL_BLEND);

	//select if render both sides of the triangles
	if (material->two_sided)
		glDisable(GL_CULL_FACE);
	else
		glEnable(GL_CULL_FACE);
	assert(glGetError() == GL_NO_ERROR);

	glEnable(GL_DEPTH_TEST);

	shader = GFX::Shader::Get(multipass ? "multipass" : "singlepass");

	assert(glGetError() == GL_NO_ERROR);

	//no shader? then nothing to render
	if (!shader)
		return;
	shader->enable();

	//upload uniforms
	shader->setUniform("u_model", model);
	cameraToShader(camera, shader);
	float t = getTime();
	shader->setUniform("u_time", t);

	shader->setUniform("u_ambient_light", scene->ambient_light);

	shader->setUniform("u_color", material->color);
	if (texture)
		shader->setUniform("u_texture", texture, 0);

	if (textureAO)
		shader->setUniform("u_roughness_metallic_texture", textureAO, 1);
	shader->setUniform("u_roughness_factor", material->roughness_factor);
	shader->setUniform("u_metallic_factor", material->metallic_factor);


	if (textureNormal)
		shader->setUniform("u_normal_texture", textureNormal, 2);

	//Si el material tiene factor emisivo y su textura es emisiva tambien, envio la flag al shader y su correspondiente textura
	if (material->emissive_factor.length() > 0 && material->textures[eTextureChannel::EMISSIVE].texture) {
		shader->setUniform("u_emissive_texture_enabled", true);
		shader->setUniform("u_emissive_texture", textureEmissive, 3);
		shader->setUniform("u_emissive_factor", material->emissive_factor);
	}

	//Si solo tengo factor emisivo
	else if (material->emissive_factor.length() > 0) {
		shader->setUniform("u_emissive_texture_enabled", false);
		shader->setUniform("u_emissive_factor", material->emissive_factor);
	}
	//Si no tengo factor emisivo, lo pongo a 0.
	else {
		shader->setUniform("u_emissive_texture_enabled", false);
		//Establecer el factor emisivo a cero para materiales no emisivos
		shader->setUniform("u_emissive_factor", Vector3f(0, 0, 0));
	}

	//this is used to say which is the alpha threshold to what we should not paint a pixel on the screen (to cut polygons according to texture alpha)
	shader->setUniform("u_alpha_cutoff", material->alpha_mode == SCN::eAlphaMode::MASK ? material->alpha_cutoff : 0.001f);

	if (render_wireframe)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);


	if (material->alpha_mode != SCN::eAlphaMode::BLEND)
	{
		glBlendFunc(GL_SRC_ALPHA, GL_ONE);
		glDisable(GL_BLEND);
	}

	glDepthFunc(GL_LEQUAL);


	if (!lights.empty()) {
		if (multipass) {

			// Multipass
			for (int i = 0; i < lights.size(); i++) {
				LightEntity* light = lights[i];
				LightToShader(light, shader);

				mesh->render(GL_TRIANGLES);

				glEnable(GL_BLEND);
				glBlendFunc(GL_ONE, GL_ONE);

				shader->setUniform("u_ambient_light", vec3(0.0));
				shader->setUniform("u_emissive_factor", Vector3f(0, 0, 0));
			}
		}
		else {

			//Singlepass 
			LightsToShader(shader);

			mesh->render(GL_TRIANGLES);
			glEnable(GL_BLEND);
		}
	}
	else {
		// No hay luces 
		shader->setUniform("u_light_type", 0);
		mesh->render(GL_TRIANGLES);
	}

	glDepthFunc(GL_LESS);


	//disable shader
	shader->disable();

	//set the render state as it was before to avoid problems with future renders
	glDisable(GL_BLEND);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}


void SCN::Renderer::cameraToShader(Camera* camera, GFX::Shader* shader)
{
	shader->setUniform("u_viewprojection", camera->viewprojection_matrix);
	shader->setUniform("u_camera_position", camera->eye);
}

void SCN::Renderer::LightsToShader(GFX::Shader* shader)
{

	const int MAX_LIGHTS = 8;
	Vector3f lightPositions[MAX_LIGHTS];
	Vector3f lightColors[MAX_LIGHTS];
	int lightTypes[MAX_LIGHTS];
	float lightMaxDistances[MAX_LIGHTS];
	vec2 lightConeInfos[MAX_LIGHTS];
	float lightAreas[MAX_LIGHTS];
	Vector3f lightFronts[MAX_LIGHTS];
	float lightShadowBias[MAX_LIGHTS];
	int lightCastShadows[MAX_LIGHTS];
	GLuint shadowMapTextures[MAX_LIGHTS];
	int num_lights = (int)lights.size();
	mat4 lightShadowViewProjections[MAX_LIGHTS];

	for (int i = 0; i < num_lights; i++) {
		LightEntity* light = lights[i];
		lightPositions[i] = light->root.model.getTranslation();
		lightColors[i] = light->color * light->intensity;
		lightTypes[i] = (int)light->light_type;
		lightMaxDistances[i] = light->max_distance;
		lightConeInfos[i] = light->cone_info;
		lightAreas[i] = light->area;
		lightFronts[i] = light->root.model.frontVector().normalize();
		lightShadowBias[i] = light->shadow_bias;
		lightCastShadows[i] = light->cast_shadows ? 1 : 0;


		if (light->cast_shadows && light->shadowmap_fbo)
		{
			lightShadowViewProjections[i] = light->shadowmap_viewprojection;
			shadowMapTextures[i] = light->shadowmap_fbo->depth_texture->texture_id;
		}
		else
			shadowMapTextures[i] = 0;
	}


	shader->setUniform3Array("u_lightPositions", (const float*)lightPositions, num_lights);
	shader->setUniform3Array("u_lightColors", (const float*)lightColors, num_lights);
	shader->setUniform1Array("u_lightTypes", lightTypes, num_lights);
	shader->setUniform1Array("u_lightMaxDistances", lightMaxDistances, num_lights);
	shader->setUniform2Array("u_lightConeInfos", (const float*)lightConeInfos, num_lights);
	shader->setUniform1Array("u_lightAreas", lightAreas, num_lights);
	shader->setUniform3Array("u_lightFronts", (const float*)lightFronts, num_lights);
	shader->setUniform1Array("u_shadowBias", lightShadowBias, num_lights);
	shader->setUniform1Array("u_lightCastShadows", lightCastShadows, num_lights);

	for (int i = 0; i < num_lights; i++) {
		//Suponiendo que a partir de la textura 8 voy a poner mas textureMaps.
		glActiveTexture(GL_TEXTURE0 + 8 + i);
		glBindTexture(GL_TEXTURE_2D, shadowMapTextures[i]);
		std::string shadowMapUniformName = "u_shadowMaps[" + std::to_string(i) + "]";
		shader->setUniform(shadowMapUniformName.c_str(), 8 + i);

		// Enviar la matriz de vista y proyección para la sombra
		std::string shadowVPUniformName = "u_shadowViewProjections[" + std::to_string(i) + "]";
		shader->setUniform(shadowVPUniformName.c_str(), lightShadowViewProjections[i]);
	}

}
void SCN::Renderer::LightToShader(LightEntity* light, GFX::Shader* shader)
{

	shader->setUniform("u_light_position", light->root.model.getTranslation());
	shader->setUniform("u_light_color", light->color * light->intensity);
	shader->setUniform("u_light_type", (int)light->light_type);
	shader->setUniform("u_light_max_distance", light->max_distance);
	shader->setUniform("u_light_cone_info", light->cone_info);
	shader->setUniform("u_light_front", light->root.model.frontVector().normalize());

	if (light->cast_shadows && light->shadowmap_fbo)
	{
		shader->setUniform("u_light_cast_shadows", 1);
		shader->setUniform("u_shadowmap", light->shadowmap_fbo->depth_texture, 8);
		shader->setUniform("u_shadow_viewprojection", light->shadowmap_viewprojection);
		shader->setUniform("u_shadow_bias", light->shadow_bias);
	}
	else
		shader->setUniform("u_light_cast_shadows", 0);


}
#ifndef SKIP_IMGUI

void Renderer::showUI()
{
	ImGui::Combo("Pipeline", (int*)&pipeline_mode, "FLAT\0FORWARD\0DEFERRED\0", ePipeLineMode::PIPE_COUNT);
	ImGui::Combo("GBuffers", (int*)&view_gbuffers, "NONE\0COLOR\0NORMAL\0EXTRA\0DEPTH\0EMISSIVE", eShowGbuffers::GBUFFERS_COUNT);


	ImGui::Checkbox("Irradiance_Scene", &irradiance_scene);

	ImGui::Checkbox("Wireframe", &render_wireframe);
	ImGui::Checkbox("Boundaries", &render_boundaries);
	ImGui::Checkbox("Multipass", &multipass);
	ImGui::Checkbox("Alpha sorting", &alpha_sorting);
	ImGui::Checkbox("ShowSSAO", &showSSAO);
	//ImGui::Checkbox("Dithering", &disable_dithering);
	ImGui::SliderFloat("SSAO_radius", &ssao_radius, 0.01f,0.0f);
	ImGui::SliderFloat("SSAO_max_distance", &ssao_max_distance, 0.01f, 0.0f);

	if (ImGui::Button("Shadowmap 256"))
		shadowmap_size = 256;

	if (ImGui::Button("Shadowmap 512"))
		shadowmap_size = 512;

	if (ImGui::Button("Shadowmap 1024"))
		shadowmap_size = 1024;

	if (ImGui::Button("Shadowmap 2048"))
		shadowmap_size = 2048;

	//add here your stuff
	//...
}

#else
void Renderer::showUI() {}
#endif