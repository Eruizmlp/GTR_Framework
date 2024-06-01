#pragma once
#include "scene.h"
#include "prefab.h"

#include "light.h"
#include "../gfx/sphericalharmonics.h"

//forward declarations
class Camera;
class Skeleton;
namespace GFX {
	class Shader;
	class Mesh;
	class FBO;
}

enum ePipeLineMode {
	FLAT,
	FORWARD,
	DEFERRED,
	PIPE_COUNT
};

enum eShowGbuffers {
	NONE,
	COLOR,
	NORMAL,
	EXTRA,
	DEPTH,
	EMISSIVE,
	GBUFFERS_COUNT
};

struct sProbe
{
	vec3 pos;
	vec3 local;
	int index;
	SphericalHarmonics sh;

};


namespace SCN {

	class Prefab;
	class Material;

	class Renderable {
	public:
		mat4 model;
		GFX::Mesh* mesh;
		SCN::Material* material;
		BoundingBox boundingbox;
		float dist_to_cam;
	};

	// This class is in charge of rendering anything in our system.
	// Separating the render from anything else makes the code cleaner
	class Renderer
	{
	public:
		bool render_wireframe;
		bool render_boundaries;

		bool irradiance_scene;

		int shadowmap_size;
		float ssao_radius;
		float ssao_max_distance;

		GFX::Texture* skybox_cubemap;
		SCN::Scene* scene;

		//Containers
		std::vector<Renderable> opaqueRenderables;
		std::vector<Renderable> transparentRenderables;
		std::vector<Renderable> renderables;

		std::vector<LightEntity*> lights;

		LightEntity* sunLight;

		ePipeLineMode pipeline_mode;
		eShowGbuffers view_gbuffers;

		//Flag para controlar el tipo de renderizado.
		bool multipass;
		bool alpha_sorting;
		bool disable_dithering;
		bool showSSAO;

		//updated every frame
		Renderer(const char* shaders_atlas_filename);

		//just to be sure we have everything ready for the rendering
		void setupScene();

		//add here your functions

		void generateShadowMaps(Camera* camera);

		void extractRenderables(SCN::Node* node, Camera* camera);

		void LightsToShader(GFX::Shader* shader);

		bool isOpaque(SCN::Material& material);

		void sortTransparentRenderables(std::vector<Renderable>& transparentRenderables);

		void extractSceneInfo(SCN::Scene* scene, Camera* camera);

		void renderMeshWithMaterialLight(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material);

		void LightToShader(LightEntity* light, GFX::Shader* shader);

		//renders several elements of the scene
		void renderScene(SCN::Scene* scene, Camera* camera);
		void renderSceneForward(SCN::Scene* scene, Camera* camera);
		void renderSceneDeferred(SCN::Scene* scene, Camera* camera);
		
		
		//render the skybox
		void renderSkybox(GFX::Texture* cubemap);

		//to render one node from the prefab and its children
		void renderNode(SCN::Node* node, Camera* camera);

		//to render one mesh given its material and transformation matrix
		void renderMeshWithMaterial(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material);
		void renderMeshWithMaterialFlat(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material);
		void renderMeshWithMaterialGBuffers(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material);



		void showUI();

		void cameraToShader(Camera* camera, GFX::Shader* shader); //sends camera uniforms to shader
	};

};