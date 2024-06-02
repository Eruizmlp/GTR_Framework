//example of some shaders compiled
flat basic.vs flat.fs
texture basic.vs texture.fs
multipass basic.vs multipass.fs
singlepass basic.vs singlepass.fs
skybox basic.vs skybox.fs
depth quad.vs depth.fs
multi basic.vs multi.fs
gbuffers basic.vs gbuffers.fs
ssao quad.vs ssao.fs
deferred_global quad.vs deferred_global.fs
tonemapping quad.vs tonemapping.fs
blur_h quad.vs blur_h.fs
blur_v quad.vs blur_v.fs


\blur_h.fs
#version 330 core

in vec2 v_uv;
out vec4 FragColor;

uniform sampler2D u_ssao_texture;
uniform vec2 u_iRes;

void main() {
    vec2 tex_offset = u_iRes; // size of one texel
    float kernel[5] = float[](0.2270270270, 0.3162162162, 0.0702702703, 0.0027027027, 0.0013501350);
    vec3 result = texture(u_ssao_texture, v_uv).rgb * kernel[0]; // central pixel

    for(int i = 1; i < 5; ++i) {
        result += texture(u_ssao_texture, v_uv + vec2(tex_offset.x * i, 0.0)).rgb * kernel[i];
        result += texture(u_ssao_texture, v_uv - vec2(tex_offset.x * i, 0.0)).rgb * kernel[i];
    }
    FragColor = vec4(result, 1.0);
}

\blur_v.fs
#version 330 core

in vec2 v_uv;
out vec4 FragColor;

uniform sampler2D u_ssao_texture;
uniform vec2 u_iRes;

void main() {
    vec2 tex_offset = u_iRes; // size of one texel
    float kernel[5] = float[](0.2270270270, 0.3162162162, 0.0702702703, 0.0027027027, 0.0013501350);
    vec3 result = texture(u_ssao_texture, v_uv).rgb * kernel[0]; // central pixel

    for(int i = 1; i < 5; ++i) {
        result += texture(u_ssao_texture, v_uv + vec2(0.0, tex_offset.y * i)).rgb * kernel[i];
        result += texture(u_ssao_texture, v_uv - vec2(0.0, tex_offset.y * i)).rgb * kernel[i];
    }
    FragColor = vec4(result, 1.0);
}


\basic.vs

#version 330 core

in vec3 a_vertex;
in vec3 a_normal;
in vec2 a_coord;
in vec4 a_color;

uniform vec3 u_camera_pos;

uniform mat4 u_model;
uniform mat4 u_viewprojection;

//this will store the color for the pixel shader
out vec3 v_position;
out vec3 v_world_position;
out vec3 v_normal;
out vec2 v_uv;
out vec4 v_color;

uniform float u_time;

void main()
{	
	//calcule the normal in camera space (the NormalMatrix is like ViewMatrix but without traslation)
	v_normal = (u_model * vec4( a_normal, 0.0) ).xyz;
	
	//calcule the vertex in object space
	v_position = a_vertex;
	v_world_position = (u_model * vec4( v_position, 1.0) ).xyz;
	
	//store the color in the varying var to use it from the pixel shader
	v_color = a_color;

	//store the texture coordinates
	v_uv = a_coord;

	//calcule the position of the vertex using the matrices
	gl_Position = u_viewprojection * vec4( v_world_position, 1.0 );
}

\quad.vs

#version 330 core

in vec3 a_vertex;
in vec2 a_coord;
out vec2 v_uv;

void main()
{	
	v_uv = a_coord;
	gl_Position = vec4( a_vertex, 1.0 );
}


\flat.fs

#version 330 core

uniform vec4 u_color;

out vec4 FragColor;

void main()
{
	FragColor = u_color;
}


\texture.fs

#version 330 core

in vec3 v_position;
in vec3 v_world_position;
in vec3 v_normal;
in vec2 v_uv;
in vec4 v_color;

uniform vec4 u_color;
uniform sampler2D u_texture;
uniform float u_time;
uniform float u_alpha_cutoff;

out vec4 FragColor;

void main()
{
	vec2 uv = v_uv;
	vec4 color = u_color;
	color *= texture( u_texture, v_uv );

	if(color.a < u_alpha_cutoff)
		discard;

	FragColor = color;
}


\skybox.fs

#version 330 core

in vec3 v_position;
in vec3 v_world_position;

uniform samplerCube u_texture;
uniform vec3 u_camera_position;
out vec4 FragColor;

void main()
{
	vec3 E = v_world_position - u_camera_position;
	vec4 color = texture( u_texture, E );
	FragColor = color;
}

\tonemapping.fs
#version 330 core

in vec2 v_uv;

uniform sampler2D u_hdr_texture;
layout(location = 0) out vec4 out_color;

// Constants for the Uncharted 2 tonemapping operator
const float A = 0.15;
const float B = 0.50;
const float C = 0.10;
const float D = 0.20;
const float E = 0.02;
const float F = 0.30;
const float W = 11.2;

vec3 Uncharted2Tonemap(vec3 x) {
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

vec3 linearToSrgb(vec3 color) {
    return pow(color, vec3(1.0 / 2.2));
}

void main()
{
    vec3 hdr_color = texture(u_hdr_texture, v_uv).xyz;

    // Apply tonemapping
    vec3 tonemapped_color = Uncharted2Tonemap(hdr_color * 2.0);
    vec3 white_scale = vec3(1.0) / Uncharted2Tonemap(vec3(W));
    tonemapped_color *= white_scale;

    // Apply gamma correction
    vec3 final_color = linearToSrgb(tonemapped_color);

    out_color = vec4(final_color, 1.0);
}


\ssao.fs
#version 330 core

in vec3 v_position;
in vec2 v_uv;

uniform sampler2D u_normal_texture;
uniform sampler2D u_depth_texture;

uniform float u_radius;
uniform vec3 u_random_points[64];
uniform float u_ssao_max_distance;

uniform vec2 u_iRes;
uniform mat4 u_inverse_viewprojection;
uniform mat4 u_viewprojection;
uniform float near; // Near plane value
uniform float far;  // Far plane value

out vec4 FragColor;

mat3 cotangent_frame(vec3 N, vec3 p, vec2 uv)
{
    // get edge vectors of the pixel triangle
    vec3 dp1 = dFdx( p );
    vec3 dp2 = dFdy( p );
    vec2 duv1 = dFdx( uv );
    vec2 duv2 = dFdy( uv );
    
    // solve the linear system
    vec3 dp2perp = cross( dp2, N );
    vec3 dp1perp = cross( N, dp1 );
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;

    // construct a scale-invariant frame 
    float invmax = inversesqrt( max( dot(T,T), dot(B,B) ) );
    return mat3( T * invmax, B * invmax, N );
}

float depthToLinear(float z)
{
    return near * (z + 1.0) / (far + near - z * (far - near));
}

void main() {
    vec2 uv = gl_FragCoord.xy * u_iRes.xy;
    
    vec3 N = texture(u_normal_texture, v_uv).xyz * 2.0 - vec3(1.0);  // Normal
    N = normalize(N);

    float depth = texture(u_depth_texture, v_uv).x;

    if (depth == 1.0)
        discard;

    vec4 screen_pos = vec4(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 proj_worldpos = u_inverse_viewprojection * screen_pos;
    vec3 world_position = proj_worldpos.xyz / proj_worldpos.w;

    // create the matrix33 to convert from tangent to world
    mat3 rotmat = cotangent_frame( N, world_position, uv );

    int num = 64;
    for(int i = 0; i < 64; i++)
    {
        // rotate a point is easy
        vec3 rotated_point = rotmat * u_random_points[i];

        vec3 p = world_position + rotated_point * u_radius;
        vec4 proj = u_viewprojection * vec4(p, 1.0);
        proj.xy /= proj.w;
        proj.z = (proj.z - 0.005) / proj.w;
        proj.xyz = proj.xyz * 0.5 + vec3(0.5); 

        vec2 proj_uv = clamp(proj.xy, 0.0, 1.0);

        float pdepth = texture(u_depth_texture, proj_uv).x;

        // Linearize depth values
        pdepth = depthToLinear(pdepth);
        float projz = depthToLinear(proj.z);

        // Calculate the depth difference and check how far it is
        float diff = pdepth - projz;
        if(diff < 0.0 && abs(diff) < 0.001)
            num--;
    }

    float ao = float(num) / 64.0;
    FragColor = vec4(ao, ao, ao, 1.0);
}

\computeShadow

uniform int u_light_cast_shadows;
uniform mat4 u_shadow_viewprojection;
uniform float u_shadow_bias;
uniform sampler2D u_shadowmap;


float computeShadows(vec3 wp){

	//project our 3D position to the shadowmap
	vec4 proj_pos = u_shadow_viewprojection * vec4(wp,1.0);

	//from homogeneus space to clip space
	vec2 shadow_uv = proj_pos.xy / proj_pos.w;

	//from clip space to uv space
	shadow_uv = shadow_uv * 0.5 + vec2(0.5);

	//get point depth [-1 .. +1] in non-linear space
	float real_depth = (proj_pos.z - u_shadow_bias) / proj_pos.w;

	//normalize from [-1..+1] to [0..+1] still non-linear
	real_depth = real_depth * 0.5 + 0.5;

	//read depth from depth buffer in [0..+1] non-linear
	float shadow_depth = texture( u_shadowmap, shadow_uv).x;

	//compute final shadow factor by comparing
	float shadow_factor = 1.0;

	//we can compare them, even if they are not linear
	if( shadow_depth < real_depth )
		shadow_factor = 0.0;
	
	return shadow_factor;
}

\computeSpecular
// Phong Specular
vec3 specularReflection(vec3 N, vec3 L, vec3 V, float shininess, vec3 lightColor, float metallic, float roughness) {
    vec3 R = reflect(-L, N); // Reflect vector
    float specAngle = max(dot(R, V), 0.0);
    float specFactor = pow(specAngle, shininess * (1.0 - roughness)); // Higher roughness, lower shininess
    vec3 specColor = lightColor * (metallic + (1.0 - metallic) * vec3(1.0)); // Blend between metal and non-metal
    return specFactor * specColor;
}

\computePixelNormals
mat3 cotangent_frame(vec3 N, vec3 p, vec2 uv)
{
	// get edge vectors of the pixel triangle
	vec3 dp1 = dFdx( p );
	vec3 dp2 = dFdy( p );
	vec2 duv1 = dFdx( uv );
	vec2 duv2 = dFdy( uv );
	
	// solve the linear system
	vec3 dp2perp = cross( dp2, N );
	vec3 dp1perp = cross( N, dp1 );
	vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
	vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
 
	// construct a scale-invariant frame 
	float invmax = inversesqrt( max( dot(T,T), dot(B,B) ) );
	return mat3( T * invmax, B * invmax, N );
}

// assume N, the interpolated vertex normal and 
// WP the world position
//vec3 normal_pixel = texture2D( normalmap, uv ).xyz; 
vec3 perturbNormal(vec3 N, vec3 WP, vec2 uv, vec3 normal_pixel)
{
	normal_pixel = normal_pixel * 255./127. - 128./127.;
	mat3 TBN = cotangent_frame(N, WP, uv);
	return normalize(TBN * normal_pixel);
}


\deferred_global.fs
#version 330 core

in vec3 v_position;
in vec2 v_uv;

uniform vec4 u_color;
uniform sampler2D u_color_texture;
uniform sampler2D u_normal_texture;
uniform sampler2D u_depth_texture;
uniform sampler2D u_emissive_texture;
uniform sampler2D u_ssao_texture;

uniform vec3 u_camera_position;
uniform vec3 u_ambient_light;
uniform vec3 u_light_position;
uniform vec3 u_light_color;
uniform int u_light_type;
uniform float u_light_max_distance;
uniform vec2 u_light_cone_info;
uniform vec3 u_light_front;

uniform vec2 u_iRes;
uniform mat4 u_inverse_viewprojection;
uniform mat4 u_light_viewprojection;

#define POINTLIGHT 1
#define SPOTLIGHT 2
#define DIRECTIONALLIGHT 3

#include "computeShadow"
#include "computeSpecular"

out vec4 FragColor;


void main() {
    vec2 uv = gl_FragCoord.xy * u_iRes.xy;
    vec4 color = texture(u_color_texture, v_uv);  // Albedo + metallic
    vec3 N = texture(u_normal_texture, v_uv).xyz * 2.0 - vec3(1.0);  // Normal
	//vec3 N = texture(u_normal_texture, v_uv).xyz;  // Normal
	N = normalize(N);
    float roughness = texture(u_normal_texture, v_uv).w;  // Roughness
    float metallic = color.w;  // Metallic

    float depth = texture(u_depth_texture, v_uv).x;
    if (depth == 1.0)
        discard;

    vec3 emissive = texture(u_emissive_texture, v_uv).xyz;
    //float ao = texture(u_emissive_texture, v_uv).w;  // AO
	float ao = texture(u_ssao_texture, v_uv).r;  // Use SSAO for AO

    vec4 screen_pos = vec4(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 proj_worldpos = u_inverse_viewprojection * screen_pos;
    vec3 v_world_position = proj_worldpos.xyz / proj_worldpos.w;

    vec3 L = u_light_position - v_world_position;
    float dist = length(L);
    L = normalize(L);

    vec3 V = normalize(u_camera_position - v_world_position);

    vec3 ambient = u_ambient_light * ao;

    float att_factor = 1.0;
    float shadow_factor = 1.0;
    float NdotL = 0.0;
    vec3 diffuse = vec3(0.0);
    vec3 specular = vec3(0.0);

	vec3 light = (ambient);

	#include "computeLights"

    vec3 final_color = light * color.xyz + emissive;
	
    FragColor = vec4(final_color, 1.0);

	gl_FragDepth = depth;
}


\computeLight

if(u_light_type == DIRECTIONALLIGHT)
{

	L = u_light_front;
	NdotL = clamp(dot(N,L),0.0,1.0);
	float shadow_factor = 1.0;
	if(u_light_cast_shadows == 1 ) 
		shadow_factor = computeShadows(v_world_position);
	light += NdotL * u_light_color * shadow_factor;
}
else if(u_light_type == POINTLIGHT || u_light_type == SPOTLIGHT )
{

	NdotL = clamp(dot(N,L), 0.0, 1.0);
	float att_factor = u_light_max_distance - dist;
	att_factor/= u_light_max_distance;
	att_factor = max(att_factor,0.0);
	if(u_light_type == SPOTLIGHT)
	{
		float cos_angle = dot(u_light_front, L);
		if(cos_angle < u_light_cone_info.x)
			NdotL = 0.0;
		else if (cos_angle < u_light_cone_info.y)
			NdotL *= (cos_angle - u_light_cone_info.x) / (u_light_cone_info.y - u_light_cone_info.x);
			
	}
	light += NdotL *u_light_color * att_factor;
}

\computeLights
	
	if (u_light_type != DIRECTIONALLIGHT) {
		
		L = u_light_position - v_world_position;
		dist = length(L);
		L = normalize(L);
		
		//compute a linear attenuation factor
		att_factor = u_light_max_distance - dist;

		//normalize factor
		att_factor /= u_light_max_distance;

		//ignore negative values
		att_factor = max( att_factor, 0.0 );
		
	} else {
		L = normalize(u_light_front); // Vector front para luz direccional
		att_factor = 1.0; // No hay attenuation para luz direccional
		
		if(u_light_cast_shadows == 1){
			shadow_factor = computeShadows(v_world_position);
		}
		
	}

	NdotL = clamp(dot(N, L), 0.0, 1.0);
	diffuse = NdotL * u_light_color * (1.0 - metallic) * att_factor * shadow_factor;
	specular = specularReflection(N, L, V, 1.0 / roughness, u_light_color, metallic, roughness) * att_factor * shadow_factor;

	if (u_light_type == SPOTLIGHT) {
		vec3 D = normalize(-u_light_front);
		float cos_theta = dot(D, -L);
		float cos_inner = cos(radians(u_light_cone_info.x));
		float cos_outer = cos(radians(u_light_cone_info.y));

		
		// Factor que delimita si hay luz o no.
		float spotlight_attenuation = 0.0;

		// Calcula el factor de atenuación angular basado en el cono de luz
		if (cos_theta > cos_inner) {
			spotlight_attenuation = 1.0; // Completa intensidad dentro del cono interno
		} else if (cos_theta > cos_outer) {
			spotlight_attenuation = (cos_theta - cos_outer) / (cos_inner - cos_outer); 
		}

		// Comprueba si estamos dentro del cono de la SPOTLIGHT
		if (spotlight_attenuation > 0.0) {
			NdotL = clamp(dot(N, L), 0.0, 1.0);
			// Aplica atenuación de la SPOTLIGHT al componente difuso y especular
			diffuse = NdotL * u_light_color * (1.0 - metallic) * att_factor * spotlight_attenuation;
			specular = specularReflection(N, L, V, 1.0 / roughness, u_light_color, metallic, roughness) * att_factor * spotlight_attenuation;
		} else {
			// La luz está fuera del cono, no hay contribución difusa o especular
			diffuse = vec3(0.0);
			specular = vec3(0.0);
		}
	}
	
	light += diffuse + specular;  //Sumo la contribución de la difusa y especular

\multipass.fs

#version 330 core

in vec3 v_position;
in vec3 v_world_position;
in vec3 v_normal;
in vec2 v_uv;
in vec4 v_color;


//MATERIAL FACTORS
uniform vec4 u_color;
uniform vec3 u_emissive_factor;
uniform float u_roughness_factor;
uniform float u_metallic_factor;

//TEXTURES
uniform sampler2D u_texture;
uniform sampler2D u_emissive_texture;
uniform sampler2D u_roughness_metallic_texture;
uniform sampler2D u_normal_texture;


uniform bool u_emissive_texture_enabled;
uniform float u_time;
uniform float u_alpha_cutoff;

uniform vec3 u_camera_position;
uniform vec3 u_ambient_light;
uniform vec3 u_light_position;
uniform vec3 u_light_color;
uniform int u_light_type;
uniform float u_light_max_distance;
uniform vec2 u_light_cone_info;
uniform vec3 u_light_front;

#include "computeShadow"
#include "computeSpecular"
#include "computePixelNormals"

#define POINTLIGHT 1
#define SPOTLIGHT 2
#define DIRECTIONALLIGHT 3

out vec4 FragColor;


void main() {
    vec2 uv = v_uv;
    vec4 color = u_color * texture(u_texture, uv);
    vec3 emissiveColor = u_emissive_texture_enabled ? texture(u_emissive_texture, uv).xyz * u_emissive_factor : u_emissive_factor;
    vec3 roughness_metallic_texture = texture(u_roughness_metallic_texture, uv).xyz;
    float ao = roughness_metallic_texture.x;
    float roughness = roughness_metallic_texture.y * u_roughness_factor;
    float metallic = texture(u_roughness_metallic_texture, uv).z * u_metallic_factor;

    if (color.a < u_alpha_cutoff) discard;

    vec3 ambient = u_ambient_light; //* (ao + 0.2);
    //vec3 N = perturbNormal(normalize(v_normal), v_world_position, uv, texture(u_normal_texture, uv).xyz);
	vec3 N = normalize(v_normal);
	
    vec3 V = normalize(u_camera_position - v_world_position);
    vec3 L;
    float dist, NdotL, att_factor;
    vec3 diffuse, specular = vec3(0.0);
	float shadow_factor = 1.0;

	vec3 light = ambient;

	#include "computeLights"
	
    vec4 final_color = vec4(color.xyz * light + emissiveColor, color.a);
    FragColor = final_color;
}

\singlepass.fs
#version 330 core

in vec3 v_position;
in vec3 v_world_position;
in vec3 v_normal;
in vec2 v_uv;
in vec4 v_color;

uniform vec3 u_camera_position;
uniform vec4 u_color;

#define MAX_LIGHTS 8

//MATERIAL FACTORS
uniform vec3 u_emissive_factor;
uniform float u_roughness_factor;
uniform float u_metallic_factor;


//TEXTURES
uniform sampler2D u_texture;
uniform sampler2D u_emissive_texture;
uniform sampler2D u_roughness_metallic_texture;
uniform sampler2D u_normal_texture;
uniform sampler2D u_shadowMaps[MAX_LIGHTS];


uniform bool u_emissive_texture_enabled;
uniform float u_time;
uniform float u_alpha_cutoff;


//SHADOWS
uniform int u_lightCastShadows[MAX_LIGHTS];
uniform mat4 u_shadowViewProjections[MAX_LIGHTS];
uniform float u_shadowBias[MAX_LIGHTS];


uniform vec3 u_ambient_light;

#include "computePixelNormals"

#define POINTLIGHT 1
#define SPOTLIGHT 2
#define DIRECTIONALLIGHT 3

// Uniforms para las luces
uniform vec3 u_lightPositions[MAX_LIGHTS];
uniform vec3 u_lightColors[MAX_LIGHTS];
uniform int u_lightTypes[MAX_LIGHTS];
uniform float u_lightMaxDistances[MAX_LIGHTS];
uniform vec2 u_lightConeInfos[MAX_LIGHTS];
uniform float u_lightAreas[MAX_LIGHTS];
uniform vec3 u_lightFronts[MAX_LIGHTS];

out vec4 FragColor;



// Phong Specular
vec3 specularReflection(vec3 N, vec3 L, vec3 V, float shininess, vec3 lightColor, float metallic, float roughness)  {
    vec3 R = reflect(-L, N); // Reflect vector
    float specAngle = max(dot(R, V), 0.0);
    float specFactor = pow(specAngle, shininess * (1.0 - roughness)); // Higher roughness, lower shininess
    vec3 specColor = lightColor * (metallic + (1.0 - metallic) * vec3(1.0)); // Blend between metal and non-metal
    return specFactor * specColor;
}

float computeShadowFactor(vec3 worldPos, int lightIndex) {
    
    vec4 proj_pos = u_shadowViewProjections[lightIndex] * vec4(worldPos, 1.0);
	
    //from homogeneus space to clip space
	vec2 shadow_uv = proj_pos.xy / proj_pos.w;
	
	//from clip space to uv space
	shadow_uv = shadow_uv * 0.5 + vec2(0.5);
	
   //get point depth [-1 .. +1] in non-linear space
	float real_depth = (proj_pos.z - u_shadowBias[lightIndex]) / proj_pos.w;

	//normalize from [-1..+1] to [0..+1] still non-linear
	real_depth = real_depth * 0.5 + 0.5;

  //read depth from depth buffer in [0..+1] non-linear
	float shadow_depth = texture( u_shadowMaps[lightIndex], shadow_uv).x;

    //compute final shadow factor by comparing
	float shadow_factor = 1.0;

	//we can compare them, even if they are not linear
	if( shadow_depth < real_depth )
		shadow_factor = 0.0;
	
	return shadow_factor;
}
void main() {
    vec2 uv = v_uv;
    vec4 color = u_color * texture(u_texture, uv);
    vec3 emissiveColor = u_emissive_texture_enabled ? texture(u_emissive_texture, uv).xyz * u_emissive_factor : u_emissive_factor;
    vec3 roughness_metallic_texture = texture(u_roughness_metallic_texture, uv).xyz;
    float ao = roughness_metallic_texture.x;
    float roughness = roughness_metallic_texture.y * u_roughness_factor;
    float metallic = texture(u_roughness_metallic_texture, uv).z * u_metallic_factor;

    if (color.a < u_alpha_cutoff) discard;

    vec3 ambient = u_ambient_light * (ao + 0.2);
    vec3 N = perturbNormal(normalize(v_normal), v_world_position, uv, texture(u_normal_texture, uv).xyz);
    vec3 V = normalize(u_camera_position - v_world_position); 
    vec3 light = ambient;
    vec3 L;
    float dist, NdotL, att_factor, angular_factor;
    vec3 diffuse, specular;
	float shadow_factor = 1.0;
	
    for (int i = 0; i < MAX_LIGHTS; ++i) {
        if (u_lightTypes[i] != DIRECTIONALLIGHT) {
		
			L = u_lightPositions[i] - v_world_position;
			dist = length(L);
			L = normalize(L);
			
			//compute a linear attenuation factor
			att_factor = u_lightMaxDistances[i] - dist;

			//normalize factor
			att_factor /= u_lightMaxDistances[i];

			//ignore negative values
			att_factor = max( att_factor, 0.0 );
			
		} else {
			L = normalize(u_lightFronts[i]); // Vector front para luz direccional
			att_factor = 1.0; // No hay attenuation para luz direccional
			if(u_lightCastShadows[i] == 1){
				shadow_factor = computeShadowFactor(v_world_position, i);
			}
		}	

		NdotL = clamp(dot(N, L), 0.0, 1.0);
		diffuse = NdotL * u_lightColors[i] * (1.0 - metallic) * att_factor * shadow_factor;
		specular = specularReflection(N, L, V, 1.0 / roughness, u_lightColors[i], metallic, roughness) * att_factor *shadow_factor;

		
	if (u_lightTypes[i] == SPOTLIGHT) {
		vec3 D = normalize(-u_lightFronts[i]);
		float cos_theta = dot(D, -L);
		float cos_inner = cos(radians(u_lightConeInfos[i].x));
		float cos_outer = cos(radians(u_lightConeInfos[i].y));

		// Calcula el factor de atenuación lineal basado en la distancia
		att_factor = max(u_lightMaxDistances[i] - dist, 0.0) / u_lightMaxDistances[i];
		att_factor = max(att_factor, 0.0); // Ignora valores negativos

		float spotlight_attenuation = 0.0;

		// Calcula el factor de atenuación angular basado en el cono de luz
		if (cos_theta > cos_inner) {
			spotlight_attenuation = 1.0; // Completa intensidad dentro del cono interno
		} else if (cos_theta > cos_outer) {
			spotlight_attenuation = (cos_theta - cos_outer) / (cos_inner - cos_outer); 
		}

		// Comprueba si estamos dentro del cono de la SPOTLIGHT
		if (spotlight_attenuation > 0.0) {
			NdotL = clamp(dot(N, L), 0.0, 1.0);
			// Aplica atenuación de la SPOTLIGHT al componente difuso y especular
			diffuse = NdotL * u_lightColors[i] * (1.0 - metallic) * att_factor * spotlight_attenuation;
			specular = specularReflection(N, L, V, 1.0 / roughness, u_lightColors[i], metallic, roughness) * att_factor * spotlight_attenuation;
		} else {
			// La luz está fuera del cono, no hay contribución difusa o especular
			diffuse = vec3(0.0);
			specular = vec3(0.0);
		}
	}
	
	light += diffuse + specular;  //Sumo la contribución de la difusa y especular
    vec4 final_color = vec4(color.xyz * light + emissiveColor, color.a);
    FragColor = final_color;
	}
}

\skybox.fs

#version 330 core

in vec3 v_position;
in vec3 v_world_position;

uniform samplerCube u_texture;
uniform vec3 u_camera_position;
out vec4 FragColor;

void main()
{
	vec3 E = v_world_position - u_camera_position;
	vec4 color = texture( u_texture, E );
	FragColor = color;
}


\gbuffers.fs
#version 330 core

in vec3 v_position;
in vec3 v_world_position;
in vec3 v_normal;
in vec2 v_uv;

uniform vec4 u_color;
uniform sampler2D u_texture;
uniform sampler2D u_normal_texture;
uniform sampler2D u_roughness_metallic_texture;
uniform sampler2D u_emissive_texture;

uniform bool u_emissive_texture_enabled;
uniform bool u_irradiance_scene_active;

uniform vec3 u_emissive_factor;
uniform float u_metallic_factor;
uniform float u_roughness_factor;

uniform float u_time;
uniform float u_alpha_cutoff;

#include "computePixelNormals"

layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 NormalColor;
layout(location = 2) out vec4 EmissiveColor;
layout(location = 3) out vec4 DepthColor;

vec3 srgbToLinear(vec3 color) {
    return pow(color, vec3(2.2));
}

void main()
{
    vec2 uv = v_uv;
    vec4 color = u_color * texture(u_texture, uv);

    // Apply degamma to the albedo texture
    color.rgb = srgbToLinear(color.rgb);

    vec3 roughness_metallic_ao = texture(u_roughness_metallic_texture, uv).xyz;

    float ao = roughness_metallic_ao.x;
    float metalness = roughness_metallic_ao.y * u_metallic_factor;
    float roughness = roughness_metallic_ao.z * u_roughness_factor;

    if (color.a < u_alpha_cutoff)
        discard;

    vec3 N = normalize(v_normal);
    vec3 normal_pixel = texture(u_normal_texture, uv).xyz;
    vec3 perturbed_normal = perturbNormal(N, v_world_position, uv, normal_pixel);

    FragColor = vec4(color.rgb, metalness);

    // If in the irradiance scene, use the normal N.
    if (u_irradiance_scene_active)
        NormalColor = vec4(N * 0.5 + vec3(0.5), roughness);
    else
        NormalColor = vec4(perturbed_normal * 0.5 + vec3(0.5), roughness);

    // Emissive
    vec3 emissive = u_emissive_factor;
    if (u_emissive_texture_enabled) {
        vec3 emissive_tex = texture(u_emissive_texture, uv).rgb;
        // Apply degamma to the emissive texture
        emissive_tex = srgbToLinear(emissive_tex);
        emissive *= emissive_tex;
    }
    EmissiveColor = vec4(emissive, ao);  // Store emissive color
}


\multi.fs

#version 330 core

in vec3 v_position;
in vec3 v_world_position;
in vec3 v_normal;
in vec2 v_uv;

uniform vec4 u_color;
uniform sampler2D u_texture;
uniform float u_time;
uniform float u_alpha_cutoff;

layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 NormalColor;

void main()
{
	vec2 uv = v_uv;
	vec4 color = u_color;
	color *= texture( u_texture, uv );

	if(color.a < u_alpha_cutoff)
		discard;

	vec3 N = normalize(v_normal);

	FragColor = color;
	NormalColor = vec4(N,1.0);
}


\depth.fs

#version 330 core

uniform vec2 u_camera_nearfar;
uniform sampler2D u_texture; //depth map
in vec2 v_uv;
out vec4 FragColor;

void main()
{
	float n = u_camera_nearfar.x;
	float f = u_camera_nearfar.y;
	float z = texture2D(u_texture,v_uv).x;
	if( n == 0.0 && f == 1.0 )
		FragColor = vec4(z);
	else
		FragColor = vec4( n * (z + 1.0) / (f + n - z * (f - n)) );
}


\instanced.vs

#version 330 core

in vec3 a_vertex;
in vec3 a_normal;
in vec2 a_coord;

in mat4 u_model;

uniform vec3 u_camera_pos;

uniform mat4 u_viewprojection;

//this will store the color for the pixel shader
out vec3 v_position;
out vec3 v_world_position;
out vec3 v_normal;
out vec2 v_uv;

void main()
{	
	//calcule the normal in camera space (the NormalMatrix is like ViewMatrix but without traslation)
	v_normal = (u_model * vec4( a_normal, 0.0) ).xyz;
	
	//calcule the vertex in object space
	v_position = a_vertex;
	v_world_position = (u_model * vec4( a_vertex, 1.0) ).xyz;
	
	//store the texture coordinates
	v_uv = a_coord;

	//calcule the position of the vertex using the matrices
	gl_Position = u_viewprojection * vec4( v_world_position, 1.0 );
}