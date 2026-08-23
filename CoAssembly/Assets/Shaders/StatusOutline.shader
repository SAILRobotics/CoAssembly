// Inverted-hull status outline (red/orange/green part-status rim).
//
// Draw the SAME mesh as a child of the part, offset outward along each vertex normal by
// _OutlineWidth, with front faces culled — only the "inside-out" backfaces survive culling,
// which are exactly the ones that poke out past the real mesh's silhouette from the camera's
// point of view. The real part (drawn separately, its own material, normal Cull Back) then
// occludes the interior of this hull via ordinary depth testing, leaving only a rim visible.
//
// Driven entirely via MaterialPropertyBlock per-instance (GearboxCommandReceiver sets
// _OutlineColor and toggles the renderer's `enabled`) — one shared Material works for every
// part, same pattern as every other per-instance color in this project (ToolColorReceiver,
// ToolSpawner's blended box faces/edges).
Shader "Custom/StatusOutline"
{
    Properties
    {
        _OutlineColor ("Outline Color", Color) = (1, 1, 1, 1)
        _OutlineWidth ("Outline Width (object space)", Float) = 0.0015
    }

    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" "Queue" = "Geometry" }

        Pass
        {
            Name "Outline"
            Cull Front
            ZWrite On
            ZTest LEqual

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            // Declared inside UnityPerMaterial so MaterialPropertyBlock overrides work per-instance
            // (mirrors how URP's own Lit/Unlit shaders expose _BaseColor).
            CBUFFER_START(UnityPerMaterial)
                float4 _OutlineColor;
                float  _OutlineWidth;
            CBUFFER_END

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS   : NORMAL;
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
            };

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                float3 posOS = IN.positionOS.xyz + normalize(IN.normalOS) * _OutlineWidth;
                OUT.positionHCS = TransformObjectToHClip(posOS);
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                return _OutlineColor;
            }
            ENDHLSL
        }
    }
}
