using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Drag renderers into "Parts" and set a per-part alpha (0 = invisible, 1 = opaque).
/// Each part gets its own instance of "Transparent Template" (assign a material already
/// configured as URP Lit / Surface Type = Transparent, e.g. alphaMaterial.mat), tinted
/// with that part's original base color so only opacity changes.
/// Updates live in the Editor (no Play mode needed) via ExecuteAlways.
/// </summary>
[ExecuteAlways]
public class PartAlphaController : MonoBehaviour
{
    [Serializable]
    public class PartEntry
    {
        public Renderer renderer;
        [Range(0f, 1f)] public float alpha = 0f;
    }

    [Tooltip("A material already set to URP Lit, Surface Type = Transparent (e.g. alphaMaterial.mat). " +
             "Each part below gets its own instance of this, tinted with its own original color.")]
    [SerializeField] private Material transparentTemplate;

    [Tooltip("Drag the renderers (or their parent GameObjects) here, then set each one's alpha.")]
    public PartEntry[] parts = new PartEntry[0];

    [Tooltip("When enabled, every child renderer uses Sim Alpha. Disable to return to the per-part alpha settings.")]
    public bool sim = false;

    [Tooltip("Alpha used for all child renderers while Sim is enabled.")]
    [Range(0f, 1f)] public float simAlpha = 0.5f;

    private static readonly int BaseColorID = Shader.PropertyToID("_BaseColor");
    private static readonly int ColorID     = Shader.PropertyToID("_Color");

    // A renderer can have several submesh material slots — keep one instanced
    // material per (renderer, slot index) so every slot gets converted, not just [0].
    private readonly Dictionary<Renderer, Material[]> _instances   = new();
    // The part's real original materials, captured once — restored whenever alpha
    // rounds to fully opaque (see OpaqueThreshold below).
    private readonly Dictionary<Renderer, Material[]> _originals   = new();
    // True (un-premultiplied) original color per slot — cached separately because once
    // alpha hits 0 the premultiplied RGB baked into the material is unrecoverable (0/0).
    private readonly Dictionary<Renderer, Color[]>    _baseColors  = new();
    private readonly HashSet<Renderer> _simRenderers = new();

    // URP's Transparent surface type always has ZWrite off, so even an alpha=1 part
    // never writes depth — it can sort/blend incorrectly next to other transparent
    // (faded) parts and look wrongly see-through. So at/near full opacity we bypass
    // the transparent template entirely and use the part's real material instead,
    // which restores normal depth-tested opaque rendering.
    private const float OpaqueThreshold = 0.999f;

    private void OnEnable()   => ApplyAll();
    private void OnValidate() => ApplyAll();
    private void Start()      => ApplyAll();

    private void ApplyAll()
    {
        if (parts == null || transparentTemplate == null) return;

        var partRenderers = new HashSet<Renderer>();
        foreach (var p in parts)
        {
            if (p?.renderer != null)
                partRenderers.Add(p.renderer);
        }

        if (sim)
        {
            var currentSimRenderers = new HashSet<Renderer>(GetComponentsInChildren<Renderer>(true));
            foreach (var renderer in currentSimRenderers)
                Apply(renderer, simAlpha);

            foreach (var renderer in _simRenderers)
            {
                if (renderer != null && !currentSimRenderers.Contains(renderer) && !partRenderers.Contains(renderer))
                    Restore(renderer);
            }

            _simRenderers.Clear();
            foreach (var renderer in currentSimRenderers)
                _simRenderers.Add(renderer);

            return;
        }

        foreach (var renderer in _simRenderers)
        {
            if (renderer != null && !partRenderers.Contains(renderer))
                Restore(renderer);
        }
        _simRenderers.Clear();

        foreach (var p in parts)
            Apply(p);
    }

    private void Apply(PartEntry p)
    {
        if (p == null || p.renderer == null) return;
        Apply(p.renderer, p.alpha);
    }

    private void Apply(Renderer renderer, float alpha)
    {
        if (renderer == null) return;

        if (!_originals.TryGetValue(renderer, out Material[] originals) || originals == null)
        {
            originals = renderer.sharedMaterials;
            _originals[renderer] = originals;

            var colors = new Color[originals.Length];
            for (int i = 0; i < originals.Length; i++)
                colors[i] = GetColor(originals[i]);
            _baseColors[renderer] = colors;
        }

        if (alpha >= OpaqueThreshold)
        {
            renderer.sharedMaterials = originals;
            return;
        }

        if (!_instances.TryGetValue(renderer, out Material[] mats) || mats == null)
        {
            mats = new Material[originals.Length];
            for (int i = 0; i < originals.Length; i++)
                mats[i] = new Material(transparentTemplate)
                    { name = transparentTemplate.name + "_" + renderer.name + "_" + i };
            _instances[renderer] = mats;
        }

        var bases = _baseColors[renderer];
        for (int i = 0; i < mats.Length; i++)
        {
            Color b = bases[i];
            SetColor(mats[i], new Color(b.r, b.g, b.b, alpha));
        }
        renderer.sharedMaterials = mats;
    }

    private void Restore(Renderer renderer)
    {
        if (renderer == null) return;
        if (_originals.TryGetValue(renderer, out Material[] originals) && originals != null)
            renderer.sharedMaterials = originals;
    }

    private static Color GetColor(Material m)
    {
        if (m == null) return Color.white;
        if (m.HasProperty(BaseColorID)) return m.GetColor(BaseColorID);
        if (m.HasProperty(ColorID))     return m.GetColor(ColorID);
        return Color.white;
    }

    private static void SetColor(Material m, Color c)
    {
        if (m.HasProperty(BaseColorID)) m.SetColor(BaseColorID, c);
        if (m.HasProperty(ColorID))     m.SetColor(ColorID, c);
    }
}
