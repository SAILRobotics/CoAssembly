using System.Collections.Generic;
using UnityEngine;

public class RobotTransparency : MonoBehaviour
{
    [Range(0f, 1f)]
    public float alpha = 0.3f;
    public bool applyOnStart = true;

    private Renderer[] _renderers;
    private Dictionary<Renderer, Material[]> _originalMaterials = new();

    void Start()
    {
        _renderers = GetComponentsInChildren<Renderer>(includeInactive: true);
        foreach (var r in _renderers)
            _originalMaterials[r] = r.sharedMaterials;

        if (applyOnStart)
            MakeTransparent();
    }

    public void MakeTransparent()
    {
        foreach (var r in _renderers)
        {
            var instances = r.materials; // creates per-instance copies
            foreach (var mat in instances)
                SetTransparent(mat, alpha);
            r.materials = instances;
        }
    }

    public void Restore()
    {
        foreach (var r in _renderers)
        {
            if (_originalMaterials.TryGetValue(r, out var orig))
                r.materials = orig;
        }
    }

    void SetTransparent(Material mat, float a)
    {
        mat.SetFloat("_Surface", 1);
        mat.SetFloat("_ZWrite", 0);
        mat.SetFloat("_SrcBlend", 5);
        mat.SetFloat("_DstBlend", 10);
        mat.SetFloat("_SrcBlendAlpha", 1);
        mat.SetFloat("_DstBlendAlpha", 10);
        mat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
        mat.renderQueue = 3000;

        Color c = mat.HasProperty("_BaseColor") ? mat.GetColor("_BaseColor") : mat.color;
        c.a = a;
        if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", c);
        mat.color = c;
    }
}
