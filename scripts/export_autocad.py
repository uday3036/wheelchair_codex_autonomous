"""Export the existing folded OBJ to millimetre 3DFACE DXF (requires ezdxf)."""
from pathlib import Path
import hashlib
import json
from collections import Counter
import numpy as np
import ezdxf

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'assets/meshes/assemblies/wheelchair_folded.obj'
    material = source.with_suffix('.mtl')
    output = ROOT / 'assets/cad'
    output.mkdir(exist_ok=True)
    colors = {}
    for line in material.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == 'newmtl':
            name = fields[1]
        elif fields[0] == 'Kd':
            rgb = [round(float(v) * 255) for v in fields[1:4]]
            colors[name] = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]
    doc = ezdxf.new('R2010')
    doc.units = ezdxf.units.MM
    doc.header['$MEASUREMENT'] = 1
    space = doc.modelspace()
    vertices, expected = [], []
    expected_components = Counter()
    layer, color = '0', 0xAAAAAA
    for line in source.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == 'o':
            layer = fields[1]
            if layer not in doc.layers:
                doc.layers.new(layer)
        elif fields[0] == 'usemtl':
            color = colors[fields[1]]
            doc.layers.get(layer).dxf.true_color = color
        elif fields[0] == 'v':
            vertices.append(tuple(float(v) * 1000 for v in fields[1:4]))
        elif fields[0] == 'f':
            ids = [int(v.split('/')[0]) for v in fields[1:]]
            polygon = [vertices[i-1 if i > 0 else i] for i in ids]
            for i in range(1, len(polygon)-1):
                triangle = [polygon[0], polygon[i], polygon[i+1]]
                space.add_3dface(triangle, dxfattribs={'layer': layer, 'true_color': color})
                expected.append(triangle)
                expected_components[layer] += 1
    destination = output / 'wheelchair_folded_mm.dxf'
    doc.set_modelspace_vport(height=2200, center=(0, 0))
    doc.saveas(destination)
    # Independently reload the serialized drawing and check all face coordinates.
    restored = ezdxf.readfile(destination)
    audit = restored.audit()
    assert not audit.errors and not audit.fixes, 'DXF audit reported issues'
    faces = list(restored.modelspace().query('3DFACE'))
    actual = np.array([[list(f.dxf.vtx0), list(f.dxf.vtx1), list(f.dxf.vtx2)] for f in faces])
    assert actual.shape == np.asarray(expected).shape
    assert np.allclose(actual, expected, rtol=0, atol=1e-9)
    assert restored.units == ezdxf.units.MM
    actual_components = Counter(f.dxf.layer for f in faces)
    assert actual_components == expected_components, 'Missing component geometry'
    manifest = json.loads((ROOT / 'assets/manifest.json').read_text())
    visible_components = {entry['name'] for entry in manifest['meshes'] if entry['visual']}
    assert set(actual_components) == visible_components, 'CAD does not contain every visible model component'
    assert not list(restored.modelspace().query('INSERT')), 'Unexpected external/block dependency'
    points = actual.reshape(-1, 3)
    report = {
        'source': str(source.relative_to(ROOT)),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'dxf_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
        'units': 'millimetres', 'pose': 'folded', 'frame': 'base_link',
        'triangles': len(faces), 'component_layers': len({f.dxf.layer for f in faces}),
        'bounds_mm': [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        'extent_mm': np.ptp(points, axis=0).tolist(),
        'roundtrip_coordinates_passed': True, 'dxf_audit_errors': len(audit.errors),
        'autocad_application_tested': False,
        'all_visible_components_present': True,
        'component_triangle_counts': dict(sorted(actual_components.items())),
        'geometry': 'Explicit world-assembled triangular 3DFACE surfaces; no joints, constraints or external references required',
        'editable_parametric_solids': False,
    }
    (output / 'export_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
