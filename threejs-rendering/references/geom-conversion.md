# MuJoCo geom → Three.js mesh

This is adapted from `rendering/GeomBuilder.ts`. It works with `@mujoco/mujoco` and `mujoco-js` (enum values are `{value}` objects).

## Type table

| `geom_type` (value) | Three.js | Notes |
|---|---|---|
| `mjGEOM_PLANE` (0) | `PlaneGeometry(2*sx \|\| 5, 2*sy \|\| 5)` | Size 0 means infinite, so pick a finite size. The Franka demo uses a `Reflector` for the floor. |
| `mjGEOM_HFIELD` (1) | Build from `hfield_data` (`hfield_nrow` × `hfield_ncol`, size in `hfield_size`) | Not handled in the Franka demo |
| `mjGEOM_SPHERE` (2) | `SphereGeometry(r)` | `size[0]` = radius |
| `mjGEOM_CAPSULE` (3) | `CapsuleGeometry(r, 2*halfLen)` then **`rotateX(π/2)`** | `size = [r, halfLen, -]`. Three.js capsules are Y-aligned. |
| `mjGEOM_ELLIPSOID` (4) | Unit `SphereGeometry(1)` scaled by `(sx, sy, sz)` | |
| `mjGEOM_CYLINDER` (5) | `CylinderGeometry(r, r, 2*halfLen)` then **`rotateX(π/2)`** | |
| `mjGEOM_BOX` (6) | `BoxGeometry(2*sx, 2*sy, 2*sz)` | **Half-extents** |
| `mjGEOM_MESH` (7) | `BufferGeometry` from `mesh_vert` / `mesh_face` | Cache by `geom_dataid`, since many geoms share meshes |

- **Visibility:** skip `geom_group === 3`. That is the Menagerie convention for collision-only geoms.
  Optionally hide group 0/1/2 to match MuJoCo's viewer toggles.
- **Colour:** use `mat_rgba[4*matId..]` if `geom_matid >= 0`, otherwise `geom_rgba[4*g..]`. Set `transparent` when alpha < 1.
- **Local pose:** `geom_pos[3g..]`, and `geom_quat[4g..]` is **wxyz**, so use `quaternion.set(q[1], q[2], q[3], q[0])`.
  Mesh vertices are stored in the mesh's own frame, and `geom_pos`/`geom_quat` already account for the re-centring
  MuJoCo does at compile time.
- **Tag** each mesh with `userData.bodyID = geom_bodyid[g]` for picking and for filtering ER raycasts.
- `mjvGeom.size` from `mjv_updateScene` is laid out differently. A cylinder is `[r, r, halfLen]` there,
  while `model.geom_size` is `[r, halfLen, -]`.

## Factory

```ts
import * as THREE from 'three';

const val = (e: any) => e?.value ?? e;                          // enum normaliser for both bindings
const meshCache = new Map<number, THREE.BufferGeometry>();

export function geomToObject(mujoco: any, m: any, g: number): THREE.Object3D | null {
  if (m.geom_group[g] === 3) return null;
  const T = mujoco.mjtGeom, type = m.geom_type[g];
  const s = m.geom_size.subarray(3 * g, 3 * g + 3);
  let geo: THREE.BufferGeometry | null = null;

  if (type === val(T.mjGEOM_PLANE)) geo = new THREE.PlaneGeometry(2 * s[0] || 5, 2 * s[1] || 5);
  else if (type === val(T.mjGEOM_SPHERE)) geo = new THREE.SphereGeometry(s[0], 24, 16);
  else if (type === val(T.mjGEOM_ELLIPSOID)) geo = new THREE.SphereGeometry(1, 24, 16).scale(s[0], s[1], s[2]);
  else if (type === val(T.mjGEOM_BOX)) geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
  else if (type === val(T.mjGEOM_CYLINDER)) geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 24).rotateX(Math.PI / 2);
  else if (type === val(T.mjGEOM_CAPSULE)) geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 16).rotateX(Math.PI / 2);
  else if (type === val(T.mjGEOM_MESH)) {
    const id = m.geom_dataid[g];
    geo = meshCache.get(id) ?? null;
    if (!geo) {
      const va = m.mesh_vertadr[id], vn = m.mesh_vertnum[id], fa = m.mesh_faceadr[id], fn = m.mesh_facenum[id];
      geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(m.mesh_vert.slice(3 * va, 3 * (va + vn)), 3)); // slice = copy
      geo.setIndex(Array.from(m.mesh_face.slice(3 * fa, 3 * (fa + fn))));
      geo.computeVertexNormals();
      meshCache.set(id, geo);
    }
  }
  if (!geo) return null;

  const matId = m.geom_matid[g];
  const rgba = matId >= 0 ? m.mat_rgba.subarray(4 * matId, 4 * matId + 4) : m.geom_rgba.subarray(4 * g, 4 * g + 4);
  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
    color: new THREE.Color(rgba[0], rgba[1], rgba[2]), opacity: rgba[3], transparent: rgba[3] < 1,
    roughness: 0.6, metalness: 0.2 }));
  mesh.castShadow = mesh.receiveShadow = true;
  const p = m.geom_pos.subarray(3 * g, 3 * g + 3), q = m.geom_quat.subarray(4 * g, 4 * g + 4);
  mesh.position.set(p[0], p[1], p[2]);
  mesh.quaternion.set(q[1], q[2], q[3], q[0]);
  mesh.userData.bodyID = m.geom_bodyid[g];
  return mesh;
}

export function buildBodies(mujoco: any, m: any, root: THREE.Group): THREE.Group[] {
  const bodies = Array.from({ length: m.nbody }, (_, i) => {
    const grp = new THREE.Group(); grp.userData.bodyID = i; root.add(grp); return grp;
  });
  for (let g = 0; g < m.ngeom; g++) { const o = geomToObject(mujoco, m, g); if (o) bodies[m.geom_bodyid[g]].add(o); }
  return bodies;
}
```

Notes:

- `three` ≥ 0.15x ships `CapsuleGeometry`. The Franka demo includes its own `CapsuleGeometry.ts` for the older import-map version.
- Flat-shaded robot meshes look faceted with `computeVertexNormals()` on STL soups. Merge vertices first
  (`BufferGeometryUtils.mergeVertices`) or accept the look.
- Textures: `mat_texid` and `tex_*` arrays hold texture data. The Franka demo ignores textures. Most Menagerie arms use plain materials.
- On model reload, dispose every geometry and material, and clear the cache.
