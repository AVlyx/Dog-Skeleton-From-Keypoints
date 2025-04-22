import numpy as np
import numpy.linalg as linalg
import trimesh
import trimesh.exchange
import optimizer

DOG_SKELETON_FILE: str = "dog_skeleton.stl"


def _make_bone(j1, j2, radius=10, sections=16):
    p1 = j1 * np.array([1, -1, 1])
    p2 = j2 * np.array([1, -1, 1])

    norm = linalg.norm(p2 - p1)
    look_to = trimesh.geometry.align_vectors([0, 0, 1], (p2 - p1) / norm)
    spawn_at = trimesh.transformations.translation_matrix((p1 + p2) / 2.0)

    bone = trimesh.creation.cylinder(radius=radius, height=norm, sections=sections)
    bone.apply_transform(look_to)
    bone.apply_transform(spawn_at)
    return bone


def _make_skull(j1, j2):
    p1 = j1 * np.array([1, -1, 1])
    p2 = j2 * np.array([1, -1, 1])

    norm = linalg.norm(p2 - p1)
    look_to = trimesh.geometry.align_vectors([0, 0, 1], (p2 - p1) / norm)
    spawn_at = trimesh.transformations.translation_matrix((p1 + p2) / 2.0)
    ellipsoid = trimesh.creation.icosphere(subdivisions=6)
    ellipsoid.apply_scale([norm / 2, norm / 2, norm])
    ellipsoid.apply_transform(look_to)
    ellipsoid.apply_transform(spawn_at)
    return ellipsoid


def export_skeleton_mesh(d2_keypoints):
    bones = None
    for progress, bone_res in optimizer.get_bones(d2_keypoints):
        if bone_res is None:
            yield progress, bone_res
        else:
            bones = bone_res
    assert bones is not None
    mesh_bones = [_make_bone(j1, j2) for j1, j2 in bones]
    nose, head = bones[0]
    mesh_bones.append(_make_skull(nose, head))
    skeleton_mesh: trimesh.Trimesh = trimesh.util.concatenate(mesh_bones)
    stl_string = trimesh.exchange.stl.export_stl_ascii(skeleton_mesh)  # type: ignore
    yield 100, stl_string
