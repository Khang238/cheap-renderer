"""
Raytracer v9 -- builds on v8 (same physics: Fresnel/Snell/Beer-Lambert,
BVH, multi-light, water caustics, post-FX, camera keyframe paths). This
version changes the SHELL around that renderer, not the renderer itself:

1) FULL ENGLISH TRANSLATION of every comment, docstring, and printed/UI
   string. No behavioural change from this alone.

2) WINDOW/INPUT BACKEND: pygame -> OpenCV (cv2). Mouse-look is removed
   entirely (as requested); camera look is keyboard-only (arrow keys),
   movement is WASD, same as before -- pygame's arrow-key look already
   existed alongside mouse-look in v8, so this mode already "worked",
   it's just now the ONLY way to look around. Because cv2 has no proper
   held-key state (unlike pygame's key.get_pressed()), movement uses a
   short "recency window": a key counts as held if it was seen within
   the last KEY_HOLD_WINDOW seconds. This relies on the OS's keyboard
   auto-repeat while a key is held down and is not as crisp as pygame's
   real key-state polling -- see KEY_HOLD_WINDOW below if it feels off
   for your keyboard's repeat rate. Any UI that used to need the mouse
   (the keyframe options popup, DoF focus-distance scroll, autofocus)
   is now keyboard-driven instead.

3) COMMAND-LINE INTERFACE (argparse): the program can now run headless
   (no window at all) for a single final-image or video render, driven
   by a scene file, with no need to open the interactive window and
   re-build a scene by hand just to toggle a setting. See --help.

4) SCENE IMPORT/EXPORT: a whole scene (geometry, camera, lights,
   spotlights, background/skybox, post-FX, camera keyframe path) can be
   saved to a single portable JSON file and reloaded elsewhere -- image
   assets (textures, skybox) referenced by the scene are embedded as
   base64 in the JSON by default, so a single .json file is enough to
   reproduce the render on a different machine without also having to
   copy loose texture files around.

5) LIVE-PREVIEW PERFORMANCE FIX: the interactive raytrace mode used to
   share the SAME max_bounce (32 in the shipped demo) as final, offline
   renders. That's fine for opaque surfaces (which terminate in ~1
   bounce) but catastrophic the moment the camera is inside a
   transparent volume (glass/water): every ray then chains Fresnel-split
   reflection/refraction bounces up to that cap, each a full BVH
   traversal, which is exactly the "drops FPS when inside something"
   symptom. Interactive rendering now uses its own, much lower
   LIVE_MAX_BOUNCE (configurable) while offline/final renders keep full
   quality via the scene's own max_bounce.

Everything NOT mentioned above (BVH build, Fresnel/Snell/Beer-Lambert
glass, tinted metal reflection, cull_internal_faces, shadow rays through
tinted glass, water caustics photon pre-pass, DoF/chromatic
aberration/lens flare/VHS post-FX, camera keyframe paths) is unchanged
physics/logic from v8, just with comments translated to English.
"""

import argparse
import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import taichi as ti
from PIL import Image

# =============================================================================
# 1) GPU init (Taichi)
# =============================================================================

def _gpu_smoke_test():
    probe = ti.field(ti.f32, shape=(4, 4))

    @ti.kernel
    def _probe_kernel(n: int):
        for py, px in ti.ndrange(4, 4):
            acc = 0.0
            for i in range(n):
                acc += ti.sin(ti.f32(i) + ti.f32(px) + ti.f32(py))
            probe[py, px] = acc

    _probe_kernel(3)
    probe.to_numpy()


def init_taichi():
    for backend in [ti.cuda, ti.cpu]:   # GPU (opengl/vulkan) skipped -- older GPUs (e.g. Haswell iGPU)
        name = str(backend).split('.')[-1]      # don't support the features Taichi needs here.
        try:
            ti.init(arch=backend, default_fp=ti.f32)
            _gpu_smoke_test()
            print(f"Taichi backend: {name}")
            return name
        except Exception as e:
            print(f"{name} failed: {e}")
    raise RuntimeError("Could not start Taichi on any backend")


BACKEND = init_taichi()

vec3 = ti.types.vector(3, ti.f32)
vec4 = ti.types.vector(4, ti.f32)
vec2 = ti.types.vector(2, ti.f32)
mat3 = ti.types.matrix(3, 3, ti.f32)

# =============================================================================
# 2) Camera / rotation convention:
#    - increasing yaw turns right (+X), increasing pitch looks up (+Y)
#    - R = Ry(yaw) * Rx(pitch) * Rz(roll), shared by camera / preview / WASD.
#      roll defaults to 0.0 (camera_matrix(yaw, pitch) still works as before)
#      and is only nonzero when driven by --camera-data (sensor replay) or
#      the ,/./ camera-roll keys in interactive mode.
#    - rotation_matrix adds Rz(roll) to rotate OBJECTS (boxes, image
#      planes) around an arbitrary axis (yaw, pitch, roll).
# =============================================================================

def camera_matrix(yaw, pitch, roll=0.0):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]], dtype=float)
    Rx = np.array([[1, 0, 0],
                   [0, cp, -sp],
                   [0, sp, cp]], dtype=float)
    if roll == 0.0:
        return Ry @ Rx
    cr, sr = math.cos(roll), math.sin(roll)
    Rz = np.array([[cr, -sr, 0],
                   [sr, cr, 0],
                   [0, 0, 1]], dtype=float)
    return Ry @ Rx @ Rz


def rotation_matrix(yaw, pitch, roll):
    """R = Ry(yaw) * Rx(pitch) * Rz(roll) -- used to rotate an OBJECT
    (box/image) around its own center; independent of camera_matrix but
    shares the same axis convention."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=float)
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=float)
    return Ry @ Rx @ Rz


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# =============================================================================
# 3) Resolutions + light-count limits.
#    These are module-level defaults; main() can override PREVIEW_RES /
#    LIVE_RENDER_RES / FINAL_RENDER_RES from CLI flags via set_resolutions()
#    below (kept as globals, same pattern as v8, so the rest of the file
#    doesn't need to thread resolution through every function signature).
# =============================================================================

PREVIEW_RES = (640, 480)         # window size + 2D painter's-algorithm preview
LIVE_RENDER_RES = (128, 96)      # interactive raytrace resolution (upscaled to window)
FINAL_RENDER_RES = (3840, 2160)  # final image (F5), saved to file
MAX_W, MAX_H = FINAL_RENDER_RES

MAX_LIGHTS = 8
MAX_SPOTLIGHTS = 4    # max number of spotlights (cone lights) per scene

WINDOW_NAME = "Raytracer v10 - CV"
KEY_HOLD_WINDOW = 0.20   # seconds -- see the module docstring: a key counts as "held"
                          # if seen within this window, since cv2 has no true held-key state.

# --- Bounce-count caps: separate for INTERACTIVE vs FINAL rendering -----
# See item 5 in the module docstring: interactive rendering used to share
# the scene's (often very high, e.g. 32) max_bounce with final renders,
# which is fine for opaque surfaces but very slow the instant the camera
# sits inside a transparent volume (glass/water), since every ray then
# chains Fresnel-split bounces up to that cap. LIVE_MAX_BOUNCE keeps the
# interactive/live view responsive; final renders still use the scene's
# own (higher-quality) max_bounce.
LIVE_MAX_BOUNCE = 3
DEFAULT_MAX_BOUNCE = 8  # sane default for RayTracer(max_bounce=...) if not overridden

# --- VIDEO (camera keyframe path -> render video, Shift+Enter) ---------
VIDEO_FPS = 29.97                    # frames/second of the exported video
VIDEO_RES = (320, 240)            # exported video resolution
VIDEO_DURATION = 17.0             # seconds -- ONLY used when there is EXACTLY 1 camera
                                  # keyframe (camera stays still, nothing to infer a
                                  # duration from via speed/distance). With >=2 keyframes,
                                  # duration is COMPUTED from distance + speed per keyframe
                                  # (see CameraPath.total_duration).
VIDEO_SAMPLES_PER_FRAME = 8       # raytrace samples per video frame

# --- WATER MOTION IN VIDEO (does not affect stills/live) ---------------
WATER_WAVE_SPEED = 1.0          # multiplier applied to the "time" fed into the water
                                 # ripple function when rendering video -- 0 = water stays
                                 # still like before, 1 = default speed, higher = faster
                                 # "running" ripples. Edit directly.
WATER_ANIMATE_CAUSTICS = False   # whether to recompute the caustic map (the light patterns
                                 # under water) in step with the ripples when rendering video
                                 # (recomputing EVERY frame would be very slow, so it's only
                                 # redone every WATER_CAUSTIC_UPDATE_INTERVAL video-seconds)
WATER_CAUSTIC_UPDATE_INTERVAL = 1.0 / VIDEO_FPS / 2   # video-seconds between caustic recomputes

# --- CAUSTICS (water) ---------------------------------------------------
# The caustic map is precomputed (not per-sample) by "firing" a simulated
# grid of rays from each point Light through the water surface (REAL
# refraction via Snell's law at both the top and bottom faces of the
# water block, using the actual ripple normal _get_water_normal), then
# comparing the density of where rays "converge" (the ripple's surface
# normal bends nearby rays together) against the density with "no
# refraction" (straight line) to get a light/dark multiplier -- this is
# the real physical cause of caustic patterns (unlike the old pre-v8
# version, which only nudged the shadow ray direction slightly and never
# actually produced light/dark regions, so nothing was visible).
MAX_WATER_BLOCKS = 4
CAUSTIC_RES = 256          # caustic grid resolution (per water block, per direction) -- raised
                           # from 192 so the sharper, more-detailed wave normals (see
                           # _get_water_normal/_gerstner_wave) don't get muddied by an
                           # under-resolved photon grid.
CAUSTIC_MARGIN = 0.15      # how far the grid extends past the water block's edge (fraction of size)
CAUSTIC_RECEIVE_RANGE = 60.0  # thickness (world units) above/below the water block that still
                               # receives caustics -- kept wide because the water block can sit
                               # any distance from the floor/ceiling (e.g. water at y=10, floor at
                               # y=-0.5); safe because _sample_caustic() already filters by surface
                               # normal direction.
CAUSTIC_NORMAL_THRESHOLD = 0.5  # |n.y| must exceed this to count as a "floor/ceiling" (a
                                 # horizontal surface) -- excludes VERTICAL surfaces like walls,
                                 # avoiding caustics being wrongly painted onto walls (a bug from
                                 # an earlier version).
CAUSTIC_BLUR_PASSES = 2    # number of 3x3 blur passes when post-processing the caustic map (denoise)
                           # -- lowered from 3: the higher CAUSTIC_RES above means each pass now
                           # blurs a physically SMALLER area, so fewer passes are needed to reach
                           # the same denoise level while keeping the caustic streaks sharper.
CAUSTIC_MAX_MULT = 6.0     # cap on the brightness multiplier (avoids runaway bright "fireflies")
                           # -- raised from 4.0 for punchier, more contrasty caustic streaks;
                           # RayTracer.caustic_strength (below) gives a runtime dial on top of this.

"""
Standard 16:9 Resolutions
 640 x 360 (nHD)
 960 x 540 (qHD)
1280 x 720 (HD / 720p)
1366 x 768 (HD Ready / WXGA)
1600 x 900 (HD+)
1920 x 1080 (Full HD / 1080p)
2560 x 1440 (QHD / 1440p)
3840 x 2160 (4K UHD)
5120 x 2880 (5K)
7680 x 4320 (8K UHD)
"""

# =============================================================================
# 4) Geometry (host-side, numpy) -- Face = 1 quad face of a box/plane.
# =============================================================================

_DEFAULT_UV = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


class Face:
    __slots__ = ("verts", "normal", "color", "roughness", "transparency",
                 "ior", "reflection_k", "uv", "texture_id")

    def __init__(self, verts, color, roughness=0.0, transparency=0.0,
                 ior=1.5, reflection_k=0.0, uv=None, texture_id=-1):
        self.verts = np.array(verts, dtype=float)
        n = np.cross(self.verts[1] - self.verts[0], self.verts[2] - self.verts[0])
        self.normal = normalize(n)
        self.color = np.array(color[:3], dtype=float) / 255.0
        self.roughness = float(roughness)
        self.transparency = float(transparency)
        self.ior = float(ior)
        self.reflection_k = float(reflection_k)
        self.uv = np.array(uv if uv is not None else _DEFAULT_UV, dtype=np.float32)
        self.texture_id = int(texture_id)

    def vertex_key(self, decimals=4):
        return frozenset(tuple(row) for row in np.round(self.verts, decimals))


# Vertex winding order for each box face -- pts[i] by bit-index:
#   0:(-,-,-) 1:(-,-,+) 2:(-,+,-) 3:(-,+,+) 4:(+,-,-) 5:(+,-,+) 6:(+,+,-) 7:(+,+,+)
_FACE_WINDING = ([0, 1, 3, 2],   # -X
                  [6, 7, 5, 4],  # +X
                  [4, 5, 1, 0],  # -Y
                  [2, 3, 7, 6],  # +Y
                  [0, 2, 6, 4],  # -Z
                  [5, 7, 3, 1])  # +Z


def cull_internal_faces(faces):
    """Removes pairs of exactly-coincident, opposite-facing faces between
    two adjacent boxes before building the BVH, avoiding computing
    Fresnel/Beer-Lambert twice at a shared boundary."""
    groups = {}
    for f in faces:
        groups.setdefault(f.vertex_key(), []).append(f)

    kept = []
    for group in groups.values():
        if len(group) == 2:
            fa, fb = group
            coincident_opposite = np.dot(fa.normal, fb.normal) < -0.99
            same_kind = (fa.transparency > 0.0) == (fb.transparency > 0.0)
            if coincident_opposite and same_kind:
                continue
        kept.extend(group)
    return kept


# =============================================================================
# 5) Texture library -- images used by Scene.add_image(); resized to a
#    fixed square size so different images can share one GPU field
#    (N_TEX, TEX_SIZE, TEX_SIZE, 3).
# =============================================================================

class TextureLibrary:
    TEX_SIZE = 512

    def __init__(self):
        self.paths = []
        self.arrays = []

    def get_id(self, path):
        if path in self.paths:
            return self.paths.index(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Texture image not found: '{path}'")

        # Loaded as RGBA (not RGB) so images with transparency work correctly.
        img = Image.open(path).convert('RGBA').resize(
            (self.TEX_SIZE, self.TEX_SIZE), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        self.paths.append(path)
        self.arrays.append(arr)
        print(f"Texture RGBA loaded: {path} -> id {len(self.arrays) - 1}")
        return len(self.arrays) - 1

    def average_color(self, tex_id):
        # Only the RGB channels are used to compute an average color for the 2D preview.
        return self.arrays[tex_id][..., :3].reshape(-1, 3).mean(axis=0)

    def stack(self):
        if not self.arrays:
            return np.zeros((1, 4, 4, 4), dtype=np.float32)
        return np.stack(self.arrays, axis=0)

# =============================================================================
# 6) Scene -- boxes (add_box, rotatable) + image planes (add_image) + water.
#
#    Scene also records every add_box/add_image/add_water/add_cube call as
#    a lightweight "op" (self.ops) purely so the whole scene can be
#    serialized to JSON and rebuilt elsewhere later (see save_scene_file /
#    load_scene_file near the bottom of this file). This has no effect on
#    rendering -- it's just a construction log.
# =============================================================================

class Scene:
    def __init__(self):
        self.faces = []
        self.boxes = []    # (8 verts in BIT-INDEX order, color 0-255) -- for the 2D preview
        self.quads = []    # (4 verts, averaged 0-255 color) -- for image-plane preview
        self.textures = TextureLibrary()
        self.water_blocks = []  # [(bmin np.float32[3], bmax np.float32[3], ior), ...] -- see add_water()
        self.ops = []       # construction log, for scene export (see module docstring)

    def add_cube(self, pts, color, roughness=0.0, transparency=0.0,
                 ior=1.5, reflection_k=0.0):
        """pts: 8 vertices in the exact bit-index order described at _FACE_WINDING."""
        pts_arr = np.array(pts, dtype=float)
        for fi in _FACE_WINDING:
            self.faces.append(Face([pts_arr[i] for i in fi], color,
                                    roughness, transparency, ior, reflection_k))
        self.boxes.append((pts_arr, tuple(int(c) for c in color[:3])))

    def add_box(self, center, size, color, roughness=0.0, transparency=0.0,
                ior=1.5, reflection_k=0.0, rotation=(0.0, 0.0, 0.0)):
        """Box given by center+size, optionally rotated about its center via
        rotation=(yaw,pitch,roll) (radians). Rotation is fully compatible
        with the BVH/raytracer (see the note at the top of this file)."""
        self.ops.append({'method': 'add_box', 'kwargs': {
            'center': list(map(float, center)), 'size': list(map(float, size)),
            'color': list(int(c) for c in color[:3]), 'roughness': float(roughness),
            'transparency': float(transparency), 'ior': float(ior),
            'reflection_k': float(reflection_k), 'rotation': list(map(float, rotation)),
        }})
        self._add_box_geometry_only(center, size, color, roughness, transparency,
                                     ior, reflection_k, rotation)

    def add_image(self, center, size, image_path, rotation=(0.0, 0.0, 0.0),
                  roughness=0.0, transparency=0.0, ior=1.5, reflection_k=0.0,
                  tint=(255, 255, 255)):
        """Adds a single 2D image plane as its own object. size=(w,h) in
        world units. rotation=(yaw,pitch,roll) radians about its center.
        The material parameters (roughness/transparency/ior/reflection_k)
        apply exactly like a box face -- e.g. you can make the image
        "frosted glass" or slightly reflective via transparency/reflection_k > 0."""
        self.ops.append({'method': 'add_image', 'kwargs': {
            'center': list(map(float, center)), 'size': list(map(float, size)),
            'image_path': image_path, 'rotation': list(map(float, rotation)),
            'roughness': float(roughness), 'transparency': float(transparency),
            'ior': float(ior), 'reflection_k': float(reflection_k),
            'tint': list(int(c) for c in tint[:3]),
        }})
        w, h = size
        hw, hh = w / 2.0, h / 2.0
        local = np.array([(-hw, -hh, 0.0), (hw, -hh, 0.0),
                           (hw, hh, 0.0), (-hw, hh, 0.0)], dtype=float)
        Rm = rotation_matrix(*rotation)
        world = local @ Rm.T + np.array(center, dtype=float)

        tex_id = self.textures.get_id(image_path)
        uv = ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0))
        f = Face(world, tint, roughness, transparency, ior, reflection_k,
                 uv=uv, texture_id=tex_id)
        self.faces.append(f)

        avg = self.textures.average_color(tex_id)
        avg_col = tuple(int(np.clip(c, 0, 1) * 255) for c in avg)
        self.quads.append((world, avg_col))
        return f

    def add_water(self, center, size, color=(160, 230, 255), transparency=0.92, ior=1.333, rotation=(0.0, 0.0, 0.0)):
        """Adds a water block with water's standard index of refraction
        (IOR = 1.333). A light tint color combined with Beer-Lambert makes
        deeper water look progressively darker/bluer."""
        self.ops.append({'method': 'add_water', 'kwargs': {
            'center': list(map(float, center)), 'size': list(map(float, size)),
            'color': list(int(c) for c in color[:3]), 'transparency': float(transparency),
            'ior': float(ior), 'rotation': list(map(float, rotation)),
        }})
        self._add_box_geometry_only(
            center=center, size=size, color=color,
            roughness=0.0,        # left at 0.0 since ripples are handled procedurally by the wave function
            transparency=transparency,
            ior=ior,              # standard IOR of water
            reflection_k=0.0,
            rotation=rotation,
        )
        # Register this water block's AABB for the CAUSTICS pass (only
        # supports NON-rotated water blocks -- if rotated, it still
        # renders/refracts normally, but is skipped by the caustic photon
        # pass below since an axis-aligned AABB no longer represents the
        # shape correctly).
        yaw, pitch, roll = rotation
        if abs(yaw) < 1e-9 and abs(pitch) < 1e-9 and abs(roll) < 1e-9:
            cx, cy, cz = center
            sx, sy, sz = size
            bmin = np.array([cx - sx / 2.0, cy - sy / 2.0, cz - sz / 2.0], dtype=np.float32)
            bmax = np.array([cx + sx / 2.0, cy + sy / 2.0, cz + sz / 2.0], dtype=np.float32)
            self.water_blocks.append((bmin, bmax, float(ior)))

    def _add_box_geometry_only(self, center, size, color, roughness, transparency,
                                ior, reflection_k, rotation):
        """Same geometry as add_box() but WITHOUT logging an 'add_box' op
        (used internally by add_water/add_image so scene export doesn't
        double-log geometry under the wrong op type)."""
        cx, cy, cz = center
        sx, sy, sz = size
        hx, hy, hz = sx / 2, sy / 2, sz / 2
        local = np.array([
            (-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, -hz), (-hx, hy, hz),
            (hx, -hy, -hz), (hx, -hy, hz), (hx, hy, -hz), (hx, hy, hz),
        ], dtype=float)
        yaw, pitch, roll = rotation
        if yaw or pitch or roll:
            Rm = rotation_matrix(yaw, pitch, roll)
            local = local @ Rm.T
        pts = local + np.array([cx, cy, cz], dtype=float)
        self.add_cube(pts, color, roughness, transparency, ior, reflection_k)

    def culled_faces(self):
        return cull_internal_faces(self.faces)

    # --- Serialization ----------------------------------------------------
    def to_dict(self, assets=None):
        """Returns {'ops': [...]}. If `assets` (a dict) is passed, any
        image_path referenced by an add_image op that exists on disk is
        base64-embedded into it (see save_scene_file for the full format)."""
        ops = [dict(op) for op in self.ops]
        if assets is not None:
            for op in ops:
                if op['method'] == 'add_image':
                    path = op['kwargs'].get('image_path')
                    _embed_asset(assets, path)
        return {'ops': ops}

    @staticmethod
    def from_dict(d, asset_dir=None):
        scene = Scene()
        for op in d.get('ops', []):
            method = getattr(scene, op['method'], None)
            if method is None:
                print(f"Warning: unknown scene op '{op['method']}' -- skipped.")
                continue
            kwargs = dict(op.get('kwargs', {}))
            if op['method'] == 'add_image' and asset_dir is not None:
                kwargs['image_path'] = _resolve_asset_path(kwargs.get('image_path'), asset_dir)
            method(**kwargs)
        return scene


def triangulate(faces):
    """Each quad Face (v0,v1,v2,v3) -> 2 triangles, carrying UV + texture_id."""
    v0s, v1s, v2s = [], [], []
    normals, colors = [], []
    rough, transp, iors, refl = [], [], [], []
    uv0s, uv1s, uv2s, texids = [], [], [], []
    for f in faces:
        v = f.verts
        uv = f.uv
        tri_defs = ((v[0], v[1], v[2], uv[0], uv[1], uv[2]),
                    (v[0], v[2], v[3], uv[0], uv[2], uv[3]))
        for (p0, p1, p2, t0, t1, t2) in tri_defs:
            v0s.append(p0); v1s.append(p1); v2s.append(p2)
            normals.append(f.normal); colors.append(f.color)
            rough.append(f.roughness); transp.append(f.transparency)
            iors.append(f.ior); refl.append(f.reflection_k)
            uv0s.append(t0); uv1s.append(t1); uv2s.append(t2)
            texids.append(f.texture_id)
    return dict(
        v0=np.array(v0s, dtype=np.float32), v1=np.array(v1s, dtype=np.float32),
        v2=np.array(v2s, dtype=np.float32), normal=np.array(normals, dtype=np.float32),
        color=np.array(colors, dtype=np.float32), roughness=np.array(rough, dtype=np.float32),
        transparency=np.array(transp, dtype=np.float32), ior=np.array(iors, dtype=np.float32),
        reflection_k=np.array(refl, dtype=np.float32),
        uv0=np.array(uv0s, dtype=np.float32), uv1=np.array(uv1s, dtype=np.float32),
        uv2=np.array(uv2s, dtype=np.float32), texture_id=np.array(texids, dtype=np.int32),
    )


# =============================================================================
# 7) BVH build (CPU, numpy) -- UNCHANGED from v8. Operates on triangles so
#    it's already compatible with rotated boxes/planes (just a bounding box).
# =============================================================================

LEAF_SIZE = 4
MAX_STACK = 64


class BVH:
    def __init__(self, tri):
        self.tri = tri
        n_tri = len(tri['v0'])
        if n_tri == 0:
            raise ValueError("Empty scene -- need at least 1 face to render")

        tmin = np.minimum(np.minimum(tri['v0'], tri['v1']), tri['v2'])
        tmax = np.maximum(np.maximum(tri['v0'], tri['v1']), tri['v2'])
        centroid = (tri['v0'] + tri['v1'] + tri['v2']) / 3.0

        nodes = []

        def build(indices):
            idx = np.asarray(indices)
            bmin = tmin[idx].min(axis=0)
            bmax = tmax[idx].max(axis=0)
            node_id = len(nodes)
            nodes.append({'min': bmin, 'max': bmax, 'left': -1, 'right': -1,
                          'start': idx, 'count': 0})
            if len(idx) <= LEAF_SIZE:
                nodes[node_id]['count'] = len(idx)
                return node_id
            extent = bmax - bmin
            axis = int(np.argmax(extent))
            order = idx[np.argsort(centroid[idx, axis])]
            mid = len(order) // 2
            if mid == 0 or mid == len(order):
                nodes[node_id]['count'] = len(idx)
                nodes[node_id]['start'] = idx
                return node_id
            left = build(order[:mid])
            right = build(order[mid:])
            nodes[node_id]['left'] = left
            nodes[node_id]['right'] = right
            nodes[node_id]['start'] = -1
            return node_id

        self.root = build(np.arange(n_tri))

        tri_order = []
        for nd in nodes:
            if nd['count'] > 0:
                start = len(tri_order)
                tri_order.extend(list(nd['start']))
                nd['start'] = start

        self.n_nodes = len(nodes)
        self.node_min = np.array([nd['min'] for nd in nodes], dtype=np.float32)
        self.node_max = np.array([nd['max'] for nd in nodes], dtype=np.float32)
        self.node_left = np.array([nd['left'] for nd in nodes], dtype=np.int32)
        self.node_right = np.array([nd['right'] for nd in nodes], dtype=np.int32)
        self.node_start = np.array([nd['start'] for nd in nodes], dtype=np.int32)
        self.node_count = np.array([nd['count'] for nd in nodes], dtype=np.int32)
        self.tri_order = np.array(tri_order, dtype=np.int32)
        self.n_tri = n_tri
        print(f"BVH: {n_tri} triangles, {self.n_nodes} nodes, depth~{math.log2(max(n_tri,1))*1.3:.0f}")

# =============================================================================
# 8) Background + Light (multiple lights, each with its own color + brightness)
# =============================================================================

class Background:
    def __init__(self, color=(20, 20, 30), image_path=None, brightness=1.0):
        self.brightness = brightness
        self.image = None
        self.img_h = self.img_w = 1
        self.image_path = image_path
        self.solid = np.array(color[:3], dtype=np.float32) / 255.0
        self.color = tuple(int(c) for c in color[:3])
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path).convert('RGB')
            self.image = np.array(img, dtype=np.float32) / 255.0
            self.img_h, self.img_w = self.image.shape[:2]
            print(f"Background image loaded: {image_path} ({self.img_w}x{self.img_h})")
        elif image_path:
            print(f"Warning: '{image_path}' not found, using solid color instead")

    @property
    def has_image(self):
        return self.image is not None

    def to_dict(self, assets=None):
        d = {'color': list(self.color), 'brightness': float(self.brightness),
             'image_path': self.image_path}
        if assets is not None and self.image_path:
            _embed_asset(assets, self.image_path)
        return d

    @staticmethod
    def from_dict(d, asset_dir=None):
        path = d.get('image_path')
        if path and asset_dir is not None:
            path = _resolve_asset_path(path, asset_dir)
        return Background(color=tuple(d.get('color', (20, 20, 30))),
                           image_path=path, brightness=float(d.get('brightness', 1.0)))


class Light:
    """Point light: position + color (0-255) + brightness (multiplies
    diffuse/specular). Max MAX_LIGHTS lights per scene."""

    def __init__(self, position, color=(255, 255, 255), brightness=1.0):
        self.position = np.array(position, dtype=np.float32)
        self.color = np.array(color[:3], dtype=np.float32) / 255.0
        self.brightness = float(brightness)

    def to_dict(self):
        return {'position': [float(x) for x in self.position],
                'color': [int(round(c * 255)) for c in self.color],
                'brightness': self.brightness}

    @staticmethod
    def from_dict(d):
        return Light(tuple(d['position']), tuple(d.get('color', (255, 255, 255))),
                      float(d.get('brightness', 1.0)))


class SpotLight:
    """Cone (spot) light: like Light (position + color + brightness) but
    only shines within a cone in one direction instead of all directions.
    Max MAX_SPOTLIGHTS per scene.

    - position:    tip of the cone (world space)
    - direction:   direction the cone points toward (will be normalized)
    - color:       light color (0-255)
    - brightness:  intensity, multiplies diffuse/specular like Light
    - cone_angle:  half-angle of the cone (degrees); fully dark outside this angle
    - softness:    0..1, how soft the cone edge is (0 = hard edge, near 1 =
                   soft falloff from center outward)
    """

    def __init__(self, position, direction, color=(255, 255, 255), brightness=2.0,
                 cone_angle=25.0, softness=0.35):
        self.position = np.array(position, dtype=np.float32)
        self.direction = normalize(np.array(direction, dtype=float)).astype(np.float32)
        self.color = np.array(color[:3], dtype=np.float32) / 255.0
        self.brightness = float(brightness)
        self.cone_angle = float(cone_angle)
        self.softness = float(min(0.97, max(0.0, softness)))

    @property
    def cos_outer(self):
        return math.cos(math.radians(self.cone_angle))

    @property
    def cos_inner(self):
        inner = self.cone_angle * (1.0 - self.softness)
        return math.cos(math.radians(inner))

    def aim_at(self, target_pos):
        self.direction = normalize(np.array(target_pos, dtype=float) - self.position).astype(np.float32)

    def to_dict(self):
        return {'position': [float(x) for x in self.position],
                'direction': [float(x) for x in self.direction],
                'color': [int(round(c * 255)) for c in self.color],
                'brightness': self.brightness, 'cone_angle': self.cone_angle,
                'softness': self.softness}

    @staticmethod
    def from_dict(d):
        return SpotLight(tuple(d['position']), tuple(d['direction']),
                          tuple(d.get('color', (255, 255, 255))),
                          float(d.get('brightness', 2.0)), float(d.get('cone_angle', 25.0)),
                          float(d.get('softness', 0.35)))


# =============================================================================
# 9) GPU fields (global)
# =============================================================================

F_V0 = F_V1 = F_V2 = F_NORMAL = None
F_COLOR = F_ROUGH = F_TRANSP = F_IOR = F_REFL = None
F_UV0 = F_UV1 = F_UV2 = F_TEXID = None
TEX_FIELD = None
N_MIN = N_MAX = N_LEFT = N_RIGHT = N_START = N_COUNT = None
BG_FIELD = None
LIGHT_POS = LIGHT_COLOR = LIGHT_BRIGHTNESS = LIGHT_VIS = None
SPOT_POS = SPOT_DIR = SPOT_COLOR = SPOT_BRIGHTNESS = None
SPOT_COS_OUTER = SPOT_COS_INNER = SPOT_VIS = None
PROBE_DEPTH = None
WB_MIN = WB_MAX = WB_IOR = N_WATER_FIELD = None
WATER_TIME = None                           # time (seconds) used to make the water ripple
                                             # "run" when rendering video -- 0.0 (default) =
                                             # ripples stay still, same as a still image/live view.
CAUSTIC_DOWN = CAUSTIC_UP = None            # histogram: REAL refracted rays (with ripples)
CAUSTIC_DOWN_BASE = CAUSTIC_UP_BASE = None  # histogram: STRAIGHT rays (no refraction) -- baseline
CAUSTIC_DOWN_MULT = CAUSTIC_UP_MULT = None  # blurred/clamped ratio, SAMPLED at render time
ACCUM = DEPTH_ACCUM = SAMPLE_COUNT = OUTPUT = DEPTH_OUT = None
ROOT_NODE = 0
N_TRI = 0


def upload_scene_geometry(bvh: BVH, background: Background, texture_lib: TextureLibrary):
    """Uploads geometry/BVH/textures/background -- called ONCE, INDEPENDENT
    of resolution (the image buffers are allocated separately in
    _alloc_buffers)."""
    global F_V0, F_V1, F_V2, F_NORMAL, F_COLOR, F_ROUGH, F_TRANSP, F_IOR, F_REFL
    global F_UV0, F_UV1, F_UV2, F_TEXID, TEX_FIELD
    global N_MIN, N_MAX, N_LEFT, N_RIGHT, N_START, N_COUNT
    global BG_FIELD, ROOT_NODE, N_TRI

    tri = bvh.tri
    order = bvh.tri_order
    n = bvh.n_tri

    F_V0 = ti.Vector.field(3, ti.f32, shape=n); F_V0.from_numpy(tri['v0'][order])
    F_V1 = ti.Vector.field(3, ti.f32, shape=n); F_V1.from_numpy(tri['v1'][order])
    F_V2 = ti.Vector.field(3, ti.f32, shape=n); F_V2.from_numpy(tri['v2'][order])
    F_NORMAL = ti.Vector.field(3, ti.f32, shape=n); F_NORMAL.from_numpy(tri['normal'][order])
    F_COLOR = ti.Vector.field(3, ti.f32, shape=n); F_COLOR.from_numpy(tri['color'][order])
    F_ROUGH = ti.field(ti.f32, shape=n); F_ROUGH.from_numpy(tri['roughness'][order])
    F_TRANSP = ti.field(ti.f32, shape=n); F_TRANSP.from_numpy(tri['transparency'][order])
    F_IOR = ti.field(ti.f32, shape=n); F_IOR.from_numpy(tri['ior'][order])
    F_REFL = ti.field(ti.f32, shape=n); F_REFL.from_numpy(tri['reflection_k'][order])
    F_UV0 = ti.Vector.field(2, ti.f32, shape=n); F_UV0.from_numpy(tri['uv0'][order])
    F_UV1 = ti.Vector.field(2, ti.f32, shape=n); F_UV1.from_numpy(tri['uv1'][order])
    F_UV2 = ti.Vector.field(2, ti.f32, shape=n); F_UV2.from_numpy(tri['uv2'][order])
    F_TEXID = ti.field(ti.i32, shape=n); F_TEXID.from_numpy(tri['texture_id'][order])

    N = bvh.n_nodes
    N_MIN = ti.Vector.field(3, ti.f32, shape=N); N_MIN.from_numpy(bvh.node_min)
    N_MAX = ti.Vector.field(3, ti.f32, shape=N); N_MAX.from_numpy(bvh.node_max)
    N_LEFT = ti.field(ti.i32, shape=N); N_LEFT.from_numpy(bvh.node_left)
    N_RIGHT = ti.field(ti.i32, shape=N); N_RIGHT.from_numpy(bvh.node_right)
    N_START = ti.field(ti.i32, shape=N); N_START.from_numpy(bvh.node_start)
    N_COUNT = ti.field(ti.i32, shape=N); N_COUNT.from_numpy(bvh.node_count)
    ROOT_NODE = bvh.root
    N_TRI = n

    bg_h = max(background.img_h, 1); bg_w = max(background.img_w, 1)
    BG_FIELD = ti.Vector.field(3, ti.f32, shape=(bg_h, bg_w))
    if background.has_image:
        BG_FIELD.from_numpy(background.image)
    else:
        BG_FIELD.from_numpy(np.tile(background.solid, (bg_h, bg_w, 1)))

    tex_stack = texture_lib.stack() if texture_lib is not None else np.zeros((1, 4, 4, 3), dtype=np.float32)
    TEX_FIELD = ti.Vector.field(4, ti.f32, shape=tex_stack.shape[:3])
    TEX_FIELD.from_numpy(tex_stack)


def alloc_light_fields():
    global LIGHT_POS, LIGHT_COLOR, LIGHT_BRIGHTNESS, LIGHT_VIS
    global SPOT_POS, SPOT_DIR, SPOT_COLOR, SPOT_BRIGHTNESS
    global SPOT_COS_OUTER, SPOT_COS_INNER, SPOT_VIS, PROBE_DEPTH
    LIGHT_POS = ti.Vector.field(3, ti.f32, shape=MAX_LIGHTS)
    LIGHT_COLOR = ti.Vector.field(3, ti.f32, shape=MAX_LIGHTS)
    LIGHT_BRIGHTNESS = ti.field(ti.f32, shape=MAX_LIGHTS)
    LIGHT_VIS = ti.field(ti.i32, shape=MAX_LIGHTS)

    SPOT_POS = ti.Vector.field(3, ti.f32, shape=MAX_SPOTLIGHTS)
    SPOT_DIR = ti.Vector.field(3, ti.f32, shape=MAX_SPOTLIGHTS)
    SPOT_COLOR = ti.Vector.field(3, ti.f32, shape=MAX_SPOTLIGHTS)
    SPOT_BRIGHTNESS = ti.field(ti.f32, shape=MAX_SPOTLIGHTS)
    SPOT_COS_OUTER = ti.field(ti.f32, shape=MAX_SPOTLIGHTS)
    SPOT_COS_INNER = ti.field(ti.f32, shape=MAX_SPOTLIGHTS)
    SPOT_VIS = ti.field(ti.i32, shape=MAX_SPOTLIGHTS)
    PROBE_DEPTH = ti.field(ti.f32, shape=())


def alloc_water_fields():
    """Allocates fields for water block AABBs + the caustic map (called once
    after Scene has finished calling add_water, independent of image resolution)."""
    global WB_MIN, WB_MAX, WB_IOR, N_WATER_FIELD
    global CAUSTIC_DOWN, CAUSTIC_UP, CAUSTIC_DOWN_BASE, CAUSTIC_UP_BASE
    global CAUSTIC_DOWN_MULT, CAUSTIC_UP_MULT, WATER_TIME
    WB_MIN = ti.Vector.field(3, ti.f32, shape=MAX_WATER_BLOCKS)
    WB_MAX = ti.Vector.field(3, ti.f32, shape=MAX_WATER_BLOCKS)
    WB_IOR = ti.field(ti.f32, shape=MAX_WATER_BLOCKS)
    N_WATER_FIELD = ti.field(ti.i32, shape=())
    WATER_TIME = ti.field(ti.f32, shape=())
    WATER_TIME[None] = 0.0

    shape3 = (MAX_WATER_BLOCKS, CAUSTIC_RES, CAUSTIC_RES)
    CAUSTIC_DOWN = ti.field(ti.f32, shape=shape3)
    CAUSTIC_UP = ti.field(ti.f32, shape=shape3)
    CAUSTIC_DOWN_BASE = ti.field(ti.f32, shape=shape3)
    CAUSTIC_UP_BASE = ti.field(ti.f32, shape=shape3)
    CAUSTIC_DOWN_MULT = ti.field(ti.f32, shape=shape3)
    CAUSTIC_UP_MULT = ti.field(ti.f32, shape=shape3)
    CAUSTIC_DOWN_MULT.fill(1.0)
    CAUSTIC_UP_MULT.fill(1.0)

# =============================================================================
# 10) Taichi funcs -- BVH traversal, ray-triangle intersection (also returns
#     (u,v) barycentric coords to interpolate UV for texturing), texture
#     sampling, Fresnel/Snell, shading.
# =============================================================================

@ti.func
def _ray_aabb_hit(ro: vec3, inv_d: vec3, bmin: vec3, bmax: vec3, tmax: ti.f32) -> ti.i32:
    t1 = (bmin - ro) * inv_d
    t2 = (bmax - ro) * inv_d
    tlo = ti.min(t1, t2)
    thi = ti.max(t1, t2)
    tmin = ti.max(ti.max(tlo[0], tlo[1]), tlo[2])
    tmaxv = ti.min(ti.min(thi[0], thi[1]), thi[2])
    return 1 if (tmaxv >= tmin and tmaxv >= 0.0 and tmin <= tmax) else 0


@ti.func
def _ray_tri_hit(ro: vec3, rd: vec3, tid: ti.i32, tmax: ti.f32):
    # Moller-Trumbore -- also returns (u,v) barycentric coords to interpolate UV for texturing.
    v0 = F_V0[tid]; v1 = F_V1[tid]; v2 = F_V2[tid]
    e1 = v1 - v0; e2 = v2 - v0
    pvec = rd.cross(e2)
    det = e1.dot(pvec)
    hit = 0
    t = tmax
    uu = 0.0
    vv = 0.0
    if ti.abs(det) > 1e-9:
        inv_det = 1.0 / det
        tvec = ro - v0
        u = tvec.dot(pvec) * inv_det
        if -1e-6 <= u <= 1.0 + 1e-6:
            qvec = tvec.cross(e1)
            v = rd.dot(qvec) * inv_det
            if v >= -1e-6 and (u + v) <= 1.0 + 1e-6:
                tt = e2.dot(qvec) * inv_det
                if 1e-4 < tt < tmax:
                    hit = 1
                    t = tt
                    uu = u
                    vv = v
    return hit, t, uu, vv


@ti.func
def _bvh_closest_hit(ro: vec3, rd: vec3, tmax_in: ti.f32):
    """Returns (tri_id, t, u, v) of the closest hit, tri_id=-1 if no hit."""
    inv_d = vec3(
        1.0 / rd[0] if ti.abs(rd[0]) > 1e-12 else 1e12 * (1.0 if rd[0] >= 0 else -1.0),
        1.0 / rd[1] if ti.abs(rd[1]) > 1e-12 else 1e12 * (1.0 if rd[1] >= 0 else -1.0),
        1.0 / rd[2] if ti.abs(rd[2]) > 1e-12 else 1e12 * (1.0 if rd[2] >= 0 else -1.0),
    )
    stack = ti.Vector.zero(ti.i32, MAX_STACK)
    sp = 1
    stack[0] = ROOT_NODE
    best_t = tmax_in
    best_tri = -1
    best_u = 0.0
    best_v = 0.0
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if _ray_aabb_hit(ro, inv_d, N_MIN[node], N_MAX[node], best_t) == 1:
            cnt = N_COUNT[node]
            if cnt > 0:
                start = N_START[node]
                for i in range(cnt):
                    tid = start + i
                    h, t, u, v = _ray_tri_hit(ro, rd, tid, best_t)
                    if h == 1 and t < best_t:
                        best_t = t
                        best_tri = tid
                        best_u = u
                        best_v = v
            else:
                if sp < MAX_STACK - 2:
                    stack[sp] = N_LEFT[node]; sp += 1
                    stack[sp] = N_RIGHT[node]; sp += 1
    return best_tri, best_t, best_u, best_v


@ti.func
def _sample_background(direction: vec3) -> vec3:
    h = BG_FIELD.shape[0]; w = BG_FIELD.shape[1]
    dx = direction[0]; dy = direction[1]; dz = direction[2]
    u = (ti.atan2(dx, dz) / (2.0 * math.pi)) % 1.0
    v = 1.0 - (ti.asin(ti.max(-1.0, ti.min(1.0, dy))) / math.pi + 0.5)
    iy = int(ti.max(0.0, ti.min(float(h - 1), v * (h - 1))))
    ix = int(ti.max(0.0, ti.min(float(w - 1), u * (w - 1))))
    px = BG_FIELD[iy, ix]
    return vec3(px[0], px[1], px[2])


@ti.func
def _sample_texture(tex_id: ti.i32, u: ti.f32, v: ti.f32) -> ti.types.vector(4, ti.f32):
    uu = u - ti.floor(u)
    vv = v - ti.floor(v)
    th = TEX_FIELD.shape[1]; tw = TEX_FIELD.shape[2]
    ix = int(ti.min(tw - 1.0, uu * tw))
    iy = int(ti.min(th - 1.0, vv * th))
    px = TEX_FIELD[tex_id, iy, ix]
    return vec4(px[0], px[1], px[2], px[3])  # Returns RGBA


@ti.func
def _gerstner_wave(x: ti.f32, z: ti.f32, t: ti.f32, dirx: ti.f32, dirz: ti.f32,
                    freq: ti.f32, amp: ti.f32, speed: ti.f32, steep: ti.f32):
    """One Gerstner (trochoidal) wave: returns (dHeight/dx, dHeight/dz, x_disp, z_disp)
    -- the height-field slope (used to build the surface normal, same as before)
    PLUS the wave's own horizontal displacement (steep waves pull the surface
    points sideways toward the crest, which is what makes real ocean/lake
    ripples look "peaked" instead of a smooth sine-blend). steep in [0,1]
    scales how much horizontal pull is applied relative to the wave height."""
    w = freq
    phase = w * (dirx * x + dirz * z) + t * speed
    s = ti.sin(phase)
    c = ti.cos(phase)
    dhdx = dirx * w * amp * c
    dhdz = dirz * w * amp * c
    qa = steep * amp
    disp_x = qa * dirx * s
    disp_z = qa * dirz * s
    return dhdx, dhdz, disp_x, disp_z


@ti.func
def _get_water_normal(p: vec3, base_n: vec3) -> vec3:
    # Only ripples horizontal surfaces (the water's top/bottom face)
    res = base_n
    if ti.abs(base_n[1]) > 0.5:
        x, z = p[0], p[2]
        t = WATER_TIME[None]  # 0.0 by default (still image/live) -- video feeds a value > 0

        # A small bit of low-frequency domain warping BEFORE the wave sum so the
        # pattern doesn't look like a perfectly regular grid of crossed sine
        # waves -- real chop is never that tidy. Cheap (2 extra sin/cos) and
        # only nudges sample position slightly.
        warp_x = x + 0.6 * ti.sin(z * 0.07 + t * 0.11)
        warp_z = z + 0.6 * ti.cos(x * 0.065 - t * 0.09)

        # Sum of several Gerstner waves at different frequencies, amplitudes,
        # directions and speeds (directions are intentionally NOT axis-aligned
        # or simple multiples of each other -- avoids the obviously-periodic
        # "checkerboard chop" look a small number of orthogonal sine waves
        # produces). Amplitude falls off roughly as frequency rises (like a
        # real wave spectrum: big slow swells + small fast capillary ripples
        # riding on top of them).
        dhdx = 0.0
        dhdz = 0.0
        dispx = 0.0
        dispz = 0.0

        d0x, d0z, r0x, r0z = _gerstner_wave(warp_x, warp_z, t,  0.80,  0.60, 0.35, 0.22, 0.55, 0.35)
        d1x, d1z, r1x, r1z = _gerstner_wave(warp_x, warp_z, t, -0.35,  0.94, 0.55, 0.14, 0.80, 0.30)
        d2x, d2z, r2x, r2z = _gerstner_wave(warp_x, warp_z, t,  0.94,  0.34, 0.95, 0.075, 1.15, 0.28)
        d3x, d3z, r3x, r3z = _gerstner_wave(warp_x, warp_z, t, -0.60, -0.80, 1.7, 0.04, 1.6, 0.22)
        d4x, d4z, r4x, r4z = _gerstner_wave(warp_x, warp_z, t,  0.20, -0.98, 3.1, 0.018, 2.3, 0.18)
        d5x, d5z, r5x, r5z = _gerstner_wave(warp_x, warp_z, t, -0.86,  0.51, 5.4, 0.008, 3.1, 0.12)

        dhdx = d0x + d1x + d2x + d3x + d4x + d5x
        dhdz = d0z + d1z + d2z + d3z + d4z + d5z

        # Tiny high-frequency jitter (not a traveling wave, just noise-like
        # texture) so dead-flat patches between the low-frequency swells
        # still show some sparkle/micro-ripple, like real water never being
        # perfectly glassy.
        jit_x = 0.006 * ti.sin(warp_x * 9.7 + warp_z * 6.1 + t * 4.3)
        jit_z = 0.006 * ti.cos(warp_z * 8.9 - warp_x * 7.3 + t * 3.7)

        wave_n = vec3(-(dhdx + jit_x), base_n[1], -(dhdz + jit_z)).normalized()
        res = wave_n
    return res


@ti.func
def _perturb_in_cone(n: vec3, roughness: ti.f32) -> vec3:
    res = n
    if roughness > 1e-4:
        up = vec3(0.0, 1.0, 0.0)
        if ti.abs(n.dot(up)) > 0.9:
            up = vec3(1.0, 0.0, 0.0)
        tangent = n.cross(up).normalized()
        bitangent = n.cross(tangent)
        r1 = (ti.random(ti.f32) * 2.0 - 1.0) * roughness
        r2 = (ti.random(ti.f32) * 2.0 - 1.0) * roughness
        res = (n + tangent * r1 + bitangent * r2).normalized()
    return res


@ti.func
def _fresnel_schlick(cos_i: ti.f32, ior: ti.f32) -> ti.f32:
    r0 = ((ior - 1.0) / (ior + 1.0)) ** 2
    return r0 + (1.0 - r0) * (ti.max(0.0, 1.0 - cos_i) ** 5)


@ti.func
def _beer_lambert_tint(color: vec3, transparency: ti.f32, distance: ti.f32) -> vec3:
    ar = -ti.log(ti.max(color[0], 1e-4)) * transparency
    ag = -ti.log(ti.max(color[1], 1e-4)) * transparency
    ab = -ti.log(ti.max(color[2], 1e-4)) * transparency
    return vec3(ti.exp(-ar * distance), ti.exp(-ag * distance), ti.exp(-ab * distance))


@ti.func
def _refract(travel_dir: vec3, n: vec3, eta: ti.f32):
    """Standard Snell refraction: travel_dir = the ray's current direction
    (unit), n = surface normal pointing AGAINST the incoming ray (i.e.
    n.dot(travel_dir) < 0), eta = n1/n2 (ratio of the refractive index of
    the medium before / after the surface). Returns (refracted direction,
    tir) -- tir=1 on total internal reflection (TIR), in which case the
    returned direction is the mirror-REFLECTED direction (used as-is;
    the refracted vector is meaningless in that case)."""
    cosi = ti.max(1e-6, travel_dir.dot(-n))
    sin2t = eta * eta * (1.0 - cosi * cosi)
    tir = 0
    out_dir = (travel_dir - 2.0 * travel_dir.dot(n) * n).normalized()
    if sin2t <= 1.0:
        cost = ti.sqrt(ti.max(0.0, 1.0 - sin2t))
        out_dir = (eta * travel_dir + (eta * cosi - cost) * n).normalized()
    else:
        tir = 1
    return out_dir, tir


@ti.func
def _bilerp_caustic(f: ti.template(), wi: ti.i32, fx: ti.f32, fz: ti.f32) -> ti.f32:
    cx = ti.max(0.0, ti.min(float(CAUSTIC_RES - 1), fx))
    cz = ti.max(0.0, ti.min(float(CAUSTIC_RES - 1), fz))
    x0 = int(cx); z0 = int(cz)
    x1 = ti.min(x0 + 1, CAUSTIC_RES - 1)
    z1 = ti.min(z0 + 1, CAUSTIC_RES - 1)
    tx = cx - x0; tz = cz - z0
    a = f[wi, z0, x0]; b = f[wi, z0, x1]
    c = f[wi, z1, x0]; d = f[wi, z1, x1]
    return a * (1.0 - tx) * (1.0 - tz) + b * tx * (1.0 - tz) + c * (1.0 - tx) * tz + d * tx * tz


@ti.func
def _sample_caustic(p: vec3, n: vec3) -> ti.f32:
    """Caustics light/dark multiplier at point p (1.0 = unchanged). Only
    applies to surfaces that are just below (receiving light from above)
    or just above (receiving light from below) a water block, within
    CAUSTIC_RECEIVE_RANGE, AND only when the surface normal points
    up/down enough -- without that normal check, a VERTICAL surface (e.g.
    a wall) that happens to fall within the water block's X,Z extent at
    roughly the right height would be wrongly treated as a "floor" and
    get caustics painted onto it (this was the actual cause of a wall
    artifact bug in an earlier version). The CAUSTIC_*_MULT maps are
    precomputed in compute_caustics_kernel(); here we only bilinearly
    interpolate them."""
    mult = 1.0
    for wi in range(N_WATER_FIELD[None]):
        bmin = WB_MIN[wi]
        bmax = WB_MAX[wi]
        sx = bmax[0] - bmin[0]
        sz = bmax[2] - bmin[2]
        mx0 = bmin[0] - sx * CAUSTIC_MARGIN
        mx1 = bmax[0] + sx * CAUSTIC_MARGIN
        mz0 = bmin[2] - sz * CAUSTIC_MARGIN
        mz1 = bmax[2] + sz * CAUSTIC_MARGIN
        if mx0 <= p[0] <= mx1 and mz0 <= p[2] <= mz1:
            # NOTE (fixed): the write side (compute_caustics_kernel /
            # _deposit_caustic_sample) places a sample at grid cell CENTER
            # (i+0.5)/CAUSTIC_RES (fxi = int(fu*CAUSTIC_RES)) -- i.e. a
            # "cell-center" convention. This read side used to sample with
            # *(CAUSTIC_RES-1)* (a "corner-to-corner" convention), which
            # mismatched the write side and caused the caustic pattern to
            # skew/drift toward the water block's edges (worse further
            # from center). Fixed to use the correct cell-center
            # convention: (u,v)=(0,0) maps to the first cell's center
            # (index 0), (u,v)=(1,1) maps to the last cell's center
            # (index CAUSTIC_RES-1) -> fx = u*CAUSTIC_RES-0.5.
            fx = (p[0] - mx0) / (mx1 - mx0) * CAUSTIC_RES - 0.5
            fz = (p[2] - mz0) / (mz1 - mz0) * CAUSTIC_RES - 0.5
            # surface facing UP (floor, table...) + below the water block -> receives light from above
            if n[1] > CAUSTIC_NORMAL_THRESHOLD and p[1] < bmin[1] and p[1] > bmin[1] - CAUSTIC_RECEIVE_RANGE:
                mult *= _bilerp_caustic(CAUSTIC_DOWN_MULT, wi, fx, fz)
            # surface facing DOWN (ceiling, underside of an object...) + above the water block -> receives light from below
            if n[1] < -CAUSTIC_NORMAL_THRESHOLD and p[1] > bmax[1] and p[1] < bmax[1] + CAUSTIC_RECEIVE_RANGE:
                mult *= _bilerp_caustic(CAUSTIC_UP_MULT, wi, fx, fz)
    return mult


MAX_SHADOW_STEPS = 8


@ti.func
def _shadow_throughput(p_from: vec3, light_pos: vec3) -> vec3:
    to_light = light_pos - p_from
    light_dist = to_light.norm()
    ldir = to_light / ti.max(light_dist, 1e-6)
    throughput = vec3(1.0)
    ro = p_from
    travelled = 0.0

    for _ in range(MAX_SHADOW_STEPS):
        # Also grab u, v from the BVH hit point
        tid, t, bu, bv = _bvh_closest_hit(ro, ldir, light_dist - travelled - 1e-3)
        if tid < 0:
            break

        transp = F_TRANSP[tid]
        texid = F_TEXID[tid]

        # --- CHECK THE TEXTURE'S ALPHA ---
        is_transparent_png = False
        if texid >= 0:
            # Interpolate UV coordinates at the hit point
            uv0 = F_UV0[tid]; uv1 = F_UV1[tid]; uv2 = F_UV2[tid]
            uvp = uv0 * (1.0 - bu - bv) + uv1 * bu + uv2 * bv

            # Sample the RGBA color at that point
            tex_rgba = _sample_texture(texid, uvp[0], uvp[1])

            # Alpha < 0.1 means a transparent region
            if tex_rgba[3] < 0.1:
                is_transparent_png = True

        # A transparent PNG region -> SKIP IT, DOES NOT COUNT AS AN OCCLUDER!
        if is_transparent_png:
            step = t + 1e-3
            ro = ro + ldir * step
            travelled += step
            if travelled >= light_dist:
                break
            continue  # jump to the next iteration to look for another occluder behind it

        # --- HANDLE A NORMAL OCCLUDER (glass or an opaque object) ---
        if transp <= 0.0:
            throughput = vec3(0.0)  # fully blocked -> a black shadow
            break

        col = F_COLOR[tid]

        # NOTE: an earlier version here had a hack that "bent" the shadow
        # ray along the ripple normal (ldir += 0.35*(wn-gn)) to fake
        # caustics. That hack never produced real light/dark patterns: it
        # only nudged the ray direction slightly (by an arbitrary amount,
        # not following Snell's law), while STILL using the original total
        # path length (light_dist) to decide when to stop -- meaning it
        # never even checked whether the bent ray still pointed at the
        # light, so in practice it had almost no effect on the final image
        # (as the user observed: no visible caustics). More importantly,
        # even with a properly Snell-corrected bend, bending a single
        # shadow ray CANNOT produce a converging bright region (caustics
        # are physically the result of many nearby rays being refracted
        # together into one point) -- that requires comparing the DENSITY
        # of rays between nearby points, not a single ray -- see
        # compute_caustics_kernel() + _sample_caustic() below, which is
        # where caustics are ACTUALLY computed (a photon-density map,
        # precomputed once per light position, not per shadow ray). So
        # this function only keeps Beer-Lambert (the water's color/opacity)
        # for the shadow ray, no more direction bending.
        throughput *= _beer_lambert_tint(col, transp, t)

        step = t + 1e-3
        ro = ro + ldir * step
        travelled += step
        if travelled >= light_dist or throughput.max() < 1e-3:
            break

    return throughput


@ti.kernel
def compute_light_visibility(cam_x: ti.f32, cam_y: ti.f32, cam_z: ti.f32, n_lights: ti.i32):
    """Checks whether each light is occluded from the CAMERA's view (used for flare)."""
    cam_pos = vec3(cam_x, cam_y, cam_z)
    for i in range(n_lights):
        lpos = LIGHT_POS[i]
        d = lpos - cam_pos
        dist = d.norm()
        dirn = d / ti.max(dist, 1e-6)
        tid, t, _, _ = _bvh_closest_hit(cam_pos, dirn, dist - 1e-3)
        LIGHT_VIS[i] = 1 if tid < 0 else 0


@ti.kernel
def compute_spot_visibility(cam_x: ti.f32, cam_y: ti.f32, cam_z: ti.f32, n_spots: ti.i32):
    """Same as compute_light_visibility but for spotlights (used for their
    flare too -- this only checks BVH occlusion; whether the spot is
    "facing" the camera is filtered separately on the CPU (cone angle)."""
    cam_pos = vec3(cam_x, cam_y, cam_z)
    for i in range(n_spots):
        spos = SPOT_POS[i]
        d = spos - cam_pos
        dist = d.norm()
        dirn = d / ti.max(dist, 1e-6)
        tid, t, _, _ = _bvh_closest_hit(cam_pos, dirn, dist - 1e-3)
        SPOT_VIS[i] = 1 if tid < 0 else 0


@ti.kernel
def probe_depth(cam_x: ti.f32, cam_y: ti.f32, cam_z: ti.f32,
                 dx: ti.f32, dy: ti.f32, dz: ti.f32):
    """Fires a single ray from the camera in direction (dx,dy,dz), writes
    the closest hit distance into PROBE_DEPTH -- used for autofocus."""
    ray_o = vec3(cam_x, cam_y, cam_z)
    ray_dir = vec3(dx, dy, dz).normalized()
    tid, t, _, _ = _bvh_closest_hit(ray_o, ray_dir, 1e18)
    PROBE_DEPTH[None] = t if tid >= 0 else 1.0e4


@ti.func
def _deposit_caustic_sample(q: vec3, din: vec3, lpos: vec3, gn: vec3, above: ti.i32,
                             entry_y: ti.f32, exit_y: ti.f32,
                             mx0: ti.f32, mx1: ti.f32, mz0: ti.f32, mz1: ti.f32,
                             wior: ti.f32, wi: ti.i32):
    """Shared logic for one photon sample (called for both Light and
    SpotLight): computes the baseline (no refraction) + the REAL
    refraction through both water surfaces, then accumulates into
    CAUSTIC_DOWN/UP (+BASE). See compute_caustics_kernel() for what each
    parameter means."""
    # --- baseline: where would a straight (unrefracted) ray land on the opposite face? ---
    if ti.abs(din[1]) > 1e-5:
        s_flat = (exit_y - lpos[1]) / din[1]
        if s_flat > 0.0:
            flat_land = lpos + din * s_flat
            fu = (flat_land[0] - mx0) / (mx1 - mx0)
            fv = (flat_land[2] - mz0) / (mz1 - mz0)
            if 0.0 <= fu < 1.0 and 0.0 <= fv < 1.0:
                fxi = int(fu * CAUSTIC_RES)
                fzi = int(fv * CAUSTIC_RES)
                if above:
                    CAUSTIC_DOWN_BASE[wi, fzi, fxi] += 1.0
                else:
                    CAUSTIC_UP_BASE[wi, fzi, fxi] += 1.0

    # --- REAL refraction through both water surfaces (entry + exit), each with its own ripple normal ---
    wn1 = _get_water_normal(q, gn)
    inside_dir, tir1 = _refract(din, wn1, 1.0 / wior)   # entering water: eta = n_air/n_water
    if tir1 == 0 and ti.abs(inside_dir[1]) > 1e-5:
        t_in = (exit_y - entry_y) / inside_dir[1]
        if t_in > 0.0:
            q2 = q + inside_dir * t_in
            wn2 = _get_water_normal(q2, gn)
            out_dir, tir2 = _refract(inside_dir, wn2, wior)  # leaving water: eta = n_water/n_air
            if tir2 == 0:
                # TIR here = light is internally reflected and never escapes
                # -> no point is added (this IS the physical reason dark
                # gaps appear between the real caustic streaks)
                lu = (q2[0] - mx0) / (mx1 - mx0)
                lv = (q2[2] - mz0) / (mz1 - mz0)
                if 0.0 <= lu < 1.0 and 0.0 <= lv < 1.0:
                    lxi = int(lu * CAUSTIC_RES)
                    lzi = int(lv * CAUSTIC_RES)
                    if above:
                        CAUSTIC_DOWN[wi, lzi, lxi] += 1.0
                    else:
                        CAUSTIC_UP[wi, lzi, lxi] += 1.0


@ti.kernel
def compute_caustics_kernel(n_water: ti.i32, n_lights: ti.i32, n_spots: ti.i32):
    """Caustics photon pre-pass: for each (water block, point Light), scatters
    a CAUSTIC_RES x CAUSTIC_RES grid of points over the water surface FACING
    the light (the top face if the light is above the block, the bottom
    face if below), refracts them for REAL (Snell, twice -- entering then
    leaving the water) using the ripple normal _get_water_normal AT EACH
    point, then accumulates (histograms) where those points "land" on the
    opposite face into CAUSTIC_DOWN/UP. In parallel, it also computes
    where each grid point WOULD land if there were NO refraction (a
    straight line from the light through the grid point) into
    CAUSTIC_*_BASE. The ratio of densities (real / straight) IS the
    caustics brightness: > 1.0 = bright, < 1.0 = dark.

    Handles BOTH Light (omnidirectional) AND SpotLight (cone-limited) --
    two separate loops but accumulating into the SAME CAUSTIC_* fields,
    since physically multiple light sources just add up their energy. For
    SpotLight, a grid point q is only counted if it falls WITHIN the
    spotlight's cone (using SPOT_DIR/SPOT_COS_OUTER, the same condition
    used in render_sample) -- without that filter, a narrow spotlight
    would be treated as a full 360-degree light, giving caustics in the
    wrong position and intensity.

    Simplifying assumption (documented for clarity): the "landing" position
    is computed from the (x,z) coordinates AT the opposite water surface
    (q2), WITHOUT continuing the ray on to the actual receiving surface.
    If that surface (floor, ceiling...) sits RIGHT AT the water surface
    (as in the original demo scene), this is exact. If it's a significant
    distance away (e.g. the water sits much higher than the floor), the
    caustic pattern still shows the correct general shape/contrast but may
    be slightly offset horizontally (since the ray keeps traveling after
    leaving the water) -- see _sample_caustic()'s normal-direction check
    and range (CAUSTIC_RECEIVE_RANGE, CAUSTIC_NORMAL_THRESHOLD).

    IMPORTANT NOTE: this pass does NOT check for occluders between the
    light and the water surface -- if something blocks the path (e.g. a
    floor directly beneath a light that's "under the water block"),
    caustics are still computed as if unobstructed, which can be
    wrong/too bright in enclosed setups. Make sure lights have a clear
    line of sight to the water surface.
    """
    for wi, li, gz, gx in ti.ndrange(n_water, n_lights, CAUSTIC_RES, CAUSTIC_RES):
        bmin = WB_MIN[wi]
        bmax = WB_MAX[wi]
        wior = WB_IOR[wi]
        lpos = LIGHT_POS[li]

        above = lpos[1] > bmax[1]
        below = lpos[1] < bmin[1]

        if above or below:
            sx = bmax[0] - bmin[0]
            sz = bmax[2] - bmin[2]
            mx0 = bmin[0] - sx * CAUSTIC_MARGIN
            mx1 = bmax[0] + sx * CAUSTIC_MARGIN
            mz0 = bmin[2] - sz * CAUSTIC_MARGIN
            mz1 = bmax[2] + sz * CAUSTIC_MARGIN

            entry_y = bmax[1] if above else bmin[1]
            exit_y = bmin[1] if above else bmax[1]
            updir = 1.0 if above else -1.0
            gn = vec3(0.0, updir, 0.0)   # "normal opposing the ray direction" -- see the note in _refract()

            u = (gx + 0.5) / CAUSTIC_RES
            v = (gz + 0.5) / CAUSTIC_RES
            wx = mx0 + u * (mx1 - mx0)
            wz = mz0 + v * (mz1 - mz0)
            q = vec3(wx, entry_y, wz)

            to_q = q - lpos
            dist0 = to_q.norm()
            din = to_q / ti.max(dist0, 1e-6)   # "photon" direction from the light to q

            _deposit_caustic_sample(q, din, lpos, gn, above, entry_y, exit_y,
                                     mx0, mx1, mz0, mz1, wior, wi)

    for wi, si, gz, gx in ti.ndrange(n_water, n_spots, CAUSTIC_RES, CAUSTIC_RES):
        bmin = WB_MIN[wi]
        bmax = WB_MAX[wi]
        wior = WB_IOR[wi]
        spos = SPOT_POS[si]
        sdir = SPOT_DIR[si]
        scos_outer = SPOT_COS_OUTER[si]

        above = spos[1] > bmax[1]
        below = spos[1] < bmin[1]

        if above or below:
            sx = bmax[0] - bmin[0]
            sz = bmax[2] - bmin[2]
            mx0 = bmin[0] - sx * CAUSTIC_MARGIN
            mx1 = bmax[0] + sx * CAUSTIC_MARGIN
            mz0 = bmin[2] - sz * CAUSTIC_MARGIN
            mz1 = bmax[2] + sz * CAUSTIC_MARGIN

            entry_y = bmax[1] if above else bmin[1]
            exit_y = bmin[1] if above else bmax[1]
            updir = 1.0 if above else -1.0
            gn = vec3(0.0, updir, 0.0)

            u = (gx + 0.5) / CAUSTIC_RES
            v = (gz + 0.5) / CAUSTIC_RES
            wx = mx0 + u * (mx1 - mx0)
            wz = mz0 + v * (mz1 - mz0)
            q = vec3(wx, entry_y, wz)

            to_q = q - spos
            dist0 = to_q.norm()
            din = to_q / ti.max(dist0, 1e-6)

            cos_theta = din.dot(sdir)   # angle between the cone's direction and the direction toward q
            if cos_theta > scos_outer:  # q falls within (or near) the spotlight's cone
                _deposit_caustic_sample(q, din, spos, gn, above, entry_y, exit_y,
                                         mx0, mx1, mz0, mz1, wior, wi)


# =============================================================================
# 11) Main render kernel -- loops over N lights, samples textures by UV,
#     accumulates DEPTH (used for DoF) alongside color.
# =============================================================================

@ti.kernel
def render_sample(
    cam_x: ti.f32, cam_y: ti.f32, cam_z: ti.f32,
    R00: ti.f32, R01: ti.f32, R02: ti.f32,
    R10: ti.f32, R11: ti.f32, R12: ti.f32,
    R20: ti.f32, R21: ti.f32, R22: ti.f32,
    half_tan: ti.f32, aspect: ti.f32,
    n_lights: ti.i32, n_spots: ti.i32,
    ambient: ti.f32, specular_k: ti.f32, shininess: ti.f32,
    bg_brightness: ti.f32,
    max_bounce: int, width: int, height: int,
    caustics_on: ti.i32,
    sky_light_strength: ti.f32, caustic_strength: ti.f32,
):
    cam_pos = vec3(cam_x, cam_y, cam_z)
    R = mat3([[R00, R01, R02], [R10, R11, R12], [R20, R21, R22]])

    for py, px in ti.ndrange(height, width):
        jx = ti.random(ti.f32) - 0.5
        jy = ti.random(ti.f32) - 0.5
        xn = ((px + jx) / (width - 1.0)) * 2.0 - 1.0
        yn = 1.0 - ((py + jy) / (height - 1.0)) * 2.0
        d_cam = vec3(xn * half_tan * aspect, yn * half_tan, 1.0)
        ray_dir = (R @ d_cam).normalized()
        ray_o = cam_pos

        throughput = vec3(1.0)
        final_color = vec3(0.0)
        terminated = False

        for bounce in range(max_bounce + 1):
            if terminated:
                continue
            tid, t, bu, bv = _bvh_closest_hit(ray_o, ray_dir, 1e18)

            if bounce == 0:
                DEPTH_ACCUM[py, px] += t if tid >= 0 else 1.0e4

            if tid < 0:
                final_color += throughput * _sample_background(ray_dir) * bg_brightness
                terminated = True
                continue

            p = ray_o + ray_dir * t
            geo_n = F_NORMAL[tid]
            col = F_COLOR[tid]
            texid = F_TEXID[tid]
            tex_alpha = 1.0
            if texid >= 0:
                uv0 = F_UV0[tid]; uv1 = F_UV1[tid]; uv2 = F_UV2[tid]
                uvp = uv0 * (1.0 - bu - bv) + uv1 * bu + uv2 * bv

                tex_color_alpha = _sample_texture(texid, uvp[0], uvp[1])
                col = col * vec3(tex_color_alpha[0], tex_color_alpha[1], tex_color_alpha[2])
                tex_alpha = tex_color_alpha[3]  # the image's alpha value (0.0 -> 1.0)

            if tex_alpha < 0.1:
                # A transparent region of the image! Push the ray origin through this plane.
                ray_o = p + ray_dir * 1e-3
                continue  # move on to the next bounce without stopping the ray here
            rough = F_ROUGH[tid]
            transp = F_TRANSP[tid]
            ior = F_IOR[tid]
            refl_k = F_REFL[tid]

            cos_i_signed = -ray_dir.dot(geo_n)
            entering = cos_i_signed > 0.0
            n = geo_n if entering else -geo_n

            if transp > 0.0:
                # --- Dielectric material (glass): random Fresnel split --- (unchanged from v7/v8)
                if bounce >= max_bounce:
                    final_color += throughput * _sample_background(ray_dir) * bg_brightness
                    terminated = True
                    continue

                ns = n
                if ior > 1.30 and ior < 1.36:
                    ns = _get_water_normal(p, n)
                else:
                    ns = _perturb_in_cone(n, rough)
                cosi = ti.max(1e-6, ray_dir.dot(-ns))
                eta = (1.0 / ior) if entering else ior
                sin2_t = eta * eta * (1.0 - cosi * cosi)

                fresnel = 1.0
                if sin2_t <= 1.0:
                    fresnel = _fresnel_schlick(cosi, ior)
                p_reflect = ti.max(0.05, ti.min(0.95, fresnel))

                if ti.random(ti.f32) < p_reflect:
                    rdir = (ray_dir - 2.0 * ray_dir.dot(ns) * ns).normalized()
                    ray_o = p + n * 1e-4
                    ray_dir = rdir
                    throughput /= p_reflect
                else:
                    cost = ti.sqrt(ti.max(0.0, 1.0 - sin2_t))
                    rdir = (eta * ray_dir + (eta * cosi - cost) * ns).normalized()
                    throughput *= _beer_lambert_tint(col, transp, t)
                    ray_o = p - n * 1e-4
                    ray_dir = rdir
                    throughput /= (1.0 - p_reflect)

                if throughput.max() < 1e-3:
                    terminated = True

            else:
                # --- Opaque material: accumulate lighting from N sources (each
                #     with its own shadow ray + color + brightness) ---
                view_dir = (-ray_dir).normalized()
                # Sky light: instead of a single flat ambient scalar (which made
                # lit-from-the-sky scenes look off -- a bright sky background but
                # a uniformly dim ambient fill, or vice versa), sample the
                # background itself in the surface normal's direction as a cheap
                # 1-tap hemispherical sky contribution. A surface facing up gets
                # tinted/lit by the sky's zenith color, a surface facing sideways
                # gets more of the horizon, and a surface facing down gets
                # whatever's "below" in the background image (ground/floor color
                # in a full panorama, or just the flat bg color otherwise) --
                # so ambient fill stays consistent with whatever's actually
                # behind the camera in that direction, sky or not.
                sky_col = _sample_background(n)
                ambient_term = col * (ambient + sky_light_strength * bg_brightness * sky_col)
                direct = vec3(0.0)
                for li in range(n_lights):
                    lpos = LIGHT_POS[li]
                    lcol = LIGHT_COLOR[li]
                    lbri = LIGHT_BRIGHTNESS[li]
                    light_dir = (lpos - p).normalized()
                    shadow_tp = _shadow_throughput(p + n * 1e-3, lpos)
                    diff = ti.max(0.0, n.dot(light_dir))
                    half_v = (light_dir + view_dir).normalized()
                    spec = specular_k * (ti.max(0.0, n.dot(half_v)) ** shininess)
                    direct += (diff * col + spec) * lcol * lbri * shadow_tp

                for si in range(n_spots):
                    spos = SPOT_POS[si]
                    sdir = SPOT_DIR[si]
                    scol = SPOT_COLOR[si]
                    sbri = SPOT_BRIGHTNESS[si]
                    cos_outer = SPOT_COS_OUTER[si]
                    cos_inner = SPOT_COS_INNER[si]

                    light_dir = (spos - p).normalized()
                    # angle between the cone's direction (sdir) and the direction FROM the light TO point p
                    cos_theta = (-light_dir).dot(sdir)
                    if cos_theta > cos_outer:
                        cone_t = (cos_theta - cos_outer) / ti.max(cos_inner - cos_outer, 1e-4)
                        cone_t = ti.max(0.0, ti.min(1.0, cone_t))
                        cone_atten = cone_t * cone_t * (3.0 - 2.0 * cone_t)  # smoothstep

                        shadow_tp = _shadow_throughput(p + n * 1e-3, spos)
                        diff = ti.max(0.0, n.dot(light_dir))
                        half_v = (light_dir + view_dir).normalized()
                        spec = specular_k * (ti.max(0.0, n.dot(half_v)) ** shininess)
                        direct += (diff * col + spec) * scol * sbri * shadow_tp * cone_atten

                caustic_mult = 1.0
                if caustics_on != 0:
                    raw_mult = _sample_caustic(p, n)
                    # caustic_strength dials the effect's contrast at runtime on
                    # top of the CAUSTIC_MAX_MULT cap: 0 = caustics off (neutral
                    # 1.0 everywhere), 1 = the full computed multiplier, >1 = even
                    # punchier streaks/shadows than the raw photon density gives.
                    caustic_mult = 1.0 + (raw_mult - 1.0) * caustic_strength
                local = ambient_term + direct * caustic_mult
                # NOTE: no longer hard-clamped to 1.0 here (only lower-bounded).
                # Strong highlights (bright specular + a punchy sky/caustic) are
                # allowed to go above "1.0" and are compressed later by the
                # filmic tonemap in resolve_output() instead of being clipped
                # flat here -- this both looks more natural (smooth highlight
                # rolloff instead of a hard-edged white blob) and gives the
                # bloom post-effect actual "how overbright is this" data to
                # work with. A generous soft ceiling still guards against
                # single-sample fireflies blowing up the accumulation buffer.
                local = ti.max(0.0, ti.min(12.0, local))

                if refl_k > 0.0:
                    final_color += throughput * (1.0 - refl_k) * local
                    throughput = throughput * col * refl_k
                    ns = _perturb_in_cone(n, rough)
                    rdir = (ray_dir - 2.0 * ray_dir.dot(ns) * ns).normalized()
                    ray_o = p + n * 1e-3
                    ray_dir = rdir
                    if throughput.max() < 1e-3:
                        terminated = True
                else:
                    final_color += throughput * local
                    terminated = True

        ACCUM[py, px] += ti.max(0.0, ti.min(16.0, final_color))
    SAMPLE_COUNT[None] += 1


@ti.func
def _aces_filmic(x: vec3) -> vec3:
    """Narkowicz's fitted ACES filmic tonemap curve. Compresses highlights
    smoothly toward 1.0 instead of hard-clipping them, and gives midtones a
    gentle filmic contrast S-curve -- this is the renderer-side "adjustment"
    that makes bright specular/sky highlights roll off naturally instead of
    burning out into flat white blobs, and gives the bloom post-effect
    (apply_bloom) a smoother brightness gradient to extract."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    num = x * (a * x + b)
    den = x * (c * x + d) + e
    return vec3(
        ti.max(0.0, ti.min(1.0, num[0] / den[0])),
        ti.max(0.0, ti.min(1.0, num[1] / den[1])),
        ti.max(0.0, ti.min(1.0, num[2] / den[2])),
    )


@ti.kernel
def resolve_output(width: int, height: int, exposure: ti.f32):
    n = ti.max(1, SAMPLE_COUNT[None])
    for py, px in ti.ndrange(height, width):
        hdr = (ACCUM[py, px] / n) * exposure
        OUTPUT[py, px] = _aces_filmic(hdr)
        DEPTH_OUT[py, px] = DEPTH_ACCUM[py, px] / n


@ti.kernel
def reset_accum(width: int, height: int):
    for py, px in ti.ndrange(height, width):
        ACCUM[py, px] = vec3(0.0)
        DEPTH_ACCUM[py, px] = 0.0
    SAMPLE_COUNT[None] = 0

# =============================================================================
# 11b) Camera keyframes / path (P / Ctrl+P / Shift+Enter) -- every camera
#      position "captured" (P) stores position + rotation (yaw, pitch) + a
#      speed value (world units/second, default 1.0) describing how FAST
#      the camera moves away from that keyframe. Motion between 2
#      keyframes is LINEAR (both position and rotation), and the duration
#      of each segment = distance / speed_of_the_starting_keyframe.
# =============================================================================

def _shortest_angle_diff(a, b):
    """Shortest angular difference from a to b (radians, result in [-pi, pi])
    -- avoids the camera spinning "all the way around" when yaw crosses the -pi/pi boundary."""
    d = (b - a) % (2 * math.pi)
    if d > math.pi:
        d -= 2 * math.pi
    return d


class CameraKeyframe:
    __slots__ = ("pos", "yaw", "pitch", "speed", "duration")

    def __init__(self, pos, yaw, pitch, speed=1.0, duration=None):
        self.pos = np.array(pos, dtype=np.float32)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.speed = max(0.01, float(speed))  # world units / second
        # Explicit duration (seconds) for the segment ENDING at this keyframe,
        # overriding the distance/speed calculation below -- the only way to
        # get a non-instant segment when the position doesn't change (a
        # "hold" -- see CameraPath.hold()), and also handy any time you just
        # want "this transition takes exactly N seconds" regardless of how
        # far apart the two keyframes are.
        self.duration = None if duration is None else max(1e-4, float(duration))

    def to_dict(self):
        d = {'pos': [float(x) for x in self.pos], 'yaw': self.yaw,
             'pitch': self.pitch, 'speed': self.speed}
        if self.duration is not None:
            d['duration'] = self.duration
        return d


def _kick(x):
    """Sharp single-lobe "impact" shape used for footstep jolts: fast rise,
    fast decay, peak of 1.0 at x=1/12, ~0 by x=0.5. x is expected in [0, 1)
    (progress through one footfall's cycle)."""
    return 32.6 * x * math.exp(-12.0 * x) if 0.0 <= x < 1.0 else 0.0


def _smoothstep(edge0, edge1, x):
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def _stride_length(speed):
    """Full stride-cycle length (2 footfalls), world units -- grows with
    speed the way a real gait's stride lengthens from a walk into a run."""
    return max(0.9, min(3.1, 0.9 + 0.55 * min(speed, 4.0)))


class CameraPath:
    """List of camera keyframes in creation order (P key appends). Does NOT
    store a "video duration" here -- duration is DERIVED from distance +
    speed (see total_duration/segment_durations), except with only 1
    keyframe (nothing to move between) where VIDEO_DURATION is used. A
    keyframe's `duration` field (see CameraKeyframe/hold()) overrides that
    derivation for its segment."""

    def __init__(self):
        self.keyframes = []
        # --- Fake handheld camera shake/sway (see _handheld_offset) ---
        # 0.0 = perfectly smooth/locked-off (unchanged behaviour). Values
        # around 0.3-0.6 read as "holding a camera by hand"; higher values
        # read as "walking/running with it". Purely additive on top of the
        # normal keyframe interpolation, so it never changes total_duration()
        # or where the path "ends up" -- only how it wobbles along the way.
        self.handheld_shake = 0.0
        self.handheld_seed = 0.0
        # Master multiplier for the FOOTSTEP-driven shake specifically (see
        # _gait_offset) -- separate dial from handheld_shake so you can turn
        # the constant idle micro-jitter up/down without changing how hard
        # footsteps hit, or vice versa. Footstep shake only ever appears
        # when handheld_shake > 0 too (it scales both together) since a
        # "locked off" camera shouldn't develop footstep jolts just because
        # it's moving fast.
        self.footstep_shake = 1.0

    def _handheld_offset(self, t):
        """Deterministic (no RNG -- see note below), multi-frequency
        sway/jitter meant to fake the small involuntary movements of a
        person holding a camera: a slow low-amplitude drift/sway (like
        weight shifting or breathing), plus a faster smaller jitter on top
        (like hand micro-tremor), plus a touch of roll wobble and a subtle
        vertical "step bob". Deterministic in t (not np.random) on purpose:
        render_video samples several sub-frame times per output frame for
        motion blur, and interactive replay scrubs t directly -- both need
        the SAME wobble for the SAME t every time, which a stateful RNG
        can't give without extra bookkeeping. This is the "idle" component
        that's present even standing still (real OIS/handheld footage never
        holds perfectly steady); see _gait_offset for the much sharper,
        speed-driven footstep component layered on top of it while moving."""
        amp = self.handheld_shake
        pos_off = np.zeros(3, dtype=np.float32)
        yaw_off = pitch_off = roll_off = 0.0
        if amp > 1e-6:
            s = self.handheld_seed

            def n(freq, ph):
                return math.sin(t * freq + ph)

            yaw_off = amp * (0.014 * n(1.7, s + 0.3) + 0.006 * n(4.3, s + 1.7) + 0.0025 * n(9.7, s + 2.9))
            pitch_off = amp * (0.011 * n(1.3, s + 1.1) + 0.005 * n(3.9, s + 0.2) + 0.002 * n(8.3, s + 3.7))
            roll_off = amp * (0.018 * n(0.9, s + 2.2) + 0.007 * n(2.6, s + 0.7))
            pos_off[0] = amp * (0.03 * n(1.1, s + 0.5) + 0.01 * n(3.3, s + 2.1))
            # Vertical "step bob" runs at roughly double the horizontal sway
            # frequency (like footsteps), plus a faster micro-tremor.
            pos_off[1] = amp * (0.02 * n(2.2, s + 1.9) + 0.008 * n(3.7, s + 0.4))
            pos_off[2] = amp * (0.012 * n(1.4, s + 2.6))
        return pos_off, yaw_off, pitch_off, roll_off

    def _gait_offset(self, t, speed, phase):
        """Footstep-driven camera shake, layered ON TOP of _handheld_offset's
        constant idle sway. Real handheld/body-worn footage gets a much
        sharper jolt each time a foot hits the ground while moving: a quick
        downward dip + a downward pitch "nod", roll that alternates
        left/right with alternating footfalls, and -- once fast enough to
        blend into a run -- a wider side-to-side arc sway (from arm swing)
        plus a slower roll "drift" that doesn't fully recenter between
        steps (fighting to keep the shot level while running).

        t: absolute path time (seconds) -- only used for the slow run-drift
           terms below, which aren't step-locked.
        speed: the CURRENT segment's speed (world units/second). 0 during a
           hold -- footsteps stop immediately, exactly like standing still.
        phase: cumulative fractional STRIDE count so far (see sample()) --
           phase advancing by 1.0 is one full left+right stride cycle, by
           0.5 is a single footfall.
        """
        intensity = self.handheld_shake * self.footstep_shake
        pos_off = np.zeros(3, dtype=np.float32)
        yaw_off = pitch_off = roll_off = 0.0
        if intensity <= 1e-6 or speed <= 1e-4:
            return pos_off, yaw_off, pitch_off, roll_off

        activity = _smoothstep(0.15, 1.1, speed)   # 0 standing -> 1 once walking-paced
        if activity <= 1e-6:
            return pos_off, yaw_off, pitch_off, roll_off
        run_blend = _smoothstep(1.7, 3.3, speed)   # 0 walking -> 1 once running-paced
        amp = intensity * activity

        step_f = phase * 2.0                        # 1.0 per footfall (2 per stride cycle)
        step_idx = math.floor(step_f)
        within_step = step_f - step_idx              # 0..1 progress since this footfall
        impact = _kick(within_step)
        foot_sign = 1.0 if (int(step_idx) % 2 == 0) else -1.0
        s = self.handheld_seed

        vert = 0.03 * (1.0 + 1.6 * run_blend)
        pos_off[1] = -amp * vert * impact                   # sharp dip at each footfall

        pitch = 0.024 * (1.0 + 1.3 * run_blend)
        pitch_off = -amp * pitch * impact                   # quick downward "nod" at footfall

        roll_step = 0.022 * (1.0 + 1.7 * run_blend)
        roll_off = amp * foot_sign * roll_step * impact      # alternating lean, opposite each foot

        lateral = 0.045 * (1.0 + 2.4 * run_blend)
        pos_off[0] = amp * lateral * math.sin(phase * 2.0 * math.pi)  # arc sway from arm swing

        fwd = 0.014 * (1.0 + 1.2 * run_blend)
        pos_off[2] = -amp * fwd * impact                     # tiny forward jerk on impact

        if run_blend > 1e-4:
            # Slower, not step-locked -- the horizon visibly wanders instead
            # of snapping back to level every step, like actually straining
            # to keep a shot steady while running.
            drift = 0.05 * run_blend * amp
            roll_off += drift * math.sin(t * 0.9 + s * 3.1)
            yaw_off += 0.012 * run_blend * amp * math.sin(t * 0.7 + s * 1.3)

        return pos_off, yaw_off, pitch_off, roll_off

    def add(self, pos, yaw, pitch, speed=1.0, duration=None):
        self.keyframes.append(CameraKeyframe(pos, yaw, pitch, speed, duration))
        return self.keyframes[-1]

    def hold(self, duration=2.0, yaw=None, pitch=None):
        """Convenience for "stay right here for `duration` seconds"
        (optionally turning to a new yaw/pitch during the hold -- e.g.
        panning to look at something while standing still, since only the
        POSITION is held fixed). Just appends a keyframe at the same
        position as the current last keyframe with an explicit duration --
        position doesn't change, so the normal distance/speed timing (which
        would give a 0-length segment) doesn't apply; footstep shake also
        naturally stops during a hold, since its speed is 0 by construction."""
        if not self.keyframes:
            raise ValueError("hold() needs at least one keyframe already in the path (nothing to hold at).")
        last = self.keyframes[-1]
        new_yaw = last.yaw if yaw is None else float(yaw)
        new_pitch = last.pitch if pitch is None else float(pitch)
        return self.add(last.pos.copy(), new_yaw, new_pitch, speed=last.speed, duration=duration)

    def is_camera_data(self):
        return False

    def remove(self, idx):
        if 0 <= idx < len(self.keyframes):
            del self.keyframes[idx]

    def clear(self):
        self.keyframes.clear()

    def segment_durations(self):
        """Duration (seconds) of each segment between 2 consecutive
        keyframes. Normally distance / speed of the STARTING keyframe of
        that segment; if the ENDING keyframe has an explicit `duration` set
        (see CameraKeyframe/hold()), that's used directly instead."""
        durs = []
        for i in range(len(self.keyframes) - 1):
            a, b = self.keyframes[i], self.keyframes[i + 1]
            if b.duration is not None:
                durs.append(b.duration)
            else:
                dist = float(np.linalg.norm(b.pos - a.pos))
                durs.append(dist / a.speed)
        return durs

    def total_duration(self):
        return sum(self.segment_durations())

    def sample(self, t):
        """t (seconds from the start of the path) -> (pos np.float32[3], yaw, pitch, roll)
        LINEARLY interpolated along the path. With only 1 keyframe, always
        returns that keyframe (camera stays still). Returns None if the path
        is empty. Handheld sway (_handheld_offset) and, once actually
        moving, footstep shake (_gait_offset) are both added on top of the
        plain interpolated pose -- see their docstrings."""
        n = len(self.keyframes)
        if n == 0:
            return None
        if n == 1:
            kf = self.keyframes[0]
            pos_off, yaw_off, pitch_off, roll_off = self._handheld_offset(t)
            gpos, gyaw, gpitch, groll = self._gait_offset(t, 0.0, 0.0)
            return (kf.pos.copy() + pos_off + gpos, kf.yaw + yaw_off + gyaw,
                    kf.pitch + pitch_off + gpitch, roll_off + groll)

        durs = self.segment_durations()
        total = sum(durs)
        t_clamped = max(0.0, min(t, total))
        acc = 0.0
        # Cumulative fractional STRIDE count, built up segment-by-segment as
        # we scan for the one containing t_clamped -- each segment's own
        # (constant, since interpolation within a segment is linear-in-time)
        # speed determines its own stride length (_stride_length), so gait
        # tempo changes naturally with how fast a given segment is moving.
        # A held segment (0 distance/explicit duration -- see hold()) has
        # speed 0 and contributes nothing, so footsteps correctly pause
        # during a hold.
        phase_acc = 0.0
        for i, d in enumerate(durs):
            a, b = self.keyframes[i], self.keyframes[i + 1]
            seg_len = float(np.linalg.norm(b.pos - a.pos))
            seg_speed = seg_len / d if d > 1e-9 else 0.0
            seg_stride = _stride_length(seg_speed)
            last = (i == len(durs) - 1)
            if t_clamped <= acc + d or last:
                local_t = (t_clamped - acc) / d if d > 1e-9 else 1.0
                local_t = max(0.0, min(1.0, local_t))
                pos = a.pos + (b.pos - a.pos) * local_t
                yaw = a.yaw + _shortest_angle_diff(a.yaw, b.yaw) * local_t
                pitch = a.pitch + (b.pitch - a.pitch) * local_t
                phase_acc += (local_t * seg_len) / seg_stride if seg_stride > 1e-6 else 0.0
                # Handheld shake is evaluated at the UN-clamped t (not
                # t_clamped) so it keeps evolving smoothly even if a caller
                # samples slightly past the path's end -- only matters at
                # the boundary, keeps the wobble continuous there.
                pos_off, yaw_off, pitch_off, roll_off = self._handheld_offset(t)
                gpos, gyaw, gpitch, groll = self._gait_offset(t, seg_speed, phase_acc)
                return (pos + pos_off + gpos, float(yaw) + yaw_off + gyaw,
                        float(pitch) + pitch_off + gpitch, roll_off + groll)
            phase_acc += seg_len / seg_stride if seg_stride > 1e-6 else 0.0
            acc += d
        kf = self.keyframes[-1]
        pos_off, yaw_off, pitch_off, roll_off = self._handheld_offset(t)
        gpos, gyaw, gpitch, groll = self._gait_offset(t, 0.0, phase_acc)
        return (kf.pos.copy() + pos_off + gpos, kf.yaw + yaw_off + gyaw,
                kf.pitch + pitch_off + gpitch, roll_off + groll)

    def to_list(self):
        return [kf.to_dict() for kf in self.keyframes]

    @staticmethod
    def from_list(items):
        cp = CameraPath()
        for kf in items:
            cp.add(kf['pos'], kf.get('yaw', 0.0), kf.get('pitch', 0.0),
                    kf.get('speed', 1.0), kf.get('duration'))
        return cp


# =============================================================================
# 11b) SensorCameraData -- camera path driven by an external sensor-record
#      JSON file (--camera-data), instead of hand-placed camera keyframes.
#      Each sample carries its own timestamp (ms, converted to seconds
#      relative to the first sample) + pos (m) + rot (deg: x=pitch, y=yaw,
#      z=roll). Loading this REPLACES the scene's camera keyframe path --
#      the two are mutually exclusive (see main()/--camera-data handling).
#      Unlike CameraPath, this does NOT get "P"-key keyframes added to it;
#      it is read-only, built once from the sensor file.
# =============================================================================

class _SensorSample:
    __slots__ = ("t", "pos", "yaw", "pitch", "roll")

    def __init__(self, t, pos, yaw, pitch, roll):
        self.t = float(t)
        self.pos = pos  # np.float32[3]
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.roll = float(roll)


class SensorCameraData:
    """Loaded from a sensor_record JSON (version 1: timestamp_unit "ms",
    position_unit "m", rotation_unit "deg", samples: [{timestamp, pos:{x,y,z},
    rot:{x,y,z}}, ...]). rot.x/.y/.z map to pitch/yaw/roll respectively (same
    axis convention as camera_matrix's Rx=pitch, Ry=yaw, Rz=roll)."""

    def __init__(self, samples, multiplier=1.0, offset=(0.0, 0.0, 0.0)):
        self.samples = samples  # list[_SensorSample], t sorted ascending, t[0] == 0.0
        self.multiplier = float(multiplier)
        self.offset = np.array(offset, dtype=np.float32)

    def is_camera_data(self):
        return True

    @staticmethod
    def load(path, multiplier=1.0, offset=(0.0, 0.0, 0.0)):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        raw_samples = data.get('samples', [])
        if not raw_samples:
            raise ValueError(f"'{path}' has no samples.")
        ts_unit = data.get('timestamp_unit', 'ms')
        ts_scale = 0.001 if ts_unit == 'ms' else 1.0
        offset_arr = np.array(offset, dtype=np.float32)

        samples = []
        t0 = raw_samples[0]['timestamp'] * ts_scale
        for s in raw_samples:
            t = s['timestamp'] * ts_scale - t0
            p = s['pos']
            r = s['rot']
            pos = (np.array([p['x'], p['y'], p['z']], dtype=np.float32) * multiplier) + offset_arr
            # rotation_unit "deg" -> radians; x=pitch, y=yaw, z=roll (see camera_matrix).
            pitch = math.radians(r['z'])
            yaw = math.radians(r['y'])
            roll = math.radians(r['x'])
            samples.append(_SensorSample(t, pos, yaw, pitch, roll))
        samples.sort(key=lambda s: s.t)
        return SensorCameraData(samples, multiplier=multiplier, offset=offset)

    def total_duration(self):
        return self.samples[-1].t if self.samples else 0.0

    def timestamps(self):
        return [s.t for s in self.samples]

    def average_fps(self):
        """Average frames/second implied by consecutive sample timestamps."""
        if len(self.samples) < 2:
            return 0.0
        total = self.samples[-1].t - self.samples[0].t
        if total <= 1e-9:
            return 0.0
        return (len(self.samples) - 1) / total

    def sample(self, t):
        """t (seconds from the first sample) -> (pos np.float32[3], yaw, pitch, roll),
        LINEARLY interpolated between the two nearest samples (roll uses the
        shortest-angle interpolation too, same as yaw, so it doesn't spin the
        long way around at the +-180deg wrap)."""
        n = len(self.samples)
        if n == 0:
            return None
        if n == 1:
            s = self.samples[0]
            return s.pos.copy(), s.yaw, s.pitch, s.roll

        t = max(0.0, min(t, self.samples[-1].t))
        # Binary search for the segment containing t.
        lo, hi = 0, n - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self.samples[mid].t <= t:
                lo = mid
            else:
                hi = mid
        a, b = self.samples[lo], self.samples[hi]
        span = b.t - a.t
        local_t = (t - a.t) / span if span > 1e-9 else 1.0
        local_t = max(0.0, min(1.0, local_t))
        pos = a.pos + (b.pos - a.pos) * local_t
        yaw = a.yaw + _shortest_angle_diff(a.yaw, b.yaw) * local_t
        pitch = a.pitch + (b.pitch - a.pitch) * local_t
        roll = a.roll + _shortest_angle_diff(a.roll, b.roll) * local_t
        return pos, float(yaw), float(pitch), float(roll)


# =============================================================================
# 11c) LiveCameraStream -- camera pose fed in over the network in REAL TIME,
#      as an alternative to SensorCameraData's prerecorded JSON file. Same
#      "is_camera_data()" interface (position/rotation/roll), so it drops
#      straight into everywhere main() already accepts --camera-data: the
#      interactive view's "I" replay, and render_video()'s camera_path
#      argument (rendering a video from a live stream just keeps sampling
#      "now" every frame instead of seeking along a fixed timeline).
#
#      Wire format: one UDP packet per pose sample, each packet's payload a
#      UTF-8 JSON object with the SAME fields as one `samples[]` entry in a
#      --camera-data file:
#          {"pos": {"x": 0.0, "y": 1.6, "z": 0.0},
#           "rot": {"x": 0.0, "y": 0.0, "z": 0.0}}   # deg: x=pitch, y=yaw, z=roll
#      Point any external source at it -- a phone IMU app, a physical rig
#      with an orientation sensor, a bridged game controller, an OpenCV
#      pose-estimation script watching a real camera -- one UDP send() per
#      new pose, no file, no "recording" step first.
# =============================================================================

class LiveCameraStream:
    """Reads camera pose samples from a UDP socket as they arrive, instead of
    from a prerecorded --camera-data file. There is no fixed timeline to
    seek into (unlike SensorCameraData/CameraPath) -- sample(t) IGNORES t
    and always returns the most recently received pose, lightly blended
    with the one before it to smooth out network jitter between packets."""

    def __init__(self, host="0.0.0.0", port=9999, multiplier=1.0, offset=(0.0, 0.0, 0.0),
                 smoothing=0.5):
        import socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.bind((host, port))
        self.host, self.port = host, port
        self.multiplier = float(multiplier)
        self.offset = np.array(offset, dtype=np.float32)
        self.smoothing = max(0.0, min(1.0, float(smoothing)))
        self._prev = None    # _SensorSample -- previous received pose
        self._latest = None  # _SensorSample -- most recently received pose
        self._start_wall = None
        self.n_received = 0

    def is_camera_data(self):
        return True

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def _drain(self):
        """Reads every packet currently waiting in the socket buffer (never
        blocks) so we always act on the FRESHEST pose, not a backlog."""
        import socket as _socket
        while True:
            try:
                data, _addr = self.sock.recvfrom(65536)
            except (BlockingIOError, _socket.error):
                break
            try:
                s = json.loads(data.decode('utf-8'))
                p = s['pos']; r = s['rot']
                pos = (np.array([p['x'], p['y'], p['z']], dtype=np.float32) * self.multiplier) + self.offset
                pitch = math.radians(r['z'])
                yaw = math.radians(r['y'])
                roll = math.radians(r['x'])
                now = time.time()
                if self._start_wall is None:
                    self._start_wall = now
                t = now - self._start_wall
                self._prev = self._latest
                self._latest = _SensorSample(t, pos, yaw, pitch, roll)
                self.n_received += 1
            except (KeyError, ValueError, TypeError, UnicodeDecodeError):
                continue  # malformed packet -- drop it and keep listening

    def has_data(self):
        self._drain()
        return self._latest is not None

    def total_duration(self):
        return 0.0  # open-ended stream -- there's no fixed "length" to report

    def sample(self, t=None):
        """t is accepted (for interface-compatibility with CameraPath/
        SensorCameraData) but ignored: a live stream has no timeline to seek,
        only a "most recent pose". Returns None until at least one packet
        has arrived."""
        self._drain()
        if self._latest is None:
            return None
        if self._prev is None or self.smoothing <= 0.0:
            s = self._latest
            return s.pos.copy(), s.yaw, s.pitch, s.roll
        a, b = self._prev, self._latest
        blend = 1.0 - self.smoothing
        pos = a.pos * (1.0 - blend) + b.pos * blend
        yaw = a.yaw + _shortest_angle_diff(a.yaw, b.yaw) * blend
        pitch = a.pitch + (b.pitch - a.pitch) * blend
        roll = a.roll + _shortest_angle_diff(a.roll, b.roll) * blend
        return pos, float(yaw), float(pitch), float(roll)


# =============================================================================
# 12) RayTracer -- Python-side wrapper. Supports set_resolution() and a
#     list of lights, same as v8. NEW: separate live_max_bounce so the
#     interactive view can use a low bounce cap (fast) independent of the
#     scene's own max_bounce (used for full-quality final renders) --
#     see item 5 in the module docstring for why this matters.
# =============================================================================

class RayTracer:
    def __init__(self, scene: Scene, width=480, height=480, fov=60,
                 max_bounce=DEFAULT_MAX_BOUNCE, background=None, lights=None, spotlights=None,
                 live_max_bounce=LIVE_MAX_BOUNCE):
        self.scene = scene
        self.fov_rad = math.radians(fov)
        self.fov_deg = fov
        self.half_tan = math.tan(self.fov_rad / 2)
        self.max_bounce = max_bounce
        self.live_max_bounce = live_max_bounce
        self.background = background if background is not None else Background()

        self.camera_pos = np.array([0.0, 5.0, -25.0], dtype=np.float32)
        self.camera_rot = np.array([0.0, 0.0], dtype=np.float32)
        self.camera_roll = 0.0  # radians -- only nonzero via --camera-data sensor replay
                                 # or the interactive ,/./ roll keys (see camera_matrix)
        self.water_time = 0.0   # seconds -- 0.0 = water ripples stay still (still image/live
                                 # raytrace, same as before); render_video() updates this value
                                 # EVERY frame (or every sub-sample when motion blur is on) so
                                 # the water "moves".
        self.ambient = 0.12
        self.specular_k = 0.6
        self.shininess = 64.0
        self.sky_light_strength = 0.5   # hemispherical sky-fill strength (see render_sample)
        self.caustic_strength = 1.0     # runtime contrast dial on top of CAUSTIC_MAX_MULT

        # --- Exposure / eye-adaptation (see render_video's autoexposure block) ---
        # `exposure` is the value actually fed to the tonemapper every frame.
        # In stills/interactive it's just used directly. In video, render_video
        # smoothly nudges it toward `_target_exposure` each frame instead of
        # jumping straight there, to fake the camera/eye needing a moment to
        # adjust to a brightness change (see EYE_ADAPT_SPEED below).
        self.exposure = 1.0
        self._target_exposure = 1.0
        self.eye_adapt_enabled = True
        self.eye_adapt_speed = 1.4      # 1/seconds -- higher = faster adaptation

        self.lights = list(lights) if lights else [Light((15, 30, -15), (255, 255, 255), 1.0)]
        if len(self.lights) > MAX_LIGHTS:
            raise ValueError(f"Max {MAX_LIGHTS} lights, got {len(self.lights)}")

        self.spotlights = list(spotlights) if spotlights else []
        if len(self.spotlights) > MAX_SPOTLIGHTS:
            raise ValueError(f"Max {MAX_SPOTLIGHTS} spotlights, got {len(self.spotlights)}")

        faces = scene.culled_faces()
        tri = triangulate(faces)
        self.bvh = BVH(tri)
        upload_scene_geometry(self.bvh, self.background, scene.textures)
        alloc_light_fields()
        self.sync_lights()

        self.water_blocks = list(scene.water_blocks)[:MAX_WATER_BLOCKS]
        if len(scene.water_blocks) > MAX_WATER_BLOCKS:
            print(f"Warning: {len(scene.water_blocks)} water blocks (non-rotated) but caustics "
                  f"only support up to {MAX_WATER_BLOCKS} blocks -- the rest will be ignored.")
        self.caustics_enabled = True
        alloc_water_fields()
        self._sync_water_blocks()
        self.compute_caustics()

        self.width = self.height = 0
        self.aspect = 1.0
        self._alloc_buffers(width, height)

    def sync_lights(self):
        """Re-uploads position/color/brightness of all lights to the GPU
        (called whenever a light is moved, e.g. after pressing K)."""
        pos = np.zeros((MAX_LIGHTS, 3), dtype=np.float32)
        col = np.ones((MAX_LIGHTS, 3), dtype=np.float32)
        bri = np.zeros(MAX_LIGHTS, dtype=np.float32)
        for i, lt in enumerate(self.lights):
            pos[i] = lt.position
            col[i] = lt.color
            bri[i] = lt.brightness
        LIGHT_POS.from_numpy(pos)
        LIGHT_COLOR.from_numpy(col)
        LIGHT_BRIGHTNESS.from_numpy(bri)

        spos = np.zeros((MAX_SPOTLIGHTS, 3), dtype=np.float32)
        sdir = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (MAX_SPOTLIGHTS, 1))
        scol = np.ones((MAX_SPOTLIGHTS, 3), dtype=np.float32)
        sbri = np.zeros(MAX_SPOTLIGHTS, dtype=np.float32)
        scos_o = np.ones(MAX_SPOTLIGHTS, dtype=np.float32)
        scos_i = np.ones(MAX_SPOTLIGHTS, dtype=np.float32)
        for i, sl in enumerate(self.spotlights):
            spos[i] = sl.position
            sdir[i] = sl.direction
            scol[i] = sl.color
            sbri[i] = sl.brightness
            scos_o[i] = sl.cos_outer
            scos_i[i] = sl.cos_inner
        SPOT_POS.from_numpy(spos)
        SPOT_DIR.from_numpy(sdir)
        SPOT_COLOR.from_numpy(scol)
        SPOT_BRIGHTNESS.from_numpy(sbri)
        SPOT_COS_OUTER.from_numpy(scos_o)
        SPOT_COS_INNER.from_numpy(scos_i)

    def _sync_water_blocks(self):
        """Uploads water block AABBs to the GPU (called once in __init__)."""
        n = len(self.water_blocks)
        mn = np.zeros((MAX_WATER_BLOCKS, 3), dtype=np.float32)
        mx = np.zeros((MAX_WATER_BLOCKS, 3), dtype=np.float32)
        iors = np.full(MAX_WATER_BLOCKS, 1.333, dtype=np.float32)
        for i, (bmin, bmax, ior) in enumerate(self.water_blocks):
            mn[i] = bmin
            mx[i] = bmax
            iors[i] = ior
        WB_MIN.from_numpy(mn)
        WB_MAX.from_numpy(mx)
        WB_IOR.from_numpy(iors)
        N_WATER_FIELD[None] = n

    def compute_caustics(self):
        """Recomputes the caustic map (photon pre-pass). Called once at
        startup, and again whenever a point Light's position changes (e.g.
        after pressing K) -- does NOT depend on image resolution so it does
        not need to be recomputed when switching between live/final resolution."""
        n_water = len(self.water_blocks)
        n_lights = len(self.lights)
        n_spots = len(self.spotlights)
        if n_water == 0 or (n_lights == 0 and n_spots == 0):
            return
        CAUSTIC_DOWN.fill(0.0)
        CAUSTIC_UP.fill(0.0)
        CAUSTIC_DOWN_BASE.fill(0.0)
        CAUSTIC_UP_BASE.fill(0.0)
        compute_caustics_kernel(n_water, n_lights, n_spots)

        down = CAUSTIC_DOWN.to_numpy()
        up = CAUSTIC_UP.to_numpy()
        down_base = CAUSTIC_DOWN_BASE.to_numpy()
        up_base = CAUSTIC_UP_BASE.to_numpy()

        def to_mult(hist, base):
            k = np.array([1.0, 2.0, 1.0], dtype=np.float32)
            k /= k.sum()
            h, b = hist, base
            for _ in range(CAUSTIC_BLUR_PASSES):
                h = np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'), 2, h)
                h = np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'), 1, h)
                b = np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'), 2, b)
                b = np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'), 1, b)
            # (h+eps)/(b+eps): tends to 1.0 (neutral) where there isn't enough
            # data (both histograms are ~0), tends to the actual density
            # ratio where there's a lot of data -- eps is calibrated around
            # the expected average density (~1 photon/cell when the grid
            # margin is ~0 and the light is directly overhead).
            eps = 0.35
            mult = (h + eps) / (b + eps)
            return np.clip(mult, 0.0, CAUSTIC_MAX_MULT).astype(np.float32)

        down_mult = to_mult(down, down_base)
        up_mult = to_mult(up, up_base)
        full_down = np.ones((MAX_WATER_BLOCKS, CAUSTIC_RES, CAUSTIC_RES), dtype=np.float32)
        full_up = np.ones((MAX_WATER_BLOCKS, CAUSTIC_RES, CAUSTIC_RES), dtype=np.float32)
        full_down[:n_water] = down_mult[:n_water]
        full_up[:n_water] = up_mult[:n_water]
        CAUSTIC_DOWN_MULT.from_numpy(full_down)
        CAUSTIC_UP_MULT.from_numpy(full_up)

    def _alloc_buffers(self, width, height):
        global ACCUM, DEPTH_ACCUM, SAMPLE_COUNT, OUTPUT, DEPTH_OUT
        self.width, self.height = width, height
        self.aspect = width / height

        # Only allocate these fields once, if they don't already exist.
        if ACCUM is None:
            ACCUM = ti.Vector.field(3, ti.f32, shape=(MAX_H, MAX_W))
            DEPTH_ACCUM = ti.field(ti.f32, shape=(MAX_H, MAX_W))
            OUTPUT = ti.Vector.field(3, ti.f32, shape=(MAX_H, MAX_W))
            DEPTH_OUT = ti.field(ti.f32, shape=(MAX_H, MAX_W))
            SAMPLE_COUNT = ti.field(ti.i32, shape=())

        self.reset_accumulation()

    def set_resolution(self, width, height):
        """Changes the raytrace resolution (e.g. switching live<->final) WITHOUT
        rebuilding the BVH/scene -- only reallocates the image buffers."""
        self._alloc_buffers(width, height)

    def reset_accumulation(self):
        reset_accum(self.width, self.height)

    def add_samples(self, n_samples=1, max_bounce_override=None):
        """max_bounce_override lets callers (e.g. the interactive live view)
        use a lower bounce cap than self.max_bounce for speed -- see
        LIVE_MAX_BOUNCE / item 5 in the module docstring."""
        bounce = self.max_bounce if max_bounce_override is None else max_bounce_override
        R = camera_matrix(*self.camera_rot, self.camera_roll)
        cp = self.camera_pos
        self.sync_lights()
        WATER_TIME[None] = float(self.water_time)
        for _ in range(n_samples):
            render_sample(
                float(cp[0]), float(cp[1]), float(cp[2]),
                float(R[0, 0]), float(R[0, 1]), float(R[0, 2]),
                float(R[1, 0]), float(R[1, 1]), float(R[1, 2]),
                float(R[2, 0]), float(R[2, 1]), float(R[2, 2]),
                float(self.half_tan), float(self.aspect),
                int(len(self.lights)), int(len(self.spotlights)),
                float(self.ambient), float(self.specular_k), float(self.shininess),
                float(self.background.brightness),
                int(bounce), int(self.width), int(self.height),
                int(self.caustics_enabled),
                float(self.sky_light_strength), float(self.caustic_strength),
            )

    def measure_target_exposure(self, target_luma=0.2):
        """Reads back the raw (pre-tonemap) accumulation buffer and returns
        the exposure multiplier that would put its average luminance at
        `target_luma` -- the same "aim for 18% gray" logic a real camera's
        auto-exposure metering uses. Cheap enough to call every frame at
        the resolutions this renderer targets (a single numpy readback +
        mean). Used two ways (see eye_adapt_enabled / DEFAULT_POST_FX):
        INSTANTLY for stills/live-preview (metering should just be correct
        immediately for a photo), and SMOOTHED frame-to-frame in
        render_video (a real eye/camera visibly takes a moment to adjust
        when the scene brightness changes)."""
        n = max(1, int(SAMPLE_COUNT[None]))
        raw = ACCUM.to_numpy()[:self.height, :self.width]
        luma = raw[..., 0] * 0.2126 + raw[..., 1] * 0.7152 + raw[..., 2] * 0.0722
        avg = float(luma.mean()) / n
        if avg < 1e-5:
            return self.exposure
        return float(np.clip(target_luma / avg, 0.05, 20.0))

    def current_image_float(self):
        if self.eye_adapt_enabled:
            # Live-preview / still-image metering: adjust INSTANTLY (no lag)
            # -- see render_video for the version that lags on purpose.
            self.exposure = self.measure_target_exposure()
        resolve_output(self.width, self.height, float(self.exposure))
        color = np.clip(OUTPUT.to_numpy()[:self.height, :self.width], 0.0, 1.0)
        depth = DEPTH_OUT.to_numpy()[:self.height, :self.width]
        return color, depth

    def current_image(self):
        color, _ = self.current_image_float()
        return (color * 255).astype(np.uint8)

    def render_to_file(self, out_path="raytrace_v9.png", samples=32, post_fx=None,
                       batch=4, progress_cb=None):
        """progress_cb (if given) is called after EVERY batch of samples with
        (done, total, elapsed_seconds, eta_seconds) -- used to draw %/ETA/
        elapsed time on screen (see draw_render_progress in main()) alongside
        the console output."""
        print(f"Rendering {self.width}x{self.height}, samples={samples} ...")
        t0 = time.time()
        self.reset_accumulation()
        done = 0
        while done < samples:
            step = min(batch, samples - done)
            self.add_samples(step)
            done += step
            elapsed = time.time() - t0
            frac = done / samples
            eta = (elapsed / frac - elapsed) if frac > 0 else 0.0
            print(f"\r  {frac*100:5.1f}%  sample {done}/{samples}  "
                  f"elapsed {elapsed:6.1f}s  ETA {eta:6.1f}s", end="", flush=True)
            if progress_cb is not None:
                progress_cb(done, samples, elapsed, eta)
        color, depth = self.current_image_float()
        if post_fx and post_fx.get('enabled', True):
            R = camera_matrix(*self.camera_rot, self.camera_roll)
            flares = compute_flare_list(self, self.camera_pos, R)
            color = apply_post_processing(color, depth, flares, post_fx)
        img = (np.clip(color, 0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(img).save(out_path)
        print(f"\nSaved: {out_path} ({time.time() - t0:.2f}s)")

    def render_video(self, camera_path, out_path="raytrace_v9_video.mp4",
                      fps=VIDEO_FPS, resolution=VIDEO_RES, duration=VIDEO_DURATION,
                      samples_per_frame=VIDEO_SAMPLES_PER_FRAME, post_fx=None,
                      progress_cb=None, camera_sync=False):
        """Renders a video following camera_path, which is EITHER a CameraPath
        (hand-placed camera keyframes) OR a SensorCameraData (--camera-data):
          - CameraPath, 0 keyframes -> does nothing (safety net; main() already
            blocks this case before calling it).
          - CameraPath, 1 keyframe  -> camera stays STILL for 'duration' seconds.
          - CameraPath, >=2 keyframes -> camera moves LINEARLY along the path,
            duration = the sum of segment_durations() (distance / speed per segment).
          - SensorCameraData -> duration = the sensor recording's own length
            (last sample timestamp). If camera_sync is True, EVERY recorded
            sample becomes exactly one output frame (fps stays as given --
            only the frame COUNT/timing source changes, see main()'s
            --camera-sync / --camera-get-fps handling for how fps itself is set);
            otherwise frames are spaced evenly at 'fps' and each frame's camera
            pose is linearly interpolated from the recording, same as a
            CameraPath. Only SensorCameraData carries roll (camera_matrix's
            3rd angle) -- CameraPath.sample() always returns roll=0.0.
        motion_blur (post_fx['motion_blur_enabled']) when enabled does NOT use
        a fake 2D blur; it takes multiple raytrace samples at different POINTS
        IN TIME within each frame's "shutter window" and accumulates them
        together -- this is "real" motion blur built on the raytracer's own
        sample-accumulation (ACCUM) system. camera_sync frames are single-pose
        (no shutter window to sample across), so motion blur is skipped for them.
        Water (if the scene has scene.add_water(...)) will RIPPLE OVER TIME
        (WATER_WAVE_SPEED) during video rendering -- stills (render_to_file)
        and the live raytrace view still stay still, since they never pass this time value."""
        is_sensor = camera_path.is_camera_data()
        n_kf = 0 if is_sensor else len(camera_path.keyframes)
        if not is_sensor and n_kf == 0:
            print("No camera keyframes -- cancelling video render.")
            return None

        if is_sensor:
            total_time = max(1e-3, camera_path.total_duration())
        else:
            total_time = duration if n_kf == 1 else max(1e-3, camera_path.total_duration())

        camera_sync = camera_sync and is_sensor
        if camera_sync:
            frame_times = camera_path.timestamps()
            n_frames = len(frame_times)
        else:
            n_frames = max(1, int(round(total_time * fps)))
            frame_times = None
        frames_dir = os.path.splitext(out_path)[0] + "_frames"
        os.makedirs(frames_dir, exist_ok=True)

        post_fx = post_fx if post_fx is not None else dict(DEFAULT_POST_FX)
        motion_blur = (not camera_sync) and (n_kf > 1 or is_sensor) and bool(post_fx.get('motion_blur_enabled', False))
        shutter = max(0.0, min(1.0, float(post_fx.get('motion_blur_shutter', 0.5))))
        dt = 1.0 / fps

        # --- Eye-adaptation (auto exposure) & autofocus, VIDEO ONLY -------
        # Unlike stills/live-preview (which meter/focus INSTANTLY, see
        # current_image_float / the F key), video is expected to visibly
        # take a moment to adjust when the scene's brightness or the focus
        # subject changes -- both are exponential ("time constant") lerps
        # toward a per-frame target, evaluated in real seconds-per-frame
        # (dt) so the adaptation SPEED looks the same regardless of fps.
        do_eye_adapt = self.eye_adapt_enabled
        self.eye_adapt_enabled = False  # avoid current_image_float()'s INSTANT metering below
        adapt_k = 1.0 - math.exp(-self.eye_adapt_speed * dt) if do_eye_adapt else 0.0

        do_autofocus = bool(post_fx.get('autofocus_enabled', False))
        af_speed = float(post_fx.get('autofocus_speed', 2.2))
        af_k = 1.0 - math.exp(-af_speed * dt) if do_autofocus else 0.0
        current_focus = float(post_fx.get('dof_focus_distance', 25.0))

        prev_w, prev_h = self.width, self.height
        self.set_resolution(*resolution)

        animate_caustics = (WATER_ANIMATE_CAUSTICS and self.water_blocks
                             and (self.lights or self.spotlights))
        last_caustic_t = None

        sync_note = " (camera-sync: 1 frame/sample)" if camera_sync else ""
        print(f"Rendering video {resolution[0]}x{resolution[1]}, {n_frames} frames "
              f"({total_time:.2f}s @ {fps}fps), {samples_per_frame} sample(s)/frame{sync_note} ...")
        t0 = time.time()
        for fi in range(n_frames):
            t_center = frame_times[fi] if camera_sync else min((fi + 0.5) * dt, total_time)
            self.reset_accumulation()

            if animate_caustics:
                wt = t_center * WATER_WAVE_SPEED
                if last_caustic_t is None or (wt - last_caustic_t) >= WATER_CAUSTIC_UPDATE_INTERVAL:
                    WATER_TIME[None] = wt
                    self.compute_caustics()
                    last_caustic_t = wt

            if motion_blur and shutter > 0.0:
                half = shutter * dt * 0.5
                t_lo = max(0.0, t_center - half)
                t_hi = min(total_time, t_center + half)
                for s in range(samples_per_frame):
                    frac = (s + np.random.random()) / samples_per_frame
                    tt = t_lo + (t_hi - t_lo) * frac
                    pos, yaw, pitch, roll = camera_path.sample(tt)
                    self.camera_pos = pos.astype(np.float32)
                    self.camera_rot = np.array([yaw, pitch], dtype=np.float32)
                    self.camera_roll = roll
                    self.water_time = tt * WATER_WAVE_SPEED
                    self.add_samples(1)
            else:
                pos, yaw, pitch, roll = camera_path.sample(t_center)
                self.camera_pos = pos.astype(np.float32)
                self.camera_rot = np.array([yaw, pitch], dtype=np.float32)
                self.camera_roll = roll
                self.water_time = t_center * WATER_WAVE_SPEED
                self.add_samples(samples_per_frame)

            if do_eye_adapt:
                target_exposure = self.measure_target_exposure()
                self.exposure += (target_exposure - self.exposure) * adapt_k

            if do_autofocus:
                R_af = camera_matrix(*self.camera_rot, self.camera_roll)
                forward = R_af[:, 2]
                probe_depth(float(self.camera_pos[0]), float(self.camera_pos[1]), float(self.camera_pos[2]),
                            float(forward[0]), float(forward[1]), float(forward[2]))
                target_focus = float(np.clip(PROBE_DEPTH[None], 0.3, 2000.0))
                current_focus += (target_focus - current_focus) * af_k
                post_fx['dof_focus_distance'] = current_focus

            color, depth = self.current_image_float()
            if post_fx.get('enabled', True):
                R = camera_matrix(*self.camera_rot, self.camera_roll)
                flares = compute_flare_list(self, self.camera_pos, R)
                color = apply_post_processing(color, depth, flares, post_fx, frame_seed=fi)
            img = (np.clip(color, 0.0, 1.0) * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(frames_dir, f"frame_{fi:05d}.png"))

            elapsed = time.time() - t0
            frac_done = (fi + 1) / n_frames
            eta = (elapsed / frac_done - elapsed) if frac_done > 0 else 0.0
            print(f"\r  Frame {fi + 1}/{n_frames}  {frac_done * 100:5.1f}%  "
                  f"elapsed {elapsed:6.1f}s  ETA {eta:6.1f}s", end="", flush=True)
            if progress_cb is not None:
                progress_cb(fi + 1, n_frames, elapsed, eta)

        self.eye_adapt_enabled = do_eye_adapt
        self.set_resolution(prev_w, prev_h)
        self.water_time = 0.0
        self.camera_roll = 0.0
        WATER_TIME[None] = 0.0
        if animate_caustics:
            self.compute_caustics()  # restore the still (t=0) caustic map, as before rendering video
        print(f"\nRendered {n_frames} frames into: {frames_dir}")

        if _encode_video_ffmpeg(frames_dir, out_path, fps):
            print(f"Encoded video (ffmpeg): {out_path}")
            return out_path
        print(f"ffmpeg not found in PATH -- PNG frames are still in "
              f"'{frames_dir}'. You can encode them yourself with:\n"
              f"  ffmpeg -framerate {fps} -i {frames_dir}/frame_%05d.png "
              f"-c:v libx264 -pix_fmt yuv420p {out_path}")
        return frames_dir

    # --- Serialization of the "shot" (camera + look/render params) --------
    def camera_dict(self):
        return {'pos': [float(x) for x in self.camera_pos],
                'yaw': float(self.camera_rot[0]), 'pitch': float(self.camera_rot[1])}


def _encode_video_ffmpeg(frames_dir, out_path, fps):
    """Stitches a sequence of PNG frames (frame_%05d.png) in frames_dir into a
    video with ffmpeg (subprocess). Returns True on success, False if ffmpeg
    isn't found in PATH or ffmpeg fails -- either way the PNG frames are KEPT
    (not deleted) so the user can handle it themselves."""
    if shutil.which('ffmpeg') is None:
        return False
    cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', os.path.join(frames_dir, 'frame_%05d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"\nError running ffmpeg: {e}")
        return False


class ProgressiveRenderer:
    """Live preview raytrace in the window: accumulates samples across
    multiple frames while the camera is still, upscales to window size,
    applies post-processing every frame. Uses tracer.live_max_bounce
    (a low bounce cap) instead of the scene's full-quality max_bounce --
    see item 5 in the module docstring."""

    def __init__(self, tracer: RayTracer, samples_per_frame=1, post_fx=None):
        self.tracer = tracer
        self.samples_per_frame = samples_per_frame
        self.active = False
        self.post_fx = post_fx if post_fx is not None else dict(DEFAULT_POST_FX)
        self._frame_seed = 0

    def start(self):
        self.tracer.reset_accumulation()
        self.active = True

    def step(self, any_input, window_size):
        """Renders one live-preview frame and returns it as a BGR uint8 numpy
        array sized `window_size` = (w, h), ready for cv2.imshow."""
        if any_input:
            self.tracer.reset_accumulation()
        self.tracer.add_samples(self.samples_per_frame, max_bounce_override=self.tracer.live_max_bounce)
        color, depth = self.tracer.current_image_float()

        R = camera_matrix(*self.tracer.camera_rot, self.tracer.camera_roll)
        flares = compute_flare_list(self.tracer, self.tracer.camera_pos, R)
        self._frame_seed += 1
        color = apply_post_processing(color, depth, flares, self.post_fx, frame_seed=self._frame_seed)

        img_rgb = (np.clip(color, 0.0, 1.0) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        w, h = window_size
        if (img_bgr.shape[1], img_bgr.shape[0]) != (w, h):
            img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        return img_bgr

# =============================================================================
# 13) Post-processing (CPU / numpy) -- lens flare, DoF, chromatic aberration,
#     VHS. All parameters default in DEFAULT_POST_FX, editable directly in
#     code or at runtime via the post_fx dict.
# =============================================================================

DEFAULT_POST_FX = {
    'enabled': True,
    'dof_enabled': False,
    'dof_focus_distance': 25.0,   # focus point (world units) -- adjustable via keys when enabled
    'dof_blur_strength': 0.5,     # how "sensitive" the blur is to focus-distance error
    'dof_max_radius': 48,          # max blur radius (px, at the resolution being rendered)
    'autofocus_enabled': False,   # video only -- see render_video's autofocus block. Photo/live
                                   # preview focus is always instant (the F key probes depth and
                                   # sets dof_focus_distance directly); this flag instead makes
                                   # render_video() itself keep re-probing screen-center depth
                                   # every frame and SMOOTHLY chase it, like a camcorder's
                                   # continuous autofocus hunting/settling instead of snapping.
    'autofocus_speed': 2.2,       # 1/seconds -- higher = focus catches up to the target faster
    'chroma_enabled': True,
    'chroma_strength': 0.004,     # how far the R/B channels shift from center (fraction of screen)
    'flare_enabled': True,
    'flare_size': 55.0,           # main glow radius (px, referenced at 360p)
    'flare_intensity': 0.9,       # additive flare intensity
    'flare_anamorphic': 0.18,     # horizontal anamorphic streak strength (0 = disabled)
    'flare_halo': 0.3,            # secondary ring/halo strength (0 = disabled)
    'fisheye_enabled': False,     # wide-angle lens barrel distortion
    'fisheye_strength': 0.32,     # 0 = rectilinear (no distortion), ~0.2-0.5 = visible bulge
    'bloom_enabled': True,        # highlight glow/bleed -- extracted from bright areas post-tonemap
    'bloom_threshold': 0.62,      # 0..1 -- brightness above which a pixel starts contributing to bloom
    'bloom_intensity': 0.45,      # additive bloom strength
    'bloom_radius': 22,           # glow spread (px, referenced at 360p)
    'vhs_enabled': True,         # VHS tape effect -- works for BOTH stills and video
    'vhs_strength': 1.0,          # overall VHS effect intensity (0..~2)
    'motion_blur_enabled': True,  # motion blur -- ONLY affects render_video()
                                   # (simulated via multiple raytrace samples at
                                   # different points in time, not a 2D image blur)
    'motion_blur_shutter': 0.1,   # "shutter open" fraction of 1 frame (0..1)
}


def _box_blur(img, radius):
    """Fast box blur via cumulative sum, edge-replicate padding."""
    if radius <= 0:
        return img

    def blur1d(a, r, axis):
        pad_width = [(0, 0)] * a.ndim
        pad_width[axis] = (r, r)
        ap = np.pad(a, pad_width, mode='edge')
        csum = np.cumsum(ap, axis=axis)
        zero_shape = list(csum.shape)
        zero_shape[axis] = 1
        csum = np.concatenate([np.zeros(zero_shape, dtype=csum.dtype), csum], axis=axis)
        n = a.shape[axis]
        sl_hi = [slice(None)] * a.ndim
        sl_lo = [slice(None)] * a.ndim
        sl_hi[axis] = slice(2 * r + 1, 2 * r + 1 + n)
        sl_lo[axis] = slice(0, n)
        summed = csum[tuple(sl_hi)] - csum[tuple(sl_lo)]
        return summed / (2 * r + 1)

    out = blur1d(img, radius, axis=0)
    out = blur1d(out, radius, axis=1)
    return out


def apply_depth_of_field(img, depth, focus_distance, blur_strength, max_radius):
    """Approximate DoF: circle-of-confusion from |depth-focus|, linearly
    interpolated between the 2 nearest box-blur levels (no external lib needed)."""
    max_radius = max(1, int(round(max_radius)))
    if blur_strength <= 0:
        return img
    coc = np.clip(np.abs(depth - focus_distance) * blur_strength, 0.0, float(max_radius))

    levels = [img]
    for r in range(1, max_radius + 1):
        levels.append(_box_blur(img, r))

    lo_idx = np.clip(np.floor(coc).astype(np.int32), 0, max_radius)
    hi_idx = np.clip(lo_idx + 1, 0, max_radius)
    frac = (coc - lo_idx)[..., None]

    lo_img = np.zeros_like(img)
    hi_img = np.zeros_like(img)
    for r in range(0, max_radius + 1):
        m_lo = (lo_idx == r)[..., None]
        lo_img = np.where(m_lo, levels[r], lo_img)
        m_hi = (hi_idx == r)[..., None]
        hi_img = np.where(m_hi, levels[r], hi_img)

    return lo_img * (1.0 - frac) + hi_img * frac


def apply_chromatic_aberration(img, strength):
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    def sample(channel, scale):
        sx = cx + (xx - cx) * scale
        sy = cy + (yy - cy) * scale
        sx = np.clip(sx, 0, w - 1).astype(np.int32)
        sy = np.clip(sy, 0, h - 1).astype(np.int32)
        return channel[sy, sx]

    r = sample(img[..., 0], 1.0 + strength)
    g = img[..., 1]
    b = sample(img[..., 2], 1.0 - strength)
    return np.stack([r, g, b], axis=-1)


def apply_lens_flare(img, flares, flare_size, flare_intensity, anamorphic=0.0, halo=0.0):
    """flares: list of (screen_x, screen_y, color[0-1] tuple, brightness).
    Draws a Gaussian glow at each light's position, a few "ghosts" along the
    line from the light to the screen center (classic lens flare), an
    optional horizontal anamorphic streak (the thin blue-ish horizontal line
    real cinema-lens flares throw across the frame), and an optional
    colored halo ring (the "rainbow donut" ghost from internal lens
    reflections) for a more camera-realistic look."""
    if not flares or flare_intensity <= 0:
        return img
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = img.copy()
    cx0, cy0 = (w - 1) / 2.0, (h - 1) / 2.0
    scale = min(w, h) / 360.0
    radius = max(2.0, flare_size * scale)

    for (sx, sy, color, brightness) in flares:
        color_arr = np.array(color, dtype=np.float32)
        dx = xx - sx; dy = yy - sy
        d2 = dx * dx + dy * dy
        glow = np.exp(-d2 / (2.0 * radius * radius))
        out += glow[..., None] * color_arr * flare_intensity * brightness

        vx, vy = cx0 - sx, cy0 - sy
        for k, frac in enumerate((0.3, 0.6, 1.0)):
            gx, gy = sx + vx * frac, sy + vy * frac
            gd2 = (xx - gx) ** 2 + (yy - gy) ** 2
            gr = max(radius * (0.4 - 0.1 * k), 1.0)
            ghost = np.exp(-gd2 / (2.0 * gr * gr))
            out += ghost[..., None] * color_arr * (flare_intensity * 0.15 * brightness)

        if anamorphic > 0.0:
            # A thin horizontal streak through the light, fading with distance
            # along X and tightly confined in Y -- the classic "cinema lens"
            # flare line. Slightly cooler/whiter than the light's own color,
            # like real anamorphic coating flares tend to look.
            streak_col = color_arr * 0.5 + np.array([0.6, 0.75, 1.0], dtype=np.float32) * 0.5
            streak = np.exp(-(dy * dy) / (2.0 * (radius * 0.12) ** 2)) * \
                np.exp(-np.abs(dx) / (radius * 7.0))
            out += streak[..., None] * streak_col * flare_intensity * anamorphic * brightness

        if halo > 0.0:
            # A faint ring at a fixed radius from the light -- a stand-in for
            # the "rainbow donut" ghost caused by light bouncing between
            # internal lens elements. Slightly desaturated so it doesn't
            # compete visually with the main glow.
            d = np.sqrt(d2)
            ring_r = radius * 2.4
            ring = np.exp(-((d - ring_r) ** 2) / (2.0 * (radius * 0.35) ** 2))
            ring_col = color_arr * 0.6 + np.array([1.0, 1.0, 1.0], dtype=np.float32) * 0.4
            out += ring[..., None] * ring_col * flare_intensity * halo * 0.5 * brightness

    return out


def compute_flare_list(tracer: RayTracer, camera_pos, R):
    """Projects each light onto the screen + checks occlusion (a single BVH
    ray from the camera to the light), returns a list usable by apply_lens_flare()."""
    if not tracer.lights:
        return []
    compute_light_visibility(float(camera_pos[0]), float(camera_pos[1]),
                              float(camera_pos[2]), len(tracer.lights))
    vis = LIGHT_VIS.to_numpy()[:len(tracer.lights)]
    hw, hh = tracer.width / 2.0, tracer.height / 2.0
    flares = []
    for i, lt in enumerate(tracer.lights):
        if not vis[i]:
            continue
        rel = lt.position - camera_pos
        cam = R.T @ rel
        xc, yc, zc = cam
        if zc <= 0.1:
            continue
        sx = (xc / (zc * tracer.half_tan * tracer.aspect)) * hw + hw
        sy = (-yc / (zc * tracer.half_tan)) * hh + hh
        if -0.2 * tracer.width <= sx <= 1.2 * tracer.width and -0.2 * tracer.height <= sy <= 1.2 * tracer.height:
            flares.append((sx, sy, tuple(lt.color), lt.brightness))

    if tracer.spotlights:
        compute_spot_visibility(float(camera_pos[0]), float(camera_pos[1]),
                                 float(camera_pos[2]), len(tracer.spotlights))
        svis = SPOT_VIS.to_numpy()[:len(tracer.spotlights)]
        for i, sl in enumerate(tracer.spotlights):
            if not svis[i]:
                continue
            rel = sl.position - camera_pos
            # Only flares when the camera is INSIDE the cone (looking back toward the light)
            dist_to_cam = np.linalg.norm(rel)
            if dist_to_cam < 1e-6:
                continue
            dir_to_cam = -rel / dist_to_cam
            if float(dir_to_cam @ sl.direction) <= sl.cos_outer:
                continue
            cam_v = R.T @ rel
            xc, yc, zc = cam_v
            if zc <= 0.1:
                continue
            sx = (xc / (zc * tracer.half_tan * tracer.aspect)) * hw + hw
            sy = (-yc / (zc * tracer.half_tan)) * hh + hh
            if -0.2 * tracer.width <= sx <= 1.2 * tracer.width and -0.2 * tracer.height <= sy <= 1.2 * tracer.height:
                flares.append((sx, sy, tuple(sl.color), sl.brightness * 0.8))
    return flares


def apply_vhs(img, strength=1.0, frame_seed=0):
    """VHS tape effect (works for BOTH stills and each video frame):
      - horizontal color shift (R/B channels slide opposite ways), like tape crosstalk
      - scanlines (every other row darkened)
      - "tracking wobble": each row is randomly shifted sideways a bit, more
        pronounced near the top/bottom edge of the frame (like real tape
        tracking errors)
      - noise + a light vignette + slightly reduced saturation
    frame_seed makes the noise/wobble CHANGE per frame when rendering video
    (instead of repeating the same static noise pattern every frame)."""
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    rng = np.random.default_rng(1000 + int(frame_seed))
    out = img.astype(np.float32).copy()

    # 1) Horizontal color shift (chroma bleed)
    shift = max(1, int(round(2 * strength)))
    r = np.roll(out[..., 0], -shift, axis=1)
    b = np.roll(out[..., 2], shift, axis=1)
    out = np.stack([r, out[..., 1], b], axis=-1)

    # 2) Scanlines
    yy = np.arange(h)
    scan = 1.0 - 0.12 * strength * (yy % 2)
    out *= scan[:, None, None]

    # 3) Tracking wobble -- randomly shifts each row along the X axis
    edge_w = np.clip(1.0 - np.abs((yy / max(h - 1, 1)) - 0.5) * 2.0, 0.0, 1.0)
    wobble = np.round(rng.normal(0.0, 1.0, size=h) * (1.0 - edge_w * 0.6) * 1.5 * strength).astype(int)
    for row in range(h):
        sh_amt = int(wobble[row])
        if sh_amt:
            out[row] = np.roll(out[row], sh_amt, axis=0)

    # 4) Noise
    out += rng.normal(0.0, 0.035 * strength, size=(h, w, 1)).astype(np.float32)

    # 5) Light edge vignette
    yy2, xx2 = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    d = np.sqrt(((xx2 - cx) / (w / 2.0)) ** 2 + ((yy2 - cy) / (h / 2.0)) ** 2)
    vig = 1.0 - 0.25 * strength * np.clip(d - 0.6, 0.0, 1.0)
    out *= vig[..., None]

    # 6) Slightly reduced saturation (a grayish, old-tape look)
    gray = out.mean(axis=-1, keepdims=True)
    out = out * (1.0 - 0.15 * strength) + gray * (0.15 * strength)

    return np.clip(out, 0.0, 1.0)


def apply_fisheye(img, depth, strength):
    """Approximate wide-angle/fisheye barrel distortion: for each OUTPUT
    pixel, samples the INPUT (rectilinear-rendered) image from a position
    pulled radially outward by an amount that grows with distance from
    center (a simple r' = r*(1+k*r^2) polynomial model, the same family
    real lens-distortion calibration uses) -- the net visual effect is the
    classic fisheye "bulge", with straight lines away from center curving
    outward and the corners compressing inward. Cheap (a single remap, no
    raytracing involved) since it's applied to the already-rendered 2D
    image, not the lens model of the raytracer itself. Returns (img, depth)
    since DoF/downstream depth-aware effects need the depth buffer warped
    the same way to stay aligned with the (now-distorted) color image."""
    if strength <= 1e-4:
        return img, depth
    h, w = img.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - cx) / (w / 2.0)
    ny = (yy - cy) / (h / 2.0)
    r2 = nx * nx + ny * ny
    factor = 1.0 + strength * r2
    src_x = (cx + nx * factor * (w / 2.0)).astype(np.float32)
    src_y = (cy + ny * factor * (h / 2.0)).astype(np.float32)
    out_img = cv2.remap(img.astype(np.float32), src_x, src_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    out_depth = cv2.remap(depth.astype(np.float32), src_x, src_y, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    return out_img, out_depth


def _fisheye_warp_point(sx, sy, w, h, strength):
    """Maps a point's RECTILINEAR screen position to where it ends up after
    apply_fisheye()'s distortion -- used so lens-flare highlights (computed
    from the undistorted camera projection) still land on the light itself
    once the image has been fisheye-warped, instead of drifting off it.
    apply_fisheye is a dest->src remap (r_src = r_dst*(1+k*r_dst^2)), so
    this is its numeric inverse (r_dst such that the above holds for a
    given r_src), found by a few fixed-point iterations -- plenty for the
    strengths this effect is meant to be used at."""
    if strength <= 1e-4:
        return sx, sy
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    nx = (sx - cx) / (w / 2.0)
    ny = (sy - cy) / (h / 2.0)
    r = math.sqrt(nx * nx + ny * ny)
    if r < 1e-6:
        return sx, sy
    dst_r = r
    for _ in range(5):
        dst_r = r / (1.0 + strength * dst_r * dst_r)
    scale = dst_r / r
    return cx + nx * scale * (w / 2.0), cy + ny * scale * (h / 2.0)


def apply_bloom(img, threshold, intensity, radius):
    """Highlight glow: extracts pixels brighter than `threshold` (with a
    soft knee so the cutoff isn't a hard edge), blurs them at 2 radii (a
    tight-ish inner glow + a wide soft outer haze, cheaply approximating
    a multi-scale/Gaussian-pyramid bloom with just 2 box blurs), and adds
    that back on top of the image -- the "hot" parts of the frame bleed
    light into their surroundings, like a real camera sensor/eye does with
    bright sources. Operates on the already-tonemapped (post
    resolve_output) display image; see the ACES tonemap in resolve_output
    for why that still gives a reasonably smooth brightness gradient to
    threshold against instead of a flat, hard-edged blob of pure white."""
    if intensity <= 0:
        return img
    h, w = img.shape[:2]
    scale = min(w, h) / 360.0
    r_small = max(1, int(round(radius * 0.35 * scale)))
    r_large = max(1, int(round(radius * scale)))

    luma = img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722
    knee = 0.2
    soft = np.clip((luma - (threshold - knee)) / max(1e-4, 2.0 * knee), 0.0, 1.0)
    soft = soft * soft * (3.0 - 2.0 * soft)  # smoothstep -- soft knee instead of a hard cutoff
    bright = img * soft[..., None]

    glow_small = _box_blur(bright, r_small)
    glow_large = _box_blur(_box_blur(bright, r_large), r_large)  # 2 passes ~= a wider, softer falloff
    glow = glow_small * 0.6 + glow_large * 0.4
    return img + glow * intensity


def apply_post_processing(img, depth, flares, post_fx, frame_seed=0):
    out = img
    if post_fx.get('dof_enabled', False):
        out = apply_depth_of_field(out, depth, post_fx['dof_focus_distance'],
                                    post_fx['dof_blur_strength'], post_fx['dof_max_radius'])
    if post_fx.get('fisheye_enabled', False):
        fh, fw = out.shape[:2]
        strength = post_fx.get('fisheye_strength', 0.32)
        out, depth = apply_fisheye(out, depth, strength)
        flares = [(*_fisheye_warp_point(sx, sy, fw, fh, strength), color, brightness)
                  for (sx, sy, color, brightness) in flares]
    if post_fx.get('bloom_enabled', False):
        out = apply_bloom(out, post_fx.get('bloom_threshold', 0.62),
                           post_fx.get('bloom_intensity', 0.45), post_fx.get('bloom_radius', 22))
    if post_fx.get('flare_enabled', False):
        out = apply_lens_flare(out, flares, post_fx['flare_size'], post_fx['flare_intensity'],
                                anamorphic=post_fx.get('flare_anamorphic', 0.0),
                                halo=post_fx.get('flare_halo', 0.0))
    if post_fx.get('chroma_enabled', False):
        out = apply_chromatic_aberration(out, post_fx['chroma_strength'])
    if post_fx.get('vhs_enabled', False):
        out = apply_vhs(out, post_fx.get('vhs_strength', 1.0), frame_seed=frame_seed)
    return np.clip(out, 0.0, 1.0)

# =============================================================================
# 14) Preview 2D (painter's algorithm, CPU) -- not raytraced, just for fast
#     navigation. Uses scene.boxes/scene.quads DIRECTLY (no np.unique() --
#     np.unique sorts vertices by coordinate value, which only happens to work
#     for axis-aligned boxes; for ROTATED boxes that order would be wrong and
#     faces would come out mismatched. Vertices are stored in the same
#     bit-index order they were created in, which works for rotated boxes too.
# =============================================================================

_CUBE_FACE_IDX = np.array([
    [0, 1, 3, 2], [6, 7, 5, 4], [4, 5, 1, 0],
    [2, 3, 7, 6], [0, 2, 6, 4], [5, 7, 3, 1],
], dtype=np.int32)


def precompute_preview_geometry(scene: Scene):
    cube_verts = np.array([b[0] for b in scene.boxes], dtype=float) if scene.boxes else np.zeros((0, 8, 3))
    cube_colors = [b[1] for b in scene.boxes]
    quad_verts = np.array([q[0] for q in scene.quads], dtype=float) if scene.quads else np.zeros((0, 4, 3))
    quad_colors = [q[1] for q in scene.quads]
    return cube_verts, cube_colors, quad_verts, quad_colors


_PREVIEW_NEAR = 0.05  # near-plane (camera space) used to clip the preview -- MUST
                       # match the threshold used to decide "behind the camera" below.


def _clip_project_face(face_cam, half_tan, aspect, hw, hh, near=_PREVIEW_NEAR):
    """Sutherland-Hodgman clip of 1 polygon (in camera space, BEFORE
    perspective projection) against the plane z=near, then perspective-
    projects the REMAINING vertices. Returns (average_depth, [(sx,sy),...])
    or None if the polygon was entirely clipped away (fully behind the
    camera) or degenerates (<3 vertices after clipping). This is why a
    face no longer "disappears" entirely just because 1-2 of its vertices
    are behind the camera -- the old version (using `.any(axis=...)`)
    dropped a face the moment EVEN 1 vertex was behind the camera, even
    though the rest of the face was still on-screen and should have been drawn."""
    n = len(face_cam)
    clipped = []
    for i in range(n):
        curr = face_cam[i]
        prev = face_cam[i - 1]
        curr_in = curr[2] > near
        prev_in = prev[2] > near
        if curr_in != prev_in:
            denom = curr[2] - prev[2]
            t = (near - prev[2]) / denom if abs(denom) > 1e-9 else 0.0
            clipped.append(prev + t * (curr - prev))
        if curr_in:
            clipped.append(curr)
    if len(clipped) < 3:
        return None
    clipped = np.asarray(clipped, dtype=float)
    z = clipped[:, 2]
    sx = (clipped[:, 0] / (z * half_tan * aspect)) * hw + hw
    sy = (-clipped[:, 1] / (z * half_tan)) * hh + hh
    depth = float(z.mean())
    return depth, list(zip(sx.tolist(), sy.tolist()))


def compute_preview_entries(geo, camera_pos, R, half_tan, aspect, sw, sh):
    """Painter's-algorithm entry list: [(depth, [(x,y),...], color_rgb), ...],
    sorted back-to-front. Pure geometry (no drawing) -- kept separate from
    rasterization so it's reusable across screen/canvas backends. Faces are
    CLIPPED against the near-plane (Sutherland-Hodgman) instead of being
    dropped whole the moment ANY vertex crosses behind the camera -- only
    faces that are ENTIRELY behind the camera are skipped, everything else
    draws its visible portion. A numpy-vectorized "fast path" is kept for
    the common case of a face fully in front of the camera, so large scenes
    (e.g. an MC schematic import with thousands of boxes) don't slow the
    preview down."""
    cube_verts, cube_colors, quad_verts, quad_colors = geo
    entries = []
    if len(cube_colors) == 0 and len(quad_colors) == 0:
        return entries
    hw, hh = sw / 2.0, sh / 2.0
    Rt = R.T
    near = _PREVIEW_NEAR

    if len(cube_colors) > 0:
        rel = cube_verts - camera_pos
        cam = rel @ Rt.T                           # (N,8,3)
        face_cam = cam[:, _CUBE_FACE_IDX, :]        # (N,6,4,3)
        zs = face_cam[:, :, :, 2]
        fully_front = (zs > near).all(axis=2)
        fully_back = (zs <= near).all(axis=2)
        mean_z = zs.mean(axis=2)
        xs = face_cam[:, :, :, 0]; ys = face_cam[:, :, :, 1]
        sx = (xs / (zs * half_tan * aspect)) * hw + hw
        sy = (-ys / (zs * half_tan)) * hh + hh
        for ci in range(len(cube_colors)):
            col = cube_colors[ci]
            for fi in range(6):
                if fully_back[ci, fi]:
                    continue
                if fully_front[ci, fi]:
                    pts = list(zip(sx[ci, fi].tolist(), sy[ci, fi].tolist()))
                    entries.append((mean_z[ci, fi], pts, col))
                else:
                    res = _clip_project_face(face_cam[ci, fi], half_tan, aspect, hw, hh, near)
                    if res is not None:
                        entries.append((res[0], res[1], col))

    if len(quad_colors) > 0:
        relq = quad_verts - camera_pos
        camq = relq @ Rt.T
        zsq = camq[:, :, 2]
        fully_front_q = (zsq > near).all(axis=1)
        fully_back_q = (zsq <= near).all(axis=1)
        mean_zq = zsq.mean(axis=1)
        xsq = camq[:, :, 0]; ysq = camq[:, :, 1]
        sxq = (xsq / (zsq * half_tan * aspect)) * hw + hw
        syq = (-ysq / (zsq * half_tan)) * hh + hh
        for qi in range(len(quad_colors)):
            if fully_back_q[qi]:
                continue
            if fully_front_q[qi]:
                pts = list(zip(sxq[qi].tolist(), syq[qi].tolist()))
                entries.append((mean_zq[qi], pts, quad_colors[qi]))
            else:
                res = _clip_project_face(camq[qi], half_tan, aspect, hw, hh, near)
                if res is not None:
                    entries.append((res[0], res[1], quad_colors[qi]))

    entries.sort(key=lambda e: e[0], reverse=True)
    return entries


def draw_preview(canvas, geo, camera_pos, R, half_tan, aspect, sw, sh, lines=False):
    """Rasterizes compute_preview_entries() onto `canvas` (a BGR uint8 numpy
    array) with cv2.fillPoly / cv2.polylines."""
    entries = compute_preview_entries(geo, camera_pos, R, half_tan, aspect, sw, sh)
    for _, pts, col in entries:
        pts_i = np.array([[int(round(px)), int(round(py))] for px, py in pts], dtype=np.int32)
        if len(pts_i) < 3:
            continue
        bgr = (int(col[2]), int(col[1]), int(col[0]))
        try:
            cv2.fillPoly(canvas, [pts_i], bgr)
            if lines:
                cv2.polylines(canvas, [pts_i], True, (0, 0, 0), 1, cv2.LINE_AA)
        except Exception:
            pass


def draw_light_indicator(canvas, lights, selected_idx, camera_pos, R, half_tan, aspect, sw, sh,
                          spotlights=None):
    """Draws all lights; the SELECTED light gets a white ring to stand out
    (Tab cycles selection, K places the selected light at the camera). Index
    selected_idx counts into the COMBINED list [lights..., spotlights...].
    Spotlights are drawn as a triangle (pointing in the cone direction)
    instead of a circle to distinguish them from regular lights."""
    hw, hh = sw / 2.0, sh / 2.0
    spotlights = spotlights or []

    for i, lt in enumerate(lights):
        color = tuple(int(np.clip(c, 0, 1) * 255) for c in lt.color)
        bgr = (color[2], color[1], color[0])
        rel = lt.position - camera_pos
        cam = R.T @ rel
        xc, yc, zc = cam
        ring_col = (255, 255, 255) if i == selected_idx else (0, 45, 60)
        if zc > 0.1:
            sx = int((xc / (zc * half_tan * aspect)) * hw + hw)
            sy = int((-yc / (zc * half_tan)) * hh + hh)
            r = 10
            cv2.circle(canvas, (sx, sy), r + 3, (30, 30, 30), -1, cv2.LINE_AA)
            cv2.circle(canvas, (sx, sy), r, ring_col, 2, cv2.LINE_AA)
            cv2.circle(canvas, (sx, sy), 3, bgr, -1, cv2.LINE_AA)
        else:
            mag = math.sqrt(xc * xc + yc * yc) + 1e-9
            dx, dy = xc / mag, -yc / mag
            margin = 24
            cx_s = int(hw + dx * (hw - margin)); cy_s = int(hh + dy * (hh - margin))
            cv2.circle(canvas, (cx_s, cy_s), 8, bgr, -1, cv2.LINE_AA)
            if i == selected_idx:
                cv2.circle(canvas, (cx_s, cy_s), 10, (255, 255, 255), 2, cv2.LINE_AA)

    base = len(lights)
    for j, sl in enumerate(spotlights):
        i = base + j
        color = tuple(int(np.clip(c, 0, 1) * 255) for c in sl.color)
        bgr = (color[2], color[1], color[0])
        rel = sl.position - camera_pos
        cam = R.T @ rel
        xc, yc, zc = cam
        ring_col = (255, 255, 255) if i == selected_idx else (0, 45, 60)
        if zc > 0.1:
            sx = int((xc / (zc * half_tan * aspect)) * hw + hw)
            sy = int((-yc / (zc * half_tan)) * hh + hh)
            r = 9
            tri = np.array([[sx, sy - r], [sx - r, sy + r], [sx + r, sy + r]], dtype=np.int32)
            cv2.fillPoly(canvas, [tri], (30, 30, 30))
            cv2.polylines(canvas, [tri], True, ring_col, 2, cv2.LINE_AA)
            cv2.circle(canvas, (sx, sy), 3, bgr, -1, cv2.LINE_AA)
        else:
            mag = math.sqrt(xc * xc + yc * yc) + 1e-9
            dx, dy = xc / mag, -yc / mag
            margin = 24
            cx_s = int(hw + dx * (hw - margin)); cy_s = int(hh + dy * (hh - margin))
            r = 8
            tri = np.array([[cx_s, cy_s - r], [cx_s - r, cy_s + r], [cx_s + r, cy_s + r]], dtype=np.int32)
            cv2.fillPoly(canvas, [tri], bgr)
            if i == selected_idx:
                cv2.polylines(canvas, [tri], True, (255, 255, 255), 2, cv2.LINE_AA)


# =============================================================================
# 14b) Camera path -- draws the camera keyframes + connecting line on the
#      preview, and finds the camera keyframe "being aimed at" (closest to
#      the crosshair) so the O key knows which keyframe to edit/delete.
# =============================================================================

def find_targeted_keyframe(camera_path, camera_pos, R, half_tan, aspect, sw, sh, max_screen_dist=48):
    """Returns the index of the keyframe CLOSEST to the crosshair (screen
    center), within max_screen_dist pixels and in front of the camera --
    or None if there is none."""
    best_idx, best_d = None, max_screen_dist
    hw, hh = sw / 2.0, sh / 2.0
    Rt = R.T
    for i, kf in enumerate(camera_path.keyframes):
        rel = kf.pos - camera_pos
        xc, yc, zc = Rt @ rel
        if zc <= _PREVIEW_NEAR:
            continue
        sx = (xc / (zc * half_tan * aspect)) * hw + hw
        sy = (-yc / (zc * half_tan)) * hh + hh
        d = math.hypot(sx - hw, sy - hh)
        if d < best_d:
            best_d, best_idx = d, i
    return best_idx


def _clip_line_near(p0_cam, p1_cam, near):
    z0, z1 = p0_cam[2], p1_cam[2]
    in0, in1 = z0 > near, z1 > near
    if not in0 and not in1:
        return None
    if in0 and in1:
        return p0_cam, p1_cam
    t = (near - z0) / (z1 - z0)
    clip_pt = p0_cam + t * (p1_cam - p0_cam)
    return (p0_cam, clip_pt) if in0 else (clip_pt, p1_cam)


def draw_camera_path(canvas, camera_path, targeted_idx, camera_pos, R, half_tan, aspect, sw, sh):
    """Draws camera keyframes (yellow squares, red when 'targeted') and a
    straight line connecting consecutive keyframes (yellow) on the preview."""
    kfs = camera_path.keyframes
    if not kfs:
        return
    hw, hh = sw / 2.0, sh / 2.0
    Rt = R.T
    near = _PREVIEW_NEAR
    cam_pts = [Rt @ (kf.pos - camera_pos) for kf in kfs]

    for i in range(len(cam_pts) - 1):
        clipped = _clip_line_near(cam_pts[i], cam_pts[i + 1], near)
        if clipped is None:
            continue
        a, b = clipped
        ax = (a[0] / (a[2] * half_tan * aspect)) * hw + hw
        ay = (-a[1] / (a[2] * half_tan)) * hh + hh
        bx = (b[0] / (b[2] * half_tan * aspect)) * hw + hw
        by = (-b[1] / (b[2] * half_tan)) * hh + hh
        cv2.line(canvas, (int(ax), int(ay)), (int(bx), int(by)), (40, 200, 255), 2, cv2.LINE_AA)

    for i, (kf, cam) in enumerate(zip(kfs, cam_pts)):
        xc, yc, zc = cam
        if zc <= near:
            continue
        sx = int((xc / (zc * half_tan * aspect)) * hw + hw)
        sy = int((-yc / (zc * half_tan)) * hh + hh)
        is_target = (i == targeted_idx)
        col = (80, 80, 255) if is_target else (40, 200, 255)
        r = 9 if is_target else 7
        cv2.circle(canvas, (sx, sy), r + 3, (25, 25, 25), -1, cv2.LINE_AA)
        cv2.circle(canvas, (sx, sy), r, col, 3 if is_target else 2, cv2.LINE_AA)
        cv2.line(canvas, (sx - 3, sy), (sx + 3, sy), col, 1, cv2.LINE_AA)
        cv2.line(canvas, (sx, sy - 3), (sx, sy + 3), col, 1, cv2.LINE_AA)
        _draw_text_shadow(canvas, str(i + 1), (sx + r + 3, sy - r - 3), scale=0.42)

# =============================================================================
# 14c) HUD (camera-viewfinder style) -- replaces the old title bar: a
#      crosshair + 4-corner viewfinder frame, and HUD text placed in the 4
#      screen corners (no background, just a drop shadow for readability).
#      Uses cv2.putText instead of pygame fonts.
# =============================================================================

_HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
_HUD_SCALE = 0.5
_HUD_SCALE_SMALL = 0.42
_HUD_THICK = 1
_HUD_LINE_H = 20


def _text_width(text, scale=_HUD_SCALE, thickness=_HUD_THICK):
    (w, _), _ = cv2.getTextSize(text, _HUD_FONT, scale, thickness)
    return w


def _draw_text_shadow(canvas, text, pos, color=(225, 235, 235), scale=_HUD_SCALE):
    """color is given as (R,G,B) 0-255 for parity with the rest of the file;
    converted to BGR for cv2 here. Draws a 1px black shadow then the text."""
    bgr = (int(color[2]), int(color[1]), int(color[0]))
    x, y = int(pos[0]), int(pos[1])
    cv2.putText(canvas, text, (x + 1, y + 1), _HUD_FONT, scale, (0, 0, 0), _HUD_THICK, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), _HUD_FONT, scale, bgr, _HUD_THICK, cv2.LINE_AA)


def draw_hud_text(canvas, sw, sh, tl_lines, tr_lines, bl_lines, br_lines):
    """Draws 4 groups of info in the 4 screen corners, no background (just
    text + a dark shadow so it stays legible over a bright background)."""
    pad = 12
    line_h = _HUD_LINE_H
    # cv2.putText anchors at the text BASELINE (bottom-left), unlike pygame's
    # top-left anchor -- offset y by one line height so top-anchored lines
    # still read top-to-bottom in the same visual order as before.

    for i, line in enumerate(tl_lines):
        _draw_text_shadow(canvas, line, (pad, pad + (i + 1) * line_h - 4))

    for i, line in enumerate(bl_lines):
        y = sh - pad - (len(bl_lines) - i - 1) * line_h
        _draw_text_shadow(canvas, line, (pad, y))

    for i, line in enumerate(tr_lines):
        w = _text_width(line)
        _draw_text_shadow(canvas, line, (sw - pad - w, pad + (i + 1) * line_h - 4))

    for i, line in enumerate(br_lines):
        w = _text_width(line)
        y = sh - pad - (len(br_lines) - i - 1) * line_h
        _draw_text_shadow(canvas, line, (sw - pad - w, y))


def draw_camera_viewfinder(canvas, sw, sh, cx, cy, focused=False):
    """Draws the outer 4-corner + crosshair viewfinder frame (camera-style):
    - 4 screen corners shaped like an L (like an autofocus box on a camera).
    - a center crosshair with a gap (no line straight through the center) + 4 small ticks.
    - a faint focus ring around the center, brightens right after autofocusing (middle mouse before; now the F key)."""
    col = (220, 230, 230)
    corner_len = max(18, int(min(sw, sh) * 0.045))
    margin = 10
    thick = 2

    corners = [
        ((margin, margin), (1, 1)),                     # top-left
        ((sw - margin, margin), (-1, 1)),                # top-right
        ((margin, sh - margin), (1, -1)),                # bottom-left
        ((sw - margin, sh - margin), (-1, -1)),          # bottom-right
    ]
    for (x, y), (sxn, syn) in corners:
        cv2.line(canvas, (x, y), (x + sxn * corner_len, y), col, thick, cv2.LINE_AA)
        cv2.line(canvas, (x, y), (x, y + syn * corner_len), col, thick, cv2.LINE_AA)

    # Viewfinder-style crosshair: a gap in the middle, 4 tick marks around the center
    gap = 6
    arm = 9
    for dxn, dyn in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        x0 = cx + dxn * gap; y0 = cy + dyn * gap
        x1 = cx + dxn * (gap + arm); y1 = cy + dyn * (gap + arm)
        cv2.line(canvas, (x0, y0), (x1, y1), col, 2, cv2.LINE_AA)

    ring_r = 16 if not focused else 20
    ring_col = (255, 255, 255) if focused else (190, 200, 200)
    cv2.circle(canvas, (cx, cy), ring_r, ring_col, 1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 1, col, -1, cv2.LINE_AA)


def _fmt_duration(seconds):
    seconds = max(0.0, seconds)
    m, s = divmod(seconds, 60.0)
    h, m = divmod(int(m), 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:04.1f}"
    if m > 0:
        return f"{m:d}:{s:04.1f}"
    return f"{s:.1f}s"


def draw_render_progress(canvas, sw, sh, done, total, elapsed, eta, label="RENDERING"):
    """Draws a progress bar + %/elapsed/ETA in the middle of the screen --
    used while rendering the final image (F5 / Shift+F5) so the user can see
    how much longer it will take instead of a frozen/blank screen while waiting."""
    frac = (done / total) if total else 0.0
    frac = max(0.0, min(1.0, frac))

    canvas[:] = (18, 14, 14)  # BGR

    bar_w = int(sw * 0.62)
    bar_h = 22
    bar_x = (sw - bar_w) // 2
    bar_y = sh // 2 - bar_h // 2

    title = f"{label} -- rendering the final image..."
    tw = _text_width(title)
    _draw_text_shadow(canvas, title, (sw // 2 - tw // 2, bar_y - 34))

    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (65, 55, 55), -1)
    fill_w = max(0, min(bar_w, int(bar_w * frac)))
    if fill_w > 0:
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (245, 175, 90), -1)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (205, 210, 210), 2)

    info = (f"{frac * 100:5.1f}%   sample {done}/{total}   "
            f"Elapsed: {_fmt_duration(elapsed)}   ETA: {_fmt_duration(eta)}")
    iw = _text_width(info)
    _draw_text_shadow(canvas, info, (sw // 2 - iw // 2, bar_y + bar_h + 22))

    hint = "(the window will return to the preview automatically when done)"
    hw2 = _text_width(hint, scale=_HUD_SCALE_SMALL)
    _draw_text_shadow(canvas, hint, (sw // 2 - hw2 // 2, bar_y + bar_h + 46), color=(155, 160, 160),
                       scale=_HUD_SCALE_SMALL)

    cv2.imshow(WINDOW_NAME, canvas)
    cv2.waitKey(1)

# =============================================================================
# 15) Scene file (JSON) import/export -- item 3/4 in the module docstring.
#
#     A scene file bundles EVERYTHING needed to reproduce a render on a
#     different machine in one .json: geometry (as a replayable op log,
#     see Scene.ops), camera position/rotation/FOV, lights, spotlights,
#     background/skybox, post-FX settings, bounce counts, and the camera
#     keyframe path -- plus, by default, every image file referenced
#     (textures used by add_image, and the background/skybox image) is
#     embedded as base64 so a single .json is enough; no need to also copy
#     loose .png files around. Use --no-embed-assets to keep the JSON
#     small and reference original file paths instead (only useful when
#     running on the SAME machine / a shared filesystem).
# =============================================================================

SCENE_FILE_FORMAT = "raytracer_scene_v1"


def _embed_asset(assets: dict, path):
    """Reads `path` and stores it base64-encoded in `assets[path]`, if it
    exists and hasn't already been embedded. Silently no-ops if missing
    (a warning is printed elsewhere when the file actually fails to load)."""
    if not path or path in assets or not os.path.exists(path):
        return
    with open(path, 'rb') as f:
        data = f.read()
    assets[path] = {
        'filename': os.path.basename(path),
        'data_b64': base64.b64encode(data).decode('ascii'),
    }


def _resolve_asset_path(path, asset_dir):
    """If `path` was extracted into asset_dir (see load_scene_file), returns
    the extracted local path; otherwise returns `path` unchanged (so scene
    files exported with --no-embed-assets still work on the same machine)."""
    if not path:
        return path
    candidate = os.path.join(asset_dir, os.path.basename(path))
    return candidate if os.path.exists(candidate) else path


def save_scene_file(path, scene: Scene, tracer: RayTracer, post_fx=None,
                     camera_path: CameraPath = None, embed_assets=True):
    """Writes a full scene (geometry + camera + lights + spotlights +
    background + post-FX + bounce counts + camera keyframe path) to a
    single JSON file at `path`. Image assets are embedded as base64 by
    default (embed_assets=True) so the file is portable to another machine."""
    assets = {} if embed_assets else None
    data = {
        'format': SCENE_FILE_FORMAT,
        'camera': tracer.camera_dict(),
        'fov': tracer.fov_deg,
        'max_bounce': tracer.max_bounce,
        'live_max_bounce': tracer.live_max_bounce,
        'ambient': tracer.ambient,
        'specular_k': tracer.specular_k,
        'shininess': tracer.shininess,
        'sky_light_strength': tracer.sky_light_strength,
        'caustic_strength': tracer.caustic_strength,
        'exposure': tracer.exposure,
        'eye_adapt_enabled': bool(tracer.eye_adapt_enabled),
        'eye_adapt_speed': tracer.eye_adapt_speed,
        'caustics_enabled': bool(tracer.caustics_enabled),
        'background': tracer.background.to_dict(assets=assets),
        'lights': [lt.to_dict() for lt in tracer.lights],
        'spotlights': [sl.to_dict() for sl in tracer.spotlights],
        'post_fx': dict(post_fx) if post_fx is not None else dict(DEFAULT_POST_FX),
        'camera_path': camera_path.to_list() if camera_path is not None else [],
        'handheld_shake': camera_path.handheld_shake if camera_path is not None else 0.0,
        'handheld_seed': camera_path.handheld_seed if camera_path is not None else 0.0,
        'footstep_shake': camera_path.footstep_shake if camera_path is not None else 1.0,
        'scene': scene.to_dict(assets=assets),
    }
    if assets is not None:
        data['assets'] = assets
    with open(path, 'w') as f:
        json.dump(data, f)
    n_assets = len(assets) if assets else 0
    print(f"Scene exported: {path} ({n_assets} embedded asset(s))")


def load_scene_file(path):
    """Reads a scene JSON written by save_scene_file(). Returns a dict with
    keys: scene (Scene), camera (dict), fov, max_bounce, live_max_bounce,
    ambient, specular_k, shininess, sky_light_strength, caustic_strength,
    exposure, eye_adapt_enabled, eye_adapt_speed, caustics_enabled,
    background (Background), lights (list[Light]), spotlights
    (list[SpotLight]), post_fx (dict), camera_path (CameraPath, with
    handheld_shake/handheld_seed already applied to it). Any embedded
    assets are extracted next to the scene file, into
    '<scene_file_stem>_assets/'."""
    with open(path, 'r') as f:
        data = json.load(f)
    if data.get('format') != SCENE_FILE_FORMAT:
        print(f"Warning: '{path}' doesn't declare the expected format "
              f"'{SCENE_FILE_FORMAT}' -- attempting to load anyway.")

    asset_dir = None
    assets = data.get('assets')
    if assets:
        asset_dir = os.path.splitext(os.path.abspath(path))[0] + "_assets"
        os.makedirs(asset_dir, exist_ok=True)
        for orig_path, entry in assets.items():
            out_path = os.path.join(asset_dir, entry['filename'])
            if not os.path.exists(out_path):
                with open(out_path, 'wb') as f:
                    f.write(base64.b64decode(entry['data_b64']))
        print(f"Extracted {len(assets)} embedded asset(s) to: {asset_dir}")

    scene = Scene.from_dict(data.get('scene', {}), asset_dir=asset_dir)
    background = Background.from_dict(data.get('background', {}), asset_dir=asset_dir)
    lights = [Light.from_dict(d) for d in data.get('lights', [])]
    spotlights = [SpotLight.from_dict(d) for d in data.get('spotlights', [])]
    post_fx = dict(DEFAULT_POST_FX)
    post_fx.update(data.get('post_fx', {}))
    camera_path = CameraPath.from_list(data.get('camera_path', []))
    camera_path.handheld_shake = data.get('handheld_shake', 0.0)
    camera_path.handheld_seed = data.get('handheld_seed', 0.0)
    camera_path.footstep_shake = data.get('footstep_shake', 1.0)

    return {
        'scene': scene, 'camera': data.get('camera', {'pos': [0, 6, -30], 'yaw': 0.0, 'pitch': 0.0}),
        'fov': data.get('fov', 65), 'max_bounce': data.get('max_bounce', DEFAULT_MAX_BOUNCE),
        'live_max_bounce': data.get('live_max_bounce', LIVE_MAX_BOUNCE),
        'ambient': data.get('ambient', 0.12), 'specular_k': data.get('specular_k', 0.6),
        'shininess': data.get('shininess', 64.0),
        'sky_light_strength': data.get('sky_light_strength', 0.5),
        'caustic_strength': data.get('caustic_strength', 1.0),
        'exposure': data.get('exposure', 1.0),
        'eye_adapt_enabled': data.get('eye_adapt_enabled', True),
        'eye_adapt_speed': data.get('eye_adapt_speed', 1.4),
        'caustics_enabled': data.get('caustics_enabled', True),
        'background': background, 'lights': lights, 'spotlights': spotlights,
        'post_fx': post_fx, 'camera_path': camera_path,
    }


# =============================================================================
# 16) MC module -- reads .schem files, merges blocks into larger boxes
#     (greedy meshing), water-aware.
# =============================================================================

_MC_COLORS = {
    'white': (249, 255, 254),
    'orange': (249, 128, 29),
    'magenta': (199, 78, 189),
    'light_blue': (58, 179, 218),
    'yellow': (254, 216, 61),
    'lime': (128, 199, 31),
    'pink': (243, 139, 170),
    'gray': (71, 79, 82),
    'light_gray': (157, 157, 151),
    'cyan': (22, 156, 156),
    'purple': (137, 50, 184),
    'brown': (131, 84, 50),
    'green': (94, 124, 22),
    'red': (176, 46, 38),
    'black': (29, 29, 33),
    # Minecraft's water color (kept in sync with the demo scene)
    'water': (120, 220, 240),
}

_MC_MATERIALS = {
    'concrete': (0.36, 0.02, 0.0, 1.0),
    'wool':     (0.05, 0.60, 0.0, 1.0),
    'glass':    (0.01, 0.00, 0.2, 1.2),
    # Watertype: (reflection_k, roughness, transparency, ior)
    # Transparency kept in sync at 0.02 with the demo scene
    'water':    (0.00, 0.00, 0.02, 1.3),
}


def _parse_mc_block(name):
    n = name.replace('minecraft:', '')

    # --- 1. CHECK FOR A WATER BLOCK ---
    if n == 'water' or n == 'flowing_water' or 'water[' in n or 'flowing_water[' in n:
        refl, rough, transp, ior = _MC_MATERIALS['water']
        color = _MC_COLORS['water']
        # Also return the 'water' identifier to distinguish it from a regular box
        return color, refl, rough, transp, ior, 'water'

    # --- 2. CHECK FOR OTHER MATERIALS ---
    mat = None
    for m in ('concrete', 'wool', 'glass'):
        if n == m or n.endswith('_' + m):
            mat = m
            break
    if mat is None:
        return None

    if mat == 'glass':
        color_key = None
        if n.endswith('_stained_glass'):
            color_key = n[:-len('_stained_glass')]
        color = _MC_COLORS.get(color_key, (250, 253, 255))
    else:
        suffix = '_' + mat
        color_key = n[:-len(suffix)] if n.endswith(suffix) else None
        if color_key not in _MC_COLORS:
            return None
        color = _MC_COLORS[color_key]

    refl, rough, transp, ior = _MC_MATERIALS[mat]
    return color, refl, rough, transp, ior, mat


def load_schematic(path, scene, offset=(0, 0, 0)):
    try:
        import nbtlib
    except ImportError:
        print("ERROR: nbtlib is not installed. Run: pip install nbtlib")
        return scene

    nbt = nbtlib.load(path)
    schem = nbt.get('Schematic', nbt)

    width = int(schem['Width'])
    height = int(schem['Height'])
    length = int(schem['Length'])

    blocks = schem['Blocks']
    palette = blocks['Palette']
    data = blocks['Data']

    palette_map = {int(v): k for k, v in palette.items()}

    grid = {}
    total_raw_blocks = 0

    for y in range(height):
        for z in range(length):
            for x in range(width):
                idx = (y * length + z) * width + x
                state = palette_map[int(data[idx])]
                name = state.split('[')[0]
                result = _parse_mc_block(name)
                if result is not None:
                    grid[(x, y, z)] = result
                    total_raw_blocks += 1

    ox, oy, oz = offset
    cx_offset = ox - width / 2.0
    cz_offset = oz - length / 2.0

    visited = set()
    boxes_added = 0

    for y in range(height):
        for z in range(length):
            for x in range(width):
                pos = (x, y, z)
                if pos in visited or pos not in grid:
                    continue
                mat_info = grid[pos]

                dx = 1
                while (x + dx < width and
                       (x + dx, y, z) not in visited and
                       grid.get((x + dx, y, z)) == mat_info):
                    dx += 1

                dz = 1
                can_expand_z = True
                while can_expand_z and (z + dz < length):
                    for kx in range(dx):
                        check_pos = (x + kx, y, z + dz)
                        if check_pos in visited or grid.get(check_pos) != mat_info:
                            can_expand_z = False
                            break
                    if can_expand_z:
                        dz += 1

                dy = 1
                can_expand_y = True
                while can_expand_y and (y + dy < height):
                    for kx in range(dx):
                        for kz in range(dz):
                            check_pos = (x + kx, y + dy, z + kz)
                            if check_pos in visited or grid.get(check_pos) != mat_info:
                                can_expand_y = False
                                break
                        if not can_expand_y:
                            break
                    if can_expand_y:
                        dy += 1

                for ky in range(dy):
                    for kz in range(dz):
                        for kx in range(dx):
                            visited.add((x + kx, y + ky, z + kz))

                real_x = cx_offset + x + dx / 2.0
                real_y = oy + y + dy / 2.0
                real_z = cz_offset + (length - (z + dz)) + dz / 2.0

                color, refl, rough, transp, bior, mat_type = mat_info

                # For water blocks, use scene.add_water so it matches the demo's format
                if mat_type == 'water':
                    scene.add_water(
                        center=(real_x, real_y, real_z),
                        size=(dx, dy, dz),
                        color=color,
                        transparency=transp
                    )
                else:
                    scene.add_box(
                        (real_x, real_y, real_z),
                        (dx, dy, dz),
                        color,
                        roughness=rough,
                        transparency=transp,
                        ior=bior,
                        reflection_k=refl
                    )
                boxes_added += 1

    print(f"Schematic loaded: {path}")
    print(f" -> Optimized {total_raw_blocks} small blocks into {boxes_added} larger boxes "
          f"({(1 - boxes_added/max(total_raw_blocks, 1))*100:.1f}% fewer objects).")
    return scene


# =============================================================================
# 17) Demo scene -- illustrates rotated boxes (add_box rotation=...),
#     image planes (add_image), multiple light sources, and a water block.
# =============================================================================

def build_demo_scene():
    scene = Scene()
    # Floor
    scene.add_box((0, -1, 0), (60, 1, 60), (200, 200, 205), roughness=0.35)
    # Back wall
    scene.add_box((0, 10, 20), (60, 20, 1), (230, 230, 235), roughness=0.6)
    # Glass block (transparent, glass IOR)
    scene.add_box((-6, 4, 0), (6, 8, 6), (235, 245, 255), roughness=0.0,
                  transparency=0.95, ior=1.5)
    # Colored glass block
    scene.add_box((4, 3, -6), (5, 6, 5), (180, 60, 60), roughness=0.0,
                  transparency=0.9, ior=1.45)
    # Gold metal block (tinted mirror)
    scene.add_box((8, 3, 4), (6, 6, 6), (230, 200, 90), roughness=0.02,
                  reflection_k=0.9)
    # Slightly rough silver metal block (glossy)
    scene.add_box((-8, 2.5, 8), (5, 5, 5), (210, 210, 220), roughness=0.15,
                  reflection_k=0.8)
    # Ordinary red solid block
    scene.add_box((0, 2, -10), (4, 4, 4), (200, 60, 60), roughness=0.9)

    # --- Demo: a ROTATED box ---
    scene.add_box((-2, 3, 14), (4, 4, 4), (90, 140, 220), roughness=0.25,
                  rotation=(math.radians(30), math.radians(15), 0))

    # --- Demo: a textured IMAGE plane ---
    demo_tex = os.path.join(os.path.dirname(os.path.abspath(__file__)), "miku_wonder.png")
    try:
        scene.add_image((0, 5.2, 8), (4.9, 12), demo_tex,
                         rotation=(0, 0, 0),
                         roughness=0.15, reflection_k=0.05)
    except FileNotFoundError as e:
        print(f"(Skipping demo texture: {e} -- replace with your own image path in build_demo_scene())")

    # --- WATER BLOCK ---
    # A light-blue transparent water block, IOR = 1.333
    scene.add_water(
        center=(0, 0.5, 0),
        size=(60, 2, 60),
        color=(120, 220, 240),
        transparency=0.02
    )

    # A red solid block half-submerged, to check refraction/distortion
    scene.add_box((0, 1.5, -2), (2, 4, 2), (220, 50, 50), roughness=0.5)

    return scene


def build_mc_scene(schem_path="lim_c.schem", decal_path="miku_wonder.png"):
    """Loads a Minecraft .schem file into a scene, robustly: if the file (or
    nbtlib) isn't available, falls back to the plain demo scene instead of
    crashing -- important if you're running this on a machine that doesn't
    have the original author's .schem/.png files."""
    if not os.path.exists(schem_path):
        print(f"'{schem_path}' not found -- using the built-in demo scene instead.")
        return build_demo_scene()
    scene = load_schematic(schem_path, Scene())
    if os.path.exists(decal_path):
        scene.add_image((0, 9.95, 4.48), (1.63, 4), decal_path,
                         rotation=(0, 0, 0),
                         roughness=0.15, reflection_k=0.05)
    return scene

# =============================================================================
# 18) Keyboard input helpers (cv2 has no true "key held" state like pygame's
#     key.get_pressed(), and its key CODES for arrows/function keys are not
#     consistent across OS/backends the way pygame's were). See the module
#     docstring (item 2) for the approach used here:
#       - movement/look keys are treated as "held" if seen within the last
#         KEY_HOLD_WINDOW seconds (relies on OS keyboard auto-repeat)
#       - one-shot actions (toggles, menus) are debounced so OS auto-repeat
#         doesn't fire them many times per physical key press
#     Run with --print-keys once if arrow-key look doesn't respond on your
#     platform -- it prints the raw code of every key you press so you can
#     add it to ARROW_CODES below.
# =============================================================================

ARROW_CODES = {
    'left':  {65361}, # 2424832, 81, 63234, 2, 100
    'up':    {65362}, # 2490368, 82, 63232, 0, 101
    'right': {65363}, # 2555904, 83, 63235, 3, 102
    'down':  {65364}, # 2621440, 84, 63233, 1, 103
}                     # ^ those values caused problems on some platforms, so only the 4 "official" cv2.waitKeyEx() codes are kept here.


class InputState:
    def __init__(self):
        self.last_seen = {}
        self.last_trigger = {}

    def note_held(self, name):
        self.last_seen[name] = time.time()

    def is_held(self, name):
        t = self.last_seen.get(name)
        return t is not None and (time.time() - t) < KEY_HOLD_WINDOW

    def one_shot(self, name, debounce=0.35):
        """Returns True at most once per `debounce` seconds for a given
        action name -- call this exactly when the key's raw event arrives."""
        now = time.time()
        t = self.last_trigger.get(name)
        if t is not None and (now - t) < debounce:
            return False
        self.last_trigger[name] = now
        return True

def classify_key(raw):
    """raw: value from cv2.waitKeyEx(). Returns ('arrow', 'left'/'up'/...),
    ('char', lowercase_char), or (None, None)."""
    if raw == -1:
        return None, None
    for name, codes in ARROW_CODES.items():
        if raw in codes:
            return 'arrow', name
    ascii_code = raw & 0xFF
    if 32 <= ascii_code < 127:
        return 'char', chr(ascii_code).lower()
    if ascii_code in (27, 13, 10, 9, 8):
        return 'char', {27: 'esc', 13: 'enter', 10: 'enter', 9: 'tab', 8: 'backspace'}[ascii_code]
    return None, None


# =============================================================================
# 19) Keyframe options "menu" -- keyboard-driven replacement for the old
#     mouse-click popup (O key). Blocks in its own small render loop until a
#     choice (or Esc) is made; digits 1-5 choose an option, matching the
#     terminal-input fallback that already existed for entering an exact speed.
# =============================================================================

def keyframe_options_menu_cv(canvas, sw, sh, kf, idx):
    labels = ["1) Speed +0.5 units/s", "2) Speed -0.5 units/s",
              "3) Enter an exact speed (terminal)", "4) Delete this keyframe", "5) Close (Esc)"]
    result = ('cancel', None)
    choosing = True
    while choosing:
        canvas[:] = (22, 18, 18)
        title = f"Camera keyframe #{idx + 1}  --  current speed: {kf.speed:.2f} units/second"
        tw = _text_width(title)
        _draw_text_shadow(canvas, title, (sw // 2 - tw // 2, sh // 2 - 90))
        for i, label in enumerate(labels):
            lw = _text_width(label)
            y = sh // 2 - 40 + i * 28
            _draw_text_shadow(canvas, label, (sw // 2 - lw // 2, y))
        hint = "Press 1-5 to choose -- Esc to close"
        hwid = _text_width(hint, scale=_HUD_SCALE_SMALL)
        _draw_text_shadow(canvas, hint, (sw // 2 - hwid // 2, sh // 2 - 40 + len(labels) * 28 + 20),
                           color=(155, 160, 160), scale=_HUD_SCALE_SMALL)
        cv2.imshow(WINDOW_NAME, canvas)
        raw = cv2.waitKeyEx(30)
        if raw == -1:
            continue
        ascii_code = raw & 0xFF
        if ascii_code == 27:
            result, choosing = ('cancel', None), False
        elif ascii_code == ord('1'):
            result, choosing = ('speed_delta', 0.5), False
        elif ascii_code == ord('2'):
            result, choosing = ('speed_delta', -0.5), False
        elif ascii_code == ord('3'):
            result, choosing = ('speed_prompt', None), False
        elif ascii_code == ord('4'):
            result, choosing = ('delete', None), False
        elif ascii_code == ord('5'):
            result, choosing = ('cancel', None), False

    if result[0] == 'speed_prompt':
        try:
            raw_in = input(f"Enter a new speed for keyframe #{idx + 1} "
                            f"(world units/second, currently {kf.speed:.2f}): ")
            result = ('set_speed', max(0.01, float(raw_in)))
        except Exception:
            print("Invalid value -- speed left unchanged.")
            result = ('cancel', None)
    return result


# =============================================================================
# 20) Command-line interface
# =============================================================================

def _parse_wh(s):
    """Parses a WIDTHxHEIGHT resolution string. Accepts 'x'/'X' (the
    documented format, e.g. '1920x1080'), but also ',' or whitespace as a
    separator (e.g. '1920,1080' or '1920 1080') since that's an easy typo
    to make coming from other tools -- gives a clear error either way."""
    parts = re.split(r'[xX,\s]+', s.strip())
    parts = [p for p in parts if p]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"invalid resolution '{s}' -- expected WIDTHxHEIGHT, e.g. 1920x1080")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid resolution '{s}' -- expected WIDTHxHEIGHT, e.g. 1920x1080")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Raytracer v9 -- interactive (cv2 window) or headless rendering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--scene', type=str, default=None,
                    help="Load a scene from a JSON file (see --export-scene) instead of the built-in demo scene.")
    p.add_argument('--export-scene', type=str, default=None,
                    help="Build the scene, save it to this JSON file, then exit (no window, no render).")
    p.add_argument('--no-embed-assets', action='store_true',
                    help="When exporting, reference image files by path instead of embedding them as base64.")
    p.add_argument('--mc-schem', type=str, default=None,
                    help="Load a Minecraft .schem file (used only if --scene is not given).")
    p.add_argument('--headless', action='store_true',
                    help="Render without opening a window, then exit. Combine with --video for a video render.")
    p.add_argument('--video', action='store_true',
                    help="With --headless: render a video along the scene's camera keyframe path instead of a still.")
    p.add_argument('--output', type=str, default="raytrace_v9.png",
                    help="Output image path for --headless still renders.")
    p.add_argument('--video-output', type=str, default="raytrace_v9_video.mp4",
                    help="Output video path for --headless --video renders.")
    p.add_argument('--resolution', type=str, default=None, help="Final-render resolution, e.g. 1920x1080.")
    p.add_argument('--video-resolution', type=str, default=None, help="Video resolution, e.g. 640x360.")
    p.add_argument('--samples', type=int, default=32, help="Sample count for --headless still renders.")
    p.add_argument('--preview-res', type=str, default=None, help="Interactive window size, e.g. 640x480.")
    p.add_argument('--live-res', type=str, default=None, help="Interactive live-raytrace resolution, e.g. 128x96.")
    p.add_argument('--live-max-bounce', type=int, default=None,
                    help="Bounce cap for interactive/live rendering (see the perf note in the module docstring).")
    p.add_argument('--print-keys', action='store_true',
                    help="Debug: print the raw code of every key pressed in the interactive window.")
    p.add_argument('--camera-data', type=str, default=None,
                    help="Load a sensor_record JSON file and drive the camera position/rotation "
                         "from it instead of the scene's camera keyframes (mutually exclusive with "
                         "camera keyframes -- any existing keyframes on the loaded/built scene are "
                         "dropped). Position is x/y/z (m), rotation is x/y/z (deg, mapped to "
                         "pitch/yaw/roll). See --camera-multiplier / --camera-offset / --camera-sync.")
    p.add_argument('--camera-multiplier', type=float, default=1.0,
                    help="With --camera-data: multiplies every sample's position (x/y/z) by this "
                         "factor before applying --camera-offset.")
    p.add_argument('--camera-offset', type=str, default=None,
                    help="With --camera-data: adds this x,y,z offset (world units) to every sample's "
                         "position AFTER --camera-multiplier, e.g. --camera-offset 0,5,0.")
    p.add_argument('--camera-sync', action='store_true',
                    help="With --camera-data --headless --video: use every recorded sample as exactly "
                         "one output video frame (instead of resampling the recording at --fps-spaced "
                         "times). The video's fps is unchanged -- only which timestamps become frames.")
    p.add_argument('--camera-get-fps', action='store_true',
                    help="With --camera-data: print the sensor recording's average fps (from its "
                         "sample timestamps) and use it as the video's fps instead of the built-in default.")
    p.add_argument('--camera-stream', type=str, default=None,
                    help="Drive the camera from a LIVE UDP pose stream (HOST:PORT, e.g. 0.0.0.0:9999) "
                         "instead of a prerecorded --camera-data file -- see LiveCameraStream's "
                         "docstring for the wire format. Mutually exclusive with --camera-data and "
                         "with camera keyframes. Not meaningful with --headless --video (there's no "
                         "fixed length to render); use it in the interactive window instead.")
    p.add_argument('--camera-stream-smoothing', type=float, default=0.5,
                    help="With --camera-stream: 0 = snap straight to each new packet, 1 = heavy "
                         "smoothing between the last 2 packets (reduces jitter, adds a little lag).")
    p.add_argument('--handheld-shake', type=float, default=0.0,
                    help="Adds fake handheld camera sway/jitter on top of camera KEYFRAME paths "
                         "(P/O/9 -- ignored for --camera-data/--camera-stream, which already carry "
                         "real motion). 0 = off, ~0.3-0.6 = handheld, higher = walking/running.")
    p.add_argument('--footstep-shake', type=float, default=None,
                    help="Master multiplier on the FOOTSTEP-driven component of --handheld-shake "
                         "specifically (the sharper per-step jolt/nod/lean while actually moving, "
                         "vs. the constant idle sway) -- 1.0 (default) = normal, 0 = idle sway only, "
                         ">1 = exaggerated footsteps. Has no effect if --handheld-shake is 0.")
    return p.parse_args(argv)


def _parse_xyz(s):
    parts = [float(x) for x in s.split(',')]
    if len(parts) != 3:
        raise ValueError(f"Expected 'x,y,z', got: {s!r}")
    return tuple(parts)


# =============================================================================
# 21) Main
# =============================================================================

def main(argv=None):
    args = parse_args(argv)

    global PREVIEW_RES, LIVE_RENDER_RES, FINAL_RENDER_RES, MAX_W, MAX_H
    if args.preview_res:
        PREVIEW_RES = _parse_wh(args.preview_res)
    if args.live_res:
        LIVE_RENDER_RES = _parse_wh(args.live_res)
    if args.resolution:
        FINAL_RENDER_RES = _parse_wh(args.resolution)
    MAX_W = max(MAX_W, FINAL_RENDER_RES[0], PREVIEW_RES[0], LIVE_RENDER_RES[0])
    MAX_H = max(MAX_H, FINAL_RENDER_RES[1], PREVIEW_RES[1], LIVE_RENDER_RES[1])

    loaded = None
    if args.scene:
        loaded = load_scene_file(args.scene)
        scene = loaded['scene']
    elif args.mc_schem:
        scene = build_mc_scene(schem_path=args.mc_schem)
    else:
        scene = build_demo_scene()
    print(f"Scene: {len(scene.faces)} faces ({len(scene.boxes)} boxes, {len(scene.quads)} image planes)")

    if loaded is not None:
        bg = loaded['background']
        lights = loaded['lights']
        spotlights = loaded['spotlights']
        fov = loaded['fov']
        max_bounce = loaded['max_bounce']
        live_max_bounce = loaded['live_max_bounce']
        cam_cfg = loaded['camera']
        post_fx = loaded['post_fx']
        camera_path = loaded['camera_path']
        ambient, specular_k, shininess = loaded['ambient'], loaded['specular_k'], loaded['shininess']
        caustics_enabled = loaded['caustics_enabled']
        sky_light_strength = loaded['sky_light_strength']
        caustic_strength = loaded['caustic_strength']
        exposure = loaded['exposure']
        eye_adapt_enabled = loaded['eye_adapt_enabled']
        eye_adapt_speed = loaded['eye_adapt_speed']
    else:
        bg = Background(color=(20, 25, 35), image_path="sky.png", brightness=1.0)
        lights = [
            Light((-18.41, 29.48, -12.73), (255, 251, 235), 2.1),   # key light, slightly warm
            #Light((-1.47, -4.69, 6.94), (128, 202, 235), 1.0),      # fill light, cooler blue
            #Light((-5.88, 16.79, -13.09), (255, 251, 235), 0.3),
        ]
        spotlights = []
        fov = 65
        max_bounce = DEFAULT_MAX_BOUNCE
        live_max_bounce = LIVE_MAX_BOUNCE
        cam_cfg = {'pos': [0.0, 6.0, -30.0], 'yaw': 0.0, 'pitch': 0.0}
        post_fx = dict(DEFAULT_POST_FX)
        camera_path = CameraPath()
        ambient, specular_k, shininess = 0.12, 0.6, 64.0
        caustics_enabled = True
        sky_light_strength, caustic_strength = 0.5, 1.0
        exposure, eye_adapt_enabled, eye_adapt_speed = 1.0, True, 1.4

    if args.live_max_bounce is not None:
        live_max_bounce = args.live_max_bounce

    init_w, init_h = (FINAL_RENDER_RES if args.headless else LIVE_RENDER_RES)
    tracer = RayTracer(scene, width=init_w, height=init_h, fov=fov, max_bounce=max_bounce,
                        background=bg, lights=lights, spotlights=spotlights,
                        live_max_bounce=live_max_bounce)
    tracer.ambient, tracer.specular_k, tracer.shininess = ambient, specular_k, shininess
    tracer.caustics_enabled = caustics_enabled
    tracer.sky_light_strength = sky_light_strength
    tracer.caustic_strength = caustic_strength
    tracer.exposure = exposure
    tracer.eye_adapt_enabled = eye_adapt_enabled
    tracer.eye_adapt_speed = eye_adapt_speed
    tracer.camera_pos = np.array(cam_cfg['pos'], dtype=np.float32)
    tracer.camera_rot = np.array([cam_cfg.get('yaw', 0.0), cam_cfg.get('pitch', 0.0)], dtype=np.float32)

    # --- --camera-data: sensor recording drives the camera instead of ------
    # hand-placed camera keyframes. The two are mutually exclusive: any
    # keyframes the loaded/built scene already had are dropped in favor of
    # the sensor recording, and the "P" add-keyframe key stays a no-op below
    # (see the interactive loop) rather than mixing a pile of new keyframes
    # into camera_path -- exactly as requested.
    if args.handheld_shake:
        camera_path.handheld_shake = float(args.handheld_shake)
    if args.footstep_shake is not None:
        camera_path.footstep_shake = float(args.footstep_shake)

    sensor_data = None
    stream_data = None    # LiveCameraStream (--camera-stream) -- see its docstring
    video_fps = VIDEO_FPS
    if args.camera_data and args.camera_stream:
        print("--camera-data and --camera-stream are mutually exclusive -- ignoring --camera-stream.")
    if args.camera_data:
        cam_multiplier = args.camera_multiplier
        cam_offset = _parse_xyz(args.camera_offset) if args.camera_offset else (0.0, 0.0, 0.0)
        sensor_data = SensorCameraData.load(args.camera_data, multiplier=cam_multiplier, offset=cam_offset)
        if len(camera_path.keyframes) > 0:
            print(f"--camera-data given -- dropping {len(camera_path.keyframes)} existing camera "
                  f"keyframe(s) from the scene in favor of the sensor recording.")
        camera_path = CameraPath()
        print(f"Loaded {len(sensor_data.samples)} camera samples from '{args.camera_data}' "
              f"({sensor_data.total_duration():.2f}s)")
        if args.camera_get_fps:
            avg_fps = sensor_data.average_fps()
            print(f"Sensor recording average fps: {avg_fps:.3f}")
            if avg_fps > 0:
                video_fps = avg_fps
        pos0, yaw0, pitch0, roll0 = sensor_data.sample(0.0)
        tracer.camera_pos = pos0.astype(np.float32)
        tracer.camera_rot = np.array([yaw0, pitch0], dtype=np.float32)
        tracer.camera_roll = roll0
    elif args.camera_stream:
        if args.headless and args.video:
            print("--camera-stream doesn't have a fixed length, so it can't drive a --headless "
                  "--video render -- use it in the interactive window instead. Ignoring it.")
        else:
            host, _, port_s = args.camera_stream.rpartition(':')
            host = host or "0.0.0.0"
            stream_data = LiveCameraStream(host, int(port_s), smoothing=args.camera_stream_smoothing)
            if len(camera_path.keyframes) > 0:
                print(f"--camera-stream given -- dropping {len(camera_path.keyframes)} existing "
                      f"camera keyframe(s) from the scene in favor of the live stream.")
            camera_path = CameraPath()
            print(f"Listening for live camera pose packets on {host}:{port_s} (UDP) -- "
                  f"see LiveCameraStream's docstring for the wire format. Waiting for the first "
                  f"packet before the camera moves...")
    elif args.camera_get_fps:
        print("--camera-get-fps has no effect without --camera-data.")

    if args.export_scene:
        save_scene_file(args.export_scene, scene, tracer, post_fx=post_fx,
                         camera_path=camera_path, embed_assets=not args.no_embed_assets)
        return

    if args.headless:
        if args.video:
            active_path = sensor_data if sensor_data is not None else camera_path
            has_path = sensor_data is not None or len(camera_path.keyframes) > 0
            if not has_path:
                print("The scene has no camera keyframes -- nothing to move the camera along, "
                      "rendering a single still image instead.")
                tracer.set_resolution(*FINAL_RENDER_RES)
                tracer.render_to_file(args.output, samples=args.samples, post_fx=post_fx)
            else:
                video_res = _parse_wh(args.video_resolution) if args.video_resolution else VIDEO_RES
                tracer.render_video(active_path, out_path=args.video_output,
                                     resolution=video_res, post_fx=post_fx,
                                     fps=video_fps, camera_sync=args.camera_sync)
        else:
            tracer.set_resolution(*FINAL_RENDER_RES)
            tracer.render_to_file(args.output, samples=args.samples, post_fx=post_fx)
        return

    # --- Interactive mode -------------------------------------------------
    WIN_W, WIN_H = PREVIEW_RES
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)

    camera_pos = tracer.camera_pos.astype(np.float64)
    camera_rot = tracer.camera_rot.astype(np.float64)
    camera_roll = float(tracer.camera_roll)  # radians -- 0 unless --camera-data supplied one at t=0
    roll_speed = 0.015   # radians per "held" tick (,/. keys) -- matches look_speed
    move_speed = 0.6
    look_speed = 0.015   # radians per "held" tick (arrow keys) -- replaces mouse_sens
    lines_mode = True
    live_render = False
    selected_light = 0          # index into the COMBINED list [lights..., spotlights...]
    autofocus_flash = 0.0        # countdown (seconds) for the focus-ring flash effect

    replaying = False           # replaying the camera path/sensor data (I key), not rendering
    replay_start_time = 0.0
    is_live_stream = stream_data is not None
    has_camera_data = sensor_data is not None or is_live_stream
    if sensor_data is not None:
        active_path = sensor_data
    elif is_live_stream:
        active_path = stream_data
    else:
        active_path = camera_path

    def _camera_data_count_str():
        if sensor_data is not None:
            return f"{len(sensor_data.samples)} samples"
        if is_live_stream:
            return f"{stream_data.n_received} samples received (live)"
        return "0 samples"

    prog = ProgressiveRenderer(tracer, samples_per_frame=1, post_fx=post_fx)
    preview_geo = precompute_preview_geometry(scene)
    inp = InputState()

    print("Controls: WASD move | arrow keys look | Q/E move down/up | L toggle live raytrace")
    print("Tab: cycle selected light/spotlight | K: place selected light/spot at the camera (+direction)")
    print("[ / ]: narrow / widen a spotlight's cone (when a spotlight is selected)")
    print("F: autofocus DoF on whatever is under the crosshair | - / =: nudge DoF focus distance")
    print("1: toggle wireframe outline | C: toggle caustics")
    print("2 DoF | 3 chromatic aberration | 4 lens flare | 6 VHS effect | 7 motion blur (video only)")
    print("0: fisheye lens | B: bloom | Y: eye adaptation/auto exposure | G: continuous autofocus (video only)")
    print("H: cycle handheld camera shake (keyframe paths only)")
    print("5: render final image | 8: quick render (lower quality)")
    print(", / . : roll camera left/right | / : reset roll to 0")
    if is_live_stream:
        print(f"Live camera stream active on {stream_data.host}:{stream_data.port} -- camera "
              f"keyframes (P/O/J) are disabled; I follows the live pose directly (no fixed replay "
              f"length -- press I again to stop following).")
    elif has_camera_data:
        print(f"Camera data loaded ({_camera_data_count_str()}) -- camera keyframes (P/O/J) "
              f"are disabled; I replays the sensor recording (roll resets to 0 when it ends/stops).")
    else:
        print("P: add a camera keyframe here | J: add a HOLD keyframe (stay put, can still turn) | "
              "O: edit/delete the targeted keyframe (nearest to crosshair)")
    print("I: replay the camera path (no render) | 9: render VIDEO along the path")
    print("X: export the current scene/camera/lights/etc to a .json file | Esc: quit")

    running = True
    while running:
        keys = inp  # alias for readability below

        any_input = False

        if replaying:
            if is_live_stream:
                # A live stream never "ends" -- keep following the freshest
                # pose every frame until the user presses I again to stop.
                live_pose = active_path.sample(None)
                if live_pose is not None:
                    rp_pos, rp_yaw, rp_pitch, rp_roll = live_pose
                    camera_pos[:] = rp_pos
                    camera_rot[0] = rp_yaw
                    camera_rot[1] = rp_pitch
                    camera_roll = rp_roll
                any_input = True
            else:
                elapsed_replay = time.time() - replay_start_time
                if has_camera_data:
                    replay_total = active_path.total_duration()
                else:
                    replay_total = (VIDEO_DURATION if len(camera_path.keyframes) <= 1
                                     else camera_path.total_duration())
                if elapsed_replay >= replay_total:
                    replaying = False
                    camera_roll = 0.0  # reset roll after a --camera-data replay ends (see module notes)
                    print("Path replay finished.")
                else:
                    rp_pos, rp_yaw, rp_pitch, rp_roll = active_path.sample(elapsed_replay)
                    camera_pos[:] = rp_pos
                    camera_rot[0] = rp_yaw
                    camera_rot[1] = rp_pitch
                    camera_roll = rp_roll
                    any_input = True
        else:
            R = camera_matrix(*camera_rot, camera_roll)
            forward = R[:, 2]
            right = R[:, 0]

            if keys.is_held('w'): camera_pos += forward * move_speed; any_input = True
            if keys.is_held('s'): camera_pos -= forward * move_speed; any_input = True
            if keys.is_held('a'): camera_pos -= right * move_speed; any_input = True
            if keys.is_held('d'): camera_pos += right * move_speed; any_input = True
            if keys.is_held('q'): camera_pos[1] -= move_speed; any_input = True
            if keys.is_held('e'): camera_pos[1] += move_speed; any_input = True

            d_yaw = d_pitch = 0.0
            if keys.is_held('arrow_left'):  d_yaw -= look_speed; any_input = True
            if keys.is_held('arrow_right'): d_yaw += look_speed; any_input = True
            if keys.is_held('arrow_up'):    d_pitch -= look_speed; any_input = True
            if keys.is_held('arrow_down'):  d_pitch += look_speed; any_input = True
            if d_yaw or d_pitch:
                camera_rot[0] += d_yaw
                camera_rot[1] += d_pitch
                camera_rot[1] = max(-1.5, min(1.5, camera_rot[1]))

            d_roll = 0.0
            if keys.is_held(','): d_roll -= roll_speed; any_input = True
            if keys.is_held('.'): d_roll += roll_speed; any_input = True
            if d_roll:
                camera_roll += d_roll

        R = camera_matrix(*camera_rot, camera_roll)

        n_pt_lights = len(tracer.lights)
        n_all_lights = n_pt_lights + len(tracer.spotlights)
        targeted_kf_idx = find_targeted_keyframe(camera_path, camera_pos, R, tracer.half_tan,
                                                  tracer.aspect, WIN_W, WIN_H)

        raw = cv2.waitKeyEx(1)
        if args.print_keys and raw != -1:
            print(f"[--print-keys] raw={raw}")
        kind, val = classify_key(raw)
        if kind == 'arrow':
            inp.note_held('arrow_' + val)
            any_input = True
        elif kind == 'char':
            ch = val
            if ch in ('w', 'a', 's', 'd', 'q', 'e', ',', '.'):
                inp.note_held(ch)
                any_input = True
            elif ch == 'esc':
                running = False
            elif ch == '1' and inp.one_shot('lines'):
                lines_mode = not lines_mode
            elif ch == 'l' and inp.one_shot('live'):
                live_render = not live_render
                if live_render:
                    tracer.set_resolution(*LIVE_RENDER_RES)
                    prog.start()
            elif ch == 'tab' and inp.one_shot('tab'):
                if n_all_lights:
                    selected_light = (selected_light + 1) % n_all_lights
            elif ch == 'c' and inp.one_shot('caustics'):
                tracer.caustics_enabled = not tracer.caustics_enabled
                print(f"Caustics: {tracer.caustics_enabled}")
            elif ch == 'k' and inp.one_shot('place_light'):
                if selected_light < n_pt_lights and tracer.lights:
                    tracer.lights[selected_light].position = camera_pos.astype(np.float32).copy()
                    print(f"Light {selected_light + 1} placed at {camera_pos}")
                    tracer.sync_lights()
                    tracer.compute_caustics()
                elif tracer.spotlights:
                    sl = tracer.spotlights[selected_light - n_pt_lights]
                    sl.position = camera_pos.astype(np.float32).copy()
                    Rk = camera_matrix(*camera_rot, camera_roll)
                    sl.direction = Rk[:, 2].astype(np.float32)
                    print(f"Spotlight {selected_light - n_pt_lights + 1} placed at {camera_pos}, "
                          f"aimed with the camera")
                    tracer.sync_lights()
                    tracer.compute_caustics()
            elif ch == '[' and inp.one_shot('cone_narrow', debounce=0.08):
                if selected_light >= n_pt_lights and tracer.spotlights:
                    sl = tracer.spotlights[selected_light - n_pt_lights]
                    sl.cone_angle = max(3.0, min(80.0, sl.cone_angle - 3.0))
                    tracer.sync_lights(); tracer.compute_caustics()
            elif ch == ']' and inp.one_shot('cone_widen', debounce=0.08):
                if selected_light >= n_pt_lights and tracer.spotlights:
                    sl = tracer.spotlights[selected_light - n_pt_lights]
                    sl.cone_angle = max(3.0, min(80.0, sl.cone_angle + 3.0))
                    tracer.sync_lights(); tracer.compute_caustics()
            elif ch == '/' and inp.one_shot('roll_reset', debounce=0.1):
                camera_roll = 0.0
            elif ch == 'f' and inp.one_shot('autofocus'):
                Rf = camera_matrix(*camera_rot, camera_roll)
                fwd = Rf[:, 2]
                probe_depth(float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2]),
                            float(fwd[0]), float(fwd[1]), float(fwd[2]))
                focus_val = float(PROBE_DEPTH[None])
                post_fx['dof_focus_distance'] = max(0.3, min(focus_val, 2000.0))
                post_fx['dof_enabled'] = True
                autofocus_flash = 0.35
            elif ch == '-' and inp.one_shot('dof_near', debounce=0.06):
                if post_fx['dof_enabled']:
                    post_fx['dof_focus_distance'] = max(0.5, post_fx['dof_focus_distance'] - 1.0)
            elif ch == '=' and inp.one_shot('dof_far', debounce=0.06):
                if post_fx['dof_enabled']:
                    post_fx['dof_focus_distance'] = max(0.5, post_fx['dof_focus_distance'] + 1.0)
            elif ch == '2' and inp.one_shot('dof_toggle'):
                post_fx['dof_enabled'] = not post_fx['dof_enabled']
                print(f"DoF: {post_fx['dof_enabled']}")
            elif ch == '3' and inp.one_shot('chroma_toggle'):
                post_fx['chroma_enabled'] = not post_fx['chroma_enabled']
                print(f"Chromatic aberration: {post_fx['chroma_enabled']}")
            elif ch == '4' and inp.one_shot('flare_toggle'):
                post_fx['flare_enabled'] = not post_fx['flare_enabled']
                print(f"Flare: {post_fx['flare_enabled']}")
            elif ch == '6' and inp.one_shot('vhs_toggle'):
                post_fx['vhs_enabled'] = not post_fx['vhs_enabled']
                print(f"VHS: {post_fx['vhs_enabled']}")
            elif ch == '7' and inp.one_shot('mblur_toggle'):
                post_fx['motion_blur_enabled'] = not post_fx['motion_blur_enabled']
                print(f"Motion blur (video only): {post_fx['motion_blur_enabled']}")
            elif ch == '0' and inp.one_shot('fisheye_toggle'):
                post_fx['fisheye_enabled'] = not post_fx['fisheye_enabled']
                print(f"Fisheye: {post_fx['fisheye_enabled']}")
            elif ch == 'b' and inp.one_shot('bloom_toggle'):
                post_fx['bloom_enabled'] = not post_fx['bloom_enabled']
                print(f"Bloom: {post_fx['bloom_enabled']}")
            elif ch == 'y' and inp.one_shot('eye_adapt_toggle'):
                tracer.eye_adapt_enabled = not tracer.eye_adapt_enabled
                print(f"Eye adaptation / auto exposure: {tracer.eye_adapt_enabled}")
            elif ch == 'g' and inp.one_shot('autofocus_video_toggle'):
                post_fx['autofocus_enabled'] = not post_fx['autofocus_enabled']
                print(f"Continuous autofocus (video renders only): {post_fx['autofocus_enabled']}")
            elif ch == 'h' and inp.one_shot('handheld_toggle'):
                if has_camera_data:
                    print("Handheld shake only applies to camera keyframe paths, not --camera-data/--camera-stream.")
                else:
                    levels = [0.0, 0.35, 0.7, 1.1]
                    cur = camera_path.handheld_shake
                    nxt = levels[(levels.index(cur) + 1) % len(levels)] if cur in levels else levels[1]
                    camera_path.handheld_shake = nxt
                    print(f"Handheld camera shake: {nxt:.2f}")
            elif ch == 'p' and inp.one_shot('add_keyframe'):
                if has_camera_data:
                    print("Camera keyframes are disabled while --camera-data is active.")
                else:
                    camera_path.add(camera_pos.copy(), camera_rot[0], camera_rot[1], speed=1.0)
                    print(f"Camera keyframe #{len(camera_path.keyframes)} added.")
            elif ch == 'j' and inp.one_shot('add_hold_keyframe'):
                if has_camera_data:
                    print("Camera keyframes are disabled while --camera-data is active.")
                elif not camera_path.keyframes:
                    print("Add a normal keyframe first (P), then J to hold in place at that position.")
                else:
                    try:
                        raw_in = input("Hold duration in seconds (camera stays put, can still turn "
                                        "-- Enter for 2.0): ").strip()
                        dur = float(raw_in) if raw_in else 2.0
                    except (ValueError, EOFError):
                        print("Invalid value -- using 2.0s.")
                        dur = 2.0
                    camera_path.hold(duration=max(0.05, dur), yaw=camera_rot[0], pitch=camera_rot[1])
                    print(f"Hold keyframe #{len(camera_path.keyframes)} added ({dur:.2f}s, "
                          f"view angle set to the current camera).")
            elif ch == 'o' and inp.one_shot('kf_menu'):
                if has_camera_data:
                    print("Camera keyframes are disabled while --camera-data is active.")
                elif targeted_kf_idx is not None:
                    kf = camera_path.keyframes[targeted_kf_idx]
                    action, value = keyframe_options_menu_cv(canvas, WIN_W, WIN_H, kf, targeted_kf_idx)
                    if action == 'speed_delta':
                        kf.speed = max(0.01, kf.speed + value)
                    elif action == 'set_speed':
                        kf.speed = value
                    elif action == 'delete':
                        camera_path.remove(targeted_kf_idx)
                        print(f"Keyframe #{targeted_kf_idx + 1} deleted.")
            elif ch == 'i' and inp.one_shot('replay'):
                if replaying:
                    replaying = False
                    camera_roll = 0.0  # reset roll when a replay is stopped early too
                    print("Replay stopped.")
                elif has_camera_data:
                    replaying = True
                    replay_start_time = time.time()
                    print("Replaying camera data...")
                elif len(camera_path.keyframes) >= 1:
                    replaying = True
                    replay_start_time = time.time()
                    print("Replaying camera path...")
                else:
                    print("No camera keyframes to replay -- press P to add one first.")
            elif ch == '9' and inp.one_shot('render_video', debounce=1.0):
                if is_live_stream:
                    print("Live camera stream has no fixed length -- can't render a video from it "
                          "(record it to a --camera-data file first, or use camera keyframes).")
                elif has_camera_data or len(camera_path.keyframes) >= 1:
                    tracer.render_video(
                        active_path, out_path="raytrace_v9_video.mp4", post_fx=post_fx,
                        fps=video_fps, camera_sync=args.camera_sync,
                        progress_cb=lambda d, t, e, eta: draw_render_progress(
                            canvas, WIN_W, WIN_H, d, t, e, eta, "VIDEO"))
                    tracer.camera_roll = 0.0
                    tracer.set_resolution(*LIVE_RENDER_RES)
                else:
                    print("No camera keyframes -- press P to add at least one before rendering a video.")
            elif ch == '5' and inp.one_shot('final_render', debounce=1.0):
                was_live = live_render
                live_render = False
                tracer.set_resolution(*FINAL_RENDER_RES)
                tracer.render_to_file(
                    "raytrace_v9.png", samples=32, post_fx=post_fx,
                    progress_cb=lambda d, t, e, eta: draw_render_progress(
                        canvas, WIN_W, WIN_H, d, t, e, eta, "FINAL RENDER"))
                tracer.set_resolution(*LIVE_RENDER_RES)
                if was_live:
                    live_render = True
                    prog.start()
            elif ch == '8' and inp.one_shot('quick_render', debounce=1.0):
                was_live = live_render
                live_render = False
                prev_final = FINAL_RENDER_RES
                tracer.set_resolution(1920, 1080)
                tracer.render_to_file(
                    "raytrace_v9_quick.png", samples=4, post_fx=post_fx,
                    progress_cb=lambda d, t, e, eta: draw_render_progress(
                        canvas, WIN_W, WIN_H, d, t, e, eta, "QUICK RENDER"))
                tracer.set_resolution(*LIVE_RENDER_RES)
                if was_live:
                    live_render = True
                    prog.start()
            elif ch == 'x' and inp.one_shot('export', debounce=1.0):
                try:
                    out_path = input("Export current scene to (path, e.g. myscene.json): ").strip()
                    if out_path:
                        tracer.camera_pos = camera_pos.astype(np.float32)
                        tracer.camera_rot = camera_rot.astype(np.float32)
                        tracer.camera_roll = camera_roll
                        save_scene_file(out_path, scene, tracer, post_fx=post_fx,
                                         camera_path=camera_path, embed_assets=True)
                except Exception as e:
                    print(f"Export failed: {e}")

        if autofocus_flash > 0.0:
            autofocus_flash = max(0.0, autofocus_flash - 1.0 / 60.0)

        # --- Draw the frame ---
        if live_render:
            tracer.camera_pos = camera_pos.astype(np.float32)
            tracer.camera_rot = camera_rot.astype(np.float32)
            tracer.camera_roll = camera_roll
            canvas = prog.step(any_input, (WIN_W, WIN_H))
        else:
            canvas[:] = (30, 25, 20)
            draw_preview(canvas, preview_geo, camera_pos, R, tracer.half_tan, tracer.aspect,
                         WIN_W, WIN_H, lines=lines_mode)
            draw_light_indicator(canvas, tracer.lights, selected_light, camera_pos, R,
                                  tracer.half_tan, tracer.aspect, WIN_W, WIN_H,
                                  spotlights=tracer.spotlights)
        draw_camera_path(canvas, camera_path, targeted_kf_idx, camera_pos, R,
                          tracer.half_tan, tracer.aspect, WIN_W, WIN_H)
        draw_camera_viewfinder(canvas, WIN_W, WIN_H, WIN_W // 2, WIN_H // 2,
                                focused=(autofocus_flash > 0.0))

        sel_kind = "spotlight" if selected_light >= n_pt_lights else "light"
        sel_num = (selected_light - n_pt_lights + 1) if selected_light >= n_pt_lights else (selected_light + 1)
        tl_lines = [
            f"pos: {camera_pos[0]:.1f}, {camera_pos[1]:.1f}, {camera_pos[2]:.1f}",
            f"yaw/pitch/roll: {math.degrees(camera_rot[0]):.0f} / {math.degrees(camera_rot[1]):.0f} "
            f"/ {math.degrees(camera_roll):.0f}",
        ]
        tr_lines = [
            f"live raytrace: {'ON' if live_render else 'off'}",
            f"caustics: {'on' if tracer.caustics_enabled else 'off'}",
        ]
        if has_camera_data:
            cam_line = f"camera data: {_camera_data_count_str()}" + (" (following)" if is_live_stream and replaying else " (replaying)" if replaying else "")
        else:
            cam_line = f"keyframes: {len(camera_path.keyframes)}" + (" (replaying)" if replaying else "")
        bl_lines = [
            f"selected {sel_kind} #{sel_num}" if n_all_lights else "no lights in scene",
            cam_line,
        ]
        br_lines = [
            f"DoF {'on' if post_fx['dof_enabled'] else 'off'}"
            + (f" @ {post_fx['dof_focus_distance']:.1f}" if post_fx['dof_enabled'] else "")
            + (" (AF)" if post_fx.get('autofocus_enabled', False) else ""),
            f"VHS {'on' if post_fx['vhs_enabled'] else 'off'}  flare {'on' if post_fx['flare_enabled'] else 'off'}",
            f"fisheye {'on' if post_fx.get('fisheye_enabled', False) else 'off'}  "
            f"bloom {'on' if post_fx.get('bloom_enabled', False) else 'off'}  "
            f"exposure {tracer.exposure:.2f}{'(auto)' if tracer.eye_adapt_enabled else ''}",
        ]
        draw_hud_text(canvas, WIN_W, WIN_H, tl_lines, tr_lines, bl_lines, br_lines)

        cv2.imshow(WINDOW_NAME, canvas)
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            running = False

    cv2.destroyAllWindows()
    if stream_data is not None:
        stream_data.close()


if __name__ == "__main__":
    main()
