"""
Raytracer v12 -- builds on v11 (same physics: Fresnel/Snell/Beer-Lambert,
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

6) WASD MOVEMENT now follows the camera's viewing direction (yaw) but is
   locked to the horizontal plane: W/S/A/D move forward/back/left/right
   relative to which way the camera is facing, same as before, EXCEPT
   pitch (looking up/down) no longer tilts that movement -- these keys
   never change camera_pos's Y themselves, so looking up and pressing W
   walks you forward along the ground instead of flying upward. Q/E
   (world Y) and look direction (arrow keys/mouse-drag) are unchanged.

7) SPOTLIGHT ANIMATION: a SpotLight can now have a SpotLightAnimation
   attached (see that class) programming its motion and/or properties
   over time -- 'orbit' (circles a pivot), 'pingpong' (eases between two
   points), 'sweep' (lighthouse-style yaw/pitch sway in place), or
   'keyframes' (explicit, looped position/target/color/brightness/
   cone_angle/softness waypoints). Configure one from the interactive
   view via a spotlight's properties menu (click/Tab-select it, then A),
   or by hand-editing/authoring a scene JSON's spotlights[i].animation
   block directly (see SpotLightAnimation.to_dict/from_dict) for a
   headless/scripted render. The interactive live view advances
   animations by wall-clock time each frame; offline video rendering
   (render_video) instead samples each animation at its exact frame (and,
   with motion blur on, sub-frame shutter) timestamp, so exported video is
   deterministic/reproducible rather than tied to render speed.

8) v12 additions:
   - SpotLightAnimation gained a 'circle_sweep' kind: like 'sweep' but the
     aim traces a full circle (yaw/pitch 90 degrees out of phase) instead
     of swinging back and forth on one line -- see SpotLightAnimation.
   - Footstep-driven camera shake (_gait_offset) now varies each footfall
     slightly (amplitude/timing/left-right asymmetry) instead of repeating
     an identical kick every step, which read as mechanical/metronomic.
   - New "optical zoom" (;/' keys) narrows/widens the camera's FOV like a
     real zoom lens, with a configurable ramp speed (--zoom-speed /
     post-fx menu) and, in rendered video only, a brief auto-focus-style
     softening while the zoom is actively racking (see set_zoom,
     apply_zoom_focal_shift).
   - apply_fisheye's barrel-distortion remap is now bounded (a normalized
     power-curve model) so it can never sample outside the source image --
     this removes the smeared/stretched edges the old r*(1+k*r^2) model
     produced once r exceeded the frame.
   - F10 opens a per-keyframe options menu (position/yaw/pitch/speed/
     duration/zoom) for whichever keyframe the camera is currently aimed
     at, alongside the existing O key. The old single `` ` `` post-FX menu
     is now three: F5 (_build_world_params -- fog/clouds/sky/sun/stars/
     caustics/god rays), F6 (_build_camera_params -- exposure/zoom/DoF/
     camera shake/resolution/samples-per-mode), and `` ` `` itself, now
     just the remaining lens/sensor-style effects (_build_postfx_params).
   - Scene.add_floor: an "infinitely extending" ground plane (really a
     big thin box -- same material properties as a box, sized to fall
     off past the horizon rather than truly unbounded) with a
     `visibility`/`fog_max_depth` limit to keep the BVH cheap;
     Scene.add_water_surface is the equivalent convenience for a big flat
     water layer (delegates to add_water, so it keeps refraction/
     ripples/caustics).
   - Background.set_sky_gradient paints a proper zenith/horizon/ground
     gradient sky (adjustable colors + a horizon-bias curve) as the SAME
     equirectangular image format _sample_background already used for a
     loaded photo -- no rendering-kernel changes needed. Background.
     set_sun bakes a dimmable sun/moon disc into that same sky (so it
     subtly brightens ambient shading via the existing sky_light_strength
     term, for free); RayTracer.add_sun_light/add_sun optionally also add
     a REAL distant point Light in the same direction for actual
     highlights/shadows (works because point lights here have no
     distance falloff -- see add_sun_light's docstring). See
     RayTracer.set_sun_intensity for dimming it at runtime.
   - Background.set_stars scatters a soft-dot star field into the same
     sky texture (density/min-size/max-size/color-variation all
     adjustable); update_stars (RayTracer.update_stars, throttled to
     ~1/second in the interactive loop) advances each star's own slow,
     seeded twinkle phase for "occasionally shifts slightly in
     color/brightness" instead of a static field or a strobing one.
   - RayTracer.set_clouds adds a REAL raymarched volumetric cloud layer
     (a noise-density band between two altitudes -- see _cloud_density/
     _cloud_march/_cloud_fbm near the top of the file, and
     alloc_cloud_fields for the CLOUD_* GPU fields it reads): visible
     from the camera with soft self-shadowed shapes, AND actually casts
     shadows on geometry below it, because every shadow ray in
     _shadow_throughput marches through the same density field on its
     way to each light (_cloud_shadow_transmittance) -- not a separate
     or faked effect. This is the one v12 feature I could not compile or
     test myself (no Taichi/GPU available in the sandbox this was
     written in) -- please test on your machine before relying on it,
     and see set_clouds' docstring for the quality/perf knobs
     (steps/light_steps) if it's slow.
"""

import argparse
import base64
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
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
    """Picks a Taichi backend and starts it -- tries real GPU backends first
    (much faster), falling back to CPU if none of them work (e.g. an old
    iGPU that doesn't support what Taichi needs here, or no GPU at all).

    Two environment variables (NOT CLI flags) can override this:
      RT_BACKEND      -- force one specific backend ('cuda', 'vulkan',
                          'metal', or 'cpu') instead of auto-trying in order.
                          Used by --multi-gpu worker processes (see
                          _spawn_mgpu_worker) to force 'cuda' on their
                          pinned GPU rather than re-running the full
                          auto-detect probe in every worker; also handy to
                          force 'cpu' yourself for debugging, e.g.:
                              RT_BACKEND=cpu python3 rfv12cv.py --scene foo.json ...
      RT_RANDOM_SEED  -- seeds Taichi's RNG explicitly. Used so --multi-gpu
                          still-image workers (which each independently
                          render an INDEPENDENT share of the total sample
                          count for the SAME image) don't all draw the exact
                          same "random" jitter/scatter sequence -- which
                          would just duplicate noise instead of actually
                          reducing it once their results are combined.
    These are env vars and not argparse flags because this function has to
    run at IMPORT time, before argparse (or even main()) exists -- the
    @ti.kernel/@ti.func definitions later in this file need an active
    Taichi runtime to be defined against, so this can't be deferred."""
    forced = os.environ.get('RT_BACKEND', '').strip().lower()
    seed_env = os.environ.get('RT_RANDOM_SEED', '').strip()
    seed = int(seed_env) if seed_env else None

    backends = {'cuda': ti.cuda, 'vulkan': ti.vulkan, 'metal': ti.metal, 'cpu': ti.cpu}
    if forced in backends:
        order = [(forced, backends[forced])]
    else:
        # GPU backends first (much faster when available); CPU last as the
        # universal fallback (this is what the older, CPU-only version of
        # this function always used, kept as the safety net for hardware
        # that doesn't support Taichi's GPU backends).
        order = [('cuda', ti.cuda), ('vulkan', ti.vulkan), ('metal', ti.metal), ('cpu', ti.cpu)]

    last_err = None
    for name, backend in order:
        try:
            kwargs = {'arch': backend, 'default_fp': ti.f32}
            if seed is not None:
                kwargs['random_seed'] = seed
            ti.init(**kwargs)
            _gpu_smoke_test()
            print(f"Taichi backend: {name}")
            return name
        except Exception as e:
            last_err = e
            print(f"{name} failed: {e}")
    raise RuntimeError(f"Could not start Taichi on any backend (last error: {last_err})")


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


def dir_from_yaw_pitch(yaw, pitch):
    """Inverse of yaw_pitch_from_dir -- builds a unit direction vector using
    the SAME yaw/pitch convention as camera_matrix (yaw=0,pitch=0 -> +Z;
    increasing yaw turns toward +X; increasing pitch looks up, i.e. -Y)."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return np.array([sy * cp, -sp, cy * cp], dtype=np.float32)


def yaw_pitch_from_dir(d):
    """Inverse of dir_from_yaw_pitch -- recovers (yaw, pitch) in radians
    from a (needn't be normalized) direction vector, using the same
    convention as camera_matrix."""
    d = normalize(np.asarray(d, dtype=float))
    pitch = math.asin(max(-1.0, min(1.0, -float(d[1]))))
    yaw = math.atan2(float(d[0]), float(d[2]))
    return yaw, pitch


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

PREVIEW_RES = (640, 360)         # window size + 2D painter's-algorithm preview
LIVE_RENDER_RES = (96, 54)       # interactive raytrace resolution (upscaled to window)
FINAL_RENDER_RES = (3840, 2160)  # final image (the '9' key), saved to file
MAX_W, MAX_H = FINAL_RENDER_RES

MAX_LIGHTS = 8
MAX_SPOTLIGHTS = 4    # max number of spotlights (cone lights) per scene

WINDOW_NAME = "Raytracer v12 - CV"
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
VIDEO_SAMPLES_PER_FRAME = 16       # raytrace samples per video frame

# --- WATER MOTION IN VIDEO (does not affect stills/live) ---------------
WATER_WAVE_SPEED = 1.0          # multiplier applied to the "time" fed into the water
                                 # ripple function when rendering video -- 0 = water stays
                                 # still like before, 1 = default speed, higher = faster
                                 # "running" ripples. Edit directly.
WATER_ANIMATE_CAUSTICS = True    # whether to recompute the caustic map (the light patterns
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

FLOOR_DEFAULT_VISIBILITY = 500.0  # world units -- half-extent of Scene.add_floor's default X/Z
                                   # square when no explicit `visibility`/`fog_max_depth` is given
                                   # (see there): far enough that the edge falls past the horizon/
                                   # off-screen for any normal camera height, without the BVH having
                                   # to hold an actually-unbounded plane (triangle meshes can't).

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

    def add_floor(self, y=0.0, color=(120, 120, 120), roughness=0.6, transparency=0.0,
                  ior=1.5, reflection_k=0.0, visibility=None, thickness=1.0,
                  fog_max_depth=None):
        """An "infinitely extending" flat ground plane at height y. In
        practice this is a big thin box -- the exact same underlying
        geometry/material system as add_box, so every parameter here
        (color/roughness/transparency/ior/reflection_k) behaves exactly
        like it would on a box -- sized just large enough that its edge
        never actually appears on screen (falls past the horizon/frame
        edge for any reasonable camera height), rather than truly
        unbounded, which no triangle-mesh BVH could hold or render
        without cost proportional to its size.

        visibility: half-extent (world units) of the floor's X/Z square.
          None (default) picks FLOOR_DEFAULT_VISIBILITY, UNLESS
          `fog_max_depth` is given (see below).
        fog_max_depth: if you're also turning on fog (the scene's
          post_fx['fog_max_depth']), pass it here so the floor only
          extends a little past where fog would fully hide it anyway
          (fog_max_depth * 1.15) instead of the full default -- this is
          the "automatically adjusted when fog is enabled" sizing: pass
          your fog settings' max depth in, and the floor shrinks its own
          footprint (and therefore the BVH's/render's cost) to match,
          without you having to size the two separately. Has no effect
          if `visibility` is given explicitly.
        thickness: how "thick" the slab is (world units) -- just gives it
          well-defined side faces like any other box; invisible from
          normal above/below viewing angles either way.
        """
        if visibility is None:
            visibility = (float(fog_max_depth) * 1.15 if fog_max_depth is not None
                          else FLOOR_DEFAULT_VISIBILITY)
        visibility = max(1.0, float(visibility))
        thickness = max(0.01, float(thickness))
        self.ops.append({'method': 'add_floor', 'kwargs': {
            'y': float(y), 'color': list(int(c) for c in color[:3]),
            'roughness': float(roughness), 'transparency': float(transparency),
            'ior': float(ior), 'reflection_k': float(reflection_k),
            'visibility': visibility, 'thickness': thickness,
        }})
        self._add_box_geometry_only(
            center=(0.0, y - thickness / 2.0, 0.0),
            size=(visibility * 2.0, thickness, visibility * 2.0),
            color=color, roughness=roughness, transparency=transparency,
            ior=ior, reflection_k=reflection_k, rotation=(0.0, 0.0, 0.0))

    def add_water_surface(self, y, color=(160, 230, 255), transparency=0.92, ior=1.333,
                          visibility=None, thickness=0.5, fog_max_depth=None):
        """Convenience for a big flat water surface (e.g. a lake/ocean
        layer sitting above an add_floor() ground) -- same "big thin box
        that reads as infinite" trick as add_floor, just delegated to
        add_water() so it keeps water's own refraction/tint/Gerstner-wave
        ripples/caustics behavior (see add_water). Counts against
        MAX_WATER_BLOCKS like any other add_water call. `visibility`/
        `fog_max_depth` mean exactly what they do on add_floor."""
        if visibility is None:
            visibility = (float(fog_max_depth) * 1.15 if fog_max_depth is not None
                          else FLOOR_DEFAULT_VISIBILITY)
        visibility = max(1.0, float(visibility))
        thickness = max(0.01, float(thickness))
        self.add_water(center=(0.0, float(y) - thickness / 2.0, 0.0),
                        size=(visibility * 2.0, thickness, visibility * 2.0),
                        color=color, transparency=transparency, ior=ior)

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
    def __init__(self, color=(20, 20, 30), image_path=None, brightness=1.0, sky=None, sun=None,
                 stars=None):
        self.brightness = brightness
        self.image = None
        self.img_h = self.img_w = 1
        self.image_path = image_path
        self.solid = np.array(color[:3], dtype=np.float32) / 255.0
        self.color = tuple(int(c) for c in color[:3])
        self.sky = None                # gradient params dict, if set_sky_gradient was called
        self.sun = None                # sun/moon disc params dict, if set_sun was called
        self.stars = None              # star-field params dict, if set_stars was called
        self._sky_base_image = None    # the plain gradient, no sun/stars -- see _repaint
        self._star_field = None        # list of (u, v, size, base_color, phase) -- see set_stars
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path).convert('RGB')
            self.image = np.array(img, dtype=np.float32) / 255.0
            self.img_h, self.img_w = self.image.shape[:2]
            print(f"Background image loaded: {image_path} ({self.img_w}x{self.img_h})")
        elif image_path:
            print(f"Warning: '{image_path}' not found, using solid color instead")
        if sky is not None:
            self.set_sky_gradient(**sky)
        if sun is not None:
            self.set_sun(**sun)
        if stars is not None:
            self.set_stars(**stars)

    @property
    def has_image(self):
        return self.image is not None

    def set_sky_gradient(self, zenith_color=(50, 90, 170), horizon_color=(210, 225, 245),
                          ground_color=(45, 40, 38), curve=1.6, resolution=(384, 192)):
        """Procedurally paints a vertical-gradient sky -- zenith_color
        straight up, fading to horizon_color at the horizon, then fading
        again to ground_color looking straight down -- and installs it as
        this Background's image. This is the SAME equirectangular image
        format/lookup _sample_background already uses for a loaded photo
        (u = azimuth, v = 0 at straight up .. 1 at straight down), so no
        rendering-kernel changes were needed to support this: it's just a
        different way of producing that image, in Python, once, at scene
        setup. Since the ambient/hemisphere-fill term in render_sample
        already samples this same background along each surface's normal
        (`sky_light_strength * bg_brightness * sky_col`), a bright zenith
        or a baked-in sun (see set_sun) SUBTLY LIGHTENS ambient shading on
        surfaces angled toward them for free, with no separate "sky
        lighting" system needed.

        `curve` biases how quickly zenith fades to horizon: 1.0 = linear,
        higher values stay closer to zenith_color longer before rushing
        to the horizon band, which is closer to how a clear real sky
        actually looks (paler near the horizon, not a straight gradient).
        `resolution` (w, h) is the baked texture size -- this is a coarse
        gradient, so the default (384x192) is already overkill; kept
        small on purpose so re-baking (e.g. every frame of a day/night
        cycle via set_sun/RayTracer.set_sun_intensity) stays cheap.

        Re-bakes a previously-set sun/star-field on top of the new
        gradient (since a fresh gradient would otherwise overwrite them)
        -- see set_sun's `intensity` / update_stars for animating either
        afterward instead of re-describing them from scratch every time."""
        w, h = max(2, int(resolution[0])), max(2, int(resolution[1]))
        zen = np.array(zenith_color[:3], dtype=np.float32) / 255.0
        hor = np.array(horizon_color[:3], dtype=np.float32) / 255.0
        gnd = np.array(ground_color[:3], dtype=np.float32) / 255.0
        curve = max(0.05, float(curve))
        rows = np.zeros((h, 3), dtype=np.float32)
        for iy in range(h):
            v = iy / max(1, h - 1)          # 0 at top (zenith) .. 1 at bottom (nadir)
            elev = 1.0 - 2.0 * v            # +1 straight up .. -1 straight down
            if elev >= 0.0:
                t = elev ** curve
                rows[iy] = hor + (zen - hor) * t
            else:
                t = min(1.0, (-elev) ** curve)
                rows[iy] = hor + (gnd - hor) * t
        self._sky_base_image = np.repeat(rows[:, None, :], w, axis=1).astype(np.float32)
        self.img_h, self.img_w = h, w
        self.image_path = None  # procedural now, not file-backed -- see to_dict/from_dict
        self.sky = {'zenith_color': [int(c) for c in zenith_color[:3]],
                    'horizon_color': [int(c) for c in horizon_color[:3]],
                    'ground_color': [int(c) for c in ground_color[:3]],
                    'curve': curve, 'resolution': [w, h]}
        if self.stars is not None:
            self._build_star_field()  # resolution changed -- star pixel positions need rebuilding
        self._repaint()
        return self

    def set_sun(self, azimuth_deg=45.0, elevation_deg=35.0, color=(255, 244, 214),
                intensity=1.0, angular_size_deg=2.0, glow_deg=6.0):
        """Bakes a bright disc for a sun/moon into this Background's sky
        image (call set_sky_gradient first -- there's no procedural image
        to bake onto otherwise; a loaded photo background isn't touched by
        this). `azimuth_deg` is measured around the horizon (0..360, same
        convention as the equirectangular u coordinate), `elevation_deg`
        is 90=straight up, 0=horizon, negative=below the horizon (still
        drawn -- useful for a sunset glow poking above the horizon band
        via `glow_deg` even once the disc itself has set). `intensity` is
        a 0..~3 dimmer -- see RayTracer.set_sun_intensity for changing
        this at runtime (day/night cycles etc.) without re-specifying the
        rest. `angular_size_deg` is the disc's apparent angular radius,
        `glow_deg` a soft halo past its edge so it doesn't look like a
        flat cutout. For the sun to ALSO cast real specular highlights/
        shadows (this only affects the visual sky image + the ambient
        term -- see set_sky_gradient's docstring), separately add a
        matching distant Light -- see RayTracer.add_sun_light, or just
        use RayTracer.add_sun which does both at once."""
        self.sun = {'azimuth_deg': float(azimuth_deg) % 360.0, 'elevation_deg': float(elevation_deg),
                    'color': [int(c) for c in color[:3]], 'intensity': max(0.0, float(intensity)),
                    'angular_size_deg': max(0.05, float(angular_size_deg)),
                    'glow_deg': max(0.0, float(glow_deg))}
        self._repaint()
        return self

    def set_stars(self, density=0.0025, min_size=0.5, max_size=1.6, brightness=1.0,
                  color_variation=0.25, twinkle=0.15, above_horizon_only=True, seed=0):
        """Scatters a star field across this Background's sky image (call
        set_sky_gradient first, same requirement as set_sun). Stars are
        tiny soft dots baked into the same equirectangular texture as the
        gradient/sun, so they cost nothing per-ray -- just more pixels in
        an already-cheap texture -- and, like the sun, they subtly add to
        ambient sky-fill lighting on surfaces facing them (usually
        negligible individually, but see set_sky_gradient's docstring for
        why that's automatic here).

        density: stars per sky-texture PIXEL (not world units) -- e.g.
          the default 0.0025 gives ~180 stars at the default 384x192
          resolution. Scales with `resolution` if you change it, so a
          higher-res sky doesn't silently get a denser field.
        min_size/max_size: each star's radius in PIXELS, picked
          uniformly at random per star between the two.
        brightness: overall dimmer on top of each star's own random
          brightness (which stars: 0..1 uniform per star, so brightness
          scales that whole range) -- also handy for a dusk/dawn fade
          alongside set_sun's intensity.
        color_variation: 0 = every star pure white, higher values let
          individual stars drift toward a warm or cool tint (a small
          per-star random RGB offset) -- real starlight isn't perfectly
          white, just close to it.
        twinkle: how far each star's brightness wanders over time when
          you call update_stars(t) (0 = static once baked, higher =
          more visible flicker) -- "occasionally shift slightly in
          color" is `twinkle` driving a slow per-star sine, not a true
          per-frame flicker; see update_stars for actually animating it.
        above_horizon_only: skip generating stars below elev=0 (default
          True -- there's usually a floor/ground down there anyway).
        seed: RNG seed -- the star FIELD (positions/sizes/base colors) is
          otherwise regenerated identically every call with the same
          seed, so re-baking (e.g. after set_sky_gradient changes
          resolution) doesn't reshuffle which stars are where."""
        self.stars = {'density': max(0.0, float(density)), 'min_size': max(0.1, float(min_size)),
                      'max_size': max(float(min_size), float(max_size)),
                      'brightness': max(0.0, float(brightness)),
                      'color_variation': max(0.0, float(color_variation)),
                      'twinkle': max(0.0, float(twinkle)),
                      'above_horizon_only': bool(above_horizon_only), 'seed': int(seed)}
        self._build_star_field()
        self._repaint()
        return self

    def update_stars(self, t):
        """Advances each star's twinkle phase to time t (seconds) and
        re-bakes -- call this occasionally (it doesn't need to be every
        frame; a star field re-bake is cheap but there's no reason to pay
        for it 60x/second for an effect that's meant to read as a slow,
        occasional shift, not a strobe) from the interactive loop or a
        video render to get "stars occasionally shift slightly in
        color/brightness" instead of a perfectly static field. No effect
        if set_stars hasn't been called (or twinkle=0 -- nothing would
        change anyway)."""
        if self.stars is None or self._star_field is None or self.stars['twinkle'] <= 1e-4:
            return
        self._repaint(star_time=t)

    def _build_star_field(self):
        """(Re)generates the star field's positions/sizes/base colors
        from self.stars' params -- see set_stars. Deterministic given the
        same seed/resolution, so calling this again (e.g. from
        set_sky_gradient when the resolution changes) doesn't reshuffle
        already-placed stars, just re-projects them onto the new size."""
        if self.stars is None or self._sky_base_image is None:
            self._star_field = None
            return
        s = self.stars
        h, w = self.img_h, self.img_w
        count = max(0, int(round(s['density'] * w * h)))
        rng = np.random.RandomState(s['seed'])
        field = []
        tries = 0
        while len(field) < count and tries < count * 4 + 100:
            tries += 1
            u = rng.uniform(0.0, 1.0)
            v = rng.uniform(0.0, 1.0)
            elev = 1.0 - 2.0 * v  # same v convention as set_sky_gradient/set_sun
            if s['above_horizon_only'] and elev < 0.0:
                continue
            size = rng.uniform(s['min_size'], s['max_size'])
            star_bri = rng.uniform(0.4, 1.0) * s['brightness']
            tint = (rng.uniform(-1.0, 1.0, size=3) * s['color_variation'])
            base_color = np.clip(1.0 + tint, 0.0, 1.4).astype(np.float32) * star_bri
            phase = rng.uniform(0.0, 2.0 * math.pi)
            twinkle_rate = rng.uniform(0.15, 0.6)  # slow -- "occasionally", not a strobe
            field.append((u, v, float(size), base_color, float(phase), float(twinkle_rate)))
        self._star_field = field

    def _paint_stars(self, star_time=0.0):
        """Paints self._star_field onto self.image (assumes the caller,
        _repaint, has already reset self.image to the clean base+sun).
        `star_time` drives each star's twinkle sine -- see update_stars."""
        if not self._star_field:
            return
        h, w = self.img_h, self.img_w
        twinkle = self.stars['twinkle']
        for (u, v, size, base_color, phase, rate) in self._star_field:
            bri_mult = 1.0
            if twinkle > 1e-4:
                bri_mult = 1.0 + twinkle * math.sin(star_time * rate + phase)
                bri_mult = max(0.0, bri_mult)
            col = np.clip(base_color * bri_mult, 0.0, 2.0)
            cx, cy = u * (w - 1), v * (h - 1)
            r = max(0.5, size)
            pad = int(math.ceil(r)) + 1
            y0, y1 = max(0, int(cy) - pad), min(h, int(cy) + pad + 1)
            for iy in range(y0, y1):
                dyp = iy - cy
                for ixo in range(-pad, pad + 1):
                    dist = math.sqrt(ixo * ixo + dyp * dyp)
                    if dist > r:
                        continue
                    ix = (int(cx) + ixo) % w
                    t = max(0.0, 1.0 - dist / max(1e-4, r))
                    t = t * t
                    # Additive-ish (via max, not a blend) so a bright star
                    # against a bright daytime sky doesn't visibly wash it
                    # out or darken it -- it only shows up where it's
                    # actually brighter than what's already there, same as
                    # how stars in reality only become visible once the
                    # sky itself is dark enough.
                    self.image[iy, ix] = np.maximum(self.image[iy, ix], col * t)

    def _paint_sun(self):
        """Paints the current self.sun disc onto self.image (assumes the
        caller, _repaint, has already reset self.image to the clean
        base)."""
        if self._sky_base_image is None or self.sun is None:
            return
        s = self.sun
        if s['intensity'] <= 1e-4:
            return  # fully dimmed -- leave the plain gradient, nothing to paint
        h, w = self.img_h, self.img_w
        az = math.radians(s['azimuth_deg'])
        el = math.radians(s['elevation_deg'])
        # Same direction/UV convention as _sample_background: u = azimuth
        # around Y, v = 0 straight up .. 1 straight down -- so the disc we
        # paint here lines up exactly with where a real add_sun_light in
        # the same azimuth/elevation would actually be in the render.
        dx, dz = math.sin(az) * math.cos(el), math.cos(az) * math.cos(el)
        dy = math.sin(el)
        u = (math.atan2(dx, dz) / (2.0 * math.pi)) % 1.0
        v = 1.0 - (math.asin(max(-1.0, min(1.0, dy))) / math.pi + 0.5)
        cx, cy = u * (w - 1), v * (h - 1)
        col = (np.array(s['color'][:3], dtype=np.float32) / 255.0) * s['intensity']
        disc_r = max(0.5, s['angular_size_deg'] / 180.0 * h)   # h spans 180 degrees, top to bottom
        glow_r = disc_r + max(0.0, s['glow_deg'] / 180.0 * h)
        pad = int(math.ceil(glow_r)) + 1
        y0, y1 = max(0, int(cy) - pad), min(h, int(cy) + pad + 1)
        for iy in range(y0, y1):
            dyp = iy - cy
            for ixo in range(-pad, pad + 1):
                dist = math.sqrt(ixo * ixo + dyp * dyp)
                if dist > glow_r:
                    continue
                ix = (int(cx) + ixo) % w  # wraps around the horizon (azimuth 360 == 0)
                if dist <= disc_r:
                    self.image[iy, ix] = col
                else:
                    t = 1.0 - (dist - disc_r) / max(1e-4, glow_r - disc_r)
                    t = t * t
                    self.image[iy, ix] = self.image[iy, ix] * (1.0 - t) + col * t

    def _repaint(self, star_time=0.0):
        """Rebuilds self.image from _sky_base_image + the current sun (if
        any) + the current star field (if any), in that order -- ALWAYS
        starting from the clean cached base rather than painting onto
        whatever self.image currently holds, so repeated calls (dimming
        the sun, twinkling the stars, ...) never accumulate a ghost trail
        of every previous bake. No-op (leaves self.image as whatever it
        already is -- e.g. a loaded photo) if set_sky_gradient hasn't
        been called."""
        if self._sky_base_image is None:
            return
        self.image = self._sky_base_image.copy()
        self._paint_sun()
        self._paint_stars(star_time=star_time)

    def to_dict(self, assets=None):
        d = {'color': list(self.color), 'brightness': float(self.brightness),

             'image_path': self.image_path}
        if assets is not None and self.image_path:
            _embed_asset(assets, self.image_path)
        if self.sky is not None:
            d['sky'] = dict(self.sky)
        if self.sun is not None:
            d['sun'] = dict(self.sun)
        if self.stars is not None:
            d['stars'] = dict(self.stars)
        return d

    @staticmethod
    def from_dict(d, asset_dir=None):
        path = d.get('image_path')
        if path and asset_dir is not None:
            path = _resolve_asset_path(path, asset_dir)
        sky = d.get('sky')
        bg = Background(color=tuple(d.get('color', (20, 20, 30))),
                        image_path=None if sky else path, brightness=float(d.get('brightness', 1.0)))
        if sky:
            bg.set_sky_gradient(zenith_color=tuple(sky.get('zenith_color', (50, 90, 170))),
                                horizon_color=tuple(sky.get('horizon_color', (210, 225, 245))),
                                ground_color=tuple(sky.get('ground_color', (45, 40, 38))),
                                curve=float(sky.get('curve', 1.6)),
                                resolution=tuple(sky.get('resolution', (384, 192))))
        sun = d.get('sun')
        if sun:
            bg.set_sun(azimuth_deg=float(sun.get('azimuth_deg', 45.0)),
                      elevation_deg=float(sun.get('elevation_deg', 35.0)),
                      color=tuple(sun.get('color', (255, 244, 214))),
                      intensity=float(sun.get('intensity', 1.0)),
                      angular_size_deg=float(sun.get('angular_size_deg', 2.0)),
                      glow_deg=float(sun.get('glow_deg', 6.0)))
        stars = d.get('stars')
        if stars:
            bg.set_stars(density=float(stars.get('density', 0.0025)),
                        min_size=float(stars.get('min_size', 0.5)),
                        max_size=float(stars.get('max_size', 1.6)),
                        brightness=float(stars.get('brightness', 1.0)),
                        color_variation=float(stars.get('color_variation', 0.25)),
                        twinkle=float(stars.get('twinkle', 0.15)),
                        above_horizon_only=bool(stars.get('above_horizon_only', True)),
                        seed=int(stars.get('seed', 0)))
        return bg


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


def _lerp(a, b, u):
    return a + (b - a) * u


def _lerp3(a, b, u):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return a + (b - a) * u


class SpotLightAnimation:
    """Programmable motion/property animation for a SpotLight. Attach with
    `spotlight.set_animation(SpotLightAnimation(kind, **params))`, then
    advance it once per frame -- the interactive view does this
    automatically (see RayTracer.update_spotlight_animations), and offline
    video renders sample it at each frame's exact timestamp instead (see
    RayTracer.set_spotlight_animation_time) so exported video is
    frame-exact/deterministic rather than tied to wall-clock time.

    `kind` is one of:

      'orbit' -- position circles around `pivot` at `radius`, on the plane
        perpendicular to `axis` ('x'/'y'/'z', default 'y' = a horizontal
        circle), completing one revolution every `period_s` seconds
        (`phase_deg` offsets the starting angle, `direction` sign flips
        which way it circles). If `aim_at_pivot` (default True) the
        spotlight keeps pointing at the pivot as it moves; otherwise
        direction is held fixed at whatever it was when first animated.
        `pivot` defaults to the spotlight's own position at attach time.

      'pingpong' -- position eases back and forth between `point_a` and
        `point_b` (both default to the spotlight's position at attach
        time -- set at least one explicitly for this to do anything), one
        full A->B->A cycle every `period_s` seconds. If `aim_target` is
        given the spotlight continuously looks at it; otherwise direction
        is held fixed.

      'sweep' -- position stays fixed (optionally set once via
        `position`); direction sweeps like a lighthouse, with yaw/pitch
        oscillating sinusoidally by +/-`yaw_amplitude_deg` /
        +/-`pitch_amplitude_deg` around `center_yaw_deg`/`center_pitch_deg`
        (both default to the spotlight's own aim at attach time), one full
        swing cycle every `period_s` seconds.

      'circle_sweep' -- like 'sweep' (position stays fixed, optionally set
        once via `position`), but yaw and pitch are driven 90 degrees out
        of phase (cos/sin of the same angle) instead of both riding the
        same sine -- the beam's aim traces a full CIRCLE (radius
        `yaw_amplitude_deg` in yaw, `pitch_amplitude_deg` in pitch, both
        default to `radius_deg`=15 when not given individually) around
        `center_yaw_deg`/`center_pitch_deg` (default: the spotlight's own
        aim at attach time), one full revolution every `period_s` seconds
        -- e.g. a spotlight scanning a slow circle across a floor or wall,
        rather than swinging back and forth along one line. `direction`
        (sign, default 1) flips which way it circles, `phase_deg` offsets
        the starting angle.

      'keyframes' -- `keyframes` is a list of dicts describing waypoints,
        each with a 'time' (seconds, non-decreasing, first should be 0.0)
        plus any of 'position', 'target' (aim point -- use this OR
        'direction', not both), 'direction', 'color' (0-255), 'brightness',
        'cone_angle', 'softness'. Every field is linearly interpolated
        between the surrounding two keyframes; a field a keyframe omits
        just keeps whatever the light already had going into it. The
        whole sequence loops back to the start after the last keyframe's
        time.

    Every spatial/anchor default above (pivot, point_a/b, position,
    center_yaw/pitch, fixed direction) is frozen into this animation's own
    params the FIRST time it runs -- not re-read from the spotlight's
    current (already-animated) state on every frame -- so the motion
    doesn't drift and is exactly reproducible from t=0."""

    def __init__(self, kind, **params):
        if kind not in ('orbit', 'pingpong', 'sweep', 'circle_sweep', 'keyframes'):
            raise ValueError(f"Unknown SpotLightAnimation kind: {kind!r}")
        self.kind = kind
        self.params = dict(params)

    def to_dict(self):
        d = dict(self.params)
        d['kind'] = self.kind
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        kind = d.pop('kind')
        return SpotLightAnimation(kind, **d)

    def apply(self, sl, t):
        """Mutates spotlight `sl` (position/direction/color/brightness/
        cone_angle/softness as applicable) to this animation's state at
        time `t` (seconds, absolute -- callers own what "absolute" means:
        accumulated wall-clock seconds for the live view, or an exact
        frame timestamp for offline video)."""
        p = self.params
        if self.kind == 'orbit':
            if 'pivot' not in p:
                p['pivot'] = [float(x) for x in sl.position]
            pivot = np.array(p['pivot'], dtype=np.float32)
            radius = float(p.get('radius', 5.0))
            axis = str(p.get('axis', 'y')).lower()
            period = max(1e-3, float(p.get('period_s', 6.0)))
            phase = math.radians(float(p.get('phase_deg', 0.0)))
            angle = phase + 2.0 * math.pi * (t / period)
            ca, sa = math.cos(angle), math.sin(angle)
            if axis == 'x':
                offset = np.array([0.0, ca, sa], dtype=np.float32) * radius
            elif axis == 'z':
                offset = np.array([ca, sa, 0.0], dtype=np.float32) * radius
            else:
                offset = np.array([ca, 0.0, sa], dtype=np.float32) * radius
            sl.position = (pivot + offset).astype(np.float32)
            if p.get('aim_at_pivot', True):
                sl.aim_at(pivot)
            else:
                if 'direction' not in p:
                    p['direction'] = [float(x) for x in sl.direction]
                sl.direction = normalize(np.array(p['direction'], dtype=float)).astype(np.float32)

        elif self.kind == 'pingpong':
            if 'point_a' not in p:
                p['point_a'] = [float(x) for x in sl.position]
            if 'point_b' not in p:
                p['point_b'] = [float(x) for x in sl.position]
            a = np.array(p['point_a'], dtype=np.float32)
            b = np.array(p['point_b'], dtype=np.float32)
            period = max(1e-3, float(p.get('period_s', 4.0)))
            phase = float(p.get('phase', 0.0))
            u = ((t / period) + phase) % 1.0
            tri = 1.0 - abs(2.0 * u - 1.0)          # 0 -> 1 -> 0 triangle wave
            ease = 0.5 - 0.5 * math.cos(math.pi * tri)  # smoothed at each turnaround
            sl.position = _lerp3(a, b, ease)
            if 'aim_target' in p:
                sl.aim_at(np.array(p['aim_target'], dtype=np.float32))
            else:
                if 'direction' not in p:
                    p['direction'] = [float(x) for x in sl.direction]
                sl.direction = normalize(np.array(p['direction'], dtype=float)).astype(np.float32)

        elif self.kind == 'sweep':
            if 'position' in p:
                sl.position = np.array(p['position'], dtype=np.float32)
            if 'center_yaw_deg' not in p or 'center_pitch_deg' not in p:
                base_yaw, base_pitch = yaw_pitch_from_dir(sl.direction)
                p.setdefault('center_yaw_deg', math.degrees(base_yaw))
                p.setdefault('center_pitch_deg', math.degrees(base_pitch))
            center_yaw = math.radians(float(p['center_yaw_deg']))
            center_pitch = math.radians(float(p['center_pitch_deg']))
            yaw_amp = math.radians(float(p.get('yaw_amplitude_deg', 30.0)))
            pitch_amp = math.radians(float(p.get('pitch_amplitude_deg', 0.0)))
            period = max(1e-3, float(p.get('period_s', 4.0)))
            phase = math.radians(float(p.get('phase_deg', 0.0)))
            ang = phase + 2.0 * math.pi * (t / period)
            yaw = center_yaw + yaw_amp * math.sin(ang)
            pitch = center_pitch + pitch_amp * math.sin(ang)
            sl.direction = dir_from_yaw_pitch(yaw, pitch)

        elif self.kind == 'circle_sweep':
            if 'position' in p:
                sl.position = np.array(p['position'], dtype=np.float32)
            if 'center_yaw_deg' not in p or 'center_pitch_deg' not in p:
                base_yaw, base_pitch = yaw_pitch_from_dir(sl.direction)
                p.setdefault('center_yaw_deg', math.degrees(base_yaw))
                p.setdefault('center_pitch_deg', math.degrees(base_pitch))
            center_yaw = math.radians(float(p['center_yaw_deg']))
            center_pitch = math.radians(float(p['center_pitch_deg']))
            radius_deg = float(p.get('radius_deg', 15.0))
            yaw_amp = math.radians(float(p.get('yaw_amplitude_deg', radius_deg)))
            pitch_amp = math.radians(float(p.get('pitch_amplitude_deg', radius_deg)))
            period = max(1e-3, float(p.get('period_s', 4.0)))
            phase = math.radians(float(p.get('phase_deg', 0.0)))
            direction_sign = 1.0 if float(p.get('direction', 1.0)) >= 0.0 else -1.0
            ang = phase + direction_sign * 2.0 * math.pi * (t / period)
            # yaw/pitch 90 degrees out of phase (cos/sin of the SAME angle,
            # unlike 'sweep' which drives both from sin(ang)) is what turns
            # a back-and-forth line swing into a full circular sweep of the
            # beam's aim point.
            yaw = center_yaw + yaw_amp * math.cos(ang)
            pitch = center_pitch + pitch_amp * math.sin(ang)
            sl.direction = dir_from_yaw_pitch(yaw, pitch)

        elif self.kind == 'keyframes':
            kfs = p.get('keyframes') or []
            if not kfs:
                return
            kfs = sorted(kfs, key=lambda k: float(k.get('time', 0.0)))
            total = float(kfs[-1].get('time', 0.0))
            if total <= 0.0:
                kf = kfs[0]
                kf1 = kf2 = kf
                u = 0.0
            else:
                tt = t % total
                i1 = 0
                for i, kf in enumerate(kfs):
                    if float(kf.get('time', 0.0)) <= tt:
                        i1 = i
                i2 = min(i1 + 1, len(kfs) - 1)
                kf1, kf2 = kfs[i1], kfs[i2]
                t1, t2 = float(kf1.get('time', 0.0)), float(kf2.get('time', 0.0))
                u = 0.0 if t2 <= t1 else (tt - t1) / (t2 - t1)

            pos1 = np.array(kf1.get('position', sl.position), dtype=np.float32)
            pos2 = np.array(kf2.get('position', pos1), dtype=np.float32)
            sl.position = _lerp3(pos1, pos2, u)

            if 'target' in kf1 or 'target' in kf2:
                default_tgt = (sl.position + sl.direction).tolist()
                tgt1 = np.array(kf1.get('target', default_tgt), dtype=np.float32)
                tgt2 = np.array(kf2.get('target', tgt1), dtype=np.float32)
                sl.aim_at(_lerp3(tgt1, tgt2, u))
            else:
                dir1 = normalize(np.array(kf1.get('direction', sl.direction), dtype=float))
                dir2 = normalize(np.array(kf2.get('direction', dir1), dtype=float))
                sl.direction = normalize(_lerp3(dir1, dir2, u))

            col1 = np.array(kf1.get('color', [c * 255.0 for c in sl.color]), dtype=np.float32)
            col2 = np.array(kf2.get('color', col1), dtype=np.float32)
            sl.color = (_lerp3(col1, col2, u) / 255.0).astype(np.float32)

            b1 = float(kf1.get('brightness', sl.brightness))
            b2 = float(kf2.get('brightness', b1))
            sl.brightness = _lerp(b1, b2, u)

            ca1 = float(kf1.get('cone_angle', sl.cone_angle))
            ca2 = float(kf2.get('cone_angle', ca1))
            sl.cone_angle = _lerp(ca1, ca2, u)

            sf1 = float(kf1.get('softness', sl.softness))
            sf2 = float(kf2.get('softness', sf1))
            sl.softness = float(min(0.97, max(0.0, _lerp(sf1, sf2, u))))


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
    - animation:   optional SpotLightAnimation programming this spotlight's
                   motion/properties over time -- see set_animation() and
                   update_animation(). None (the default) means static,
                   exactly like before this feature existed.
    """

    def __init__(self, position, direction, color=(255, 255, 255), brightness=2.0,
                 cone_angle=25.0, softness=0.35, animation=None):
        self.position = np.array(position, dtype=np.float32)
        self.direction = normalize(np.array(direction, dtype=float)).astype(np.float32)
        self.color = np.array(color[:3], dtype=np.float32) / 255.0
        self.brightness = float(brightness)
        self.cone_angle = float(cone_angle)
        self.softness = float(min(0.97, max(0.0, softness)))
        self.animation = animation      # SpotLightAnimation or None
        self.anim_time = 0.0            # accumulated seconds since attached/reset

    @property
    def cos_outer(self):
        return math.cos(math.radians(self.cone_angle))

    @property
    def cos_inner(self):
        inner = self.cone_angle * (1.0 - self.softness)
        return math.cos(math.radians(inner))

    def aim_at(self, target_pos):
        self.direction = normalize(np.array(target_pos, dtype=float) - self.position).astype(np.float32)

    def set_animation(self, animation):
        """Attaches (or replaces) this spotlight's animation and resets its
        internal clock to 0 so the new animation starts from its own t=0,
        regardless of how long the old one had been running."""
        self.animation = animation
        self.anim_time = 0.0

    def clear_animation(self):
        self.animation = None
        self.anim_time = 0.0

    def update_animation(self, dt):
        """Advances the attached animation (if any) by dt wall-clock
        seconds and updates position/direction/color/etc. in place.
        No-op if no animation is attached. Used by the interactive live
        view -- see RayTracer.update_spotlight_animations. Offline video
        rendering uses set_animation_time() instead for frame-exact,
        deterministic timing."""
        if self.animation is None:
            return
        self.anim_time += max(0.0, dt)
        self.animation.apply(self, self.anim_time)

    def set_animation_time(self, t):
        """Jumps the attached animation (if any) directly to absolute time
        `t` seconds, ignoring anim_time. No-op if no animation is
        attached. Used for deterministic offline video rendering -- see
        RayTracer.set_spotlight_animation_time."""
        if self.animation is None:
            return
        self.anim_time = t
        self.animation.apply(self, t)

    def to_dict(self):
        d = {'position': [float(x) for x in self.position],
             'direction': [float(x) for x in self.direction],
             'color': [int(round(c * 255)) for c in self.color],
             'brightness': self.brightness, 'cone_angle': self.cone_angle,
             'softness': self.softness}
        if self.animation is not None:
            d['animation'] = self.animation.to_dict()
        return d

    @staticmethod
    def from_dict(d):
        sl = SpotLight(tuple(d['position']), tuple(d['direction']),
                        tuple(d.get('color', (255, 255, 255))),
                        float(d.get('brightness', 2.0)), float(d.get('cone_angle', 25.0)),
                        float(d.get('softness', 0.35)))
        anim_d = d.get('animation')
        if anim_d:
            sl.animation = SpotLightAnimation.from_dict(anim_d)
        return sl


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
CLOUD_ENABLED = CLOUD_BASE = CLOUD_TOP = CLOUD_DENSITY = CLOUD_COVERAGE = None
CLOUD_SCALE = CLOUD_COLOR = CLOUD_WIND = CLOUD_STEPS = CLOUD_LIGHT_STEPS = CLOUD_SUN_DIR = None
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


def alloc_cloud_fields():
    """Allocates the volumetric-cloud parameter fields (called once,
    unconditionally, same as alloc_light_fields/alloc_water_fields -- the
    cloud layer defaults to disabled (CLOUD_ENABLED=0), see
    RayTracer.set_clouds to turn it on). See the "Volumetric clouds"
    section above (near _hash1/_cloud_density/_cloud_march) for how these
    are actually used."""
    global CLOUD_ENABLED, CLOUD_BASE, CLOUD_TOP, CLOUD_DENSITY, CLOUD_COVERAGE
    global CLOUD_SCALE, CLOUD_COLOR, CLOUD_WIND, CLOUD_STEPS, CLOUD_LIGHT_STEPS, CLOUD_SUN_DIR
    CLOUD_ENABLED = ti.field(ti.i32, shape=())
    CLOUD_BASE = ti.field(ti.f32, shape=())
    CLOUD_TOP = ti.field(ti.f32, shape=())
    CLOUD_DENSITY = ti.field(ti.f32, shape=())
    CLOUD_COVERAGE = ti.field(ti.f32, shape=())
    CLOUD_SCALE = ti.field(ti.f32, shape=())
    CLOUD_COLOR = ti.Vector.field(3, ti.f32, shape=())
    CLOUD_WIND = ti.Vector.field(2, ti.f32, shape=())
    CLOUD_STEPS = ti.field(ti.i32, shape=())
    CLOUD_LIGHT_STEPS = ti.field(ti.i32, shape=())
    CLOUD_SUN_DIR = ti.Vector.field(3, ti.f32, shape=())
    CLOUD_ENABLED[None] = 0
    CLOUD_BASE[None] = 100.0
    CLOUD_TOP[None] = 160.0
    CLOUD_DENSITY[None] = 0.08
    CLOUD_COVERAGE[None] = 0.5
    CLOUD_SCALE[None] = 60.0
    CLOUD_COLOR[None] = [1.0, 1.0, 1.0]
    CLOUD_WIND[None] = [0.0, 0.0]
    CLOUD_STEPS[None] = 24
    CLOUD_LIGHT_STEPS[None] = 4
    CLOUD_SUN_DIR[None] = [0.0, 1.0, 0.0]

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


# =============================================================================
# Volumetric clouds (v12) -- a horizontal noise-density layer between
# CLOUD_BASE and CLOUD_TOP (world Y), raymarched twice per relevant ray:
#   - render_sample composites it visually over whatever the primary ray
#     resolves to (sky or geometry), with a short secondary march toward
#     the sun for self-shadowing (see _cloud_march).
#   - _shadow_throughput marches through it too, on every light's shadow
#     ray -- this is what actually makes it cast real shadows on geometry
#     below it (see _cloud_shadow_transmittance), not a separate/faked
#     effect layered on top.
# All parameters (CLOUD_BASE/TOP/DENSITY/COVERAGE/SCALE/COLOR/WIND/STEPS/
# LIGHT_STEPS/SUN_DIR/ENABLED) are global fields (see alloc_cloud_fields),
# the same pattern as LIGHT_POS/SPOT_POS etc. -- set via
# RayTracer.set_clouds, not kernel arguments.
# =============================================================================

@ti.func
def _hash1(p: vec3) -> ti.f32:
    # Cheap deterministic pseudo-random scalar in [0, 1) from a 3D point --
    # doesn't need to be a good general-purpose PRNG, just look unrelated
    # from one grid cell to its neighbors. Same "irrational-frequency sine
    # hash" trick as the Python-side _step_noise (footstep shake variation),
    # just written for use inside a kernel.
    n = p[0] * 127.1 + p[1] * 311.7 + p[2] * 74.7
    x = ti.sin(n) * 43758.5453
    return x - ti.floor(x)


@ti.func
def _value_noise3(p: vec3) -> ti.f32:
    # Trilinear value noise: hash the 8 corners of the unit cell containing
    # p, blend with a smootherstep-eased fractional part (6t^5-15t^4+10t^3,
    # Perlin's improved easing curve -- avoids the grid-aligned creases a
    # plain linear blend leaves). Standard/cheap; written from scratch since
    # a Taichi kernel can't call out to an external noise library.
    i = ti.floor(p)
    f = p - i
    u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)

    c000 = _hash1(i + vec3(0.0, 0.0, 0.0))
    c100 = _hash1(i + vec3(1.0, 0.0, 0.0))
    c010 = _hash1(i + vec3(0.0, 1.0, 0.0))
    c110 = _hash1(i + vec3(1.0, 1.0, 0.0))
    c001 = _hash1(i + vec3(0.0, 0.0, 1.0))
    c101 = _hash1(i + vec3(1.0, 0.0, 1.0))
    c011 = _hash1(i + vec3(0.0, 1.0, 1.0))
    c111 = _hash1(i + vec3(1.0, 1.0, 1.0))

    x00 = c000 + (c100 - c000) * u[0]
    x10 = c010 + (c110 - c010) * u[0]
    x01 = c001 + (c101 - c001) * u[0]
    x11 = c011 + (c111 - c011) * u[0]
    y0 = x00 + (x10 - x00) * u[1]
    y1 = x01 + (x11 - x01) * u[1]
    return y0 + (y1 - y0) * u[2]


@ti.func
def _cloud_fbm(p: vec3) -> ti.f32:
    # 4-octave fractal sum -- each octave doubles frequency and halves
    # amplitude, which is what turns smooth blobby value-noise into the
    # more detailed "cauliflower" look real cloud density fields have.
    total = 0.0
    amp = 0.5
    freq = 1.0
    for _ in range(4):
        total += _value_noise3(p * freq) * amp
        freq *= 2.02   # NOT exactly 2.0 -- keeps the octaves' grid cells
                        # from ever lining back up on top of each other
        amp *= 0.5
    return total


@ti.func
def _cloud_density(p: vec3) -> ti.f32:
    # Zero outside [CLOUD_BASE, CLOUD_TOP], with a soft fade-in/out near
    # each edge (rather than a hard cutoff) so the layer doesn't look like
    # a slab sliced off flat top and bottom.
    base = CLOUD_BASE[None]
    top = CLOUD_TOP[None]
    thickness = ti.max(1e-3, top - base)
    edge = ti.min(thickness * 0.3, 8.0)
    y = p[1]
    vert = 0.0
    if base <= y <= top:
        vert = ti.min(1.0, (y - base) / ti.max(1e-3, edge)) * \
               ti.min(1.0, (top - y) / ti.max(1e-3, edge))
    d = 0.0
    if vert > 0.0:
        scale = 1.0 / ti.max(1e-3, CLOUD_SCALE[None])
        wind = CLOUD_WIND[None]
        wp = vec3((p[0] + wind[0]) * scale, y * scale * 0.5, (p[2] + wind[1]) * scale)
        n = _cloud_fbm(wp)
        # `coverage` (0..1) raises/lowers how much of the noise has to
        # clear before it counts as any density at all -- low coverage =
        # mostly clear sky with a few dense puffs, high coverage = a
        # solid overcast sheet.
        shaped = ti.max(0.0, n - (1.0 - CLOUD_COVERAGE[None]))
        d = shaped * vert * CLOUD_DENSITY[None]
    return d


@ti.func
def _cloud_slab_interval(ro: vec3, rd: vec3, tmax: ti.f32):
    # Clips (0, tmax) down to whatever sub-segment of the ray actually
    # lies inside the [CLOUD_BASE, CLOUD_TOP] altitude band -- a horizontal
    # slab is just two Y planes, so this is a 1D interval clip against
    # each plane's t value. Returns (t0, t1); t1 <= t0 means "doesn't
    # cross the slab at all" (ray runs parallel to it from outside, or the
    # whole (0, tmax) segment is on one side).
    base = CLOUD_BASE[None]
    top = CLOUD_TOP[None]
    t0 = 0.0
    t1 = tmax
    if ti.abs(rd[1]) < 1e-6:
        if ro[1] < base or ro[1] > top:
            t1 = t0 - 1.0
    else:
        ta = (base - ro[1]) / rd[1]
        tb = (top - ro[1]) / rd[1]
        t0 = ti.max(t0, ti.min(ta, tb))
        t1 = ti.min(t1, ti.max(ta, tb))
    return t0, t1


@ti.func
def _cloud_march(ro: vec3, rd: vec3, tmax: ti.f32):
    # Marches the ray segment that lies inside the cloud slab (clipped to
    # tmax, so it never marches past whatever the camera ray already hit),
    # front-to-back accumulating optical depth (-> transmittance) and a
    # single-scatter estimate of in-scattered sunlight. Each step's
    # contribution to `scattered` is weighted by the transmittance left
    # AND by a short secondary march toward the sun (self-shadowing: a
    # thick cloud's underside reads darker than its sunlit top/edges).
    # Returns (transmittance, scattered_color) -- composite as
    # `result = behind_color * transmittance + scattered_color`.
    transmittance = 1.0
    scattered = vec3(0.0)
    t0, t1 = _cloud_slab_interval(ro, rd, tmax)

    if t1 > t0:
        steps = CLOUD_STEPS[None]
        seg = (t1 - t0) / steps
        # A random jitter on the starting offset breaks up the visible
        # banding a fixed-step march would otherwise leave in thick cloud.
        t = t0 + seg * ti.random(ti.f32)
        light_steps = CLOUD_LIGHT_STEPS[None]
        light_dir = CLOUD_SUN_DIR[None]
        cloud_col = CLOUD_COLOR[None]
        for _ in range(steps):
            if transmittance > 1e-3:
                p = ro + rd * t
                dens = _cloud_density(p)
                if dens > 1e-4:
                    step_optical = dens * seg
                    step_transmit = ti.exp(-step_optical)
                    light_seg = ti.max(1e-3, CLOUD_TOP[None] - CLOUD_BASE[None]) / light_steps
                    light_optical = 0.0
                    lt = light_seg * 0.5
                    for _l in range(light_steps):
                        light_optical += _cloud_density(p + light_dir * lt) * light_seg
                        lt += light_seg
                    light_transmit = ti.exp(-light_optical)
                    # 0.35 floor -- even a fully self-shadowed cloud interior
                    # still gets some ambient sky-bounce light, real clouds
                    # are never lit purely from one direction.
                    lit_color = cloud_col * (0.35 + 0.65 * light_transmit)
                    scattered += transmittance * (1.0 - step_transmit) * lit_color
                    transmittance *= step_transmit
            t += seg
    return transmittance, scattered


@ti.func
def _cloud_shadow_transmittance(ro: vec3, rd: vec3, tmax: ti.f32) -> ti.f32:
    # Lighter-weight than _cloud_march (no self-shadow inner loop, no color
    # accumulation -- a shadow ray only needs "how much light gets
    # through", not what the cloud looks like) -- this is what makes a
    # dense/thick cloud actually darken the ground underneath it: every
    # shadow ray (see _shadow_throughput) marches through this same
    # density field between the surface point and the light.
    transmittance = 1.0
    t0, t1 = _cloud_slab_interval(ro, rd, tmax)
    if t1 > t0:
        steps = CLOUD_LIGHT_STEPS[None] * 2  # coarser than the visual march is fine -- shadow-only
        seg = (t1 - t0) / steps
        # Random (not fixed-offset) starting jitter -- same reason as
        # _cloud_march's: a FIXED per-step sample offset makes every
        # shadow ray in a given direction quantize against the same
        # underlying step grid, which shows up as hard, faceted/terraced
        # blob edges (structured aliasing) rather than soft ones. Random
        # jitter turns that into unstructured grain instead, which many
        # samples (or a denoiser) average into a smooth gradient.
        t = t0 + seg * ti.random(ti.f32)
        optical = 0.0
        for _ in range(steps):
            optical += _cloud_density(ro + rd * t) * seg
            t += seg
        transmittance = ti.exp(-optical)
    return transmittance


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

    if CLOUD_ENABLED[None] != 0:
        # This is the actual "clouds cast shadows on objects" mechanism --
        # not a separate/precomputed effect, just this same shadow ray
        # additionally marching through the cloud density field between
        # the surface and the light (see _cloud_shadow_transmittance).
        # Runs once per shadow ray regardless of whether an opaque
        # occluder is also found below, same as the BVH occluder loop.
        # (No early-return on a fully-clouded-out ray here -- Taichi
        # doesn't support `return` inside a non-static if/for -- the loop
        # below already checks throughput.max() < 1e-3 each iteration and
        # exits immediately via its own `break`, at the cost of one wasted
        # BVH query if the clouds alone already zeroed it out.)
        throughput *= _cloud_shadow_transmittance(ro, ldir, light_dist)

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
        depth_written = False
        primary_dir = ray_dir   # saved before the bounce loop mutates ray_dir (reflections) --
        primary_t = 1.0e18      # see the cloud composite after the loop, below

        for bounce in range(max_bounce + 1):
            if terminated:
                continue
            tid, t, bu, bv = _bvh_closest_hit(ray_o, ray_dir, 1e18)

            if tid < 0:
                if not depth_written:
                    DEPTH_ACCUM[py, px] += 1.0e4
                    depth_written = True
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
                # A transparent region of the image! Push the ray origin
                # through this plane WITHOUT recording depth here -- an
                # alpha-cutout pixel isn't really "there", so the depth
                # buffer (which DoF uses for its blur radius) should reflect
                # whatever is actually visible once the ray keeps going
                # (the real background/geometry behind it), not this
                # transparent quad's own (usually much nearer) distance.
                # Recording depth unconditionally at bounce 0 here used to
                # be exactly why DoF left a sharp, unblurred rectangle
                # showing an image plane's full bounding box: the see-
                # through parts kept that near depth and were treated as
                # "in focus" even though the color actually shown there
                # comes from something far behind it.
                ray_o = p + ray_dir * 1e-3
                continue  # move on to the next bounce without stopping the ray here

            if not depth_written:
                DEPTH_ACCUM[py, px] += t
                depth_written = True
                primary_t = t

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

        if CLOUD_ENABLED[None] != 0:
            # Composited once per pixel/sample against the PRIMARY ray only
            # (not each reflection bounce -- see set_clouds' docstring) --
            # tmax is the first real (non-cutout) hit's distance, or the
            # 1e18 default if the ray never hit anything (open sky), so
            # clouds correctly appear in front of whatever's behind them
            # without needing to know what that was.
            cloud_tr, cloud_col = _cloud_march(cam_pos, primary_dir, primary_t)
            final_color = final_color * cloud_tr + cloud_col

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
    __slots__ = ("pos", "yaw", "pitch", "speed", "duration", "zoom")

    def __init__(self, pos, yaw, pitch, speed=1.0, duration=None, zoom=1.0):
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
        # Optical zoom multiplier (see RayTracer.set_zoom) AT this keyframe --
        # interpolated linearly across a segment by CameraPath.zoom_at, so a
        # video can rack the zoom in/out over a path the same way it does
        # yaw/pitch/position. 1.0 = unzoomed. Kept as a plain per-keyframe
        # value (not folded into sample()'s return tuple -- see zoom_at's
        # docstring for why) so it doesn't disturb every existing caller of
        # sample().
        self.zoom = max(1.0, float(zoom))

    def to_dict(self):
        d = {'pos': [float(x) for x in self.pos], 'yaw': self.yaw,
             'pitch': self.pitch, 'speed': self.speed}
        if self.duration is not None:
            d['duration'] = self.duration
        if abs(self.zoom - 1.0) > 1e-6:
            d['zoom'] = self.zoom
        return d


def _kick(x):
    """Footstep "impact" shape: a sharp single-lobe jolt (fast rise, fast
    decay, peak of 1.0 at x=1/12, ~0 by x=0.5) plus a light damped-
    oscillation tail after it -- a real body's mass keeps gently
    rebounding for a moment after each footfall instead of the motion
    stopping dead the instant the sharp impact decays, which is what made
    the un-tailed version read as a metronome click rather than a step.
    Both terms are ~0 at x=0 and x->1, so consecutive footfalls (x wraps
    back to 0 every cycle) still join up continuously. x is expected in
    [0, 1) (progress through one footfall's cycle)."""
    if not (0.0 <= x < 1.0):
        return 0.0
    primary = 32.6 * x * math.exp(-12.0 * x)
    tail = 0.12 * math.sin(x * 18.0) * math.exp(-6.0 * x)
    return primary + tail


def _smoothstep(edge0, edge1, x):
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def _step_noise(seed_val):
    """Deterministic pseudo-random value in [-1, 1] from a float seed --
    used to give each individual footfall its own small, repeatable
    variation (see _gait_offset) instead of every step reproducing the
    exact same kick shape/amplitude/timing, which is what made the old
    footstep shake read as a metronome rather than an actual gait. Not
    np.random (see _handheld_offset's docstring for why: needs to be a
    pure function of its input, not stateful, so motion-blur sub-samples
    and replays agree). A cheap irrational-frequency sine hash is plenty
    for this -- it doesn't need to be a good general-purpose PRNG, just
    unrelated-looking from one integer step index to the next."""
    x = math.sin(seed_val * 127.1 + 43758.5453) * 43758.5453
    return 2.0 * (x - math.floor(x)) - 1.0


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
        # --- Camera "inertia" (momentum/overshoot on direction changes) ---
        # 0.0 = old behaviour, dead-straight linear interpolation between
        # keyframes with instantaneous direction changes at each one. > 0
        # runs the raw linear path through a damped spring (see
        # _build_inertia_cache) so position/yaw/pitch keep some momentum
        # instead of snapping onto the new segment's velocity -- a fast
        # pan left immediately followed by a pan right will overshoot left
        # a bit and "bounce" back onto the new direction, the way a real
        # handheld camera (or its operator's arm) can't instantly reverse.
        # Distinct from handheld_shake (that's noise layered on top of an
        # otherwise-followed path; this instead changes HOW the path
        # itself is followed) -- the two stack fine together.
        self.inertia = 0.0
        # How springy/bouncy the overshoot is: 0 = critically damped (no
        # overshoot at all, just a smoothed-out lag), higher = underdamped
        # (visibly bounces past the target 1+ times before settling).
        self.inertia_bounce = 0.35
        self._inertia_cache = None  # see _build_inertia_cache

    def _raw_pose_at(self, t):
        """Plain linear keyframe interpolation at time t, WITHOUT inertia
        smoothing or handheld/footstep shake -- returns (pos, yaw, pitch,
        phase_acc, seg_speed) where phase_acc/seg_speed feed _gait_offset.
        This is the ground-truth trajectory that _build_inertia_cache
        resamples and smooths; sample() below adds shake back on top of
        the (possibly inertia-smoothed) result."""
        n = len(self.keyframes)
        if n == 0:
            return None
        if n == 1:
            kf = self.keyframes[0]
            return kf.pos.copy(), kf.yaw, kf.pitch, 0.0, 0.0
        durs = self.segment_durations()
        total = sum(durs)
        t_clamped = max(0.0, min(t, total))
        acc = 0.0
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
                return pos, float(yaw), float(pitch), phase_acc, seg_speed
            phase_acc += seg_len / seg_stride if seg_stride > 1e-6 else 0.0
            acc += d
        kf = self.keyframes[-1]
        return kf.pos.copy(), kf.yaw, kf.pitch, phase_acc, 0.0

    def zoom_at(self, t):
        """Plain linear interpolation of each keyframe's `zoom` field at
        time t -- kept separate from _raw_pose_at/sample() rather than
        folded into their return tuple, since sample() is called from a
        handful of places (interactive live view, render_to_file,
        render_video, and SensorCameraData's own sample() with a DIFFERENT
        signature) and changing that shared tuple's shape would risk
        breaking all of them. render_video's zoom ramp (zoom_speed) uses
        this as the RACK TARGET for each frame -- the actual rendered zoom
        eases toward it rather than snapping, so an abrupt jump between two
        keyframes' zoom values still reads as a lens racking, not a cut.
        No inertia smoothing (unlike pos/yaw/pitch) -- zoom_speed already
        provides the "can't change instantly" easing."""
        n = len(self.keyframes)
        if n == 0:
            return 1.0
        if n == 1:
            return self.keyframes[0].zoom
        durs = self.segment_durations()
        total = sum(durs)
        t_clamped = max(0.0, min(t, total))
        acc = 0.0
        for i, d in enumerate(durs):
            a, b = self.keyframes[i], self.keyframes[i + 1]
            last = (i == len(durs) - 1)
            if t_clamped <= acc + d or last:
                local_t = (t_clamped - acc) / d if d > 1e-9 else 1.0
                local_t = max(0.0, min(1.0, local_t))
                return a.zoom + (b.zoom - a.zoom) * local_t
            acc += d
        return self.keyframes[-1].zoom

    def _inertia_signature(self):
        """Cheap "did anything the cache depends on change" fingerprint --
        rebuilt lazily in _sample_smoothed only when this changes, so a
        long interactive session doesn't re-simulate the spring every
        single call."""
        return (id(self.keyframes), len(self.keyframes), self.total_duration(),
                self.inertia, self.inertia_bounce,
                tuple(round(kf.yaw, 6) for kf in self.keyframes),
                tuple(round(float(x), 6) for kf in self.keyframes for x in kf.pos))

    def _build_inertia_cache(self):
        """Pre-computes a smoothed (pos, yaw, pitch) trajectory by running
        the RAW linear path (see _raw_pose_at) through a damped spring
        that chases it over time, then samples/caches that spring's
        output on a dense fixed time grid.

        Why precompute on a grid instead of smoothing live inside
        sample(): sample() is called with arbitrary, sometimes
        out-of-order t (render_video's motion blur jitters t backward/
        forward within each frame's shutter window, and --multi-gpu
        splits a video into independent per-process time BLOCKS) -- a
        naive "step the spring forward from the last t seen" approach
        would make the result depend on call order/history, so different
        frames (or GPU workers) of the same render could disagree about
        where the camera was at the same t. Simulating the whole path
        once, forward in time, up front sidesteps that: every later
        sample(t) just looks up the same precomputed curve regardless of
        what was queried before or by whom, so motion-blur sub-samples
        and multi-GPU blocks all agree, exactly like the un-smoothed path
        already did.

        The spring is a standard mass-spring-damper (critically-damped-
        style harmonic oscillator, semi-implicit Euler integration):
        stiffness comes from self.inertia (higher = takes LONGER to snap
        back onto the raw path, i.e. more perceived momentum -- so
        "inertia" here is really a settling-time dial, not literal mass),
        and self.inertia_bounce sets the damping ratio (low damping = more
        overshoot/bounce past the raw path before it settles).
        """
        n = len(self.keyframes)
        if n < 2 or self.inertia <= 1e-6:
            self._inertia_cache = None
            return
        total = self.total_duration()
        if total <= 1e-6:
            self._inertia_cache = None
            return
        # Fixed grid fine enough to resolve fast direction changes (short
        # segments/holds) without needing an adaptive step -- 240 Hz is
        # comfortably above any video frame rate and any plausible
        # keyframe-segment duration used interactively.
        sim_hz = 240.0
        sim_dt = 1.0 / sim_hz
        # A little pre-roll and post-roll so the spring can settle onto
        # the FIRST keyframe's pose before t=0 (instead of starting the
        # simulation already "at rest" exactly on the raw path, which
        # would silently disable inertia for the very first segment) and
        # keep bouncing a bit past the last keyframe instead of being cut
        # off mid-overshoot.
        pre_roll = 1.5
        post_roll = 1.5
        t0 = -pre_roll
        t1 = total + post_roll
        n_steps = max(2, int(math.ceil((t1 - t0) / sim_dt)) + 1)

        # Spring tuning: inertia in [0, 3] (same range as handheld_shake's
        # slider) maps to a settling time constant -- higher inertia means
        # the camera takes LONGER to catch up to the raw path (more
        # perceived momentum), so stiffness is inversely related to it.
        # omega_n is the spring's natural frequency (rad/s); zeta is the
        # damping ratio (0 = undamped/never settles, 1 = critically
        # damped/no overshoot, >1 = overdamped/sluggish with no overshoot).
        time_constant = 0.08 + 0.6 * min(self.inertia, 3.0)
        omega_n = 1.0 / max(1e-3, time_constant)
        zeta = max(0.05, 1.0 - max(0.0, min(1.0, self.inertia_bounce)) * 0.85)

        raw0 = self._raw_pose_at(max(0.0, t0))
        pos = raw0[0].astype(np.float64).copy()
        yaw = float(raw0[1])
        pitch = float(raw0[2])
        vel_pos = np.zeros(3, dtype=np.float64)
        vel_yaw = 0.0
        vel_pitch = 0.0

        grid_pos = np.empty((n_steps, 3), dtype=np.float32)
        grid_yaw = np.empty(n_steps, dtype=np.float32)
        grid_pitch = np.empty(n_steps, dtype=np.float32)
        grid_phase = np.empty(n_steps, dtype=np.float32)
        grid_speed = np.empty(n_steps, dtype=np.float32)

        k = omega_n * omega_n
        c = 2.0 * zeta * omega_n
        prev_target_yaw = yaw
        # Footstep phase/speed are re-derived from the SPRING's own smoothed
        # velocity here (see below), not the raw path's declared per-segment
        # speed -- see the note on grid_phase/grid_speed just after the loop
        # for why: without this, footstep shake pops in intensity/timing at
        # every keyframe boundary even though the position itself is smooth.
        smoothed_phase = 0.0
        for i in range(n_steps):
            t = t0 + i * sim_dt
            t_query = max(0.0, min(total, t))
            r_pos, r_yaw, r_pitch, _raw_phase, _raw_speed = self._raw_pose_at(t_query)
            r_pos = r_pos.astype(np.float64)
            # Unwrap the raw target's yaw against the PREVIOUS target (not
            # against the spring's own yaw) so a raw path that itself
            # wraps through +-pi doesn't confuse the spring into thinking
            # it needs to swing all the way around.
            r_yaw = prev_target_yaw + _shortest_angle_diff(prev_target_yaw, r_yaw)
            prev_target_yaw = r_yaw

            acc_pos = k * (r_pos - pos) - c * vel_pos
            vel_pos = vel_pos + acc_pos * sim_dt
            pos = pos + vel_pos * sim_dt

            acc_yaw = k * (r_yaw - yaw) - c * vel_yaw
            vel_yaw = vel_yaw + acc_yaw * sim_dt
            yaw = yaw + vel_yaw * sim_dt

            acc_pitch = k * (r_pitch - pitch) - c * vel_pitch
            vel_pitch = vel_pitch + acc_pitch * sim_dt
            pitch = pitch + vel_pitch * sim_dt

            # Speed actually being displayed right now (the spring's own
            # velocity), not the target segment's nominal speed -- these
            # differ noticeably right after a keyframe boundary (or any
            # abrupt speed change) while the spring is still catching up,
            # which is exactly the moment footstep shake used to pop.
            smoothed_speed = float(np.linalg.norm(vel_pos))
            stride = _stride_length(smoothed_speed)
            if stride > 1e-6:
                smoothed_phase += (smoothed_speed * sim_dt) / stride

            grid_pos[i] = pos
            grid_yaw[i] = yaw
            grid_pitch[i] = pitch
            grid_phase[i] = smoothed_phase
            grid_speed[i] = smoothed_speed

        self._inertia_cache = {
            'sig': self._inertia_signature(),
            't0': t0, 'dt': sim_dt, 'n': n_steps,
            'pos': grid_pos, 'yaw': grid_yaw, 'pitch': grid_pitch,
            # NOTE: unlike pos/yaw/pitch, `phase`/`speed` here are derived
            # from the SPRING's smoothed velocity (see the loop above), not
            # a resample of _raw_pose_at's declared per-segment speed --
            # _gait_offset uses these to decide how hard/fast to shake, so
            # feeding it the actual on-screen motion (instead of a value
            # that can jump the instant a new segment starts) keeps the
            # footstep shake's timing and intensity continuous through
            # keyframe transitions, matching what inertia already did for
            # position/yaw/pitch.
            'phase': grid_phase, 'speed': grid_speed,
        }

    def _sample_smoothed(self, t):
        """Like _raw_pose_at, but with inertia smoothing applied when
        self.inertia > 0 (rebuilding the cached spring trajectory first
        if keyframes/inertia settings changed since it was last built).
        Falls back to _raw_pose_at unchanged when inertia is off, so
        inertia == 0.0 reproduces the exact old behaviour."""
        if len(self.keyframes) < 2 or self.inertia <= 1e-6:
            return self._raw_pose_at(t)
        if self._inertia_cache is None or self._inertia_cache['sig'] != self._inertia_signature():
            self._build_inertia_cache()
        cache = self._inertia_cache
        if cache is None:
            return self._raw_pose_at(t)
        # Look up the smoothed curve at t via linear interpolation between
        # the two nearest grid samples -- the grid is dense (240 Hz)
        # relative to anything that queries it, so this is visually
        # indistinguishable from re-simulating at the exact t.
        local = (t - cache['t0']) / cache['dt']
        i0 = max(0, min(cache['n'] - 1, int(math.floor(local))))
        i1 = min(cache['n'] - 1, i0 + 1)
        frac = max(0.0, min(1.0, local - i0))
        pos = cache['pos'][i0] + (cache['pos'][i1] - cache['pos'][i0]) * frac
        yaw = cache['yaw'][i0] + (cache['yaw'][i1] - cache['yaw'][i0]) * frac
        pitch = cache['pitch'][i0] + (cache['pitch'][i1] - cache['pitch'][i0]) * frac
        phase = cache['phase'][i0] + (cache['phase'][i1] - cache['phase'][i0]) * frac
        speed = cache['speed'][i0] + (cache['speed'][i1] - cache['speed'][i0]) * frac
        return pos.astype(np.float32), float(yaw), float(pitch), float(phase), float(speed)

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

        Each footfall's impact is additionally given small, deterministic
        per-step variation (amplitude/timing/left-right asymmetry -- see
        the step_amp_jit/step_time_jit/leftright_bias block below and
        _step_noise) so consecutive steps don't repeat an identical shape;
        without that, footstep shake reads as a mechanical metronome tick
        rather than an actual gait.
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
        s = self.handheld_seed

        # Per-footfall variation: a REAL gait never repeats the exact same
        # impact twice -- force, timing and left/right balance all wobble
        # a little step to step. Feeding _kick() the identical shape every
        # single time (the old behaviour) is what made footstep shake read
        # as a mechanical metronome instead of an actual walk/run. These
        # are small, deterministic per-step offsets (see _step_noise) keyed
        # off the integer step index (plus the path's own seed, so
        # different paths don't all wobble in lockstep): +/-16% impact
        # strength, +/-5% of a step's timing, and a persistent-but-noisy
        # left/right asymmetry (real gaits are rarely perfectly symmetric
        # between feet).
        step_amp_jit = 1.0 + 0.16 * _step_noise(step_idx + s * 7.0)
        step_time_jit = 0.05 * _step_noise(step_idx + s * 3.3 + 11.0)
        leftright_jit = (1.0 if (int(step_idx) % 2 == 0) else -1.0)
        leftright_bias = 1.0 + 0.08 * leftright_jit + 0.05 * _step_noise(step_idx + s * 5.1 + 23.0)
        within_step_j = min(0.999, max(0.0, within_step + step_time_jit))
        impact = _kick(within_step_j) * step_amp_jit * leftright_bias
        foot_sign = 1.0 if (int(step_idx) % 2 == 0) else -1.0

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

    def add(self, pos, yaw, pitch, speed=1.0, duration=None, zoom=1.0):
        self.keyframes.append(CameraKeyframe(pos, yaw, pitch, speed, duration, zoom))
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
        return self.add(last.pos.copy(), new_yaw, new_pitch, speed=last.speed,
                         duration=duration, zoom=last.zoom)

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
        interpolated along the path -- LINEARLY when self.inertia == 0.0
        (old behaviour, unchanged), or through a damped-spring "momentum"
        filter when self.inertia > 0 (see _build_inertia_cache: direction
        changes then ease/overshoot instead of snapping instantly). With
        only 1 keyframe, always returns that keyframe (camera stays still
        -- inertia has nothing to smooth with just one pose). Returns None
        if the path is empty. Handheld sway (_handheld_offset) and, once
        actually moving, footstep shake (_gait_offset) are both added on
        top of the (possibly inertia-smoothed) interpolated pose -- see
        their docstrings."""
        if not self.keyframes:
            return None
        pos, yaw, pitch, phase_acc, seg_speed = self._sample_smoothed(t)
        # Handheld shake is evaluated at the UN-clamped t (not any
        # internally-clamped local time) so it keeps evolving smoothly even
        # if a caller samples slightly past the path's end -- only matters
        # at the boundary, keeps the wobble continuous there.
        pos_off, yaw_off, pitch_off, roll_off = self._handheld_offset(t)
        gpos, gyaw, gpitch, groll = self._gait_offset(t, seg_speed, phase_acc)
        return (pos + pos_off + gpos, yaw + yaw_off + gyaw,
                pitch + pitch_off + gpitch, roll_off + groll)

    def to_list(self):
        return [kf.to_dict() for kf in self.keyframes]

    @staticmethod
    def from_list(items):
        cp = CameraPath()
        for kf in items:
            cp.add(kf['pos'], kf.get('yaw', 0.0), kf.get('pitch', 0.0),
                    kf.get('speed', 1.0), kf.get('duration'), kf.get('zoom', 1.0))
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

    def zoom_at(self, t):
        """Sensor-recorded camera data doesn't carry a zoom track (it's an
        external position/rotation log, not something authored with this
        renderer's zoom in mind), so this is a constant 1.0 -- exists purely
        so callers (render_video's zoom ramp) can treat CameraPath and
        SensorCameraData the same way without a hasattr check."""
        return 1.0

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

def _compute_video_frame_plan(camera_path, fps, duration, camera_sync):
    """Returns (n_frames, frame_times_or_None, total_time, camera_sync) --
    the same frame-count/timing logic render_video() uses to decide how
    many frames a video will have, factored out so the --multi-gpu video
    orchestrator (_run_multi_gpu_video) can compute frame-block boundaries
    WITHOUT actually rendering anything."""
    is_sensor = camera_path.is_camera_data()
    n_kf = 0 if is_sensor else len(camera_path.keyframes)
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
    return n_frames, frame_times, total_time, camera_sync


class RayTracer:
    def __init__(self, scene: Scene, width=480, height=480, fov=60,
                 max_bounce=DEFAULT_MAX_BOUNCE, background=None, lights=None, spotlights=None,
                 live_max_bounce=LIVE_MAX_BOUNCE):
        self.scene = scene
        self.fov_rad = math.radians(fov)
        self.fov_deg = fov
        self.half_tan = math.tan(self.fov_rad / 2)
        # --- Optical zoom (;/' keys -- see set_zoom) ---
        # `base_fov_deg` is the UNZOOMED field of view (the `fov` this
        # tracer was constructed with); `zoom` is the current multiplier
        # applied on top of it (1.0 = no zoom, e.g. 2.0 = half the FOV,
        # "2x zoomed in"), and fov_deg/fov_rad/half_tan above always
        # reflect base_fov_deg / zoom -- see set_zoom for how that's kept
        # in sync. `zoom_target` is what the current zoom is chasing
        # (held ;/' keys move the target; render_video eases the actual
        # zoom toward it at `zoom_speed` per second rather than snapping,
        # like a real zoom lens racking, not a hard cut).
        self.base_fov_deg = fov
        self.zoom = 1.0
        self.zoom_target = 1.0
        self.zoom_speed = 1.0   # zoom-multiplier change per second
        self.zoom_min = 1.0
        self.zoom_max = 8.0
        self._zoom_kick_smooth = 0.0  # see render_video's zoom ramp / apply_zoom_focal_shift
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

        self.clouds = None  # see set_clouds -- disabled by default
        alloc_cloud_fields()
        self._sync_cloud_fields()

        self.width = self.height = 0
        self.aspect = 1.0
        self._alloc_buffers(width, height)

    def update_spotlight_animations(self, dt):
        """Advances every spotlight's attached SpotLightAnimation (if any)
        by dt wall-clock seconds -- used by the interactive live view, once
        per frame. Does NOT itself call sync_lights()/compute_caustics():
        sync_lights() already runs on every add_samples() call when live
        raytracing is on, so it's cheap to let that pick up the new
        position/color/etc. automatically; compute_caustics() is a full
        photon pre-pass and deliberately NOT re-run every frame (same
        limitation as moving a point light with K -- caustics only update
        on an explicit action) since doing so continuously would tank
        interactive framerate. Returns True if anything was animated (so
        callers that DON'T already call sync_lights() every frame, e.g. a
        non-live preview, know they should call it themselves)."""
        moved = False
        for sl in self.spotlights:
            if sl.animation is not None:
                sl.update_animation(dt)
                moved = True
        return moved

    def set_spotlight_animation_time(self, t):
        """Jumps every animated spotlight directly to absolute time `t`
        seconds -- used for deterministic, frame-exact spotlight motion
        during offline video rendering (render_video), as opposed to
        update_spotlight_animations()'s wall-clock dt stepping for the
        live view. Returns True if anything was animated."""
        moved = False
        for sl in self.spotlights:
            if sl.animation is not None:
                sl.set_animation_time(t)
                moved = True
        return moved

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

    def add_sun_light(self, azimuth_deg=45.0, elevation_deg=35.0, color=(255, 244, 214),
                      brightness=1.0, distance=4000.0):
        """Adds a REAL point Light very far away in the sun's direction, so
        it casts actual diffuse/specular highlights and shadows on scene
        geometry -- not just the visual/ambient glow from
        Background.set_sun. This works with NO rendering-kernel changes
        because point lights here have no distance falloff at all (see
        render_sample's direct-lighting loop: `lcol * lbri`, no 1/d^2
        term), so one placed far enough away behaves exactly like a true
        directional/sun light while reusing the existing point-light code
        path unchanged. `azimuth_deg`/`elevation_deg` use the same
        convention as Background.set_sun -- pass the same values to both
        (or just use add_sun, which does so for you) so the visual disc
        and the real light line up. Counts against MAX_LIGHTS like any
        other light -- sync_lights() will raise if you go over.
        Returns the Light (dim it later with `light.brightness = ...;
        tracer.sync_lights()`)."""
        az = math.radians(azimuth_deg)
        el = math.radians(elevation_deg)
        dx, dz = math.sin(az) * math.cos(el), math.cos(az) * math.cos(el)
        dy = math.sin(el)
        pos = np.array([dx, dy, dz], dtype=np.float64) * float(distance)
        light = Light(pos, color=color, brightness=brightness)
        self.lights.append(light)
        self.sync_lights()
        return light

    def add_sun(self, azimuth_deg=45.0, elevation_deg=35.0, color=(255, 244, 214),
               brightness=1.5, angular_size_deg=2.0, glow_deg=6.0, distance=4000.0,
               cast_light=True):
        """Convenience: bakes a sun/moon disc into the sky
        (Background.set_sun -- call set_sky_gradient on this tracer's
        background first, there's nothing to bake it onto otherwise) AND,
        if `cast_light` (default True), also adds a real distant Light in
        the same direction (add_sun_light) so it casts actual highlights/
        shadows, not just the ambient sky glow. `elevation_deg` above ~0
        reads as a sun, a low/negative one with a cooler `color` reads as
        a moon -- there's no separate "moon" type, just different
        parameters on the same call (see the module docstring's v12
        notes). Returns the Light if cast_light else None -- dim the sun
        later with set_sun_intensity() (visual) and/or
        `light.brightness = ...; tracer.sync_lights()` (real light),
        typically together for a day/night cycle."""
        self.background.set_sun(azimuth_deg=azimuth_deg, elevation_deg=elevation_deg,
                                color=color, intensity=1.0, angular_size_deg=angular_size_deg,
                                glow_deg=glow_deg)
        if BG_FIELD is not None:
            BG_FIELD.from_numpy(self.background.image)
        if self.clouds is not None:
            self._sync_cloud_fields()  # keep the clouds' self-shadow direction pointed at the sun
        light = None
        if cast_light:
            light = self.add_sun_light(azimuth_deg, elevation_deg, color, brightness, distance)
        self.reset_accumulation()
        return light

    def set_sun_intensity(self, intensity):
        """Dims/brightens the sun/moon disc baked into the sky
        (Background.set_sun) and re-uploads the sky texture to the GPU --
        also resets accumulation (live view), since the background
        changed. No effect if no sun has been set (Background.set_sun /
        add_sun). Does NOT touch a separate add_sun_light's brightness
        (that's real diffuse/specular light, not the sky's visual/ambient
        glow) -- dim that directly with `light.brightness = ...;
        tracer.sync_lights()` if you added one, so a day/night cycle can
        drive both together."""
        if self.background.sun is None:
            return
        self.background.set_sun(**{**self.background.sun, 'intensity': max(0.0, float(intensity))})
        if BG_FIELD is not None:
            BG_FIELD.from_numpy(self.background.image)
        self.reset_accumulation()

    def add_stars(self, density=0.0025, min_size=0.5, max_size=1.6, brightness=1.0,
                 color_variation=0.25, twinkle=0.15, above_horizon_only=True, seed=0):
        """Convenience: Background.set_stars + re-uploads the sky texture
        to the GPU + resets accumulation. Requires set_sky_gradient to
        already be set on this tracer's background (a star field bakes
        into that same procedural texture, same requirement as add_sun).
        See Background.set_stars for what each parameter does; see
        update_stars for actually animating the "occasionally shift"
        twinkle afterward."""
        self.background.set_stars(density=density, min_size=min_size, max_size=max_size,
                                  brightness=brightness, color_variation=color_variation,
                                  twinkle=twinkle, above_horizon_only=above_horizon_only, seed=seed)
        if BG_FIELD is not None:
            BG_FIELD.from_numpy(self.background.image)
        self.reset_accumulation()

    def update_stars(self, t):
        """Advances the star field's twinkle to time t (seconds) and
        re-uploads the sky texture -- see Background.update_stars. Cheap,
        but there's no reason to call this every single frame for an
        effect that's meant to read as occasional, not a strobe; the
        interactive live view throttles its own calls to roughly once a
        second (see the main loop) -- call it yourself at whatever
        cadence suits a script or video render. No effect if no star
        field has been set."""
        if self.background.stars is None:
            return
        self.background.update_stars(t)
        if BG_FIELD is not None:
            BG_FIELD.from_numpy(self.background.image)
        self.reset_accumulation()

    def set_clouds(self, enabled=True, base=100.0, top=160.0, density=0.08, coverage=0.5,
                   scale=60.0, color=(255, 255, 255), steps=24, light_steps=4, wind=(0.0, 0.0)):
        """Enables/configures a real, GPU-raymarched volumetric cloud layer
        -- a horizontal band between world-Y `base` and `top`, filled with
        a noise-based density field (see _cloud_fbm/_cloud_density near
        the top of the file), that:
          - Is actually VISIBLE from the camera: composited over whatever
            the primary ray resolves to, sky or geometry (see
            render_sample's post-bounce-loop cloud composite), with soft,
            patchy shapes rather than a flat translucent deck.
          - CASTS REAL SHADOWS on geometry below it: every shadow ray (in
            _shadow_throughput -- used for every light, every surface
            point) marches through this SAME density field between the
            surface and the light, so a dense/thick cloud genuinely
            darkens the ground underneath it -- this is the actual
            mechanism, not a separate/baked effect layered on afterward.
          - Self-shadows: each point's own brightness (in the camera-
            visible composite) also does a short secondary march toward
            the sun, so a thick cloud's underside reads darker than a
            thin wisp (see _cloud_march).

        base/top: altitude (world Y) of the layer's bottom/top -- `top -
          base` is its thickness (thicker both looks puffier and casts a
          more solid shadow -- more depth for density to accumulate over).
        density: this is a Beer-Lambert EXTINCTION COEFFICIENT (optical
          depth per world unit of travel through a fully-dense patch),
          not an arbitrary 0..1 slider -- the actual darkness a shadow ray
          sees is roughly `exp(-density * path_length_through_cloud)`, and
          path_length is usually much longer than the layer's own
          thickness (a shadow ray travels the slab at whatever angle the
          sun is at, e.g. thickness/sin(sun_elevation) -- at a 40-degree
          sun through a 60-unit-thick layer that's already ~90 units).
          That's why the default is small (0.08): at ~90 units, even that
          gives optical depths of several, i.e. proper shadow. If shadows
          look like solid black cutout blobs instead of soft graded ones,
          `density` is too high for your base/top/sun angle -- turn it
          DOWN (try halving it repeatedly), not up; if clouds barely
          shadow anything, turn it up. `top - base` and `density` trade
          off against each other (thicker layer needs less density for
          the same look).
        coverage: 0..1 -- how much of the noise field counts as "cloud"
          at all. Low = wispy scattered puffs with lots of clear sky,
          high = a solid overcast sheet.
        scale: world units per noise feature -- bigger = larger, fewer
          individual cloud shapes; smaller = more, finer detail.
        color: the cloud's base (fully lit) color -- tint this for a
          sunset/stormy look.
        steps/light_steps: raymarch quality knobs -- more steps = smoother
          gradients and more accurate self-shadowing, at a roughly linear
          cost in render time. Lower these first if clouds are the
          slowest part of a render. (light_steps is doubled internally
          for the shadow-ray march since it's a cheaper query than the
          camera-visible one -- see _cloud_shadow_transmittance.)
        wind: (x, z) world-space offset added to the noise sample
          position -- see set_cloud_wind for animating this over time
          (e.g. from render_video) to drift the cloud shapes without
          regenerating them (the noise itself doesn't change, just where
          in it you're sampling, so shapes stay self-consistent frame to
          frame instead of flickering).

        Uses whatever sun direction Background.set_sun last configured
        (straight up if none) for the self-shadow term -- re-calling this
        (or add_sun) after changing the sun's angle keeps them in sync;
        add_sun/set_sun_intensity also refresh it automatically if clouds
        are already enabled when they're called."""
        self.clouds = {'enabled': bool(enabled), 'base': float(base),
                       'top': max(float(base) + 1.0, float(top)),
                       'density': max(0.0, float(density)), 'coverage': max(0.0, min(1.0, float(coverage))),
                       'scale': max(1.0, float(scale)), 'color': [int(c) for c in color[:3]],
                       'steps': max(1, int(steps)), 'light_steps': max(1, int(light_steps)),
                       'wind': [float(wind[0]), float(wind[1])]}
        self._sync_cloud_fields()
        self.reset_accumulation()
        return self

    def set_cloud_wind(self, wind):
        """Updates just the wind offset (see set_clouds) without touching
        any other cloud parameter or resetting the whole config -- the
        cheap call to make from a per-frame animation loop / render_video
        for slowly drifting clouds. No effect if set_clouds hasn't been
        called."""
        if self.clouds is None:
            return
        self.clouds['wind'] = [float(wind[0]), float(wind[1])]
        CLOUD_WIND[None] = list(self.clouds['wind'])
        self.reset_accumulation()

    def _sync_cloud_fields(self):
        """Uploads self.clouds (see set_clouds) to the CLOUD_* GPU fields,
        including the sun direction (read fresh from self.background.sun
        each call, so it stays correct even if the sun changed since
        set_clouds was last called)."""
        c = self.clouds
        if c is None or not c.get('enabled', True):
            CLOUD_ENABLED[None] = 0
            return
        CLOUD_ENABLED[None] = 1
        CLOUD_BASE[None] = c['base']
        CLOUD_TOP[None] = c['top']
        CLOUD_DENSITY[None] = c['density']
        CLOUD_COVERAGE[None] = c['coverage']
        CLOUD_SCALE[None] = c['scale']
        CLOUD_COLOR[None] = [ch / 255.0 for ch in c['color']]
        CLOUD_WIND[None] = list(c['wind'])
        CLOUD_STEPS[None] = c['steps']
        CLOUD_LIGHT_STEPS[None] = c['light_steps']
        sun = self.background.sun
        if sun is not None:
            az = math.radians(sun['azimuth_deg'])
            el = math.radians(sun['elevation_deg'])
            dx, dz = math.sin(az) * math.cos(el), math.cos(az) * math.cos(el)
            dy = math.sin(el)
            CLOUD_SUN_DIR[None] = [dx, dy, dz]
        else:
            CLOUD_SUN_DIR[None] = [0.0, 1.0, 0.0]

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

    def set_zoom(self, zoom):
        """Sets the current OPTICAL ZOOM multiplier (1.0 = the tracer's
        base_fov_deg unzoomed, higher = narrower FOV / more "zoomed in",
        clamped to [zoom_min, zoom_max]) and immediately recomputes
        fov_deg/fov_rad/half_tan from it -- every projection in the file
        reads half_tan, so this is the one place zoom actually takes
        effect. Also updates zoom_target to match, since an explicit
        set_zoom() (e.g. loading a saved scene, or a keyframe's `zoom`
        field being applied directly) shouldn't leave a stale ease target
        fighting the value that was just set -- callers that specifically
        want an EASED transition should set zoom_target instead and let
        the interactive loop / render_video's per-frame ramp (see
        zoom_speed) carry `zoom` toward it over time."""
        z = float(max(self.zoom_min, min(self.zoom_max, zoom)))
        self.zoom = z
        self.zoom_target = z
        self.fov_deg = self.base_fov_deg / z
        self.fov_rad = math.radians(self.fov_deg)
        self.half_tan = math.tan(self.fov_rad / 2)
        return z

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
        the exposure multiplier that would put the scene's metered
        luminance at `target_luma` -- the same "aim for 18% gray" logic a
        real camera's auto-exposure metering uses. Cheap enough to call
        every frame at the resolutions this renderer targets (a single
        numpy readback + a couple of reductions). Used two ways (see
        eye_adapt_enabled / DEFAULT_POST_FX): INSTANTLY for stills/live-
        preview (metering should just be correct immediately for a photo),
        and SMOOTHED frame-to-frame in render_video (a real eye/camera
        visibly takes a moment to adjust when the scene brightness
        changes).

        The metering itself is a CENTER-WEIGHTED LOG-AVERAGE, not a flat
        arithmetic mean of every pixel -- a plain mean is what made outdoor
        shots come out dark: a bright sky filling the top third of frame
        drags the arithmetic average way up, so the exposure that makes
        THAT average hit 18% gray ends up crushing the actual foreground/
        ground into shadow. Real camera meters don't do a flat mean either,
        for the same reason:
          - log-average (geometric mean of luminance) is the standard fix
            used by both camera metering and HDR tonemapping's "key value"
            calculation -- it's dominated by the bulk of mid-brightness
            pixels instead of getting dragged around by a relatively small
            number of very bright ones (sky, sun, hot highlights).
          - center-weighting biases the reading toward the middle of frame
            (where the subject usually is) and away from the edges/top,
            further reducing how much a bright sky band can skew things."""
        n = max(1, int(SAMPLE_COUNT[None]))
        raw = ACCUM.to_numpy()[:self.height, :self.width]
        luma = (raw[..., 0] * 0.2126 + raw[..., 1] * 0.7152 + raw[..., 2] * 0.0722) / n
        h, w = luma.shape
        if h < 2 or w < 2:
            avg = float(luma.mean())
        else:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
            d = np.sqrt(((xx - cx) / (w / 2.0)) ** 2 + ((yy - cy) / (h / 2.0)) ** 2)
            weight = 1.0 - 0.65 * np.clip(d, 0.0, 1.0)  # center-weighted, like a real meter
            log_luma = np.log(np.maximum(luma, 1e-4))
            avg = float(math.exp(np.average(log_luma, weights=weight)))
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

    def load_accum(self, accum_np, depth_accum_np, n_samples):
        """Loads EXTERNALLY-combined accumulation buffers into this
        tracer's fields, standing in for actually having called
        add_samples() here. Used ONLY by the --multi-gpu still-image
        orchestrator (_run_multi_gpu_still), which sums each worker GPU's
        raw ACCUM/DEPTH_ACCUM (each worker independently accumulated its
        own share of the total sample count) before calling this, so the
        combined result can be resolved/tonemapped/post-processed exactly
        once, here, instead of once per worker."""
        padded_accum = np.zeros((MAX_H, MAX_W, 3), dtype=np.float32)
        padded_accum[:self.height, :self.width] = accum_np
        padded_depth = np.zeros((MAX_H, MAX_W), dtype=np.float32)
        padded_depth[:self.height, :self.width] = depth_accum_np
        ACCUM.from_numpy(padded_accum)
        DEPTH_ACCUM.from_numpy(padded_depth)
        SAMPLE_COUNT[None] = int(n_samples)

    def dump_raw_accum(self):
        """Raw (pre-tonemap) ACCUM/DEPTH_ACCUM + the current SAMPLE_COUNT,
        sliced to this tracer's active resolution. Used ONLY by --multi-gpu
        still-image WORKERS to hand their independently-accumulated share
        of samples back to the orchestrator (see load_accum, the combining
        side, in the parent process)."""
        accum_np = ACCUM.to_numpy()[:self.height, :self.width]
        depth_np = DEPTH_ACCUM.to_numpy()[:self.height, :self.width]
        return accum_np, depth_np, int(SAMPLE_COUNT[None])

    def finalize_and_save(self, out_path, post_fx=None):
        """Resolve -> post-fx -> save, with no sampling of its own. Split
        out of render_to_file so the --multi-gpu still-image path (which
        combines several workers' ACCUM buffers via load_accum() instead
        of calling add_samples itself) can reuse the exact same
        resolve/post-processing/save logic without duplicating it."""
        color, depth = self.current_image_float()
        if post_fx and post_fx.get('enabled', True):
            R = camera_matrix(*self.camera_rot, self.camera_roll)
            flares = compute_flare_list(self, self.camera_pos, R)
            camera_pose = (self.camera_pos, R, self.half_tan, self.aspect)
            color = apply_post_processing(color, depth, flares, post_fx, camera_pose=camera_pose)
        img = (np.clip(color, 0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(img).save(out_path)

    def render_to_file(self, out_path="raytrace_v12.png", samples=32, post_fx=None,
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
        self.finalize_and_save(out_path, post_fx=post_fx)
        print(f"\nSaved: {out_path} ({time.time() - t0:.2f}s)")

    def render_video(self, camera_path, out_path="raytrace_v12_video.mp4",
                      fps=VIDEO_FPS, resolution=VIDEO_RES, duration=VIDEO_DURATION,
                      samples_per_frame=VIDEO_SAMPLES_PER_FRAME, post_fx=None,
                      progress_cb=None, camera_sync=False, frame_subset=None, skip_encode=False):
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
        and the live raytrace view still stay still, since they never pass this time value.

        frame_subset (--multi-gpu video worker use ONLY): an iterable of frame
        indices to actually render -- every OTHER frame index in range(n_frames)
        is skipped entirely (not even sampled for timing). Meant for a
        CONTIGUOUS range (e.g. range(0, 40)), one per GPU -- see the
        _run_multi_gpu_video/_compute_video_frame_plan module note for why
        contiguous blocks (not interleaved frames) are used and what that
        costs (eye-adaptation/autofocus lag start "cold" at each block's
        first frame, same as they would at the very start of a normal render).
        skip_encode: if True, this call is one --multi-gpu worker's block --
        out_path is that worker's own segment file (already a complete,
        independently playable video on its own), and the --multi-gpu
        orchestrator joins every worker's segment into the final file
        afterward (see _run_multi_gpu_video). Doesn't change how frames are
        written -- every render_video call streams straight to ffmpeg via
        VideoStreamWriter regardless of skip_encode; it only changes what
        out_path refers to and the log message at the end."""
        is_sensor = camera_path.is_camera_data()
        n_kf = 0 if is_sensor else len(camera_path.keyframes)
        if not is_sensor and n_kf == 0:
            print("No camera keyframes -- cancelling video render.")
            return None

        n_frames, frame_times, total_time, camera_sync = _compute_video_frame_plan(
            camera_path, fps, duration, camera_sync)
        frame_range = range(n_frames) if frame_subset is None else list(frame_subset)
        # Frames are streamed straight into an ffmpeg subprocess as they're
        # produced (see VideoStreamWriter) instead of being saved as
        # individual PNGs and stitched afterward -- this removes the
        # per-frame PIL/PNG-compression step that was pegging the CPU and
        # stalling the GPU between frames. `out_path` is either the final
        # video (single-process render) or this worker's own segment file
        # (when frame_subset/skip_encode indicate a --multi-gpu block).
        writer = VideoStreamWriter(out_path, resolution[0], resolution[1], fps)
        pipeline = PostProcessPipeline(writer)

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
        n_render = len(frame_range)
        subset_note = f" [{n_render}/{n_frames} frames, this worker's block]" if frame_subset is not None else ""
        print(f"Rendering video {resolution[0]}x{resolution[1]}, {n_frames} frames "
              f"({total_time:.2f}s @ {fps}fps), {samples_per_frame} sample(s)/frame{sync_note}{subset_note} ...")
        t0 = time.time()
        # Previous frame's (yaw, pitch, roll), for apply_analog_camera's motion
        # smear -- None on this worker's very first rendered frame (no prior
        # pose to diff against yet, so that frame gets no smear).
        prev_rot_roll = None
        # Whether `camera_path` has an authored zoom track worth following
        # (>=2 keyframes -- zoom_at() only varies between at least two) --
        # if so each frame's zoom ramp target below tracks the path itself
        # (e.g. keyframes placed with different tracer.zoom values, or set
        # via the F10/O keyframe menu's "Set zoom" option); otherwise it
        # falls back to self.zoom_target, a single constant zoom for the
        # whole render (whatever was last set interactively or via
        # --zoom/--zoom-speed for a headless run).
        _path_has_zoom_track = hasattr(camera_path, 'keyframes') and len(camera_path.keyframes) >= 2
        try:
            for ri, fi in enumerate(frame_range):
                t_center = frame_times[fi] if camera_sync else min((fi + 0.5) * dt, total_time)
                self.reset_accumulation()

                # --- Optical zoom ramp, VIDEO ONLY (interactive live view
                # does its own per-frame ramp in main()'s loop instead) ---
                # Eases self.zoom toward self.zoom_target at (at most)
                # zoom_speed zoom-multiplier/second, instead of snapping --
                # a real zoom lens takes a moment to rack even if you ask
                # for a big change all at once. `_zoom_focal_kick` (fed to
                # apply_post_processing/apply_zoom_focal_shift) is derived
                # from how fast the zoom is CURRENTLY moving relative to
                # zoom_speed, smoothed a little so it doesn't flicker
                # on/off frame to frame, then decays to 0 once the zoom
                # settles onto its target -- the camcorder-style
                # auto-focus "hunt" only appears while actually zooming.
                prev_zoom = self.zoom
                zoom_ramp_target = camera_path.zoom_at(t_center) if _path_has_zoom_track else self.zoom_target
                zoom_speed = max(1e-4, float(self.zoom_speed))
                max_delta = zoom_speed * dt
                if abs(zoom_ramp_target - self.zoom) > max_delta:
                    self.zoom += max_delta if zoom_ramp_target > self.zoom else -max_delta
                else:
                    self.zoom = zoom_ramp_target
                self.zoom = max(self.zoom_min, min(self.zoom_max, self.zoom))
                self.fov_deg = self.base_fov_deg / self.zoom
                self.fov_rad = math.radians(self.fov_deg)
                self.half_tan = math.tan(self.fov_rad / 2)
                zoom_rate = abs(self.zoom - prev_zoom) / dt if dt > 0 else 0.0
                kick_raw = min(1.0, zoom_rate / zoom_speed)
                self._zoom_kick_smooth += (kick_raw - self._zoom_kick_smooth) * min(1.0, 6.0 * dt)
                post_fx['_zoom_focal_kick'] = self._zoom_kick_smooth

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
                        # Sample spotlight animations at this exact
                        # sub-frame shutter time too, so an animated
                        # spotlight motion-blurs consistently with the
                        # camera instead of jumping once per whole frame.
                        self.set_spotlight_animation_time(tt)
                        self.add_samples(1)
                else:
                    pos, yaw, pitch, roll = camera_path.sample(t_center)
                    self.camera_pos = pos.astype(np.float32)
                    self.camera_rot = np.array([yaw, pitch], dtype=np.float32)
                    self.camera_roll = roll
                    self.water_time = t_center * WATER_WAVE_SPEED
                    # Deterministic, frame-exact spotlight animation (see
                    # SpotLightAnimation) -- NOT wall-clock based like the
                    # interactive view, so re-rendering the same video is
                    # reproducible regardless of how fast the machine runs.
                    self.set_spotlight_animation_time(t_center)
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
                curr_rot_roll = (float(self.camera_rot[0]), float(self.camera_rot[1]), float(self.camera_roll))
                post_fx_on = post_fx.get('enabled', True)
                if post_fx_on:
                    R = camera_matrix(*self.camera_rot, self.camera_roll)
                    flares = compute_flare_list(self, self.camera_pos, R)
                    camera_pose = (self.camera_pos, R, self.half_tan, self.aspect)
                    motion_px = _camera_motion_px(prev_rot_roll, curr_rot_roll,
                                                   self.width, self.height, self.half_tan, self.aspect)
                else:
                    flares, camera_pose, motion_px = None, None, None
                prev_rot_roll = curr_rot_roll
                # The heavy CPU work (post-fx) goes to the background
                # pipeline thread -- submit() returns right away, so this
                # loop can go straight on to frame fi+1's raytrace kernels
                # instead of blocking on bloom/DoF/VHS/etc. first. Snapshot
                # post_fx now (dict(post_fx)) since autofocus/eye-adapt
                # will keep mutating the live dict for later frames while
                # this one is still queued.
                pipeline.submit(color, depth, flares, dict(post_fx), post_fx_on, fi, camera_pose, motion_px)

                elapsed = time.time() - t0
                frac_done = (ri + 1) / n_render
                eta = (elapsed / frac_done - elapsed) if frac_done > 0 else 0.0
                print(f"\r  Frame {fi + 1}/{n_frames}  {frac_done * 100:5.1f}%  "
                      f"elapsed {elapsed:6.1f}s  ETA {eta:6.1f}s", end="", flush=True)
                if progress_cb is not None:
                    progress_cb(ri + 1, n_render, elapsed, eta)
        except Exception:
            # Something went wrong mid-render -- shut both stages down
            # (ignoring their own errors, e.g. an already-broken pipe)
            # rather than leaving threads/subprocesses hanging around
            # waiting for work that will never arrive, then let the
            # original exception propagate. Pipeline first: it may still
            # be holding queued frames that reference `writer`.
            try:
                pipeline.close()
            except Exception:
                pass
            try:
                writer.close()
            except Exception:
                pass
            raise

        self.eye_adapt_enabled = do_eye_adapt
        self.set_resolution(prev_w, prev_h)
        self.water_time = 0.0
        self.camera_roll = 0.0
        WATER_TIME[None] = 0.0
        if animate_caustics:
            self.compute_caustics()  # restore the still (t=0) caustic map, as before rendering video

        # Pipeline first: waits for every already-queued frame to finish
        # post-fx and reach writer.submit(). Only THEN is it safe to tell
        # the writer no more frames are coming.
        pipeline.close()
        writer.close()
        label = "segment" if skip_encode else "video"
        print(f"\nEncoded {n_render} frame(s) directly to {label}: {out_path}")
        return out_path

    # --- Serialization of the "shot" (camera + look/render params) --------
    def camera_dict(self):
        return {'pos': [float(x) for x in self.camera_pos],
                'yaw': float(self.camera_rot[0]), 'pitch': float(self.camera_rot[1])}


class PostProcessPipeline:
    """Runs apply_post_processing (DoF/bloom/god-rays/VHS/chromatic
    aberration/etc.) on a background thread, instead of inline in the
    render loop. That function is pure NumPy/OpenCV -- it never touches
    Taichi/the GPU -- so it's safe to run concurrently with the main
    thread, which is the only thread allowed to issue Taichi kernel calls.

    Why this matters: previously each frame was raytrace (GPU) -> readback
    -> post-fx (CPU, several box-blur passes etc.) -> next frame,
    strictly serial. The GPU sat idle for the entire post-fx stretch of
    EVERY frame, which is exactly the "CPU maxed, GPU half-idle" pattern.
    With post-fx moved here, the main thread hands off frame N's
    raw color/depth and immediately moves on to dispatching frame N+1's
    raytrace kernels, so the GPU keeps working while the CPU chews on the
    previous frame's effects. A single consumer thread pulling off a FIFO
    queue.Queue guarantees frames still reach the video writer in order.

    This does NOT reduce total CPU work (the post-fx math is exactly the
    same) -- it overlaps that CPU work with GPU work that was previously
    just waiting on it, which is where the wall-clock win comes from. If
    CPU post-fx work were the single true bottleneck of the whole
    pipeline, this thread's queue would fill and the main thread would
    block on submit() until it drains -- that's expected backpressure, not
    a bug, and it still keeps the GPU fed as fast as the CPU can consume."""

    def __init__(self, writer, queue_size=3):
        self._writer = writer
        self._queue = queue.Queue(maxsize=queue_size)
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while True:
                job = self._queue.get()
                if job is None:
                    break
                color, depth, flares, post_fx_snapshot, post_fx_on, fi, camera_pose, motion_px = job
                if post_fx_on:
                    color = apply_post_processing(color, depth, flares, post_fx_snapshot,
                                                   frame_seed=fi, camera_pose=camera_pose,
                                                   motion_px=motion_px)
                img = (np.clip(color, 0.0, 1.0) * 255).astype(np.uint8)
                self._writer.submit(img)
        except Exception as e:
            self._error = e

    def submit(self, color, depth, flares, post_fx_snapshot, post_fx_on, fi, camera_pose, motion_px):
        """color/depth: this frame's arrays from current_image_float() --
        fresh allocations each call (to_numpy()), so no aliasing risk with
        the next frame's arrays. post_fx_snapshot MUST be a fresh dict copy
        (not the live, still-being-mutated post_fx dict) -- autofocus/eye
        adaptation update post_fx's shared dict every frame in the main
        thread, and without a snapshot this thread could read a LATER
        frame's focus distance while processing an EARLIER frame."""
        if self._error is not None:
            raise RuntimeError(f"Post-processing pipeline failed: {self._error}")
        self._queue.put((color, depth, flares, post_fx_snapshot, post_fx_on, fi, camera_pose, motion_px))

    def close(self):
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise RuntimeError(f"Post-processing pipeline failed: {self._error}")


class VideoStreamWriter:
    """Streams finished RGB frames straight into a persistent ffmpeg
    subprocess, which encodes them to out_path AS THEY ARRIVE -- no
    per-frame PNG (or any other) image file is ever written to disk.

    Why: saving each frame as a PNG (PIL's Image.fromarray(...).save(...))
    then running ffmpeg over the whole directory afterward does two full
    CPU-bound encode passes (PNG per frame, then H.264 once at the end) in
    a single thread that also has to synchronously wait on the GPU/BVH
    raytrace for that frame. On a machine with fast GPU(s) but few CPU
    cores (e.g. a Kaggle dual-GPU instance), that PNG-encode step is
    exactly what pins the CPU at/near 100% per worker while the GPU(s)
    idle waiting for it to finish before the next frame can start.

    This class removes BOTH problems at once:
      - no PNG encode at all -- raw RGB bytes go straight into ffmpeg's
        rawvideo input, so the only per-frame CPU cost on the caller's
        side is a numpy copy plus a queue.put(), and the H.264 encode
        happens exactly once, incrementally, instead of as a separate
        pass over saved files;
      - a background thread owns the actual pipe write, so submit()
        returns almost immediately and the caller (the raytrace loop) can
        start the NEXT frame's GPU work while ffmpeg is still busy
        encoding the previous one on the CPU, instead of the two
        serializing on each other every single frame.

    Memory footprint: the queue only ever holds a handful of frames
    in-flight (queue_size), not the whole video -- at 640x480 that's a few
    MB, not the ~0.9 GB that 1000 buffered raw frames would take. There's
    no need to stage frames in GPU VRAM either; the raytracer already
    copies each finished frame out to host RAM (to_numpy) before this
    class ever sees it, so VRAM was never actually where the time was
    going."""

    def __init__(self, out_path, width, height, fps, queue_size=4,
                 crf=18, preset='medium'):
        if shutil.which('ffmpeg') is None:
            raise RuntimeError(
                "ffmpeg not found in PATH -- required to stream-encode video "
                "(there's no PNG-frames fallback anymore; install ffmpeg "
                "and re-run).")
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
            '-an', '-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
            '-pix_fmt', 'yuv420p', out_path,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._queue = queue.Queue(maxsize=queue_size)
        self._error = None
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self):
        """Runs on the background thread: pulls frames off the queue and
        writes their raw bytes to ffmpeg's stdin, one at a time, until it
        sees the None sentinel close() sends."""
        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    break
                self._proc.stdin.write(frame.tobytes())
        except Exception as e:
            self._error = e

    def submit(self, frame_rgb_uint8):
        """Hands off one HxWx3 uint8 RGB frame (same channel order PIL's
        Image.fromarray expects) to the background writer thread. Returns
        as soon as there's room in the queue -- it does NOT wait for the
        frame to actually be written/encoded."""
        if self._error is not None:
            raise RuntimeError(f"Video encoder failed: {self._error}")
        self._queue.put(np.ascontiguousarray(frame_rgb_uint8))

    def close(self):
        """Signals no more frames are coming, waits for the background
        thread to drain the queue and finish writing, then waits for
        ffmpeg itself to finish encoding and checks its exit code."""
        self._queue.put(None)
        self._thread.join()
        try:
            self._proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        # NOT self._proc.communicate() here -- it would try to flush/close
        # stdin itself, and stdin is already closed above (closing it is
        # what tells ffmpeg "no more input, finish encoding and exit").
        # Read stderr to EOF (blocks until ffmpeg actually exits and closes
        # its end) instead, then wait() just reaps the now-dead process.
        stderr = self._proc.stderr.read()
        self._proc.wait()
        if self._error is not None:
            raise RuntimeError(f"Video encoder failed: {self._error}")
        if self._proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {self._proc.returncode}: "
                f"{stderr.decode(errors='replace')}")


def _concat_video_segments(segment_paths, out_path):
    """Losslessly joins already-encoded MP4 segments (one per --multi-gpu
    worker's contiguous frame block, each already a complete, independently
    playable H.264 file written by that worker's own VideoStreamWriter)
    into a single final file, using ffmpeg's concat demuxer with `-c copy`
    (stream copy -- no re-encode, no quality loss, negligible CPU/time).
    Returns True on success, False if ffmpeg isn't found in PATH."""
    if shutil.which('ffmpeg') is None:
        return False
    list_path = out_path + ".concat_list.txt"
    with open(list_path, "w") as f:
        for p in segment_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
           '-i', list_path, '-c', 'copy', out_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"\nError concatenating video segments: {e}")
        return False
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass


# =============================================================================
# 11d) --multi-gpu orchestration
#
# Taichi doesn't support spreading a SINGLE kernel launch across multiple
# GPUs -- one ti.init() binds to one device. So this works by launching N
# separate OS processes (this same script, re-invoked with the ORIGINAL
# command line plus hidden --_mgpu-* flags), pinning each one to a
# different physical GPU via the CUDA_VISIBLE_DEVICES environment variable,
# and combining their results afterward:
#
#   - STILL IMAGES: each worker renders the SAME image with an INDEPENDENT
#     share of the total sample count (each GPU accumulates its own,
#     uncorrelated noise -- see RT_RANDOM_SEED in init_taichi -- rather than
#     splitting a single sample's work across GPUs, which would need much
#     finer-grained cross-device synchronization for no real benefit at
#     this granularity). The orchestrator sums every worker's raw
#     ACCUM/DEPTH_ACCUM buffers and resolves/tonemaps/post-processes ONCE,
#     in this process, via RayTracer.load_accum + finalize_and_save.
#
#   - VIDEO: each worker renders one CONTIGUOUS block of frames (start..end,
#     not interleaved) so that per-frame temporal state -- eye adaptation
#     (measure_target_exposure smoothing) and continuous autofocus lag --
#     stays internally consistent within each worker's block (both use the
#     real elapsed time between CONSECUTIVE rendered frames, which is only
#     correct if consecutive output frames are actually adjacent in time;
#     interleaving frames round-robin across GPUs would silently break that
#     math). The tradeoff: each block starts that smoothing "cold" (default
#     exposure/focus, same as the very start of any render), so there can
#     be a brief, visible re-adjustment right at each seam between GPUs'
#     blocks -- more GPUs means more (narrower) blocks, means more seams.
#     Motion blur is unaffected either way (it only samples time WITHIN a
#     single frame, never across frames).
# =============================================================================

def _detect_gpu_indices():
    """NVIDIA GPU indices visible to nvidia-smi, or [] if it's not
    available or fails (no NVIDIA GPU, missing/broken driver, etc.)."""
    if shutil.which('nvidia-smi') is None:
        return []
    try:
        out = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True,
                              timeout=10, check=True)
    except (subprocess.SubprocessError, OSError):
        return []
    indices = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith('GPU '):
            try:
                indices.append(int(line.split(':')[0].split()[1]))
            except (IndexError, ValueError):
                continue
    return indices


def _resolve_multi_gpu_indices(spec):
    """Parses --multi-gpu's value into a concrete list of GPU indices to
    use, or [] meaning "render normally, single process": 'off' (disabled),
    'auto' (use all nvidia-smi-detected GPUs if there are 2+, otherwise
    same as off), a bare integer ('3' -> GPUs 0,1,2), or an explicit
    comma-separated list ('0,2,3', for picking specific GPUs or working
    around a broken/undetectable one)."""
    spec = (spec or 'auto').strip().lower()
    if spec in ('off', 'none', ''):
        return []
    if ',' in spec:
        try:
            return [int(x) for x in spec.split(',') if x.strip() != '']
        except ValueError:
            print(f"Warning: couldn't parse --multi-gpu '{spec}' as a GPU index list -- disabling multi-GPU.")
            return []
    if spec.isdigit():
        n = int(spec)
        return list(range(n)) if n >= 2 else []
    if spec == 'auto':
        found = _detect_gpu_indices()
        return found if len(found) >= 2 else []
    print(f"Warning: unrecognized --multi-gpu value '{spec}' -- disabling multi-GPU.")
    return []


def _spawn_mgpu_worker(extra_args, gpu_index, seed):
    """Launches one worker: this same script, the CURRENT process's
    original command line (sys.argv[1:], whatever the user actually typed),
    plus extra_args (the hidden --_mgpu-* flags telling it what slice of
    work to do) -- pinned to physical GPU gpu_index via CUDA_VISIBLE_DEVICES
    and seeded via RT_RANDOM_SEED (see init_taichi). Non-blocking (returns
    the Popen handle) -- callers launch a whole batch of these before
    waiting on any of them, so the GPUs actually run in parallel."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
    env['RT_BACKEND'] = 'cuda'
    env['RT_RANDOM_SEED'] = str(seed)
    cmd = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:] + extra_args
    return subprocess.Popen(cmd, env=env)


def _wait_mgpu_workers(procs):
    rc = [p.wait() for p in procs]
    if any(r != 0 for r in rc):
        raise RuntimeError(f"One or more --multi-gpu worker processes exited with an error "
                            f"(exit codes: {rc}) -- see their output above for the actual cause.")


def _run_multi_gpu_still(gpu_indices, out_path, samples, post_fx, tracer):
    """Orchestrates a still-image render across gpu_indices: splits `samples`
    as evenly as possible (one share per GPU), waits for every worker to
    finish rendering its independent share, sums their raw accumulation
    buffers, and resolves/post-processes/saves ONCE, here."""
    n = len(gpu_indices)
    base, extra = divmod(samples, n)
    shares = [base + (1 if i < extra else 0) for i in range(n)]
    tmp_dir = tempfile.mkdtemp(prefix="rfv12cv_mgpu_")
    print(f"Multi-GPU still render across {n} GPU(s) {gpu_indices}: sample split {shares}")
    procs = []
    out_paths = []
    try:
        for i, (gpu_idx, share) in enumerate(zip(gpu_indices, shares)):
            if share <= 0:
                continue
            npz_path = os.path.join(tmp_dir, f"worker_{i}.npz")
            out_paths.append(npz_path)
            extra_args = ['--_mgpu-role', 'still', '--_mgpu-samples', str(share),
                          '--_mgpu-out', npz_path]
            seed = (os.getpid() * 7919 + gpu_idx * 104729 + i * 1000003) & 0x7fffffff
            procs.append(_spawn_mgpu_worker(extra_args, gpu_idx, seed))
        _wait_mgpu_workers(procs)

        total_accum = None
        total_depth = None
        total_n = 0
        for p in out_paths:
            data = np.load(p)
            a, d, n_s = data['accum'], data['depth_accum'], int(data['n'])
            total_accum = a if total_accum is None else total_accum + a
            total_depth = d if total_depth is None else total_depth + d
            total_n += n_s
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    tracer.load_accum(total_accum, total_depth, total_n)
    tracer.finalize_and_save(out_path, post_fx=post_fx)
    print(f"Saved: {out_path} (combined {total_n} samples across {n} GPU(s))")


def _run_multi_gpu_video(gpu_indices, out_path, fps, camera_sync, active_path, duration):
    """Orchestrates a video render across gpu_indices: splits the frame
    range into one CONTIGUOUS block per GPU (see the module note above for
    why contiguous rather than interleaved). Each worker now streams its
    own block straight to its own complete, independently-playable MP4
    segment (via VideoStreamWriter -- no PNG files, no shared frames
    directory), so once every worker is done there's nothing left to
    "encode" here, just N finished segments to join in order. That join
    uses ffmpeg's concat demuxer with `-c copy` (_concat_video_segments),
    which is a stream copy -- no re-encode, so it costs almost no CPU/time
    regardless of how many frames were rendered."""
    n_frames, _frame_times, _total_time, _cs = _compute_video_frame_plan(
        active_path, fps, duration, camera_sync)
    n = len(gpu_indices)
    base, extra = divmod(n_frames, n)
    bounds = []
    start = 0
    for i in range(n):
        count = base + (1 if i < extra else 0)
        bounds.append((start, start + count))
        start += count
    tmp_dir = tempfile.mkdtemp(prefix="rfv12cv_mgpu_video_")
    print(f"Multi-GPU video render across {n} GPU(s) {gpu_indices}: "
          f"{n_frames} frames split into contiguous blocks {bounds}")
    procs = []
    segment_paths = []
    try:
        for i, (gpu_idx, (s, e)) in enumerate(zip(gpu_indices, bounds)):
            if e <= s:
                continue
            seg_path = os.path.join(tmp_dir, f"segment_{i:03d}.mp4")
            segment_paths.append(seg_path)
            extra_args = ['--_mgpu-role', 'video', '--_mgpu-start', str(s), '--_mgpu-end', str(e),
                          '--_mgpu-out', seg_path]
            seed = (os.getpid() * 7919 + gpu_idx * 104729 + i * 1000003) & 0x7fffffff
            procs.append(_spawn_mgpu_worker(extra_args, gpu_idx, seed))
        _wait_mgpu_workers(procs)

        print(f"All {n} GPU(s) finished -- {n_frames} frames encoded across "
              f"{len(segment_paths)} segment(s), joining ...")
        if _concat_video_segments(segment_paths, out_path):
            print(f"Encoded video (streamed per-GPU + concatenated): {out_path}")
        else:
            raise RuntimeError(
                "ffmpeg not found in PATH -- can't join the per-GPU video "
                "segments into the final file. Install ffmpeg and re-run "
                "(each worker's own segment render already succeeded).")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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
        self._prev_rot_roll = None  # see apply_analog_camera's motion smear

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
        camera_pose = (self.tracer.camera_pos, R, self.tracer.half_tan, self.tracer.aspect)
        curr_rot_roll = (float(self.tracer.camera_rot[0]), float(self.tracer.camera_rot[1]),
                          float(self.tracer.camera_roll))
        motion_px = _camera_motion_px(self._prev_rot_roll, curr_rot_roll,
                                       self.tracer.width, self.tracer.height,
                                       self.tracer.half_tan, self.tracer.aspect)
        self._prev_rot_roll = curr_rot_roll
        color = apply_post_processing(color, depth, flares, self.post_fx, frame_seed=self._frame_seed,
                                       camera_pose=camera_pose, motion_px=motion_px)

        img_rgb = (np.clip(color, 0.0, 1.0) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        w, h = window_size
        if (img_bgr.shape[1], img_bgr.shape[0]) != (w, h):
            img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        return img_bgr

# =============================================================================
# 13) Post-processing (CPU / numpy) -- lens flare, DoF, chromatic aberration,
#     fog, god rays, bloom, VHS, analog-camera film look. All parameters
#     default in DEFAULT_POST_FX, editable directly in code or at runtime
#     via the post_fx dict.
# =============================================================================

DEFAULT_POST_FX = {
    'enabled': True,
    'dof_enabled': False,
    'dof_focus_distance': 25.0,   # focus point (world units) -- adjustable via keys when enabled
    'dof_blur_strength': 0.2,     # how "sensitive" the blur is to focus-distance error
    'dof_max_radius': 16,          # max blur radius (px, at the resolution being rendered)
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
    'flare_size': 20.0,           # main glow radius (px, referenced at 360p)
    'flare_intensity': 0.28,       # additive flare intensity
    'flare_anamorphic': 0.52,     # horizontal anamorphic streak strength (0 = disabled)
    'flare_halo': 0.62,            # secondary ring/halo strength (0 = disabled)
    'fisheye_enabled': False,     # wide-angle lens barrel distortion
    'fisheye_strength': 0.32,     # 0 = rectilinear (no distortion), ~0.2-0.5 = visible bulge
    'bloom_enabled': False,        # highlight glow/bleed -- extracted from bright areas post-tonemap
    'bloom_threshold': 0.72,      # 0..1 -- brightness above which a pixel starts contributing to bloom
    'bloom_intensity': 0.25,      # additive bloom strength (screen-blended -- see apply_bloom)
    'bloom_radius': 22,           # glow spread (px, referenced at 360p)
    'fog_enabled': False,          # depth-based atmospheric fog (exponential, optionally height-falloff)
    'fog_color': (0.62, 0.66, 0.72),  # RGB 0..1, a neutral overcast-sky haze by default
    'fog_density': 0.006,         # exponential fog coefficient -- higher = thicker/closer-in fog
    'fog_max_depth': 260.0,       # world units at which fog reaches full strength (also caps how far
                                   # sky/background pixels -- which report a huge sentinel depth -- get pulled in)
    'fog_height_falloff': 0.0,    # 0 = uniform fog at all heights, >0 = fog thins out above fog_base_height
                                   # (an exponential falloff in world Y, like ground mist/haze layers)
    'fog_base_height': 0.0,       # world Y the height falloff is measured from (ignored if fog_height_falloff is 0)
    'godrays_enabled': True,      # screen-space crepuscular rays / light shafts from visible lights through sky
    'godrays_intensity': 0.35,    # additive strength of the ray contribution
    'godrays_decay': 0.97,        # per-sample falloff along each ray (closer to 1 = rays reach further)
    'godrays_density': 0.9,       # sample step size as a fraction of the full source->pixel distance
    'godrays_samples': 24,        # radial samples per pixel per light -- higher = smoother rays, slower
    'godrays_sky_depth': 5000.0,  # depth (world units) above which a pixel counts as unoccluded "sky"
                                   # and can act as a ray source (matches the renderer's 1e4 miss sentinel)
    'vhs_enabled': False,         # VHS tape effect -- works for BOTH stills and video
    'vhs_strength': 1.0,          # overall VHS effect intensity (0..~2)
    'analog_enabled': True,       # Y2K-camcorder look: crushed/lifted low dynamic range, hard highlight
                                   # blowout, halation, heavy grain, and (video/live only) a directional
                                   # motion smear driven by actual camera movement between frames -- distinct
                                   # from vhs_enabled, which simulates a VIDEO TAPE's electronic artifacts
                                   # (scanlines, tracking wobble); this simulates the CAMERA/SENSOR side of a
                                   # cheap consumer camcorder instead, and the two can be combined (e.g. "worn
                                   # VHS dub of camcorder footage")
    'analog_strength': 1.0,       # overall intensity multiplier for the whole effect bundle below
    'analog_contrast': 0.3,       # dynamic-range compression: lifts blacks and soft-knees the top end, the
                                   # "flat, milky" low-DR look of a small CCD sensor (0 = full range, off)
    'analog_highlight_clip': 0.28,# 0..1 -- how little headroom highlights get before blowing out to a hard,
                                   # detail-less white (lower = blows out sooner/harder)
    'analog_grain': 0.06,         # grain amount (0..~0.15), stronger in shadows/midtones than highlights
    'analog_grain_size': 1.6,     # grain "clump" size in px -- 1 = fine per-pixel noise, >1 = coarser blobs
                                   # closer to real CCD/tape noise than flat per-pixel grain
    'analog_halation': 0.3,       # warm red-orange bleed around bright/blown-out highlights (0 = off)
    'analog_vignette': 0.4,       # soft optical vignette strength (0..~1)
    'analog_warmth': 0.00,        # warm-highlight/cool-shadow split-tone strength (0 = neutral)
    'analog_smear': 0.6,          # extra DIRECTIONAL motion smear, sized from the camera's actual frame-to-
                                   # frame movement (rotation dominates, like a handheld camcorder) -- video
                                   # and live preview only (needs a previous pose); stands in for a slow
                                   # electronic shutter/CCD smear on top of the "real" sampled motion_blur_*
                                   # below, since that would otherwise need very high samples_per_frame to
                                   # resolve fast handheld shake smoothly. 0 = off.
    'analog_smear_max_px': 40,    # cap on smear length (px, at the resolution being rendered) so a fast
                                   # spin/cut doesn't smear into an unreadable blur or blow up render time
    'motion_blur_enabled': True,  # motion blur -- ONLY affects render_video()
                                   # (simulated via multiple raytrace samples at
                                   # different points in time, not a 2D image blur)
    'motion_blur_shutter': 1.0,   # "shutter open" fraction of 1 frame (0..1)
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
    pulled radially outward, using a BOUNDED power-curve model (see below)
    instead of the old unbounded r' = r*(1+k*r^2) polynomial -- the net
    visual effect is the classic fisheye "bulge", with straight lines away
    from center curving outward and the corners compressing inward. Cheap
    (a single remap, no raytracing involved) since it's applied to the
    already-rendered 2D image, not the lens model of the raytracer itself.
    Returns (img, depth) since DoF/downstream depth-aware effects need the
    depth buffer warped the same way to stay aligned with the (now-
    distorted) color image.

    Why a power curve, not r*(1+k*r^2): that polynomial has no upper bound
    -- at the frame corners (the largest on-screen radius) `factor` can
    comfortably exceed 1.6-2x even at moderate strength, which pulls the
    source sample position well OUTSIDE the actual rendered image. With
    BORDER_REPLICATE that means a growing band near the edges/corners just
    repeats whatever pixel happened to be at the source image's boundary,
    which reads as a smeared/stretched streak radiating from the corners
    exactly where the distortion is strongest -- the "unwanted stretching
    artifacts" this fixes. The fix: normalize radius by the frame's own
    corner distance (r_norm in [0, 1], 1 = exactly the corner) and warp it
    with src_r_norm = r_norm ** (1 / (1 + strength)) -- a curve that still
    passes through (0, 0) and (1, 1) for ANY strength, so a corner dest
    pixel always samples from (at most) the corner of the source image,
    never beyond it. Every other radius bulges outward (grabs more
    peripheral content, same "bulge" look as before) but can never escape
    the image bounds, so there's nothing left for BORDER_REPLICATE to
    smear."""
    if strength <= 1e-4:
        return img, depth
    h, w = img.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    half_w, half_h = w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - cx) / half_w
    ny = (yy - cy) / half_h
    r = np.sqrt(nx * nx + ny * ny)
    r_max = math.sqrt(2.0)  # corner radius in this aspect-normalized space
    r_norm = np.clip(r / r_max, 0.0, 1.0)
    power = 1.0 / (1.0 + strength)
    src_r_norm = np.power(r_norm, power)
    scale = np.ones_like(r)
    mask = r > 1e-6
    scale[mask] = (src_r_norm[mask] * r_max) / r[mask]
    src_x = (cx + nx * scale * half_w).astype(np.float32)
    src_y = (cy + ny * scale * half_h).astype(np.float32)
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
    apply_fisheye is a dest->src remap of src_r_norm = dst_r_norm**power
    (power = 1/(1+strength)), so its exact inverse is a closed form --
    dst_r_norm = src_r_norm**(1+strength) -- no iteration needed (unlike
    the old unbounded model, which had to solve it numerically)."""
    if strength <= 1e-4:
        return sx, sy
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    half_w, half_h = w / 2.0, h / 2.0
    nx = (sx - cx) / half_w
    ny = (sy - cy) / half_h
    r = math.sqrt(nx * nx + ny * ny)
    if r < 1e-6:
        return sx, sy
    r_max = math.sqrt(2.0)
    r_norm = min(1.0, r / r_max)
    dst_r_norm = r_norm ** (1.0 + strength)
    scale = (dst_r_norm * r_max) / r
    return cx + nx * scale * half_w, cy + ny * scale * half_h


def apply_zoom_focal_shift(img, kick):
    """Camcorder-style auto-focus "hunt" while the optical zoom (;/' keys /
    zoom_speed, see RayTracer.set_zoom) is actively racking -- a cheap
    stand-in for the brief re-focus wobble a real camcorder's autofocus
    visibly does the moment you zoom, rather than staying perfectly sharp
    through the whole zoom the way a raytraced image otherwise would.
    `kick` (0 = fully sharp, ~1 = softest) is derived by the caller from
    how fast the zoom is currently changing (render_video's per-frame zoom
    ramp) and decays back to 0 once the zoom settles -- this function
    itself is just a one-shot blend toward a Gaussian-blurred copy of the
    frame, scaled by `kick`; ONLY called from render_video (see
    apply_post_processing's `_zoom_focal_kick` read), so stills and the
    interactive live view are never softened by this."""
    if kick <= 1e-4:
        return img
    a = min(1.0, kick)
    radius = max(1, int(round(2 + 5 * a)))
    ksize = radius * 2 + 1
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return img * (1.0 - a) + blurred * a


def apply_bloom(img, threshold, intensity, radius):
    """Highlight glow: extracts pixels brighter than `threshold` (with a
    soft knee so the cutoff isn't a hard edge), blurs them at 2 radii (a
    tight-ish inner glow + a wide soft outer haze, cheaply approximating
    a multi-scale/Gaussian-pyramid bloom with just 2 box blurs), and blends
    that back on top of the image -- the "hot" parts of the frame bleed
    light into their surroundings, like a real camera sensor/eye does with
    bright sources. Operates on the already-tonemapped (post
    resolve_output) display image; see the ACES tonemap in resolve_output
    for why that still gives a reasonably smooth brightness gradient to
    threshold against instead of a flat, hard-edged blob of pure white.

    The glow is SCREEN-BLENDED onto the image (1-(1-a)*(1-b)) instead of
    just added on top. Straight addition is what made this look like a
    flashbang: once a bright area's glow pushed a pixel's value past 1.0,
    the final clamp in apply_post_processing just flattened it to solid
    white with a hard edge wherever the glow happened to cross that line --
    visually indistinguishable from overexposure. A screen blend
    asymptotically approaches 1.0 instead of blowing through it, so bright
    regions get a smooth, self-limiting highlight rolloff (closer to real
    sensor/film halation) no matter how strong `intensity` is, while still
    behaving like ordinary additive glow for the darker majority of the
    frame where a and b are both small."""
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
    glow = np.clip((glow_small * 0.6 + glow_large * 0.4) * intensity, 0.0, 1.0)
    return 1.0 - (1.0 - img) * (1.0 - glow)


def _reconstruct_world_dir_y(depth, half_tan, aspect, R):
    """Rebuilds the world-space Y component of each pixel's primary ray
    direction from the camera pose alone (no separate world-position
    G-buffer exists -- only depth). Matches render_sample's own ray
    construction exactly (xn/yn normalized screen coords, d_cam = (xn *
    half_tan * aspect, yn * half_tan, 1.0), ray_dir = R @ d_cam) so that
    cam_pos.y + world_dir_y * depth reproduces the world-space Y of the
    point that was actually shaded there. Used by apply_fog's optional
    height falloff."""
    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = (xx / max(w - 1.0, 1.0)) * 2.0 - 1.0
    yn = 1.0 - (yy / max(h - 1.0, 1.0)) * 2.0
    d_cam = np.stack([xn * half_tan * aspect, yn * half_tan, np.ones_like(xn)], axis=-1)
    d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)
    # d_cam is a row-vector per pixel; world = R @ d_cam (column-vector form)
    # is the same as d_cam @ R.T in row-vector form.
    world_dir = d_cam @ R.T
    return world_dir[..., 1]


def apply_fog(img, depth, post_fx, camera_pose=None):
    """Exponential depth fog: blends the image toward `fog_color` as depth
    increases (fog_factor = 1 - exp(-depth * density), the standard
    atmospheric-scattering approximation), with an optional height
    falloff so the fog can thin out above `fog_base_height` (ground mist)
    instead of sitting uniformly at every altitude. Sky/background pixels
    report a huge sentinel depth (see the 1e4 miss case in render_sample),
    so depth is capped at `fog_max_depth` first -- otherwise those pixels
    would always compute fog_factor ~= 1 regardless of density and the
    whole sky would flatten to a solid fog-colored wall instead of a
    distant haze. camera_pose, if given, is (camera_pos, R, half_tan,
    aspect) and is only needed when fog_height_falloff > 0."""
    density = post_fx.get('fog_density', 0.0)
    if not post_fx.get('fog_enabled', False) or density <= 0:
        return img
    max_depth = max(1e-3, post_fx.get('fog_max_depth', 260.0))
    d = np.clip(depth, 0.0, max_depth)
    fog_amount = 1.0 - np.exp(-d * density)

    falloff = post_fx.get('fog_height_falloff', 0.0)
    if falloff > 0.0 and camera_pose is not None:
        camera_pos, R, half_tan, aspect = camera_pose
        world_dir_y = _reconstruct_world_dir_y(depth, half_tan, aspect, R)
        world_y = camera_pos[1] + world_dir_y * d
        base_h = post_fx.get('fog_base_height', 0.0)
        fog_amount = fog_amount * np.exp(-np.maximum(world_y - base_h, 0.0) * falloff)

    fog_amount = np.clip(fog_amount, 0.0, 1.0)[..., None]
    color = np.array(post_fx.get('fog_color', (0.6, 0.65, 0.75)), dtype=np.float32)
    return img * (1.0 - fog_amount) + color * fog_amount


def apply_god_rays(img, depth, flares, post_fx):
    """Screen-space crepuscular rays / volumetric light shafts (the classic
    "Volumetric Light Scattering as a Post-Process" radial-blur technique):
    for each visible light, masks the image down to just the SKY pixels
    (depth above `godrays_sky_depth` -- i.e. the camera ray reached the
    background/miss case, matching the renderer's 1e4 sentinel depth, so
    this is genuinely "can see the light/sky here" and not just "this pixel
    is bright") near that light, then radially resamples that masked
    buffer along the line from each screen pixel back toward the light's
    screen position, accumulating samples with a per-step decay. Occluded
    (non-sky) geometry between a pixel and the light contributes ~0 along
    that ray, so solid objects naturally cast dark shaft-shaped shadows
    through the brighter sky around them -- exactly the "rays streaming
    around a tree/building" look. Reuses the same `flares` list
    apply_lens_flare uses (already screen-projected + occlusion-checked
    against the light itself), so a light has to actually be visible from
    the camera to cast rays."""
    intensity = post_fx.get('godrays_intensity', 0.0)
    if not post_fx.get('godrays_enabled', False) or not flares or intensity <= 0:
        return img
    h, w = img.shape[:2]
    sky_depth = post_fx.get('godrays_sky_depth', 5000.0)
    n_samples = max(1, int(post_fx.get('godrays_samples', 24)))
    density = post_fx.get('godrays_density', 0.9)
    decay = post_fx.get('godrays_decay', 0.97)

    is_sky = depth > sky_depth
    luma = img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722
    src = np.where(is_sky, np.clip(luma, 0.0, 4.0), 0.0).astype(np.float32)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = img.copy()
    for (sx, sy, color, brightness) in flares:
        if not (-0.3 * w <= sx <= 1.3 * w and -0.3 * h <= sy <= 1.3 * h):
            continue
        step_x = (sx - xx) * (density / n_samples)
        step_y = (sy - yy) * (density / n_samples)
        cur_x, cur_y = xx.copy(), yy.copy()
        accum = np.zeros((h, w), dtype=np.float32)
        falloff = 1.0
        for _ in range(n_samples):
            cur_x = cur_x + step_x
            cur_y = cur_y + step_y
            ix = np.clip(cur_x, 0, w - 1).astype(np.int32)
            iy = np.clip(cur_y, 0, h - 1).astype(np.int32)
            accum += src[iy, ix] * falloff
            falloff *= decay
        accum /= n_samples
        color_arr = np.array(color, dtype=np.float32)
        out += accum[..., None] * color_arr * intensity * (0.3 + 0.7 * min(1.0, brightness))
    return out


def apply_analog_motion_smear(img, motion_px, strength, max_px):
    """Cheap DIRECTIONAL smear standing in for a cheap camcorder's slow
    electronic shutter + CCD charge-smear -- averages several copies of
    `img` shifted along the (dx, dy) apparent-motion vector (in pixels,
    computed by _camera_motion_px from the camera's actual pose change
    since the previous displayed frame). This is intentionally NOT the
    physically-correct per-object motion blur that render_video's
    motion_blur_* system already does via real time-jittered raytrace
    samples (see render_video's docstring) -- that system is limited by
    samples_per_frame, so resolving FAST handheld shake smoothly through
    it alone would need many more samples than is practical every frame.
    This adds a smooth, deterministic streak on top of however many real
    samples were taken, at the cost of being a uniform whole-frame shift
    rather than a true per-pixel/per-object blur (fine for the camcorder
    look being targeted here, where the whole sensor smears together).
    motion_px: (dx, dy) in screen pixels, or None/zero to skip (stills and
    the very first frame of a sequence have no previous pose to diff)."""
    if strength <= 0 or max_px <= 0 or motion_px is None:
        return img
    dx, dy = motion_px
    mag = math.hypot(dx, dy)
    if mag < 0.05:
        return img
    length = min(max_px, mag * strength)
    if length < 0.4:
        return img
    ux, uy = dx / mag, dy / mag
    n_taps = max(3, min(16, int(round(length)) + 3))
    h, w = img.shape[:2]
    src = img.astype(np.float32)
    acc = np.zeros_like(src)
    for i in range(n_taps):
        f = (i / (n_taps - 1)) - 0.5  # -0.5 .. +0.5 along the motion vector
        ox, oy = ux * length * f, uy * length * f
        M = np.float32([[1.0, 0.0, ox], [0.0, 1.0, oy]])
        acc += cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    return acc / n_taps


def _camera_motion_px(prev_pose, curr_pose, w, h, half_tan, aspect):
    """Approximates the on-screen pixel shift a point near the center of
    frame would show between two camera poses -- used only to size
    apply_analog_motion_smear's streak, not for anything physically exact
    (a true per-pixel motion vector would also depend on scene depth and
    camera translation, which this ignores: rotation dominates the blur in
    handheld/camcorder footage, and that's the dominant term here).
    prev_pose/curr_pose: (yaw, pitch, roll) radians, or prev_pose is None
    for "no previous frame" (returns zero motion)."""
    if prev_pose is None:
        return 0.0, 0.0
    p_yaw, p_pitch, _ = prev_pose
    c_yaw, c_pitch, _ = curr_pose
    d_yaw = _shortest_angle_diff(p_yaw, c_yaw)
    d_pitch = c_pitch - p_pitch
    focal_px_x = (w / 2.0) / max(1e-4, half_tan * aspect)
    focal_px_y = (h / 2.0) / max(1e-4, half_tan)
    return -d_yaw * focal_px_x, d_pitch * focal_px_y


def apply_analog_camera(img, post_fx, frame_seed=0, motion_px=None):
    """Y2K-camcorder 'analog camera' look -- deliberately distinct from
    apply_vhs's tape degradation (which simulates a video SIGNAL's
    electronic artifacts: scanlines, tracking wobble, chroma crosstalk).
    This instead models the physical quirks of a cheap consumer camcorder's
    LENS/SENSOR, applied in this order:
      1) Dynamic-range compression: lifts blacks and soft-knees the top of
         the range, the flat/milky "can't hold true black or a graceful
         highlight rolloff" look of a small CCD sensor + a low-bitrate
         signal chain (a much narrower usable range than the renderer's
         native output).
      2) Highlight blowout: pixels above `analog_highlight_clip` rush to a
         hard, detail-less white -- a small sensor's clip point, not
         film's graceful halation rolloff (that's a separate step below).
      3) Halation: bright/blown highlights bleed a soft, warm red-orange
         halo into the frame (still present -- footage shot on a camcorder
         through a cheap lens gets bloom too, just harder-edged than film).
      4) A gentle warm-highlight / cool-shadow split-tone.
      5) Grain: coarse-ish (see analog_grain_size), mostly-monochromatic
         noise plus a little per-channel chroma noise, stronger in
         shadows/midtones than highlights -- the way real sensor/tape
         noise behaves (a flat, fine, colorless noise floor reads as
         clean digital noise instead of a cheap CCD's).
      6) A soft, smoothly-falling optical vignette.
      7) Motion smear (video/live only, see apply_analog_motion_smear) --
         a directional blur sized from the camera's actual movement since
         the previous frame, standing in for the slow shutter/CCD smear a
         cheap camcorder adds on top of the renderer's own sampled motion
         blur.
    Meant to be combinable with apply_vhs (e.g. "a VHS dub of camcorder
    footage"), so this never touches scanlines or the overall saturation
    cut VHS applies -- it stays in its own lane. frame_seed varies the
    grain per frame for video, same convention as apply_vhs."""
    strength = post_fx.get('analog_strength', 0.0)
    if not post_fx.get('analog_enabled', False) or strength <= 0:
        return img
    h, w = img.shape[:2]
    rng = np.random.default_rng(3000 + int(frame_seed))
    out = img.astype(np.float32).copy()

    def _luma(a):
        return a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722

    # 1) Dynamic-range compression -- lift blacks, soft-knee the top half
    dr = post_fx.get('analog_contrast', 0.0) * strength
    if dr > 0.0:
        black_lift = 0.14 * dr
        out = out * (1.0 - black_lift) + black_lift
        knee = 0.55
        over = np.clip(out - knee, 0.0, None)
        out = np.where(out > knee, knee + over / (1.0 + over * (4.0 * dr)), out)

    # 2) Highlight blowout -- a hard, fast rush to white above the clip point
    hc = post_fx.get('analog_highlight_clip', 0.0)
    if hc > 0.0:
        luma = np.clip(_luma(out), 0.0, 4.0)
        thresh = max(0.02, 1.0 - hc * 0.9)
        blow = np.clip((luma - thresh) / max(1e-4, 1.0 - thresh), 0.0, 1.0) ** 0.35
        out = out + (1.15 - out) * (blow[..., None] * strength)

    # 3) Halation
    halation = post_fx.get('analog_halation', 0.0)
    if halation > 0.0:
        luma_h = _luma(out)
        hot = np.clip((luma_h - 0.7) / 0.3, 0.0, 1.0)
        hot = hot * hot
        scale = min(w, h) / 360.0
        r = max(2, int(round(16 * scale)))
        glow = _box_blur(_box_blur(out * hot[..., None], r), r)
        halation_tint = np.array([1.0, 0.35, 0.15], dtype=np.float32)
        out = out + glow * halation_tint * halation * strength

    # 4) Warm-highlight / cool-shadow split-tone
    warmth = post_fx.get('analog_warmth', 0.0)
    if warmth > 0.0:
        luma2 = np.clip(_luma(out), 0.0, 1.0)
        shadow_tint = np.array([-0.02, -0.01, 0.03], dtype=np.float32)
        highlight_tint = np.array([0.05, 0.02, -0.03], dtype=np.float32)
        l = luma2[..., None]
        out = out + (shadow_tint * (1.0 - l) + highlight_tint * l) * warmth * strength

    # 5) Grain -- coarser clumps + a touch of chroma noise, weighted so it
    #    shows up more in shadows/midtones than highlights
    grain = post_fx.get('analog_grain', 0.0)
    if grain > 0.0:
        luma3 = np.clip(_luma(out), 0.0, 1.0)
        grain_weight = 1.0 - 0.6 * luma3
        grain_size = max(1.0, post_fx.get('analog_grain_size', 1.0))
        gh, gw = max(1, int(round(h / grain_size))), max(1, int(round(w / grain_size)))
        small = rng.normal(0.0, 1.0, size=(gh, gw)).astype(np.float32)
        noise = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR) if grain_size > 1.0001 else small
        out = out + (noise * grain)[..., None] * grain_weight[..., None] * strength
        small_c = rng.normal(0.0, 1.0, size=(gh, gw, 3)).astype(np.float32)
        chroma_noise = (cv2.resize(small_c, (w, h), interpolation=cv2.INTER_LINEAR)
                         if grain_size > 1.0001 else small_c)
        out = out + chroma_noise * (grain * 0.35) * grain_weight[..., None] * strength

    # 6) Soft optical vignette
    vignette = post_fx.get('analog_vignette', 0.0)
    if vignette > 0.0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        d = np.sqrt(((xx - cx) / (w / 2.0)) ** 2 + ((yy - cy) / (h / 2.0)) ** 2)
        vig = 1.0 - vignette * strength * 0.5 * np.clip(d, 0.0, 1.4) ** 2
        out = out * vig[..., None]

    # 7) Motion smear (video/live only -- motion_px is None for stills)
    smear = post_fx.get('analog_smear', 0.0) * strength
    if smear > 0.0:
        out = apply_analog_motion_smear(out, motion_px, smear, post_fx.get('analog_smear_max_px', 40))

    return out


def apply_post_processing(img, depth, flares, post_fx, frame_seed=0, camera_pose=None, motion_px=None):
    out = img
    if post_fx.get('dof_enabled', False):
        out = apply_depth_of_field(out, depth, post_fx['dof_focus_distance'],
                                    post_fx['dof_blur_strength'], post_fx['dof_max_radius'])
    # Only ever set by render_video (see its zoom-speed ramp) -- absent
    # (.get default 0.0) for stills and the interactive live view, so this
    # is a video-only effect as intended.
    zoom_kick = float(post_fx.get('_zoom_focal_kick', 0.0))
    if zoom_kick > 1e-4:
        out = apply_zoom_focal_shift(out, zoom_kick)
    if post_fx.get('fisheye_enabled', False):
        fh, fw = out.shape[:2]
        strength = post_fx.get('fisheye_strength', 0.32)
        out, depth = apply_fisheye(out, depth, strength)
        flares = [(*_fisheye_warp_point(sx, sy, fw, fh, strength), color, brightness)
                  for (sx, sy, color, brightness) in flares]
    out = apply_fog(out, depth, post_fx, camera_pose=camera_pose)
    out = apply_god_rays(out, depth, flares, post_fx)
    if post_fx.get('bloom_enabled', False):
        out = apply_bloom(out, post_fx.get('bloom_threshold', 0.62),
                           post_fx.get('bloom_intensity', 0.45), post_fx.get('bloom_radius', 22))
    if post_fx.get('flare_enabled', False):
        out = apply_lens_flare(out, flares, post_fx['flare_size'], post_fx['flare_intensity'],
                                anamorphic=post_fx.get('flare_anamorphic', 0.0),
                                halo=post_fx.get('flare_halo', 0.0))
    out = apply_analog_camera(out, post_fx, frame_seed=frame_seed, motion_px=motion_px)
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


def draw_orientation_gizmo(canvas, cx, cy, R, radius=28):
    """Small 3-axis compass (world X=red, Y=green, Z=blue, BGR-adjusted for
    cv2) showing the camera's current orientation -- drawn as a fixed HUD
    element (screen-space, not scene-space) near the position/rotation text.
    Axes are the world unit axes rotated into camera space (R.T, the same
    convention used everywhere else here for projecting world directions),
    drawn back-to-front so whichever axis currently points more toward the
    camera overlaps the others correctly, like a typical 3D-viewport gizmo."""
    axes = [
        (np.array([1.0, 0.0, 0.0]), (60, 60, 230), "X"),   # red (BGR)
        (np.array([0.0, 1.0, 0.0]), (80, 200, 80), "Y"),   # green
        (np.array([0.0, 0.0, 1.0]), (230, 140, 60), "Z"),  # blue
    ]
    cv2.circle(canvas, (cx, cy), radius + 8, (35, 32, 28), -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), radius + 8, (95, 85, 75), 1, cv2.LINE_AA)
    Rt = R.T
    items = []
    for axis, color, label in axes:
        cam = Rt @ axis  # this world axis, expressed in camera space
        items.append((cam[2], cam, color, label))
    items.sort(key=lambda it: it[0])  # farthest (most "into the screen") drawn first
    for _, cam, color, label in items:
        tip = (cx + int(cam[0] * radius), cy - int(cam[1] * radius))
        thick = 3 if cam[2] > 0 else 1  # bolder when this axis points toward the viewer
        cv2.line(canvas, (cx, cy), tip, color, thick, cv2.LINE_AA)
        cv2.circle(canvas, tip, 4, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, label, (tip[0] + 5, tip[1] + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, color, 1, cv2.LINE_AA)


# =============================================================================
# 14b) Camera path -- draws the camera keyframes + connecting line on the
#      preview, and finds the camera keyframe "being aimed at" (closest to
#      the crosshair) so the O key knows which keyframe to edit/delete.
# =============================================================================

def find_targeted_keyframe(camera_path, camera_pos, R, half_tan, aspect, sw, sh, max_screen_dist=48,
                            aim_x=None, aim_y=None):
    """Returns the index of the keyframe CLOSEST to the aim point (screen
    center by default, or an explicit (aim_x, aim_y) -- used for mouse-click
    picking), within max_screen_dist pixels and in front of the camera --
    or None if there is none."""
    best_idx, best_d = None, max_screen_dist
    hw, hh = sw / 2.0, sh / 2.0
    ax = hw if aim_x is None else aim_x
    ay = hh if aim_y is None else aim_y
    Rt = R.T
    for i, kf in enumerate(camera_path.keyframes):
        rel = kf.pos - camera_pos
        xc, yc, zc = Rt @ rel
        if zc <= _PREVIEW_NEAR:
            continue
        sx = (xc / (zc * half_tan * aspect)) * hw + hw
        sy = (-yc / (zc * half_tan)) * hh + hh
        d = math.hypot(sx - ax, sy - ay)
        if d < best_d:
            best_d, best_idx = d, i
    return best_idx


def find_targeted_light(lights, spotlights, camera_pos, R, half_tan, aspect, sw, sh,
                         aim_x=None, aim_y=None, max_screen_dist=26):
    """Returns an index into the COMBINED [lights..., spotlights...] list for
    whichever light/spotlight marker is closest to the aim point (screen
    center by default, or an explicit (aim_x, aim_y) for mouse-click
    picking), within max_screen_dist pixels -- or None. Mirrors the same
    screen-space projection draw_light_indicator uses (including its
    off-screen edge-marker fallback for lights behind/beside the camera,
    so a light's edge arrow is clickable too, not just its on-screen dot)."""
    best_idx, best_d = None, max_screen_dist
    hw, hh = sw / 2.0, sh / 2.0
    ax = hw if aim_x is None else aim_x
    ay = hh if aim_y is None else aim_y
    Rt = R.T
    combined = list(lights) + list(spotlights)
    for i, lt in enumerate(combined):
        rel = lt.position - camera_pos
        xc, yc, zc = Rt @ rel
        if zc > 0.1:
            sx = (xc / (zc * half_tan * aspect)) * hw + hw
            sy = (-yc / (zc * half_tan)) * hh + hh
        else:
            mag = math.sqrt(xc * xc + yc * yc) + 1e-9
            dx, dy = xc / mag, -yc / mag
            margin = 24
            sx = hw + dx * (hw - margin)
            sy = hh + dy * (hh - margin)
        d = math.hypot(sx - ax, sy - ay)
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
    used while rendering the final image ('9'), a quick preview ('8'), or a
    video ('0') so the user can see how much longer it will take instead of
    a frozen/blank screen while waiting."""
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
        'inertia': camera_path.inertia if camera_path is not None else 0.0,
        'inertia_bounce': camera_path.inertia_bounce if camera_path is not None else 0.35,
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
    handheld_shake/handheld_seed/inertia/inertia_bounce already applied
    to it). Any embedded assets are extracted next to the scene file,
    into '<scene_file_stem>_assets/'."""
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
    camera_path.inertia = data.get('inertia', 0.0)
    camera_path.inertia_bounce = data.get('inertia_bounce', 0.35)

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
    #scene.add_box((0, -1.2, 0), (60, 1, 60), (200, 200, 205), roughness=0.35)
    # Back wall
    #scene.add_box((0, 10, 20), (60, 20, 1), (230, 230, 235), roughness=0.6)
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

    demo_tex = os.path.join(os.path.dirname(os.path.abspath(__file__)), "osage.png")
    try:
        scene.add_image((-6.2, 8.4, -3.12), (4.20, 8), demo_tex,
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

def build_custom_scene():
    scene = Scene()
    water_f_pos = (-15, 3.8, 30)
    water_s_pos = (-47, 0, -31)
    water_size = (
        abs(water_f_pos[0] - water_s_pos[0]),
        abs(water_f_pos[1] - water_s_pos[1]),
        abs(water_f_pos[2] - water_s_pos[2]),
    )
    water_center = (
        min(water_f_pos[0], water_s_pos[0]) + water_size[0] / 2,
        min(water_f_pos[1], water_s_pos[1]) + water_size[1] / 2,
        min(water_f_pos[2], water_s_pos[2]) + water_size[2] / 2,
    )
    scene.add_water(center=water_center, size=water_size, color=(120, 220, 240), transparency=0.8)
    scene = load_schematic("/home/khang238/Documents/python/watar.schem", scene)
    scene.add_image((-18.6, 4.32, -7), (2.10, 4), "osage.png", rotation=(math.radians(90), 0, 0), roughness=0.15, reflection_k=0.05)
    # scene.add_image((-11, 5.98, -4), (1.63, 4), "miku_wonder.png", rotation=(math.radians(90), 0, 0), roughness=0.15, reflection_k=0.05)
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

# Same caveat as ARROW_CODES above -- F-key raw codes from cv2.waitKeyEx()
# are not consistent across OS/backends either. These are the common X11/
# GTK keysym values (Linux); if an F-key doesn't respond on your platform,
# run with --print-keys, press it, and add the printed raw code here.
FUNCTION_CODES = {
    'f5': {65474},
    'f6': {65475},
    'f10': {65479},
}


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
    ('func', 'f10'), ('char', lowercase_char), or (None, None)."""
    if raw == -1:
        return None, None
    for name, codes in ARROW_CODES.items():
        if raw in codes:
            return 'arrow', name
    for name, codes in FUNCTION_CODES.items():
        if raw in codes:
            return 'func', name
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
              "3) Enter an exact speed (terminal)", "4) Set zoom (terminal)",
              "5) Delete this keyframe", "6) Close (Esc)"]
    result = ('cancel', None)
    choosing = True
    while choosing:
        canvas[:] = (22, 18, 18)
        title = (f"Camera keyframe #{idx + 1}  --  speed: {kf.speed:.2f} units/second"
                  f"  --  zoom: {kf.zoom:.2f}x")
        tw = _text_width(title)
        _draw_text_shadow(canvas, title, (sw // 2 - tw // 2, sh // 2 - 90))
        for i, label in enumerate(labels):
            lw = _text_width(label)
            y = sh // 2 - 40 + i * 28
            _draw_text_shadow(canvas, label, (sw // 2 - lw // 2, y))
        hint = "Press 1-6 to choose -- Esc to close"
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
            result, choosing = ('zoom_prompt', None), False
        elif ascii_code == ord('5'):
            result, choosing = ('delete', None), False
        elif ascii_code == ord('6'):
            result, choosing = ('cancel', None), False

    if result[0] == 'speed_prompt':
        try:
            raw_in = input(f"Enter a new speed for keyframe #{idx + 1} "
                            f"(world units/second, currently {kf.speed:.2f}): ")
            result = ('set_speed', max(0.01, float(raw_in)))
        except Exception:
            print("Invalid value -- speed left unchanged.")
            result = ('cancel', None)
    elif result[0] == 'zoom_prompt':
        try:
            raw_in = input(f"Enter a new zoom for keyframe #{idx + 1} "
                            f"(1.0 = unzoomed, currently {kf.zoom:.2f}): ")
            result = ('set_zoom', max(1.0, float(raw_in)))
        except Exception:
            print("Invalid value -- zoom left unchanged.")
            result = ('cancel', None)
    return result


# =============================================================================
# 19b) Light properties "menu" -- keyboard-driven, opened by clicking a
#      light/spotlight marker directly (or Tab-selecting it, then this key).
#      Same blocking-loop pattern as keyframe_options_menu_cv.
# =============================================================================

def _configure_spotlight_animation_cli(light):
    """Terminal Q&A flow (same plain-input style as the export path / hold-
    keyframe-duration prompts elsewhere in this file, since there's no
    mouse to drag sliders with) to attach or replace a SpotLightAnimation
    on `light`. Returns True if the animation was changed."""
    print("\nSpotlight animation -- choose a kind:")
    print("  1) Orbit     -- circles around a pivot point")
    print("  2) Ping-pong -- eases back and forth between two points")
    print("  3) Sweep        -- lighthouse-style yaw/pitch sway, position fixed")
    print("  4) Keyframes    -- explicit position/color/etc. waypoints, looped")
    print("  5) Circle sweep -- aim traces a full circle, position fixed")
    print("  0) Cancel")
    try:
        choice = input("Choice: ").strip()
    except EOFError:
        choice = ''

    def _f(prompt, default):
        raw_in = input(prompt).strip()
        return float(raw_in) if raw_in else default

    try:
        if choice == '1':
            raw_in = input(f"Pivot X Y Z (Enter = current position "
                            f"{light.position[0]:.2f} {light.position[1]:.2f} "
                            f"{light.position[2]:.2f}): ").split()
            pivot = [float(v) for v in raw_in] if raw_in else light.position.tolist()
            radius = _f("Radius (Enter = 5.0): ", 5.0)
            axis = (input("Orbit plane axis x/y/z (Enter = y, horizontal): ").strip() or 'y').lower()
            period = _f("Seconds per full revolution (Enter = 6.0): ", 6.0)
            aim_raw = input("Keep aimed at the pivot while orbiting? (Y/n): ").strip().lower()
            aim = aim_raw != 'n'
            light.set_animation(SpotLightAnimation('orbit', pivot=pivot, radius=radius, axis=axis,
                                                    period_s=period, aim_at_pivot=aim))
            print("Orbit animation attached.")
            return True
        elif choice == '2':
            raw_a = input("Point A X Y Z (Enter = current position): ").split()
            point_a = [float(v) for v in raw_a] if raw_a else light.position.tolist()
            point_b = [float(v) for v in input("Point B X Y Z: ").split()]
            period = _f("Seconds per full A->B->A cycle (Enter = 4.0): ", 4.0)
            raw_t = input("Aim target X Y Z (Enter = keep direction fixed): ").split()
            kwargs = {'aim_target': [float(v) for v in raw_t]} if raw_t else {}
            light.set_animation(SpotLightAnimation('pingpong', point_a=point_a, point_b=point_b,
                                                    period_s=period, **kwargs))
            print("Ping-pong animation attached.")
            return True
        elif choice == '3':
            yaw_amp = _f("Yaw sweep amplitude, degrees (Enter = 30): ", 30.0)
            pitch_amp = _f("Pitch sweep amplitude, degrees (Enter = 0): ", 0.0)
            period = _f("Seconds per full swing cycle (Enter = 4.0): ", 4.0)
            light.set_animation(SpotLightAnimation('sweep', yaw_amplitude_deg=yaw_amp,
                                                    pitch_amplitude_deg=pitch_amp, period_s=period))
            print("Sweep animation attached (centered on the spotlight's current aim).")
            return True
        elif choice == '4':
            print("Enter keyframes one at a time (need >= 2); blank time finishes the list.")
            kfs = []
            while True:
                raw_t = input(f"  Keyframe {len(kfs) + 1} time in seconds (blank = done): ").strip()
                if not raw_t:
                    break
                kf = {'time': float(raw_t)}
                raw_p = input("    Position X Y Z (Enter = inherit): ").split()
                if raw_p:
                    kf['position'] = [float(v) for v in raw_p]
                raw_tg = input("    Aim target X Y Z (Enter = inherit): ").split()
                if raw_tg:
                    kf['target'] = [float(v) for v in raw_tg]
                raw_c = input("    Color R G B 0-255 (Enter = inherit): ").split()
                if raw_c:
                    kf['color'] = [float(v) for v in raw_c]
                raw_b = input("    Brightness (Enter = inherit): ").strip()
                if raw_b:
                    kf['brightness'] = float(raw_b)
                raw_ca = input("    Cone angle, degrees (Enter = inherit): ").strip()
                if raw_ca:
                    kf['cone_angle'] = float(raw_ca)
                raw_sf = input("    Softness 0-1 (Enter = inherit): ").strip()
                if raw_sf:
                    kf['softness'] = float(raw_sf)
                kfs.append(kf)
            if len(kfs) < 2:
                print("Need at least 2 keyframes -- animation not changed.")
                return False
            light.set_animation(SpotLightAnimation('keyframes', keyframes=kfs))
            print(f"Keyframe animation attached ({len(kfs)} keyframes, loops after "
                  f"{kfs[-1]['time']:.2f}s).")
            return True
        elif choice == '5':
            radius = _f("Circle radius, degrees (Enter = 15): ", 15.0)
            period = _f("Seconds per full revolution (Enter = 4.0): ", 4.0)
            dir_raw = input("Direction, clockwise/counter-clockwise (Enter = clockwise): ").strip().lower()
            direction = -1.0 if dir_raw.startswith('counter') or dir_raw.startswith('ccw') else 1.0
            light.set_animation(SpotLightAnimation('circle_sweep', radius_deg=radius,
                                                    period_s=period, direction=direction))
            print("Circle sweep animation attached (centered on the spotlight's current aim).")
            return True
        else:
            print("Cancelled.")
            return False
    except Exception as e:
        print(f"Invalid input -- animation not changed ({e}).")
        return False


def light_options_menu_cv(canvas, sw, sh, light, idx, camera_pos, camera_rot, camera_roll,
                           is_spot=False):
    """Blocking modal menu for editing one light/spotlight's properties in
    place. Position/color are entered via the terminal (not practical to
    drag numeric text in a plain cv2 window); brightness and (for
    spotlights) cone angle/softness have quick +/- keys since those get
    tweaked far more often. Returns True if anything changed (caller is
    responsible for tracer.sync_lights()/compute_ss() afterward,
    since those are somewhat expensive and shouldn't run on every keypress)."""
    changed = False
    choosing = True
    while choosing:
        canvas[:] = (18, 20, 22)
        kind = "Spotlight" if is_spot else "Light"
        title = f"{kind} #{idx + 1}  --  brightness {light.brightness:.2f}" + (
            f"  cone {light.cone_angle:.0f} deg  soft {light.softness:.2f}" if is_spot else "")
        tw = _text_width(title)
        _draw_text_shadow(canvas, title, (sw // 2 - tw // 2, sh // 2 - 110))

        labels = ["1) Brightness +0.25", "2) Brightness -0.25",
                  "3) Move to the camera's position" + (" (+aim)" if is_spot else ""),
                  "4) Set color (terminal, R G B 0-255)",
                  "5) Set position (terminal, exact X Y Z)"]
        if is_spot:
            anim_note = f"[{light.animation.kind}]" if light.animation is not None else "[none]"
            labels += ["6) Cone angle +3 deg", "7) Cone angle -3 deg",
                       "8) Softness +0.1", "9) Softness -0.1",
                       f"A) Configure animation (terminal)  {anim_note}",
                       "Z) Clear animation"]
        labels.append("0) Close (Esc)")

        for i, label in enumerate(labels):
            lw = _text_width(label)
            y = sh // 2 - 60 + i * 26
            _draw_text_shadow(canvas, label, (sw // 2 - lw // 2, y))
        cv2.imshow(WINDOW_NAME, canvas)
        raw = cv2.waitKeyEx(30)
        if raw == -1:
            continue
        ascii_code = raw & 0xFF
        if ascii_code == 27 or ascii_code == ord('0'):
            choosing = False
        elif ascii_code == ord('1'):
            light.brightness = max(0.0, light.brightness + 0.25); changed = True
        elif ascii_code == ord('2'):
            light.brightness = max(0.0, light.brightness - 0.25); changed = True
        elif ascii_code == ord('3'):
            light.position = camera_pos.astype(np.float32).copy()
            if is_spot:
                light.direction = camera_matrix(*camera_rot, camera_roll)[:, 2].astype(np.float32)
            changed = True
        elif ascii_code == ord('4'):
            try:
                raw_in = input(f"New color for {kind.lower()} #{idx + 1} "
                                f"(R G B, each 0-255, currently "
                                f"{int(light.color[0]*255)} {int(light.color[1]*255)} "
                                f"{int(light.color[2]*255)}): ").split()
                r, g, b = (max(0.0, min(255.0, float(x))) for x in raw_in)
                light.color = np.array([r, g, b], dtype=np.float32) / 255.0
                changed = True
            except Exception:
                print("Invalid color -- left unchanged.")
        elif ascii_code == ord('5'):
            try:
                raw_in = input(f"New position for {kind.lower()} #{idx + 1} "
                                f"(X Y Z, currently {light.position[0]:.2f} "
                                f"{light.position[1]:.2f} {light.position[2]:.2f}): ").split()
                x, y, z = (float(v) for v in raw_in)
                light.position = np.array([x, y, z], dtype=np.float32)
                changed = True
            except Exception:
                print("Invalid position -- left unchanged.")
        elif is_spot and ascii_code == ord('6'):
            light.cone_angle = max(3.0, min(80.0, light.cone_angle + 3.0)); changed = True
        elif is_spot and ascii_code == ord('7'):
            light.cone_angle = max(3.0, min(80.0, light.cone_angle - 3.0)); changed = True
        elif is_spot and ascii_code == ord('8'):
            light.softness = max(0.0, min(1.0, light.softness + 0.1)); changed = True
        elif is_spot and ascii_code == ord('9'):
            light.softness = max(0.0, min(1.0, light.softness - 0.1)); changed = True
        elif is_spot and ascii_code in (ord('a'), ord('A')):
            if _configure_spotlight_animation_cli(light):
                changed = True
        elif is_spot and ascii_code in (ord('z'), ord('Z')):
            if light.animation is not None:
                light.clear_animation()
                print("Animation cleared.")
                changed = True
    return changed


# =============================================================================
# 19c) Post-FX / renderer properties menu (` key) -- a live, scrollable
#      list of (almost) every tunable in DEFAULT_POST_FX plus the handful
#      of renderer-level dials on RayTracer/CameraPath (exposure, ambient,
#      sky light, caustic strength, handheld/footstep shake). Reading and
#      writing go straight through closures into the SAME post_fx dict /
#      tracer / camera_path the rest of the program already uses -- there's
#      no separate copy to sync back, so a change here takes effect the
#      moment you make it (dof_enabled flips on immediately, exposure
#      changes the very next frame, etc.), same as the old dedicated toggle
#      keys did, just for every parameter instead of only on/off ones.
# =============================================================================

def _build_world_params(tracer, post_fx, camera_path, has_camera_data):
    """World/atmosphere/lighting settings -- opened with F5 (see
    _build_camera_params for F6's camera/renderer settings, and
    _build_postfx_params for the `` ` `` key's remaining stylistic
    post-processing effects)."""
    params = []

    def add(label, get, set_, kind, step=0.0, lo=None, hi=None):
        params.append({'label': label, 'get': get, 'set': set_, 'kind': kind,
                        'step': step, 'lo': lo, 'hi': hi})

    def pf(key):
        return lambda: post_fx[key]

    def spf(key):
        def setter(v):
            post_fx[key] = v
        return setter

    add("Ambient light", lambda: tracer.ambient, lambda v: setattr(tracer, 'ambient', v),
        'float', 0.02, 0.0, 2.0)
    add("Sky light strength", lambda: tracer.sky_light_strength,
        lambda v: setattr(tracer, 'sky_light_strength', v), 'float', 0.05, 0.0, 5.0)
    if tracer.background.sun is not None:
        add("  Sun/moon intensity", lambda: tracer.background.sun['intensity'],
            lambda v: tracer.set_sun_intensity(v), 'float', 0.05, 0.0, 3.0)
    if tracer.background.stars is not None:
        add("  Star brightness", lambda: tracer.background.stars['brightness'],
            lambda v: tracer.add_stars(**{**tracer.background.stars, 'brightness': v}),
            'float', 0.05, 0.0, 3.0)
        add("  Star twinkle", lambda: tracer.background.stars['twinkle'],
            lambda v: tracer.add_stars(**{**tracer.background.stars, 'twinkle': v}),
            'float', 0.05, 0.0, 1.0)

    add("[Clouds] Enabled", lambda: bool(tracer.clouds and tracer.clouds.get('enabled')),
        lambda v: tracer.set_clouds(**{**(tracer.clouds or {}), 'enabled': v}), 'bool')
    if tracer.clouds is not None:
        add("  Cloud density", lambda: tracer.clouds['density'],
            lambda v: tracer.set_clouds(**{**tracer.clouds, 'density': v}), 'float', 0.01, 0.0, 0.6)
        add("  Cloud coverage", lambda: tracer.clouds['coverage'],
            lambda v: tracer.set_clouds(**{**tracer.clouds, 'coverage': v}), 'float', 0.02, 0.0, 1.0)
        add("  Cloud altitude (base)", lambda: tracer.clouds['base'],
            lambda v: tracer.set_clouds(**{**tracer.clouds, 'base': v,
                                           'top': max(v + 1.0, tracer.clouds['top'])}),
            'float', 2.0, 0.0, 2000.0)
        add("  Cloud thickness", lambda: tracer.clouds['top'] - tracer.clouds['base'],
            lambda v: tracer.set_clouds(**{**tracer.clouds, 'top': tracer.clouds['base'] + max(1.0, v)}),
            'float', 2.0, 1.0, 500.0)

    add("[C] Caustics enabled", lambda: tracer.caustics_enabled,
        lambda v: setattr(tracer, 'caustics_enabled', v), 'bool')
    add("  Caustic strength", lambda: tracer.caustic_strength,
        lambda v: setattr(tracer, 'caustic_strength', v), 'float', 0.1, 0.0, 5.0)

    add("[3] Fog", pf('fog_enabled'), spf('fog_enabled'), 'bool')
    add("  Density", pf('fog_density'), spf('fog_density'), 'float', 0.001, 0.0, 0.2)
    add("  Max depth", pf('fog_max_depth'), spf('fog_max_depth'), 'float', 5.0, 1.0, 2000.0)
    add("  Height falloff", pf('fog_height_falloff'), spf('fog_height_falloff'), 'float', 0.01, 0.0, 2.0)
    add("  Base height", pf('fog_base_height'), spf('fog_base_height'), 'float', 0.5, -500.0, 500.0)

    add("[4] God rays", pf('godrays_enabled'), spf('godrays_enabled'), 'bool')
    add("  Intensity", pf('godrays_intensity'), spf('godrays_intensity'), 'float', 0.02, 0.0, 2.0)
    add("  Decay", pf('godrays_decay'), spf('godrays_decay'), 'float', 0.005, 0.5, 0.999)
    add("  Density", pf('godrays_density'), spf('godrays_density'), 'float', 0.02, 0.1, 1.5)
    add("  Samples", pf('godrays_samples'), spf('godrays_samples'), 'int', 1, 1, 96)

    return params


def _build_camera_params(tracer, post_fx, camera_path, has_camera_data, res_getset, sample_getset):
    """Camera/renderer settings -- opened with F6 (see _build_world_params
    for F5's world/atmosphere settings, and _build_postfx_params for the
    `` ` `` key's remaining stylistic post-processing effects).
    res_getset/sample_getset: dicts of the extra getter/setter closures
    main() builds for resolution and per-mode sample counts (these live as
    plain locals in main(), not on tracer/post_fx/camera_path, since
    LIVE_RENDER_RES/FINAL_RENDER_RES are module globals and final_samples/
    video_samples are local to the '9'/'0' key handlers -- see there)."""
    params = []

    def add(label, get, set_, kind, step=0.0, lo=None, hi=None):
        params.append({'label': label, 'get': get, 'set': set_, 'kind': kind,
                        'step': step, 'lo': lo, 'hi': hi})

    def pf(key):
        return lambda: post_fx[key]

    def spf(key):
        def setter(v):
            post_fx[key] = v
        return setter

    add("Exposure", lambda: tracer.exposure, lambda v: setattr(tracer, 'exposure', v),
        'float', 0.05, 0.02, 20.0)
    add("[Y] Eye adaptation (auto exposure)", lambda: tracer.eye_adapt_enabled,
        lambda v: setattr(tracer, 'eye_adapt_enabled', v), 'bool')
    add("  Eye adapt speed", lambda: tracer.eye_adapt_speed,
        lambda v: setattr(tracer, 'eye_adapt_speed', v), 'float', 0.1, 0.05, 20.0)

    add("[;/'] Optical zoom", lambda: tracer.zoom, lambda v: tracer.set_zoom(v),
        'float', 0.1, 1.0, 8.0)
    add("  Zoom speed (x/second)", lambda: tracer.zoom_speed,
        lambda v: setattr(tracer, 'zoom_speed', max(1e-4, v)), 'float', 0.1, 0.05, 20.0)

    add("[U] Depth of field", pf('dof_enabled'), spf('dof_enabled'), 'bool')
    add("  [-/=] Focus distance", pf('dof_focus_distance'), spf('dof_focus_distance'), 'float', 0.5, 0.1, 2000.0)
    add("  Blur strength", pf('dof_blur_strength'), spf('dof_blur_strength'), 'float', 0.05, 0.0, 5.0)
    add("  Max blur radius (px)", pf('dof_max_radius'), spf('dof_max_radius'), 'int', 2, 0, 200)
    add("[G] Continuous autofocus (video)", pf('autofocus_enabled'), spf('autofocus_enabled'), 'bool')
    add("  Autofocus speed", pf('autofocus_speed'), spf('autofocus_speed'), 'float', 0.2, 0.05, 20.0)

    if not has_camera_data:
        add("Handheld camera shake", lambda: camera_path.handheld_shake,
            lambda v: setattr(camera_path, 'handheld_shake', v), 'float', 0.05, 0.0, 3.0)
        add("  Footstep shake multiplier", lambda: camera_path.footstep_shake,
            lambda v: setattr(camera_path, 'footstep_shake', v), 'float', 0.05, 0.0, 3.0)
        add("Camera inertia", lambda: camera_path.inertia,
            lambda v: setattr(camera_path, 'inertia', v), 'float', 0.05, 0.0, 3.0)
        add("  Bounce (overshoot)", lambda: camera_path.inertia_bounce,
            lambda v: setattr(camera_path, 'inertia_bounce', v), 'float', 0.05, 0.0, 1.0)

    add("Live resolution width (px)", res_getset['live_w_get'], res_getset['live_w_set'],
        'int', 16, 16, 1920)
    add("  height (px)", res_getset['live_h_get'], res_getset['live_h_set'], 'int', 16, 16, 1080)
    add("Final render width (px)", res_getset['final_w_get'], res_getset['final_w_set'],
        'int', 64, 64, 7680)
    add("  height (px)", res_getset['final_h_get'], res_getset['final_h_set'], 'int', 64, 64, 4320)

    add("Live samples/frame", sample_getset['live_get'], sample_getset['live_set'], 'int', 1, 1, 64)
    add("Final render samples ('9' key)", sample_getset['final_get'], sample_getset['final_set'],
        'int', 4, 1, 4096)
    add("Video samples/frame ('0' key)", sample_getset['video_get'], sample_getset['video_set'],
        'int', 1, 1, 256)

    return params


def _build_postfx_params(tracer, post_fx, camera_path, has_camera_data):
    """Everything left over: stylistic image post-processing that isn't a
    world/lighting property (F5) or a core camera/renderer setting (F6) --
    lens/sensor artifacts and color-grade-style effects."""
    params = []

    def add(label, get, set_, kind, step=0.0, lo=None, hi=None):
        params.append({'label': label, 'get': get, 'set': set_, 'kind': kind,
                        'step': step, 'lo': lo, 'hi': hi})

    def pf(key):
        return lambda: post_fx[key]

    def spf(key):
        def setter(v):
            post_fx[key] = v
        return setter

    add("[R] Chromatic aberration", pf('chroma_enabled'), spf('chroma_enabled'), 'bool')
    add("  Strength", pf('chroma_strength'), spf('chroma_strength'), 'float', 0.001, 0.0, 0.05)

    add("[N] Lens flare", pf('flare_enabled'), spf('flare_enabled'), 'bool')
    add("  Size", pf('flare_size'), spf('flare_size'), 'float', 2.0, 0.0, 300.0)
    add("  Intensity", pf('flare_intensity'), spf('flare_intensity'), 'float', 0.05, 0.0, 5.0)
    add("  Anamorphic streak", pf('flare_anamorphic'), spf('flare_anamorphic'), 'float', 0.02, 0.0, 1.0)
    add("  Halo", pf('flare_halo'), spf('flare_halo'), 'float', 0.02, 0.0, 1.0)

    add("[F] Fisheye lens", pf('fisheye_enabled'), spf('fisheye_enabled'), 'bool')
    add("  Strength", pf('fisheye_strength'), spf('fisheye_strength'), 'float', 0.02, 0.0, 1.0)

    add("[B] Bloom", pf('bloom_enabled'), spf('bloom_enabled'), 'bool')
    add("  Threshold", pf('bloom_threshold'), spf('bloom_threshold'), 'float', 0.02, 0.0, 1.0)
    add("  Intensity", pf('bloom_intensity'), spf('bloom_intensity'), 'float', 0.02, 0.0, 3.0)
    add("  Radius (px)", pf('bloom_radius'), spf('bloom_radius'), 'int', 1, 1, 100)

    add("[V] VHS effect", pf('vhs_enabled'), spf('vhs_enabled'), 'bool')
    add("  Strength", pf('vhs_strength'), spf('vhs_strength'), 'float', 0.05, 0.0, 3.0)

    add("[5] Analog camera (camcorder)", pf('analog_enabled'), spf('analog_enabled'), 'bool')
    add("  Strength", pf('analog_strength'), spf('analog_strength'), 'float', 0.05, 0.0, 2.0)
    add("  Dynamic range crush", pf('analog_contrast'), spf('analog_contrast'), 'float', 0.02, 0.0, 1.0)
    add("  Highlight clip", pf('analog_highlight_clip'), spf('analog_highlight_clip'), 'float', 0.02, 0.0, 1.0)
    add("  Grain", pf('analog_grain'), spf('analog_grain'), 'float', 0.005, 0.0, 0.2)
    add("  Grain size", pf('analog_grain_size'), spf('analog_grain_size'), 'float', 0.1, 1.0, 6.0)
    add("  Halation", pf('analog_halation'), spf('analog_halation'), 'float', 0.02, 0.0, 1.0)
    add("  Vignette", pf('analog_vignette'), spf('analog_vignette'), 'float', 0.02, 0.0, 1.5)
    add("  Warmth", pf('analog_warmth'), spf('analog_warmth'), 'float', 0.01, 0.0, 1.0)
    add("  Motion smear", pf('analog_smear'), spf('analog_smear'), 'float', 0.05, 0.0, 2.0)

    add("[M] Motion blur (video only)", pf('motion_blur_enabled'), spf('motion_blur_enabled'), 'bool')
    add("  Shutter", pf('motion_blur_shutter'), spf('motion_blur_shutter'), 'float', 0.05, 0.0, 1.0)

    return params


def _postfx_row_geometry(params):
    """Derives tree depth/branch info for each row from the existing
    label-indentation convention (a label with no leading spaces is an FX
    "root"; a label starting with 2+ leading spaces is that root's child --
    see _build_postfx_params, unchanged there) so the menu can draw
    tree-style branch lines without needing a separate data structure.
    Returns a list of dicts (same length/order as params), one per row:
      'depth'   -- 0 for a root/FX row, 1 for a sub-property row.
      'text'    -- the label with its leading indent stripped (the tree
                   lines themselves now carry that visual nesting instead).
      'is_last' -- True if this is the LAST child in its root's group
                   (its branch connector is an "L" corner instead of a
                   "T" tee), or True for a childless root.
    """
    geom = []
    n = len(params)
    for i, p in enumerate(params):
        label = p['label']
        stripped = label.lstrip(' ')
        depth = 1 if (len(label) - len(stripped)) >= 2 else 0
        is_last = True
        if depth == 1:
            nxt = params[i + 1]['label'] if i + 1 < n else ''
            nxt_depth = 1 if (len(nxt) - len(nxt.lstrip(' '))) >= 2 else 0
            is_last = (nxt_depth == 0)
        geom.append({'depth': depth, 'text': stripped, 'is_last': is_last})
    return geom


def postfx_menu_cv(canvas, sw, sh, params):
    """Blocking modal menu (` key): Up/Down selects a row, Left/Right
    adjusts its value (bool rows just flip either way), Enter also flips a
    bool row, Esc closes. Every row edits the actual live tracer/post_fx/
    camera_path object directly (see _build_postfx_params) -- closing this
    menu doesn't "apply" anything, it already IS applied.

    Layout: root rows (an FX's own on/off/name row) sit at a fixed left
    margin; sub-property rows are indented further right and connected to
    their root with tree-style branch lines (a vertical "trunk" running
    down the group with "|-" tees off of it, an "L" corner on the last
    child) so a run of properties visually reads as belonging to one FX.
    Each row's value is right-aligned in its own column, joined to the
    label by a short dark-gray leader line (like a table of contents)
    instead of arbitrary whitespace, so values line up regardless of how
    long each label is.
    """
    sel = 0
    n = len(params)
    geom = _postfx_row_geometry(params)
    header_h = 68
    footer_h = 14
    # Row height / viewport size are DERIVED from the actual window height
    # (sh) instead of a fixed 18 rows / 24px -- that fixed pairing assumed
    # a taller window than PREVIEW_RES's current 640x360 and got the
    # bottom rows cut off below the visible canvas. This keeps every row
    # on-screen (down to a legible minimum row height) no matter the
    # window size.
    row_h = max(14, min(24, (sh - header_h - footer_h) // max(1, min(n, 18))))
    viewport = max(1, (sh - header_h - footer_h) // row_h)

    tree_x = 42        # left edge of the tree-line gutter
    root_text_x = 42   # root (FX) label text -- one step in from the gutter
    child_indent = 26  # extra right-shift for a sub-property vs. its root
    child_text_x = root_text_x + child_indent
    value_right_x = sw - 28  # right edge that every value is right-aligned against
    leader_color = (90, 90, 95)  # dark gray leader line, BGR handled by cv2 line below

    open_ = True
    while open_ and n > 0:
        canvas[:] = (16, 16, 20)
        title = "Post-FX / renderer properties  --  Up/Down select, Left/Right adjust, Esc close"
        subtitle = ("[key] before a row = its own hotkey outside this menu, e.g. [Y] toggles "
                    "the same eye-adaptation flag as pressing Y. A few other hotkeys don't have "
                    "a matching row here: [T] focuses instantly (one-shot, not a persistent "
                    "setting), [H] cycles handheld shake and [2] types in an exact camera "
                    "position/angle (both are on CameraPath, not post-fx).")
        _draw_text_shadow(canvas, title, (24, 32))
        _draw_text_shadow(canvas, subtitle, (24, 50), color=(150, 150, 150), scale=_HUD_SCALE_SMALL)
        top = max(0, min(sel - viewport // 2, max(0, n - viewport)))
        bottom = min(n, top + viewport)

        # --- Tree trunk lines: one continuous vertical segment per FX
        # group that has at least one visible child row in [top, bottom),
        # running from the group's root row down to its last visible
        # child row. Drawn before the rows/text so text sits on top.
        i = top
        while i < bottom:
            if geom[i]['depth'] == 0:
                root_row = i - top
                j = i + 1
                last_child_row = None
                while j < bottom and geom[j]['depth'] == 1:
                    last_child_row = j - top
                    j += 1
                if last_child_row is not None:
                    trunk_x = tree_x + 6
                    y_top = 68 + root_row * row_h + row_h // 2
                    y_bot = 68 + last_child_row * row_h + row_h // 2
                    cv2.line(canvas, (trunk_x, y_top), (trunk_x, y_bot), (80, 80, 88), 1, cv2.LINE_AA)
                i = j if j > i + 1 else i + 1
            else:
                i += 1

        for row, i in enumerate(range(top, bottom)):
            p = params[i]
            g = geom[i]
            y = 68 + row * row_h
            y_mid = y - 5  # roughly vertical-center of the text baseline for line-drawing
            val = p['get']()
            if p['kind'] == 'bool':
                val_s = "ON" if val else "off"
            elif p['kind'] == 'int':
                val_s = f"{int(val)}"
            else:
                val_s = f"{val:.3f}"
            is_sel = (i == sel)
            color = (255, 255, 130) if is_sel else (195, 195, 195)
            marker = ">" if is_sel else " "

            # --- Branch connector + label text ---
            if g['depth'] == 0:
                _draw_text_shadow(canvas, marker, (tree_x - 16, y), color=color, scale=_HUD_SCALE_SMALL)
                text_x = root_text_x
            else:
                trunk_x = tree_x + 6
                corner_x = trunk_x + 14
                if g['is_last']:
                    # "L" corner: down-then-right, stops here (last child).
                    cv2.line(canvas, (trunk_x, y_mid - row_h // 2 + 2), (trunk_x, y_mid), (80, 80, 88), 1, cv2.LINE_AA)
                    cv2.line(canvas, (trunk_x, y_mid), (corner_x, y_mid), (80, 80, 88), 1, cv2.LINE_AA)
                else:
                    # "T" tee: trunk keeps going past this row.
                    cv2.line(canvas, (trunk_x, y_mid), (corner_x, y_mid), (80, 80, 88), 1, cv2.LINE_AA)
                if is_sel:
                    _draw_text_shadow(canvas, marker, (tree_x - 16, y), color=color, scale=_HUD_SCALE_SMALL)
                text_x = child_text_x
            _draw_text_shadow(canvas, g['text'], (text_x, y), color=color, scale=_HUD_SCALE_SMALL)

            # --- Leader line + right-aligned value ---
            label_end_x = text_x + _text_width(g['text'], scale=_HUD_SCALE_SMALL)
            val_w = _text_width(val_s, scale=_HUD_SCALE_SMALL)
            val_x = value_right_x - val_w
            leader_gap = 6
            leader_x0 = label_end_x + leader_gap
            leader_x1 = val_x - leader_gap
            if leader_x1 > leader_x0:
                cv2.line(canvas, (leader_x0, y_mid), (leader_x1, y_mid), leader_color, 1, cv2.LINE_AA)
            _draw_text_shadow(canvas, val_s, (val_x, y), color=color, scale=_HUD_SCALE_SMALL)
        cv2.imshow(WINDOW_NAME, canvas)
        raw = cv2.waitKeyEx(30)
        kind, val = classify_key(raw)
        if kind == 'arrow':
            if val == 'up':
                sel = (sel - 1) % n
            elif val == 'down':
                sel = (sel + 1) % n
            elif val in ('left', 'right'):
                p = params[sel]
                if p['kind'] == 'bool':
                    p['set'](not p['get']())
                else:
                    step = p['step'] if val == 'right' else -p['step']
                    newv = p['get']() + step
                    if p['lo'] is not None:
                        newv = max(p['lo'], newv)
                    if p['hi'] is not None:
                        newv = min(p['hi'], newv)
                    if p['kind'] == 'int':
                        newv = int(round(newv))
                    p['set'](newv)
        elif kind == 'char':
            if val == 'enter':
                p = params[sel]
                if p['kind'] == 'bool':
                    p['set'](not p['get']())
            elif val == 'esc':
                open_ = False


# =============================================================================
# 19d) Manual camera position/angle entry (the '2' key) -- terminal prompt,
#      same pattern as the other "type an exact value" prompts (keyframe
#      speed, light color/position) rather than an in-canvas text field,
#      since cv2 has no text-input widget to build on.
# =============================================================================

def prompt_manual_camera(camera_pos, camera_rot, camera_roll):
    """Blank input on either line leaves that value unchanged. Returns
    (new_pos [np.float64 array], new_yaw, new_pitch, new_roll) -- all in
    radians for the angles, matching camera_rot/camera_roll's own units."""
    print(f"Current position: {camera_pos[0]:.3f} {camera_pos[1]:.3f} {camera_pos[2]:.3f}")
    print(f"Current yaw/pitch/roll (degrees): {math.degrees(camera_rot[0]):.1f} "
          f"{math.degrees(camera_rot[1]):.1f} {math.degrees(camera_roll):.1f}")
    pos = camera_pos.copy()
    yaw, pitch, roll = camera_rot[0], camera_rot[1], camera_roll
    try:
        raw_in = input("New position (X Y Z, Enter to keep current): ").strip()
        if raw_in:
            x, y, z = (float(v) for v in raw_in.split())
            pos = np.array([x, y, z], dtype=np.float64)
    except Exception:
        print("Invalid position -- left unchanged.")
    try:
        raw_in = input("New yaw/pitch/roll in DEGREES (Enter to keep current): ").strip()
        if raw_in:
            yd, pd, rd = (float(v) for v in raw_in.split())
            yaw, pitch, roll = math.radians(yd), math.radians(pd), math.radians(rd)
            pitch = max(-1.5, min(1.5, pitch))
    except Exception:
        print("Invalid rotation -- left unchanged.")
    return pos, yaw, pitch, roll


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
        description="Raytracer v12 -- interactive (cv2 window) or headless rendering.",
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
    p.add_argument('--output', type=str, default="raytrace_v12.png",
                    help="Output image path for --headless still renders.")
    p.add_argument('--video-output', type=str, default="raytrace_v12_video.mp4",
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
    p.add_argument('--inertia', type=float, default=0.0,
                    help="Adds 'momentum' to camera KEYFRAME paths (P/O -- ignored for "
                         "--camera-data/--camera-stream): instead of instantly snapping onto each "
                         "new segment's direction, the camera eases/overshoots through direction "
                         "changes like a real handheld camera can't instantly reverse. 0 = off "
                         "(exact old linear interpolation), higher = more momentum/lag. Distinct "
                         "from --handheld-shake (that's noise on top of the path; this changes how "
                         "the path itself is followed) -- the two stack fine together.")
    p.add_argument('--inertia-bounce', type=float, default=None,
                    help="Damping for --inertia's spring: 0 = critically damped (smooth ease, no "
                         "overshoot), 1 (default 0.35) = strongly underdamped (visibly bounces past "
                         "the target before settling). Has no effect if --inertia is 0.")
    p.add_argument('--zoom', type=float, default=1.0,
                    help="Starting optical zoom multiplier (see the ;/' keys), 1.0 = unzoomed. Also "
                         "used as the constant zoom for a --headless render/video unless the camera "
                         "path's keyframes carry their own zoom (F10/O menu's 'Set zoom').")
    p.add_argument('--zoom-speed', type=float, default=1.0,
                    help="How fast the optical zoom can rack, in zoom-multiplier/second -- both for "
                         "the interactive ;/' keys and for render_video's per-frame zoom ramp (also "
                         "editable live from the ` post-fx menu). Higher = snappier zoom, lower = "
                         "slower/more cinematic racking.")
    p.add_argument('--multi-gpu', type=str, default='off',
                    help="Split a --headless render across multiple NVIDIA GPUs in separate "
                         "processes (Taichi can't span one kernel launch across GPUs -- see the "
                         "module note above _run_multi_gpu_still for how this actually works). "
                         "'off' (default): single process/GPU, as normal. 'auto': use every GPU "
                         "nvidia-smi finds (only kicks in with 2+). A bare integer, e.g. '2': use "
                         "that many GPUs (indices 0..N-1). An explicit list, e.g. '0,2,3': use "
                         "exactly those GPU indices. Ignored outside --headless (interactive mode "
                         "is inherently single-process/real-time).")
    # --- Internal: --multi-gpu worker flags. Not meant to be set by hand --
    # the orchestrator (_run_multi_gpu_still/_run_multi_gpu_video) adds
    # these itself when it re-invokes this script as a worker process.
    p.add_argument('--_mgpu-role', type=str, default=None, choices=['still', 'video'],
                    help=argparse.SUPPRESS)
    p.add_argument('--_mgpu-samples', type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument('--_mgpu-out', type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument('--_mgpu-start', type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument('--_mgpu-end', type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def _parse_xyz(s):
    parts = [float(x) for x in s.split(',')]
    if len(parts) != 3:
        raise ValueError(f"Expected 'x,y,z', got: {s!r}")
    return tuple(parts)


# =============================================================================
# 21) Main
# =============================================================================

def _do_still_render(tracer, args, post_fx, is_worker, mgpu_indices):
    """Dispatches a still-image render 1 of 3 ways: as a --multi-gpu WORKER
    (renders its assigned sample share, dumps raw accum to --_mgpu-out,
    returns without saving an image), as the --multi-gpu ORCHESTRATOR
    (spawns workers and combines their results -- see _run_multi_gpu_still),
    or normally (single process, as before --multi-gpu existed)."""
    tracer.set_resolution(*FINAL_RENDER_RES)
    if is_worker:
        print(f"[worker] Rendering {tracer.width}x{tracer.height}, "
              f"{args._mgpu_samples} sample(s) (this GPU's share) ...")
        tracer.reset_accumulation()
        tracer.add_samples(args._mgpu_samples)
        accum_np, depth_np, n = tracer.dump_raw_accum()
        np.savez(args._mgpu_out, accum=accum_np, depth_accum=depth_np, n=n)
        print(f"[worker] Done: {n} samples -> {args._mgpu_out}")
        return
    if mgpu_indices:
        _run_multi_gpu_still(mgpu_indices, args.output, args.samples, post_fx, tracer)
        return
    tracer.render_to_file(args.output, samples=args.samples, post_fx=post_fx)


def _do_video_render(tracer, active_path, args, post_fx, is_worker, mgpu_indices, video_res, video_fps):
    """Dispatches a video render the same 3 ways as _do_still_render (see
    above), except a worker's "share" is a CONTIGUOUS block of frame
    indices (--_mgpu-start/--_mgpu-end) rather than a sample count -- see
    the module note above _run_multi_gpu_still for why."""
    if is_worker:
        frame_subset = range(args._mgpu_start, args._mgpu_end)
        tracer.render_video(active_path, out_path=args._mgpu_out, resolution=video_res,
                             post_fx=post_fx, fps=video_fps, camera_sync=args.camera_sync,
                             frame_subset=frame_subset, skip_encode=True)
        return
    if mgpu_indices:
        _run_multi_gpu_video(mgpu_indices, args.video_output, video_fps, args.camera_sync,
                              active_path, VIDEO_DURATION)
        return
    tracer.render_video(active_path, out_path=args.video_output, resolution=video_res,
                         post_fx=post_fx, fps=video_fps, camera_sync=args.camera_sync)


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
        # scene = build_mc_scene(schem_path=args.mc_schem)
        scene = build_custom_scene()
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
        bg = Background(color=(20, 25, 35), brightness=1.0)
        bg.set_sky_gradient(curve=0.2)
        lights = [
            Light((265, 324, -141), (255, 251, 235), 1.1),   # key light, slightly warm
            #Light((-1.47, -4.69, 6.94), (255, 251, 235), 1.0),      # fill light, cooler blue
            #Light((-5.88, 16.79, -13.09), (255, 251, 235), 0.3),
        ]
        spotlights = [
            #SpotLight((0.0, 20.0, -5.0), (0.15, -1.0, 0.25), color=(16, 255, 255), brightness=0.3, cone_angle=14.0, softness=0.04),
            #SpotLight((0.0, 20.0, -5.0), (0.15, -1.0, 0.25), color=(255, 16, 255), brightness=0.3, cone_angle=14.0, softness=0.04),
            #SpotLight((0.0, 20.0, -5.0), (0.15, -1.0, 0.25), color=(255, 255, 16), brightness=0.3, cone_angle=14.0, softness=0.04),
        ]
        fov = 65
        max_bounce = DEFAULT_MAX_BOUNCE
        live_max_bounce = LIVE_MAX_BOUNCE
        cam_cfg = {'pos': [0.0, 6.0, -30.0], 'yaw': 0.0, 'pitch': 0.0}
        post_fx = dict(DEFAULT_POST_FX)
        camera_path = CameraPath()
        ambient, specular_k, shininess = 0.12, 0.6, 64.0
        caustics_enabled = True
        sky_light_strength, caustic_strength = 0.5, 1.0
        exposure, eye_adapt_enabled, eye_adapt_speed = 1.0, False, 1.4

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
    tracer.zoom_speed = max(1e-4, float(args.zoom_speed))
    tracer.set_zoom(float(args.zoom))

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
    if args.inertia:
        camera_path.inertia = float(args.inertia)
    if args.inertia_bounce is not None:
        camera_path.inertia_bounce = float(args.inertia_bounce)

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

    is_mgpu_worker = args._mgpu_role is not None
    mgpu_indices = []
    if args.headless and not is_mgpu_worker:
        mgpu_indices = _resolve_multi_gpu_indices(args.multi_gpu)
    elif (not is_mgpu_worker) and args.multi_gpu not in (None, 'off'):
        print("--multi-gpu is ignored outside --headless (interactive mode is inherently "
              "single-process/real-time).")

    if args.headless:
        if args.video:
            active_path = sensor_data if sensor_data is not None else camera_path
            has_path = sensor_data is not None or len(camera_path.keyframes) > 0
            if not has_path:
                print("The scene has no camera keyframes -- nothing to move the camera along, "
                      "rendering a single still image instead.")
                _do_still_render(tracer, args, post_fx, is_mgpu_worker, mgpu_indices)
            else:
                video_res = _parse_wh(args.video_resolution) if args.video_resolution else VIDEO_RES
                _do_video_render(tracer, active_path, args, post_fx, is_mgpu_worker, mgpu_indices,
                                  video_res, video_fps)
        else:
            _do_still_render(tracer, args, post_fx, is_mgpu_worker, mgpu_indices)
        return

    # --- Interactive mode -------------------------------------------------
    WIN_W, WIN_H = PREVIEW_RES
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)

    # --- Mouse: left-press-and-DRAG to look around; a left-press that stays
    # within a few pixels of where it started (no real drag) is a CLICK
    # instead, used to select/edit whatever light or keyframe marker is
    # under the cursor. Both live on the left button and are disambiguated
    # by movement distance -- NOT by a separate button -- because the right
    # button is unusable for this: on most OpenCV backends (GTK in
    # particular) a right-click first pops up a built-in context menu,
    # which swallows the button-down event before this callback ever sees a
    # clean down/up pair, and can leave OpenCV's own mouse state stuck
    # thinking the button is still held (the "camera keeps following the
    # mouse until I restart the script" symptom). Left-click has no such
    # built-in menu, so drag-detection via cv2's per-move `flags` bitmask
    # (EVENT_FLAG_LBUTTON) is reliable instead of tracking down/up state by
    # hand. A plain dict (not a class) so the callback closure below can
    # mutate it without a `nonlocal` per field -- cv2's callback only gets
    # (event, x, y, flags, userdata), so this dict IS userdata.
    _MOUSE_CLICK_SLOP = 4  # px -- movement within this radius still counts as a click, not a drag
    mouse_state = {'down': False, 'down_x': 0, 'down_y': 0, 'last_x': 0, 'last_y': 0,
                   'accum_dx': 0, 'accum_dy': 0, 'moved_far': False, 'click': None}

    def _mouse_callback(event, x, y, flags, ms):
        if event == cv2.EVENT_LBUTTONDOWN:
            ms['down'] = True
            ms['down_x'], ms['down_y'] = x, y
            ms['last_x'], ms['last_y'] = x, y
            ms['moved_far'] = False
        elif event == cv2.EVENT_MOUSEMOVE and ms['down'] and (flags & cv2.EVENT_FLAG_LBUTTON):
            if not ms['moved_far'] and math.hypot(x - ms['down_x'], y - ms['down_y']) > _MOUSE_CLICK_SLOP:
                ms['moved_far'] = True
            if ms['moved_far']:
                ms['accum_dx'] += x - ms['last_x']
                ms['accum_dy'] += y - ms['last_y']
            ms['last_x'], ms['last_y'] = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            if ms['down'] and not ms['moved_far']:
                ms['click'] = (x, y)
            ms['down'] = False
            ms['moved_far'] = False

    cv2.setMouseCallback(WINDOW_NAME, _mouse_callback, mouse_state)
    mouse_sens = 0.0028  # radians per pixel of drag -- similar feel to look_speed's key-repeat rate

    camera_pos = tracer.camera_pos.astype(np.float64)
    camera_rot = tracer.camera_rot.astype(np.float64)
    camera_roll = float(tracer.camera_roll)  # radians -- 0 unless --camera-data supplied one at t=0
    roll_speed = 0.015   # radians per "held" tick (,/. keys) -- matches look_speed
    move_speed = 0.6
    look_speed = 0.015   # radians per "held" tick (arrow keys) -- also used as a base for mouse-drag
    lines_mode = True
    live_render = False
    selected_light = 0          # index into the COMBINED list [lights..., spotlights...]
    autofocus_flash = 0.0        # countdown (seconds) for the focus-ring flash effect

    fps_smooth = 0.0
    last_frame_time = time.time()
    _last_zoom_tick = [time.time()]  # mutable cell for the ;/' zoom ramp below (see there)
    _star_clock_start = time.time()
    _next_star_update = 0.0  # wall-clock time.time() timestamp -- see the star-twinkle throttle below

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

    # Mutable "camera/renderer" settings the F6 menu (see _build_camera_params)
    # can adjust -- these used to be hardcoded literals/module constants at
    # their one call site each (the '9'/'0' key handlers below), which meant
    # there was nowhere to change them short of editing the source. Promoted
    # to plain locals here instead.
    final_samples = 32                        # samples for the '9' (final render) key
    video_samples = VIDEO_SAMPLES_PER_FRAME    # samples/frame for the '0' (render video) key

    def _set_live_res(w=None, h=None):
        # LIVE_RENDER_RES is a module global (see set_resolutions()) --
        # `global` here refers to the MODULE namespace regardless of this
        # being a nested function, which is exactly what's needed since
        # every other read of LIVE_RENDER_RES in main() (the live<->final
        # switches below) looks it up fresh each time, not a cached copy.
        global LIVE_RENDER_RES
        w = int(w) if w is not None else LIVE_RENDER_RES[0]
        h = int(h) if h is not None else LIVE_RENDER_RES[1]
        LIVE_RENDER_RES = (max(16, w), max(16, h))
        if live_render:
            tracer.set_resolution(*LIVE_RENDER_RES)

    def _set_final_res(w=None, h=None):
        global FINAL_RENDER_RES
        w = int(w) if w is not None else FINAL_RENDER_RES[0]
        h = int(h) if h is not None else FINAL_RENDER_RES[1]
        FINAL_RENDER_RES = (max(16, w), max(16, h))

    def _set_final_samples(v):
        nonlocal final_samples   # a plain local of main(), not a module global -- see above
        final_samples = max(1, int(v))

    def _set_video_samples(v):
        nonlocal video_samples
        video_samples = max(1, int(v))

    # Bundled once here rather than rebuilt per-menu-open -- _build_camera_params
    # takes these as plain getter/setter closures since LIVE_RENDER_RES/
    # FINAL_RENDER_RES/final_samples/video_samples don't live on tracer,
    # post_fx, or camera_path the way everything else in the menus does.
    res_getset = {
        'live_w_get': lambda: LIVE_RENDER_RES[0], 'live_w_set': lambda v: _set_live_res(w=v),
        'live_h_get': lambda: LIVE_RENDER_RES[1], 'live_h_set': lambda v: _set_live_res(h=v),
        'final_w_get': lambda: FINAL_RENDER_RES[0], 'final_w_set': lambda v: _set_final_res(w=v),
        'final_h_get': lambda: FINAL_RENDER_RES[1], 'final_h_set': lambda v: _set_final_res(h=v),
    }
    sample_getset = {
        'live_get': lambda: prog.samples_per_frame,
        'live_set': lambda v: setattr(prog, 'samples_per_frame', max(1, int(v))),
        'final_get': lambda: final_samples, 'final_set': _set_final_samples,
        'video_get': lambda: video_samples, 'video_set': _set_video_samples,
    }

    print("Controls: WASD move (follows facing/yaw, stays level -- pitch doesn't tilt it) | "
          "arrow keys OR left-drag mouse to look | Q/E move down/up | L toggle live raytrace")
    print("Tab: cycle selected light/spotlight | K: place selected light/spot at the camera (+direction)")
    print("Left-click a light/spotlight or keyframe marker to open its properties menu directly")
    print("  -- in a spotlight's menu: A) configure a looping/keyframed animation, Z) clear it")
    print("[ / ]: narrow / widen a spotlight's cone (when a spotlight is selected)")
    print("T: autofocus DoF on whatever is under the crosshair | - / =: nudge DoF focus distance")
    print("1: toggle wireframe outline | C: toggle caustics")
    print("U: DoF | R: chromatic aberration | N: lens flare | V: VHS | M: motion blur (video only)")
    print("F: fisheye lens | B: bloom | Y: eye adaptation/auto exposure | G: continuous autofocus (video only)")
    print("3: fog | 4: god rays | 5: analog camera (Y2K camcorder: LDR/blowout/grain/smear)")
    print("H: cycle handheld camera shake (keyframe paths only)")
    print("2: type in an exact camera position/angle | `: open the post-FX properties menu")
    print("F5: world/atmosphere settings (fog, clouds, sky, sun, stars) | "
          "F6: camera/renderer settings (exposure, zoom, DoF, resolution, samples)")
    print("9: render final image | 8: quick render (lower quality) | 0: render video")
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
              "O or F10: edit/delete the targeted keyframe (nearest to crosshair)")
    print("I: replay the camera path (no render)")
    print("; / ' : zoom out / in (optical zoom -- narrows/widens FOV, see --zoom-speed)")
    print("X: export the current scene/camera/lights/etc to a .json file | Esc: quit")

    running = True
    while running:
        keys = inp  # alias for readability below

        any_input = False

        if replaying:
            # Not consuming drag-look while replaying (movement is
            # driven by the path itself here) -- but still DRAIN it, or it'd
            # jump the view the instant replay ends and this starts being
            # read again.
            mouse_state['accum_dx'] = 0
            mouse_state['accum_dy'] = 0
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

            # WASD follows the camera's viewing direction (yaw) but is
            # locked to the horizontal (X/Z) plane -- pitch (looking up/
            # down) does NOT tilt movement, so W/S/A/D never change
            # camera_pos[1] (Y) themselves. Built from yaw alone (not the
            # full R, which would include pitch/roll) so forward/right stay
            # unit-length and perfectly level regardless of look angle.
            # Q/E remain the only keys that move along Y.
            yaw = camera_rot[0]
            cy, sy = math.cos(yaw), math.sin(yaw)
            forward = np.array([sy, 0.0, cy], dtype=np.float32)
            right = np.array([cy, 0.0, -sy], dtype=np.float32)

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
            # Left-drag mouselook -- same yaw/pitch sign convention as the
            # arrow keys above (dx>0 = mouse moved right = turn right =
            # yaw increases; dy>0 = mouse moved down = look down = pitch
            # increases), so it feels consistent whichever you use.
            if mouse_state['accum_dx'] or mouse_state['accum_dy']:
                d_yaw += mouse_state['accum_dx'] * mouse_sens
                d_pitch += mouse_state['accum_dy'] * mouse_sens
                mouse_state['accum_dx'] = 0
                mouse_state['accum_dy'] = 0
                any_input = True
            if d_yaw or d_pitch:
                camera_rot[0] += d_yaw
                camera_rot[1] += d_pitch
                camera_rot[1] = max(-1.5, min(1.5, camera_rot[1]))

            d_roll = 0.0
            if keys.is_held(','): d_roll -= roll_speed; any_input = True
            if keys.is_held('.'): d_roll += roll_speed; any_input = True
            if d_roll:
                camera_roll += d_roll

        # --- Optical zoom (;/' keys) -- narrows/widens the camera's FOV
        # like a real zoom lens (see RayTracer.set_zoom). Held keys nudge
        # the TARGET zoom (same "held tick" pattern as WASD/roll above);
        # the ACTUAL zoom then eases toward that target at tracer.zoom_speed
        # (zoom-multiplier/second, adjustable in the post-fx menu /
        # --zoom-speed) using this loop's own inter-frame time, so it racks
        # smoothly like a real lens instead of snapping -- render_video
        # does the equivalent per-frame ramp for recorded video. Runs
        # whether or not a replay is active, so you can zoom while
        # previewing a path too. No focal-shift "auto-focus hunt" blur here
        # -- that effect is video-only (see apply_zoom_focal_shift).
        zoom_key_step = 0.05
        if keys.is_held(';'):
            tracer.zoom_target = max(tracer.zoom_min, tracer.zoom_target - zoom_key_step)
        if keys.is_held("'"):
            tracer.zoom_target = min(tracer.zoom_max, tracer.zoom_target + zoom_key_step)
        _zoom_now = time.time()
        _zoom_dt = min(0.25, max(0.0, _zoom_now - _last_zoom_tick[0]))
        _last_zoom_tick[0] = _zoom_now
        _prev_zoom = tracer.zoom
        _zoom_max_delta = max(1e-4, tracer.zoom_speed) * _zoom_dt
        if abs(tracer.zoom_target - tracer.zoom) > _zoom_max_delta:
            tracer.zoom += _zoom_max_delta if tracer.zoom_target > tracer.zoom else -_zoom_max_delta
        else:
            tracer.zoom = tracer.zoom_target
        tracer.zoom = max(tracer.zoom_min, min(tracer.zoom_max, tracer.zoom))
        if abs(tracer.zoom - _prev_zoom) > 1e-6:
            tracer.fov_deg = tracer.base_fov_deg / tracer.zoom
            tracer.fov_rad = math.radians(tracer.fov_deg)
            tracer.half_tan = math.tan(tracer.fov_rad / 2)
            any_input = True

        R = camera_matrix(*camera_rot, camera_roll)

        n_pt_lights = len(tracer.lights)
        n_all_lights = n_pt_lights + len(tracer.spotlights)
        targeted_kf_idx = find_targeted_keyframe(camera_path, camera_pos, R, tracer.half_tan,
                                                  tracer.aspect, WIN_W, WIN_H)

        def _open_keyframe_menu(kf_idx):
            kf = camera_path.keyframes[kf_idx]
            action, value = keyframe_options_menu_cv(canvas, WIN_W, WIN_H, kf, kf_idx)
            if action == 'speed_delta':
                kf.speed = max(0.01, kf.speed + value)
            elif action == 'set_speed':
                kf.speed = value
            elif action == 'set_zoom':
                kf.zoom = value
            elif action == 'delete':
                camera_path.remove(kf_idx)
                print(f"Keyframe #{kf_idx + 1} deleted.")

        # --- Mouse click: select + open a properties menu for whatever
        # light/spotlight or keyframe marker is under the cursor (lights take
        # priority when both are close, since they're usually the smaller/
        # more precise target). Equivalent to Tab-selecting + K, or aiming +
        # O, just without needing the crosshair centered on it first.
        if mouse_state['click'] is not None:
            mx, my = mouse_state['click']
            mouse_state['click'] = None
            hit_light = find_targeted_light(tracer.lights, tracer.spotlights, camera_pos, R,
                                             tracer.half_tan, tracer.aspect, WIN_W, WIN_H,
                                             aim_x=mx, aim_y=my)
            if hit_light is not None:
                selected_light = hit_light
                is_spot = hit_light >= n_pt_lights
                lt = (tracer.spotlights[hit_light - n_pt_lights] if is_spot
                      else tracer.lights[hit_light])
                if light_options_menu_cv(canvas, WIN_W, WIN_H, lt, hit_light, camera_pos,
                                          camera_rot, camera_roll, is_spot=is_spot):
                    tracer.sync_lights()
                    tracer.compute_caustics()
            elif not has_camera_data:
                hit_kf = find_targeted_keyframe(camera_path, camera_pos, R, tracer.half_tan,
                                                 tracer.aspect, WIN_W, WIN_H, aim_x=mx, aim_y=my)
                if hit_kf is not None:
                    _open_keyframe_menu(hit_kf)

        raw = cv2.waitKeyEx(1)
        if args.print_keys and raw != -1:
            print(f"[--print-keys] raw={raw}")
        kind, val = classify_key(raw)
        if kind == 'arrow':
            inp.note_held('arrow_' + val)
            any_input = True
        elif kind == 'char':
            ch = val
            if ch in ('w', 'a', 's', 'd', 'q', 'e', ',', '.', ';', "'"):
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
            elif ch == 't' and inp.one_shot('autofocus'):
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
            elif ch == 'u' and inp.one_shot('dof_toggle'):
                post_fx['dof_enabled'] = not post_fx['dof_enabled']
                print(f"DoF: {post_fx['dof_enabled']}")
            elif ch == 'r' and inp.one_shot('chroma_toggle'):
                post_fx['chroma_enabled'] = not post_fx['chroma_enabled']
                print(f"Chromatic aberration: {post_fx['chroma_enabled']}")
            elif ch == 'n' and inp.one_shot('flare_toggle'):
                post_fx['flare_enabled'] = not post_fx['flare_enabled']
                print(f"Flare: {post_fx['flare_enabled']}")
            elif ch == 'v' and inp.one_shot('vhs_toggle'):
                post_fx['vhs_enabled'] = not post_fx['vhs_enabled']
                print(f"VHS: {post_fx['vhs_enabled']}")
            elif ch == 'm' and inp.one_shot('mblur_toggle'):
                post_fx['motion_blur_enabled'] = not post_fx['motion_blur_enabled']
                print(f"Motion blur (video only): {post_fx['motion_blur_enabled']}")
            elif ch == 'f' and inp.one_shot('fisheye_toggle'):
                post_fx['fisheye_enabled'] = not post_fx['fisheye_enabled']
                print(f"Fisheye: {post_fx['fisheye_enabled']}")
            elif ch == 'b' and inp.one_shot('bloom_toggle'):
                post_fx['bloom_enabled'] = not post_fx['bloom_enabled']
                print(f"Bloom: {post_fx['bloom_enabled']}")
            elif ch == '3' and inp.one_shot('fog_toggle'):
                post_fx['fog_enabled'] = not post_fx['fog_enabled']
                print(f"Fog: {post_fx['fog_enabled']}")
            elif ch == '4' and inp.one_shot('godrays_toggle'):
                post_fx['godrays_enabled'] = not post_fx['godrays_enabled']
                print(f"God rays: {post_fx['godrays_enabled']}")
            elif ch == '5' and inp.one_shot('analog_toggle'):
                post_fx['analog_enabled'] = not post_fx['analog_enabled']
                print(f"Analog camera: {post_fx['analog_enabled']}")
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
            elif ch == '2' and inp.one_shot('manual_camera', debounce=0.3):
                new_pos, new_yaw, new_pitch, new_roll = prompt_manual_camera(
                    camera_pos, camera_rot, camera_roll)
                camera_pos[:] = new_pos
                camera_rot[0], camera_rot[1] = new_yaw, new_pitch
                camera_roll = new_roll
            elif ch == '`' and inp.one_shot('postfx_menu', debounce=0.3):
                params = _build_postfx_params(tracer, post_fx, camera_path, has_camera_data)
                postfx_menu_cv(canvas, WIN_W, WIN_H, params)
            elif ch == 'p' and inp.one_shot('add_keyframe'):
                if has_camera_data:
                    print("Camera keyframes are disabled while --camera-data is active.")
                else:
                    camera_path.add(camera_pos.copy(), camera_rot[0], camera_rot[1], speed=5.0,
                                     zoom=tracer.zoom)
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
                    _open_keyframe_menu(targeted_kf_idx)
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
            elif ch == '0' and inp.one_shot('render_video', debounce=1.0):
                if is_live_stream:
                    print("Live camera stream has no fixed length -- can't render a video from it "
                          "(record it to a --camera-data file first, or use camera keyframes).")
                elif has_camera_data or len(camera_path.keyframes) >= 1:
                    tracer.render_video(
                        active_path, out_path="raytrace_v12_video.mp4", post_fx=post_fx,
                        fps=video_fps, samples_per_frame=video_samples, camera_sync=args.camera_sync,
                        progress_cb=lambda d, t, e, eta: draw_render_progress(
                            canvas, WIN_W, WIN_H, d, t, e, eta, "VIDEO"))
                    tracer.camera_roll = 0.0
                    tracer.set_resolution(*LIVE_RENDER_RES)
                else:
                    print("No camera keyframes -- press P to add at least one before rendering a video.")
            elif ch == '9' and inp.one_shot('final_render', debounce=1.0):
                was_live = live_render
                live_render = False
                tracer.set_resolution(*FINAL_RENDER_RES)
                tracer.render_to_file(
                    "raytrace_v12.png", samples=final_samples, post_fx=post_fx,
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
                    "raytrace_v12_quick.png", samples=4, post_fx=post_fx,
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
        elif kind == 'func':
            # F10: per-keyframe options menu -- an alternate binding for the
            # same action as O (see _open_keyframe_menu above), just reachable
            # without needing a letter key free of other bindings.
            if val == 'f10' and inp.one_shot('kf_menu'):
                if has_camera_data:
                    print("Camera keyframes are disabled while --camera-data is active.")
                elif targeted_kf_idx is not None:
                    _open_keyframe_menu(targeted_kf_idx)
            elif val == 'f5' and inp.one_shot('world_menu', debounce=0.3):
                # World/atmosphere/lighting settings (fog, clouds, sky, sun,
                # stars, caustics, god rays) -- see _build_world_params.
                params = _build_world_params(tracer, post_fx, camera_path, has_camera_data)
                postfx_menu_cv(canvas, WIN_W, WIN_H, params)
            elif val == 'f6' and inp.one_shot('camera_menu', debounce=0.3):
                # Camera/renderer settings (exposure, zoom, DoF, camera
                # shake, resolution, samples-per-mode) -- see
                # _build_camera_params.
                params = _build_camera_params(tracer, post_fx, camera_path, has_camera_data,
                                              res_getset, sample_getset)
                postfx_menu_cv(canvas, WIN_W, WIN_H, params)

        if autofocus_flash > 0.0:
            autofocus_flash = max(0.0, autofocus_flash - 1.0 / 60.0)

        now_t = time.time()
        frame_dt = now_t - last_frame_time
        last_frame_time = now_t
        if frame_dt > 0.0:
            inst_fps = 1.0 / frame_dt
            fps_smooth = inst_fps if fps_smooth <= 0.0 else fps_smooth * 0.9 + inst_fps * 0.1

        # Advance any spotlight animations (orbit/ping-pong/sweep/keyframes,
        # see SpotLightAnimation) by this frame's wall-clock delta. Cheap to
        # call unconditionally -- it's a no-op for any spotlight that
        # doesn't have one attached.
        if tracer.spotlights and tracer.update_spotlight_animations(frame_dt):
            any_input = True

        # Star twinkle: throttled to roughly once a second (see
        # RayTracer.update_stars' docstring for why) rather than every
        # frame -- "occasionally shift slightly", not a strobe.
        if tracer.background.stars is not None and now_t >= _next_star_update:
            tracer.update_stars(now_t - _star_clock_start)
            _next_star_update = now_t + 1.0
            any_input = True

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
        # Small orientation compass, placed just under the pos/yaw-pitch-roll
        # text (2 lines) in the top-left corner.
        draw_orientation_gizmo(canvas, 12 + 45, 12 + 2 * _HUD_LINE_H + 45, R, radius=28)

        sel_kind = "spotlight" if selected_light >= n_pt_lights else "light"
        sel_num = (selected_light - n_pt_lights + 1) if selected_light >= n_pt_lights else (selected_light + 1)
        tl_lines = [
            f"pos: {camera_pos[0]:.1f}, {camera_pos[1]:.1f}, {camera_pos[2]:.1f}",
            f"yaw/pitch/roll: {math.degrees(camera_rot[0]):.0f} / {math.degrees(camera_rot[1]):.0f} "
            f"/ {math.degrees(camera_roll):.0f}",
            f"zoom: {tracer.zoom:.2f}x" + (f" -> {tracer.zoom_target:.2f}x"
                                            if abs(tracer.zoom_target - tracer.zoom) > 1e-3 else ""),
        ]
        tr_lines = [
            f"live raytrace: {'ON' if live_render else 'off'}",
            f"FPS: {fps_smooth:.0f}",
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
            f"VHS {'on' if post_fx['vhs_enabled'] else 'off'}  flare {'on' if post_fx['flare_enabled'] else 'off'}  "
            f"caustics {'on' if tracer.caustics_enabled else 'off'}",
            f"fisheye {'on' if post_fx.get('fisheye_enabled', False) else 'off'}  "
            f"bloom {'on' if post_fx.get('bloom_enabled', False) else 'off'}  "
            f"exposure {tracer.exposure:.2f}{'(auto)' if tracer.eye_adapt_enabled else ''}",
            f"fog {'on' if post_fx.get('fog_enabled', False) else 'off'}  "
            f"god rays {'on' if post_fx.get('godrays_enabled', False) else 'off'}  "
            f"analog {'on' if post_fx.get('analog_enabled', False) else 'off'}",
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