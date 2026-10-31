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
   - F9 opens a light LIST menu (light_list_menu_cv), the light-source
     counterpart of F10's keyframe list: Up/Down picks any point
     light/spotlight, Enter opens its existing properties menu.
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

9) CAMERA ORIENTATION IS NOW QUATERNION-BASED. A unit quaternion
   [w, x, y, z] is the canonical representation throughout the core
   camera logic -- storage (RayTracer._camera_quat), composition
   (camera_matrix/rotation_matrix both build their matrix from one),
   interpolation (CameraPath and SensorCameraData both slerp instead of
   lerping yaw/pitch/roll independently), and inertia smoothing (a real
   rotational spring in quaternion space, _quat_spring_step). Euler
   angles survive only as a human-readable format at the edges: HUD
   readouts, the F5/F6/` menus, the manual-camera prompt, scene-file
   fields, and the interactive arrow-key look (which is literally a
   yaw/pitch delta, so it stays Euler and is converted once per frame at
   that boundary). See the comment block above camera_matrix.
   - Backward compatible: scene JSONs from older versions carry only
     yaw/pitch, and are converted to a quaternion on load. Scenes saved
     by this version write yaw/pitch/roll (readable) AND an exact 'quat'
     field, preferring 'quat' on load when present.
   - The new quaternion code reproduces the old trig matrices to ~1e-15,
     so no existing scene renders differently because of this change.

10) --camera-data now reads a plain-text quaternion track (e.g.
   'Map_Race_#01.txt') as well as the original sensor_record JSON --
   formats are auto-detected. Text format: one ':'-separated sample per
   line, `t:pos_x:pos_y:pos_z:qx:qy:qz:qw`, decimal comma or dot, with
   an optional trailing 'END:<n>' line. Its orientation is already a
   quaternion, so it needs no Euler conversion on the way in.
   - --camera-offset now takes 6 values as well as 3: 'x,y,z' (position
     only, as before) or 'x,y,z,r,p,y' / 'x,y,z/r,p,y' to add a rotation
     offset in degrees. The new --camera-rot-offset 'r,p,y' flag is the
     preferred way to give the rotation half, and wins if both appear.
   - --inertia/--inertia-bounce now apply to --camera-data replay, not
     just to keyframe paths (previously they were set on a camera_path
     that --camera-data then discarded, so they silently did nothing).
     SensorCameraData runs the same position spring plus the quaternion
     rotational spring over the recording, smoothing its own jitter and
     abrupt direction changes; 0 keeps the exact raw-data behaviour.
     Also editable live from the F6 camera menu.

11) SUN AND STARS ARE TOGGLEABLE, and the star twinkle cadence is
   configurable. Background.sun_enabled/stars_enabled hide either layer
   while KEEPING all of its settings (angle, color, size, density, seed),
   so it can be switched straight back on instead of being re-described
   -- unlike set_sun(intensity=0), which overwrites the stored intensity.
   RayTracer.set_sun_enabled/toggle_sun also mute/unmute the real distant
   sun Light (remembering its brightness), so the toggle covers the disc
   AND the light it casts rather than leaving an invisible sun still
   throwing shadows. The interactive view's star-twinkle re-bake, which
   was hard-coded to once a second, is now
   RayTracer.star_update_interval (0 = frozen field, and cheapest).
   New flags: --no-sun, --no-stars, --star-update-interval.
   In the F5 world menu, sun/moon and stars are now their own top-level
   trees ("[Sun/moon]", "[Stars]"), each led by an Enabled row, instead
   of the star rows hanging off "Sky light" as though they belonged to
   it.

11b) VIDEO PLANES + EMISSIVE (SELF-ILLUMINATED) MATERIALS.
   Scene.add_video(center, (w, h|None), "clip.mp4") adds a plane that plays a
   looping video (decoded with cv2, audio ignored; h=None takes the video's
   aspect ratio). All video planes share one clock: SPACE pauses/plays them in
   the live view (the F5 menu has the same toggle); --headless --video plays
   them automatically from t=0 along the output timeline (also with parallel
   frame blocks and motion blur -- playback is a pure function of time);
   stills show --video-time T. --video-start-paused opens the live view paused.
   Video frames are resized to TextureLibrary.TEX_SIZE (--texture-size) squared.
   emissive=/emissive_color=/unlit= on add_box/add_image/add_video: emissive is a
   glow strength (0 = off), emissive_color overrides the glow colour (default =
   the surface's own colour/texture/video frame), unlit ignores scene lighting.
   Emissive faces also act as AREA LIGHTS (one power-weighted light sample +
   shadow ray per shaded point, only paid when the scene has emitters), so a
   screen tints the room with its picture. --no-emissive-light, or
   tracer.emissive_lighting_enabled / emissive_light_strength (F5 menu), keep
   the glow but drop the lighting. Scene export embeds video files.

12) SCENE EXPORT (X key / --export-scene) IS COMPLETE. save_scene_file
   additionally writes the volumetric cloud layer, the full optical-zoom
   state (base_fov_deg/zoom/zoom_target/zoom_speed/zoom_min/zoom_max),
   star_update_interval, the sun/stars enable flags, and a
   sun_light_index identifying which entry in lights[] is the sun's own
   distant Light, so a reload can re-point tracer.sun_light and keep
   set_sun_angle/set_sun_enabled working on it. Every field added here
   defaults on load to the renderer's previous behaviour, so older scene
   files still load identically.
   - Fixed alongside this: main() called tracer.add_sun()/add_stars()
     UNCONDITIONALLY, so loading a scene overwrote its sun/star field
     with the defaults and appended a SECOND sun Light on every reload.
     Those defaults now only run for a freshly-built scene.

13) GIMBAL-LOCK FIX in euler_from_quat. The standard decomposition reads
   yaw from R[0,2]/R[2,2] and roll from R[1,0]/R[1,1], but all four of
   those elements carry a cos(pitch) factor, so at pitch = +-90deg they
   are all zero and atan2(0, 0) quietly returns 0 -- reporting
   yaw = roll = 0 for a camera that is actually rotated, and recomposing
   to the WRONG rotation (off by up to 90deg). Recordings do hit this
   exactly: 'Map_Race_#01.txt' passes through pitch = 90deg at t ~= 7.4s.
   Now yaw and roll are folded together into yaw (roll = 0) there, which
   round-trips exactly. This affected the HUD readout, the readable
   yaw/pitch/roll in exported scenes, and free-look resuming after a
   replay ended at such a pose.
   - Note on that file specifically: its one visible "snap" is NOT gimbal
     lock and not a renderer bug. It stores each quaternion component to
     ONE decimal place (position gets three), i.e. ~5.7deg of orientation
     resolution, and the rounding produces a genuine 42.8deg step between
     two consecutive samples at t ~= 16.2s. No interpolation can recover
     that -- slerp faithfully reproduces what it is given. SensorCameraData
     .load now detects coarse rotation precision and prints the resolution,
     the worst step and its timestamp, suggesting --inertia; measured at
     that spot, per-frame motion drops from 7.1deg to 3.2deg at
     --inertia 0.3 and 1.8deg at 0.6. Re-exporting the track with more
     decimals is the real fix.

14) FIXES (clouds + alpha cutouts):
   - Clouds no longer collapse into one flat sheet. Cause: _cloud_march
     used a uniform step of (slab ray length)/steps, and that length grows
     like 1/sin(elevation), so near the horizon each step spanned several
     noise features and averaged them away. Steps are now graded
     (dense near the camera, coarser far away), the density field gets
     height-dependent coverage (rounded domes on flat bases), a steeper
     edge remap with edge-weighted erosion, height-based ambient shading,
     and a little distance haze. No new parameters; scene files unchanged.
     Shadows use the same _cloud_density, so ground shadows follow the
     new shapes.
   - Glass/water behind an alpha=0 region of an RGBA image is no longer
     paler. Cause: passing through a cutout pixel moved ray_o to the sprite
     plane, so the next hit's `t` (used for Beer-Lambert absorption) only
     measured from that plane. The skipped distance is now carried
     (carry_t / seg_carry) in render_sample and _shadow_throughput. This
     also corrects the depth used for DoF and the cloud composite tmax for
     rays that pass through a cutout first.

15) POST-FX NOW RUN ON THE GPU (section 13b). The whole post-processing
   chain -- DoF, zoom focus-hunt blur, fisheye, fog, god rays, bloom, lens
   flare, the analog-camera look (range squash, blowout, halation,
   split-tone, grain, vignette, motion smear), chromatic aberration, VHS,
   the final clip/8-bit/BGR conversion and the live view's resize to window
   size -- is now a chain of Taichi kernels on right-sized GPU arrays, fed
   straight from the resolved OUTPUT/DEPTH_OUT fields. Only the finished,
   packed 8-bit frame is read back (previously: the whole float32
   MAX_W x MAX_H colour + depth fields, every frame, then ~dozens of
   full-frame CPU passes). Eye-adaptation metering (a full ACCUM readback
   per frame) is a GPU reduction too. The NumPy chain is unchanged and is
   both the reference the kernels were checked against and an automatic
   fallback (a warning is printed and the CPU chain takes over if a GPU
   kernel ever fails on some backend). --postfx-backend cpu (or
   RT_POSTFX_BACKEND=cpu) forces the old CPU chain.
   - render_video no longer needs the PostProcessPipeline thread on the GPU
     path: post-fx is a few asynchronous kernel launches on the same
     stream as the raytrace, and the frame goes straight to the (already
     threaded) ffmpeg writer. The thread is kept for the CPU fallback.
   - VHS / analog-grain noise is hash-generated on the GPU, so it is
     deterministic per frame but not the same random stream as before
     (see the notes at the top of section 13b).

16) MINECRAFT SKINS + POSEABLE CHARACTER. Scene.add_minecraft_character(skin_png,
   position=, rotation=, scale=, pose={joint: (yaw, pitch, roll)}, slim=...) adds a
   player model built from six textured boxes (head, body, 2 arms, 2 legs), each with
   its BASE layer and its inflated OVERLAY layer (hat/jacket/sleeves/pants). Both layers
   of a part go through the very same joint transform, so they stay aligned under any
   pose; joints rotate about the game's own pivots (body about the waist, so a bent
   torso stays attached to the legs) and are parented body -> head/arms. Skins can be
   64x64, legacy 64x32 (left limbs are mirrored from the right, as in the game) or HD
   (128x128 ...); slim (Alex) arms are auto-detected. Skin textures are scaled with
   NEAREST (TextureLibrary.get_skin_id) so the pixel art stays crisp, the base layer is
   forced opaque, and overlay alpha becomes see-through cut-outs. The pose is applied
   while the geometry is built -- the BVH is static once the RayTracer exists -- so to
   change a pose, build a new scene (see build_skin_demo_scene for a worked example).
   Scene export/import round-trips the character (skin embedded as an asset).
   CLI: --skin FILE [--skin-pos x,y,z] [--skin-yaw DEG] [--skin-pose 'head=20,10,0;
   right_arm=0,0,-150'] [--skin-scale S] [--skin-model auto|classic|slim]
   [--no-skin-overlay] [--skin-demo].

17) RENDER HUD, CRT EFFECT, OPTICAL LENS FLARE.
   - HUD (section 13c): text elements that are part of the render (live view, stills,
     video). F7 opens the HUD menu: [HUD] on/off with sub-options "Affected by FX" and
     "Camera lag (s)", plus a separate "HUD text" branch listing every element (text, size,
     X/Y in % of the frame, anchor, colour, opacity; add/delete rows). Default HUD: timestamp
     + zoom (top-left), a "+" crosshair (centre), "Noir" (bottom-right), all white.
     Text variables: {hh} {mm} {ss} (video/live timestamp), {rh} {rm} {rs} (wall clock; in
     video it starts at the render's start time and runs at video speed), {t} (seconds),
     {zoom}/{z}, {w} {h} {width} {height}, {dist} (distance under the crosshair, 9999 if
     far/sky) + extras {frame} {fps} {fov} {yaw} {pitch} {roll} {px} {py} {pz}. Python format
     specs work ({zoom:.1f}); "\\n" is a line break.
     Font: --hud-font PATH, else ./font.ttf, else a monospace system font, else any font.
     --no-hud turns it off for a run. Everything is stored in post_fx (hud_*), so scene
     export/import carries it. Size is in points at a 360-px-tall frame and scales with the
     frame, so live/still/video look the same.
   - CRT (key 6, ` menu): curved glass, scanlines (beam widens with brightness), RGB phosphor
     stripes, glow, convergence error, vignette, flicker + rolling bar. GPU kernel _gp_crt,
     NumPy fallback apply_crt.
   - Lens flare rebuilt as an optical model: long-tail glow, aperture-blade starburst,
     aperture-shaped ghosts along the lens axis with coating tints and dispersion,
     dispersed halo ring, smooth fade as the light leaves the frame. New settings:
     flare_ghosts, flare_starburst, flare_blades.
"""

import argparse
import base64
import copy
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
from PIL import Image, ImageDraw, ImageFont

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
#
#    QUATERNIONS: as of v12's camera-data/inertia rework, a unit quaternion
#    [w, x, y, z] (Hamilton, np.float64) is the CANONICAL representation of
#    camera (and camera-path/camera-data) orientation everywhere in the
#    core logic below -- composition (camera_matrix/rotation_matrix),
#    interpolation (CameraPath/SensorCameraData sampling), and inertia
#    smoothing (the rotational spring in _quat_spring_step) all operate on
#    quaternions directly, never on yaw/pitch/roll. Euler angles are kept
#    ONLY as a human-readable in/out format at the edges (HUD text, menus,
#    scene-file readability, older scene files, keyboard input deltas) and
#    are converted to/from a quaternion right at that boundary -- see
#    RayTracer.camera_rot/camera_roll (properties backed by camera_quat)
#    and euler_from_quat/quat_from_euler below. This avoids gimbal lock in
#    interpolation/smoothing and means a single slerp (not three separate,
#    independently-unwrapped angle lerps) drives all path/replay blending.
# =============================================================================

def quat_identity():
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else quat_identity()


def quat_from_axis_angle(axis, angle):
    axis = normalize(np.asarray(axis, dtype=np.float64))
    h = angle * 0.5
    s = math.sin(h)
    return np.array([math.cos(h), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


def quat_multiply(q1, q2):
    """Hamilton product q1*q2: applying the result to a vector matches
    applying q2's rotation first, then q1's -- same composition order as
    matrix multiplication R1 @ R2 (so q_total = qy*qx*qz mirrors
    R = Ry*Rx*Rz below)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def quat_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)


def quat_from_euler(yaw, pitch, roll=0.0):
    """Quaternion equivalent of camera_matrix's R = Ry(yaw)*Rx(pitch)*
    Rz(roll) -- same axis convention/order, composed as quaternions."""
    qy = quat_from_axis_angle((0.0, 1.0, 0.0), yaw)
    qx = quat_from_axis_angle((1.0, 0.0, 0.0), pitch)
    qz = quat_from_axis_angle((0.0, 0.0, 1.0), roll)
    return quat_normalize(quat_multiply(quat_multiply(qy, qx), qz))


def quat_to_matrix(q):
    """3x3 rotation matrix for a unit quaternion, equal to camera_matrix's
    output when q == quat_from_euler(yaw, pitch, roll)."""
    w, x, y, z = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=float)


def euler_from_quat(q):
    """Inverse of quat_from_euler -- (yaw, pitch, roll) radians, purely for
    display/scene-file readability.

    Handles the gimbal-lock pose (pitch = +-90deg) explicitly. The usual
    formulas read yaw from R[0,2]/R[2,2] and roll from R[1,0]/R[1,1], but
    every one of those four elements carries a cos(pitch) factor, so at
    pitch = +-90 they are all zero and atan2(0, 0) silently returns 0 --
    which would report yaw = roll = 0 for a camera that is in fact
    rotated, and recompose to the WRONG rotation. Real --camera-data
    recordings do hit this exactly ('Map_Race_#01.txt' does, at t ~= 7.4s),
    so it is not a theoretical edge case. There, yaw and roll are not
    separately determined -- only (yaw - roll) at pitch = +90 and
    (yaw + roll) at pitch = -90 are -- so roll is pinned to 0 and the
    whole rotation is folded into yaw, which round-trips exactly through
    quat_from_euler."""
    R = quat_to_matrix(q)
    sin_pitch = max(-1.0, min(1.0, -R[1, 2]))
    pitch = math.asin(sin_pitch)
    if abs(sin_pitch) > 1.0 - 1e-9:
        # Gimbal lock: fold yaw and roll together into yaw, roll = 0.
        if sin_pitch > 0.0:   # pitch = +90: R[0,0]=cos(yaw-roll), R[0,1]=sin(yaw-roll)
            yaw = math.atan2(R[0, 1], R[0, 0])
        else:                 # pitch = -90: R[0,0]=cos(yaw+roll), R[0,1]=-sin(yaw+roll)
            yaw = math.atan2(-R[0, 1], R[0, 0])
        return yaw, pitch, 0.0
    yaw = math.atan2(R[0, 2], R[2, 2])
    roll = math.atan2(R[1, 0], R[1, 1])
    return yaw, pitch, roll


def quat_slerp(q0, q1, t):
    """Spherical linear interpolation, shortest arc. This is what lets
    CameraPath/SensorCameraData interpolate yaw+pitch(+roll) TOGETHER, in
    one operation, instead of three separately shortest-angle-unwrapped
    Euler lerps."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return quat_normalize(q0 + (q1 - q0) * t)
    theta0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta0 = math.sin(theta0)
    theta = theta0 * t
    s0 = math.cos(theta) - dot * math.sin(theta) / sin_theta0
    s1 = math.sin(theta) / sin_theta0
    return quat_normalize(q0 * s0 + q1 * s1)


def _quat_spring_step(quat, ang_vel, target_quat, k, c, dt):
    """One semi-implicit-Euler step of a rotational spring-damper chasing
    target_quat, done in quaternion/angular-velocity space (the rotational
    analogue of the plain position spring in _build_inertia_cache) --
    used by CameraPath and SensorCameraData's inertia smoothing so camera
    ROTATION gets real momentum/overshoot too, not just position, without
    ever falling back to per-axis Euler springs (which would need manual
    angle-wrap handling and can behave oddly near +-90deg pitch)."""
    q_target = quat_normalize(target_quat)
    q_cur = quat_normalize(quat)
    dq = quat_multiply(q_target, quat_conjugate(q_cur))
    if dq[0] < 0.0:
        dq = -dq  # shortest arc
    w, x, y, z = dq
    sin_half = math.sqrt(max(0.0, x * x + y * y + z * z))
    half_angle = math.atan2(sin_half, w)
    axis = (np.array([x, y, z], dtype=np.float64) / sin_half
            if sin_half > 1e-9 else np.zeros(3, dtype=np.float64))
    error_vec = axis * (2.0 * half_angle)  # world-space axis-angle error, radians

    ang_acc = k * error_vec - c * ang_vel
    ang_vel = ang_vel + ang_acc * dt
    delta = ang_vel * dt
    delta_mag = float(np.linalg.norm(delta))
    dq_step = (quat_from_axis_angle(delta / delta_mag, delta_mag)
               if delta_mag > 1e-12 else quat_identity())
    q_new = quat_normalize(quat_multiply(dq_step, q_cur))
    return q_new, ang_vel


def camera_matrix(yaw, pitch, roll=0.0):
    """R = Ry(yaw) * Rx(pitch) * Rz(roll). Built via quat_from_euler /
    quat_to_matrix -- quaternions are the canonical rotation
    representation in the core logic (see the block comment above); this
    function stays as the plain-float entry point every HUD/menu/keyframe
    call site already uses, and always agrees with anything computed
    straight from a quaternion since they share the same math underneath."""
    return quat_to_matrix(quat_from_euler(yaw, pitch, roll))


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
    (box/image) around its own center; shares camera_matrix's axis
    convention and, like it, is built via a quaternion under the hood."""
    return quat_to_matrix(quat_from_euler(yaw, pitch, roll))


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

PREVIEW_RES = (960, 540)         # window size + 2D painter's-algorithm preview
LIVE_RENDER_RES = (96, 54)       # interactive raytrace resolution (upscaled to window)
FINAL_RENDER_RES = (1920, 1080)  # final image (the '9' key), saved to file
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
CUTOUT_TUNNEL_BUDGET = 48  # extra loop iterations (beyond max_bounce) reserved purely for
                           # passing through alpha-cutout pixels -- see render_sample's
                           # real_bounce counter. Needs to comfortably cover the messiest
                           # case (a ray grazing along several overlapping RGBA sprite edges,
                           # e.g. wispy hair strands) without ever being mistaken for real
                           # reflect/refract bounces.
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
WATER_LOOP_FRAMES = 681          # water is a SEAMLESS LOOP of exactly this many video frames
                                 # (681 / VIDEO_FPS = ~22.72 s). Every water time-frequency
                                 # below is an integer multiple of W0, so frame N == frame 0.
                                 # Keep WATER_WAVE_SPEED = 1.0 and render N*681 frames.
W0 = 2.0 * math.pi * VIDEO_FPS / WATER_LOOP_FRAMES   # rad/s, base loop frequency
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

# --- GLASS CAUSTICS (photon pass for glass solids; see compute_glass_caustics) ---
MAX_GLASS_CLUSTERS = 8          # glass blocks that overlap are merged into one cluster; this caps
                                 # how many separate clusters get their own receiver map
GLASS_CAUSTIC_RES = 128         # receiver-map resolution per cluster (cells per side)
GLASS_PHOTON_STEPS = 12         # max glass interactions per photon (a cube needs 2; internal bounces add more)
GLASS_MAX_CLUSTER_RADIUS = 30.0 # bigger "glass" (e.g. a giant slab) is skipped: photons would be
                                 # spread too thin to resolve anything
GLASS_Y_TOL = 0.05              # a surface counts as the receiver plane if within this of its height

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
                 "ior", "reflection_k", "uv", "texture_id",
                 "emissive", "emissive_color", "unlit", "two_sided")

    def __init__(self, verts, color, roughness=0.0, transparency=0.0,
                 ior=1.5, reflection_k=0.0, uv=None, texture_id=-1,
                 emissive=0.0, emissive_color=None, unlit=False, two_sided=False):
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
        # Self-illumination. `emissive` is a radiance multiplier (0 = none, 1 = the
        # surface glows at its own colour's full brightness, >1 = overbright/bloomy).
        # `emissive_color` (0-255 rgb) overrides WHAT colour it glows; None means "glow
        # in the surface's own colour" (including its texture/video, per texel).
        # `unlit` makes the face ignore scene lighting entirely (a screen/lamp shade
        # that shows only its own emission); it still lights OTHER surfaces.
        self.emissive = max(0.0, float(emissive))
        self.emissive_color = (np.array(emissive_color[:3], dtype=float) / 255.0
                               if emissive_color is not None else np.array([-1.0, -1.0, -1.0]))
        self.unlit = bool(unlit)
        # Flat planes (images/videos) emit from both sides; box faces only outward, which
        # also stops light samples being wasted on a box's hidden back faces.
        self.two_sided = bool(two_sided)

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

class VideoSource:
    """A looping video decoded with cv2, giving random access by frame index.

    frame(idx) is cheap when idx is the frame right after the last one (plain
    sequential decode -- the normal playback case), skips a few frames with
    grab() when playback fell slightly behind, and seeks for anything else
    (offline renders that jump around, e.g. parallel frame blocks). Indices
    past the end wrap, so the video loops forever. Returns a uint8 RGBA
    (size, size, 4) array, or None while nothing is decodable yet.
    Audio is ignored."""

    def __init__(self, path, size):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: '{path}'")
        self.path = path
        self.size = int(size)
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video '{path}' (is it a format your OpenCV build can decode?)")
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if 1.0 <= fps <= 240.0 else 30.0
        self.native_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.native_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.n_frames = n if n > 0 else 0       # 0 = unknown; learned the first time playback hits the end
        self._next = 0          # index the next sequential read() returns (None = unknown -> seek)
        self.cur_idx = -1       # index of self.cur
        self.cur = None
        self.uploaded_idx = -1  # which frame the GPU texture currently holds (set by RayTracer)
        self.first = self.frame(0)
        if self.first is None:
            raise RuntimeError(f"Video '{path}' opened but no frame could be decoded.")

    @property
    def duration(self):
        return (self.n_frames / self.fps) if self.n_frames > 0 else 0.0

    def index_at(self, t):
        """Frame index shown at playback time t (seconds), with looping."""
        idx = int(max(0.0, t) * self.fps + 1e-6)
        return idx % self.n_frames if self.n_frames > 0 else idx

    def _read(self):
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            return None
        small = cv2.resize(bgr, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGBA))

    def frame(self, idx):
        """uint8 RGBA frame `idx` (wrapped to the video's length), or the last good frame."""
        if self.n_frames > 0:
            idx %= self.n_frames
        if idx == self.cur_idx and self.cur is not None:
            return self.cur
        for _attempt in range(2):
            nxt = self._next
            if nxt is None or idx < nxt or idx - nxt > 8:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)      # rewind / big jump
            else:
                for _ in range(idx - nxt):                       # slightly behind: skip cheaply
                    self.cap.grab()
            arr = self._read()
            if arr is not None:
                self.cur, self.cur_idx, self._next = arr, idx, idx + 1
                return arr
            # Read failed: we ran off the end (or the container's frame count was missing or
            # wrong). Frames 0..idx-1 exist, so that is the real length; wrap and retry.
            # This is how videos with an unknown length learn it and then loop forever.
            if idx <= 0:
                break
            self.n_frames = idx if self.n_frames <= 0 else min(self.n_frames, idx)
            idx %= self.n_frames
            self._next = None
        return self.cur


class TextureLibrary:
    TEX_SIZE = 512

    def __init__(self):
        self.paths = []
        self.arrays = []
        self.videos = {}   # tex_id -> VideoSource (video textures are re-uploaded as playback advances)
        self.skin_slim = {}  # tex_id -> bool, "this Minecraft skin looks like a slim (Alex) model"

    def get_video_id(self, path):
        """Texture slot for a looping video. The slot starts out holding frame 0 and
        is refreshed by RayTracer.set_video_time(). Planes that use the same file
        share one slot (one decoder, always in sync)."""
        key = "video:" + path
        if key in self.paths:
            return self.paths.index(key)
        src = VideoSource(path, self.TEX_SIZE)
        self.paths.append(key)
        self.arrays.append(src.first.astype(np.float32) / 255.0)
        tex_id = len(self.arrays) - 1
        self.videos[tex_id] = src
        src.uploaded_idx = src.cur_idx   # the GPU texture is baked from frame 0
        print(f"Video loaded: {path} -> id {tex_id} ({src.native_w}x{src.native_h} @ {src.fps:.2f} fps, "
              f"{'%.1fs' % src.duration if src.n_frames else 'length unknown'}, loops)")
        return tex_id

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

    def get_skin_id(self, path):
        """Texture slot for a Minecraft skin (see load_minecraft_skin). Differs from
        get_id() in two ways that matter for skins: (1) the resize to TEX_SIZE uses
        NEAREST, not LANCZOS -- a skin is 64x64 pixel art, and a smooth filter would
        blur every texel edge (with the default TEX_SIZE of 512 each skin pixel becomes
        an exact 8x8 block); (2) legacy 64x32 skins are upgraded to the 64x64 layout and
        the base layer is forced opaque, exactly as the game does. Also records whether
        the skin looks slim-armed, so Scene.add_minecraft_character can auto-detect it."""
        key = "skin:" + path
        if key in self.paths:
            return self.paths.index(key)
        rgba, slim, legacy = load_minecraft_skin(path)
        img = Image.fromarray(rgba, 'RGBA').resize((self.TEX_SIZE, self.TEX_SIZE), Image.NEAREST)
        self.paths.append(key)
        self.arrays.append(np.array(img, dtype=np.float32) / 255.0)
        tex_id = len(self.arrays) - 1
        self.skin_slim[tex_id] = bool(slim)
        print(f"Minecraft skin loaded: {path} -> id {tex_id} "
              f"({rgba.shape[1]}x{rgba.shape[0]}{', upgraded from legacy 64x32' if legacy else ''}, "
              f"{'slim' if slim else 'classic'} arms detected)")
        return tex_id

    def average_color(self, tex_id):
        # Only the RGB channels are used to compute an average color for the 2D preview.
        return self.arrays[tex_id][..., :3].reshape(-1, 3).mean(axis=0)

    def stack(self):
        if not self.arrays:
            return np.zeros((1, 4, 4, 4), dtype=np.float32)
        return np.stack(self.arrays, axis=0)

# =============================================================================
# 5b) Minecraft skins -- loading, and the character rig used by
#     Scene.add_minecraft_character.
#
#     COORDINATE CONVENTION (character-local, before position/rotation/scale are
#     applied): units are SKIN PIXELS (1 px = 1/16 of a Minecraft block), +Y is up
#     with the origin on the ground midway between the feet, and the character
#     FACES -Z -- toward a default camera, which looks down +Z -- so a character
#     with rotation=(0,0,0) shows its front to the default view, just like an
#     add_image() plane does. The character's own LEFT side is +X and its RIGHT side
#     is -X, so from the default camera you see the character's right arm on the
#     left of the screen.
#
#     JOINT ROTATIONS use the same (yaw, pitch, roll) = Ry*Rx*Rz convention as
#     rotation_matrix/add_box, applied about the joint's pivot:
#         pitch (+) : swing about the X axis. Arms/legs swing FORWARD, the head tips
#                     UP (looks up -- the same "positive pitch looks up" as the camera)
#                     and the body leans BACK; use a negative pitch to bow forward.
#         yaw   (+) : twist about the vertical axis, toward the character's RIGHT.
#         roll  (+) : tilt about the Z axis, toward the character's LEFT (+X): raises
#                     the LEFT arm outward and swings the RIGHT arm inward, so to
#                     raise the right arm outward use a NEGATIVE roll.
#     Hierarchy: body (waist pivot) carries head + both arms; legs hang from the root.
# =============================================================================

MC_JOINTS = ('body', 'head', 'right_arm', 'left_arm', 'right_leg', 'left_leg')
_MC_JOINT_ALIASES = {'torso': 'body', 'waist': 'body', 'neck': 'head',
                     'r_arm': 'right_arm', 'l_arm': 'left_arm',
                     'r_leg': 'right_leg', 'l_leg': 'left_leg'}
_MC_PARENT = {'body': None, 'head': 'body', 'right_arm': 'body', 'left_arm': 'body',
              'right_leg': None, 'left_leg': None}
_SKIN_UV_EPS = 0.01   # px inset on every face's UV rect so a hit exactly on a face edge can't sample the neighbouring texel


def _mc_joint_name(name):
    key = str(name).strip().lower().replace('-', '_').replace(' ', '_')
    key = _MC_JOINT_ALIASES.get(key, key)
    if key not in _MC_PARENT:
        raise ValueError(f"Unknown joint '{name}'. Valid joints: {', '.join(MC_JOINTS)}")
    return key


def _mc_pivots(slim):
    """Rest-pose joint pivots in skin pixels. These are the vanilla model's own pivots,
    except 'body', which the game pivots at the neck: a WAIST pivot is used here so a
    bent torso stays attached to the legs."""
    ay = 21.5 if slim else 22.0
    return {
        'body': np.array([0.0, 12.0, 0.0]), 'head': np.array([0.0, 24.0, 0.0]),
        'right_arm': np.array([-5.0, ay, 0.0]), 'left_arm': np.array([5.0, ay, 0.0]),
        'right_leg': np.array([-1.9, 12.0, 0.0]), 'left_leg': np.array([1.9, 12.0, 0.0]),
    }


def _mc_parts(slim):
    """The six body parts. size=(w,h,d) in px, center = rest-pose cube centre, uv/uv2 =
    texture origin of the BASE / OVERLAY layer in the 64x64 skin, grow = how far the
    overlay cube is inflated on every side (the game uses 0.5 for the hat, 0.25 for the
    jacket/sleeves/pants). Both layers share the same joint, which is what keeps an
    overlay glued to its base part under any pose."""
    aw = 3.0 if slim else 4.0
    ax = 5.5 if slim else 6.0
    return (
        dict(name='head', joint='head', size=(8, 8, 8), center=(0.0, 28.0, 0.0), uv=(0, 0), uv2=(32, 0), grow=0.5),
        dict(name='body', joint='body', size=(8, 12, 4), center=(0.0, 18.0, 0.0), uv=(16, 16), uv2=(16, 32), grow=0.25),
        dict(name='right_arm', joint='right_arm', size=(aw, 12, 4), center=(-ax, 18.0, 0.0), uv=(40, 16), uv2=(40, 32), grow=0.25),
        dict(name='left_arm', joint='left_arm', size=(aw, 12, 4), center=(ax, 18.0, 0.0), uv=(32, 48), uv2=(48, 48), grow=0.25),
        dict(name='right_leg', joint='right_leg', size=(4, 12, 4), center=(-1.9, 6.0, 0.0), uv=(0, 16), uv2=(0, 32), grow=0.25),
        dict(name='left_leg', joint='left_leg', size=(4, 12, 4), center=(1.9, 6.0, 0.0), uv=(16, 48), uv2=(0, 48), grow=0.25),
    )


def _skin_face_rects(u0, v0, w, h, d):
    """Where each face of a w*h*d box lives in the 64x64 skin, as (x0, y0, x1, y1) px --
    the standard Minecraft box net:   [  top ][bottom]
                                      [right][front][left][back]"""
    return {
        'top': (u0 + d, v0, u0 + d + w, v0 + d),
        'bottom': (u0 + d + w, v0, u0 + d + 2 * w, v0 + d),
        'right': (u0, v0 + d, u0 + d, v0 + d + h),
        'front': (u0 + d, v0 + d, u0 + d + w, v0 + d + h),
        'left': (u0 + d + w, v0 + d, u0 + 2 * d + w, v0 + d + h),
        'back': (u0 + 2 * d + w, v0 + d, u0 + 2 * d + 2 * w, v0 + d + h),
    }


def _skin_cube_corners(half):
    """(face name, outward axis, [TL, TR, BR, BL] corner offsets from the cube centre) for
    each face, where TL/TR/BR/BL are the corners of that face's texture rect as the face
    reads when looked at FROM OUTSIDE (u runs left->right, v top->bottom). Front is -Z,
    the character's right is -X. In BOTH the top and the bottom image the BOTTOM edge is
    the front edge (the game samples the bottom face's v in reverse, which is what makes
    this so) -- checked against real skins: with this orientation the head-bottom image of
    Steve/Alex continues the chin/beard of the front face and the hair of the back."""
    x, y, z = half
    return (
        ('front', (0, 0, -1), [(-x, y, -z), (x, y, -z), (x, -y, -z), (-x, -y, -z)]),
        ('back', (0, 0, 1), [(x, y, z), (-x, y, z), (-x, -y, z), (x, -y, z)]),
        ('right', (-1, 0, 0), [(-x, y, z), (-x, y, -z), (-x, -y, -z), (-x, -y, z)]),
        ('left', (1, 0, 0), [(x, y, -z), (x, y, z), (x, -y, z), (x, -y, -z)]),
        ('top', (0, 1, 0), [(-x, y, z), (x, y, z), (x, y, -z), (-x, y, -z)]),
        ('bottom', (0, -1, 0), [(-x, -y, z), (x, -y, z), (x, -y, -z), (-x, -y, -z)]),
    )


def _mc_pose_value(val):
    """A joint pose may be a bare number (= pitch, the common swing) or (yaw, pitch, roll)."""
    if isinstance(val, (int, float, np.floating, np.integer)):
        return np.array([0.0, float(val), 0.0])
    arr = np.array(val, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.array([0.0, arr[0], 0.0])
    if arr.size != 3:
        raise ValueError(f"A joint pose must be a number (pitch) or (yaw, pitch, roll), got {val!r}")
    return arr


def load_minecraft_skin(path):
    """Reads a Minecraft skin PNG and returns (rgba uint8 array, slim_detected, was_legacy).

    * Accepts the modern 64x64 layout, the legacy 64x32 layout, and HD skins that are
      an integer multiple of either (128x128, 128x64, ...). Everything is returned in
      the SQUARE 64k x 64k layout; legacy skins get their left arm/leg built by
      mirroring the right ones (what the game does), and have no overlay layer.
    * The BASE layer is forced fully opaque (the game ignores base-layer alpha); the
      OVERLAY layer (hat, jacket, sleeves, pants) keeps its alpha, which becomes a
      see-through cut-out in the renderer.
    * slim_detected: Alex-style skins leave a 2px-wide strip of each arm's texture block
      unused (transparent), since their arms are 3px wide. Skins that aren't transparent
      there are treated as classic -- pass slim=True/False explicitly if that guess is
      wrong for a particular skin."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Skin image not found: '{path}'")
    arr = np.array(Image.open(path).convert('RGBA'), dtype=np.uint8)
    h, w = arr.shape[:2]
    if w < 64 or w % 64 != 0 or h not in (w, w // 2):
        raise ValueError(f"'{path}' is {w}x{h}: not a Minecraft skin (expected 64x64 or legacy 64x32, "
                         f"or an integer multiple such as 128x128).")
    k = w // 64
    legacy = (h == w // 2)
    if legacy:
        full = np.zeros((w, w, 4), dtype=np.uint8)
        full[:h] = arr
        arr = full

        def mirror_limb(src, dst):
            """Copy a limb's whole texture block, flipped left<->right; the right and left
            side faces swap places because the limb is mirrored."""
            for fa, fb in (('top', 'top'), ('bottom', 'bottom'), ('front', 'front'),
                           ('back', 'back'), ('right', 'left'), ('left', 'right')):
                sx0, sy0, sx1, sy1 = (v * k for v in _skin_face_rects(*src, 4, 12, 4)[fa])
                dx0, dy0, dx1, dy1 = (v * k for v in _skin_face_rects(*dst, 4, 12, 4)[fb])
                arr[dy0:dy1, dx0:dx1] = arr[sy0:sy1, sx0:sx1][:, ::-1]
        mirror_limb((0, 16), (16, 48))    # right leg -> left leg
        mirror_limb((40, 16), (32, 48))   # right arm -> left arm
        slim = False
    else:
        a = arr[..., 3]
        right_unused = a[20 * k:32 * k, 54 * k:56 * k]
        left_unused = a[52 * k:64 * k, 46 * k:48 * k]
        slim = bool((right_unused == 0).all() and (left_unused == 0).all())

    for (x0, y0, x1, y1) in ((0, 0, 32, 16), (0, 16, 64, 32), (16, 48, 48, 64)):   # base-layer blocks
        arr[y0 * k:y1 * k, x0 * k:x1 * k, 3] = 255
    return np.ascontiguousarray(arr), slim, legacy


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
        self.glass_blocks = []  # [(center np.float64[3], bounding radius), ...] -- non-water
                                # transparent solids, used by the glass-caustic photon pass

    def add_cube(self, pts, color, roughness=0.0, transparency=0.0,
                 ior=1.5, reflection_k=0.0, emissive=0.0, emissive_color=None, unlit=False):
        """pts: 8 vertices in the exact bit-index order described at _FACE_WINDING."""
        pts_arr = np.array(pts, dtype=float)
        if transparency > 0.0 and not (1.30 < ior < 1.36):   # same water test as render_sample
            ctr = pts_arr.mean(axis=0)
            self.glass_blocks.append((ctr, float(np.linalg.norm(pts_arr - ctr, axis=1).max())))
        for fi in _FACE_WINDING:
            self.faces.append(Face([pts_arr[i] for i in fi], color,
                                    roughness, transparency, ior, reflection_k,
                                    emissive=emissive, emissive_color=emissive_color, unlit=unlit))
        self.boxes.append((pts_arr, tuple(int(c) for c in color[:3])))

    def add_box(self, center, size, color, roughness=0.0, transparency=0.0,
                ior=1.5, reflection_k=0.0, rotation=(0.0, 0.0, 0.0),
                emissive=0.0, emissive_color=None, unlit=False):
        """Box given by center+size, optionally rotated about its center via
        rotation=(yaw,pitch,roll) (radians). Rotation is fully compatible
        with the BVH/raytracer (see the note at the top of this file).

        emissive: self-illumination strength (0 = off). The box glows in its own
          `color` (or `emissive_color`, 0-255 rgb, if given) and, unless
          RayTracer.emissive_lighting_enabled is off, also LIGHTS nearby surfaces
          like an area light (soft shadows included). 1.0 = glows at the colour's
          full brightness; try 2-6 for a lamp that reads as clearly lit.
        unlit: True = ignore scene lighting entirely (shows only its glow)."""
        op = {
            'center': list(map(float, center)), 'size': list(map(float, size)),
            'color': list(int(c) for c in color[:3]), 'roughness': float(roughness),
            'transparency': float(transparency), 'ior': float(ior),
            'reflection_k': float(reflection_k), 'rotation': list(map(float, rotation)),
        }
        if emissive > 0.0:      # only logged when used, so older scene files stay byte-identical
            op['emissive'] = float(emissive)
            op['emissive_color'] = None if emissive_color is None else [int(c) for c in emissive_color[:3]]
            op['unlit'] = bool(unlit)
        elif unlit:
            op['unlit'] = True
        self.ops.append({'method': 'add_box', 'kwargs': op})
        self._add_box_geometry_only(center, size, color, roughness, transparency,
                                     ior, reflection_k, rotation,
                                     emissive=emissive, emissive_color=emissive_color, unlit=unlit)

    def add_minecraft_character(self, skin_path, position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0),
                                scale=1.0 / 16.0, pose=None, slim=None,
                                base_layer=True, overlay_layer=True,
                                roughness=0.9, reflection_k=0.0, tint=(255, 255, 255),
                                degrees=False):
        """Adds a Minecraft player model textured with a skin PNG (64x64, legacy 64x32, or HD).

        position: world point on the GROUND between the character's feet.
        rotation: (yaw, pitch, roll) of the WHOLE character about that point, same
          convention as add_box. yaw=0 faces -Z (toward the default camera); positive
          yaw turns the character toward its own right.
        scale: world units per skin pixel. The default 1/16 makes one Minecraft block
          equal one world unit, so the model is 2 units tall; use e.g. 0.25 for a 8-unit
          giant in this file's demo scenes.
        pose: {joint: rotation} posing the joints. Joints: 'body' (pivot at the waist,
          carries head and arms), 'head', 'right_arm', 'left_arm', 'right_leg',
          'left_leg'. A rotation is (yaw, pitch, roll), or a bare number meaning pitch.
          Pitch (+) swings arms/legs forward, tips the head up and leans the body back
          (negative bows forward); yaw (+) twists toward
          the character's right; roll (+) moves the tip of a limb toward the character's
          LEFT (+X), so it raises the left arm outward but the right arm inward --
          raise a right arm outward with a negative roll. Angles are radians unless
          degrees=True (which applies to `rotation` as well). Each joint rotates about
          its own pivot, and the BASE and OVERLAY layers of a part are built from the same
          transform, so a sleeve/jacket/pants/hat always moves exactly with its limb.
        slim: True = 3px-wide (Alex) arms, False = 4px-wide (Steve) arms, None = guess
          from the skin (see load_minecraft_skin).
        base_layer/overlay_layer: set overlay_layer=False to drop the hat/jacket/sleeves/
          pants layer (or base_layer=False to render only it).
        roughness/reflection_k/tint: material, as on add_image.

        The pose is applied WHILE THE GEOMETRY IS BUILT: the BVH is built once and the
        scene is immutable after the RayTracer is constructed, so to change a pose, build
        a new Scene (see build_skin_demo_scene for a worked example) -- or add several
        copies of the character in different poses in the same scene.

        Overlay pixels with alpha below 0.1 are see-through cut-outs; anything more
        opaque renders fully opaque (the renderer has no partial alpha blending).

        Returns {joint: world-space pivot position (np.ndarray)} for the posed model, so
        you can attach other geometry (e.g. a held item at the hand) to it."""
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        k_ang = math.pi / 180.0 if degrees else 1.0
        rotation = np.array(rotation, dtype=float).reshape(3) * k_ang
        pose_rot = {}
        for name, val in (pose or {}).items():
            pose_rot[_mc_joint_name(name)] = _mc_pose_value(val) * k_ang

        tex_id = self.textures.get_skin_id(skin_path)
        if slim is None:
            slim = self.textures.skin_slim.get(tex_id, False)
        slim = bool(slim)

        self.ops.append({'method': 'add_minecraft_character', 'kwargs': {
            'skin_path': skin_path, 'position': list(map(float, position)),
            'rotation': list(map(float, rotation)), 'scale': float(scale),
            'pose': {j: list(map(float, v)) for j, v in pose_rot.items()},
            'slim': slim, 'base_layer': bool(base_layer), 'overlay_layer': bool(overlay_layer),
            'roughness': float(roughness), 'reflection_k': float(reflection_k),
            'tint': list(int(c) for c in tint[:3]), 'degrees': False,
        }})

        pivots = _mc_pivots(slim)
        pose_R = {j: rotation_matrix(*v) for j, v in pose_rot.items()}
        R_root = rotation_matrix(*rotation)
        origin = np.array(position, dtype=float)

        def to_world(pts, joint):
            # Rotate about this joint's pivot, then about its parent's pivot, ... up to the
            # root. Rest-pose px -> posed px -> world units.
            p = np.asarray(pts, dtype=float)
            while joint is not None:
                R = pose_R.get(joint)
                if R is not None:
                    p = (p - pivots[joint]) @ R.T + pivots[joint]
                joint = _MC_PARENT[joint]
            return origin + float(scale) * (p @ R_root.T)

        tex = self.textures.arrays[tex_id]
        tex_n = tex.shape[0]
        layers = []
        if base_layer:
            layers.append(('base', False))
        if overlay_layer:
            layers.append(('overlay', True))
        for part in _mc_parts(slim):
            w, h, d = part['size']
            centre = np.array(part['center'], dtype=float)
            rects = {}
            for layer, is_overlay in layers:
                u0, v0 = part['uv2'] if is_overlay else part['uv']
                grow = part['grow'] if is_overlay else 0.0
                half = np.array([w, h, d], dtype=float) / 2.0 + grow
                rects = _skin_face_rects(u0, v0, w, h, d)    # the UV layout depends on the un-inflated size
                for fname, outward, corners in _skin_cube_corners(half):
                    rest = centre + np.array(corners, dtype=float)
                    x0, y0, x1, y1 = rects[fname]
                    e = _SKIN_UV_EPS
                    uv = [((x0 + e) / 64.0, (y0 + e) / 64.0), ((x1 - e) / 64.0, (y0 + e) / 64.0),
                          ((x1 - e) / 64.0, (y1 - e) / 64.0), ((x0 + e) / 64.0, (y1 - e) / 64.0)]
                    # Keep the Face normal pointing OUT of the cube: the corner order above is
                    # chosen for the texture, so flip order (and UVs with it) when needed.
                    if np.dot(np.cross(rest[1] - rest[0], rest[2] - rest[0]), outward) < 0.0:
                        rest = rest[::-1]
                        uv = uv[::-1]
                    self.faces.append(Face(to_world(rest, part['joint']), tint, roughness, 0.0, 1.5,
                                           reflection_k, uv=uv, texture_id=tex_id))
            if base_layer:
                # 2D-preview box (8 corners in bit-index order, see _FACE_WINDING), base layer only.
                hb = np.array([w, h, d], dtype=float) / 2.0
                pts = np.array([[(hb[0] if i & 4 else -hb[0]), (hb[1] if i & 2 else -hb[1]),
                                 (hb[2] if i & 1 else -hb[2])] for i in range(8)]) + centre
                fx0, fy0, fx1, fy1 = _skin_face_rects(*part['uv'], w, h, d)['front']
                region = tex[int(fy0 / 64.0 * tex_n):max(int(fy1 / 64.0 * tex_n), int(fy0 / 64.0 * tex_n) + 1),
                             int(fx0 / 64.0 * tex_n):max(int(fx1 / 64.0 * tex_n), int(fx0 / 64.0 * tex_n) + 1), :3]
                avg = tuple(int(np.clip(c, 0.0, 1.0) * 255) for c in region.reshape(-1, 3).mean(axis=0))
                self.boxes.append((to_world(pts, part['joint']), avg))

        return {j: to_world(pivots[j], _MC_PARENT[j]) for j in MC_JOINTS}

    def add_image(self, center, size, image_path, rotation=(0.0, 0.0, 0.0),
                  roughness=0.0, transparency=0.0, ior=1.5, reflection_k=0.0,
                  tint=(255, 255, 255), emissive=0.0, emissive_color=None, unlit=False):
        """Adds a single 2D image plane as its own object. size=(w,h) in
        world units. rotation=(yaw,pitch,roll) radians about its center.
        The material parameters (roughness/transparency/ior/reflection_k)
        apply exactly like a box face -- e.g. you can make the image
        "frosted glass" or slightly reflective via transparency/reflection_k > 0.

        Auto-sizing (like add_video): pass h=None, e.g. size=(4, None), to take the
        height from the picture's own aspect ratio -- or w=None, size=(None, 3), to
        take the width. A bare number (size=4) means width 4, height automatic."""
        if isinstance(size, (int, float)):
            size = (float(size), None)
        if size[0] is None or size[1] is None:
            if size[0] is None and size[1] is None:
                raise ValueError("add_image: size needs at least one of width/height (the other may be None).")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Texture image not found: '{image_path}'")
            with Image.open(image_path) as _im:   # header only -- no full decode
                iw, ih = _im.size
            aspect = (ih / iw) if iw > 0 and ih > 0 else 1.0     # height / width
            if size[1] is None:
                size = (float(size[0]), float(size[0]) * aspect)
            else:
                size = (float(size[1]) / aspect, float(size[1]))
        op = {
            'center': list(map(float, center)), 'size': list(map(float, size)),
            'image_path': image_path, 'rotation': list(map(float, rotation)),
            'roughness': float(roughness), 'transparency': float(transparency),
            'ior': float(ior), 'reflection_k': float(reflection_k),
            'tint': list(int(c) for c in tint[:3]),
        }
        if emissive > 0.0:
            op['emissive'] = float(emissive)
            op['emissive_color'] = None if emissive_color is None else [int(c) for c in emissive_color[:3]]
        if unlit:
            op['unlit'] = True
        self.ops.append({'method': 'add_image', 'kwargs': op})
        w, h = size
        hw, hh = w / 2.0, h / 2.0
        local = np.array([(-hw, -hh, 0.0), (hw, -hh, 0.0),
                           (hw, hh, 0.0), (-hw, hh, 0.0)], dtype=float)
        Rm = rotation_matrix(*rotation)
        world = local @ Rm.T + np.array(center, dtype=float)

        tex_id = self.textures.get_id(image_path)
        uv = ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0))
        f = Face(world, tint, roughness, transparency, ior, reflection_k,
                 uv=uv, texture_id=tex_id,
                 emissive=emissive, emissive_color=emissive_color, unlit=unlit, two_sided=True)
        self.faces.append(f)

        avg = self.textures.average_color(tex_id)
        avg_col = tuple(int(np.clip(c, 0, 1) * 255) for c in avg)
        self.quads.append((world, avg_col))
        return f

    def add_video(self, center, size, video_path, rotation=(0.0, 0.0, 0.0),
                  emissive=1.0, unlit=True, emissive_color=None,
                  roughness=0.0, transparency=0.0, ior=1.5, reflection_k=0.0,
                  tint=(255, 255, 255)):
        """A flat plane that plays a video, looping forever -- like add_image, but the
        picture changes over time. size=(w, h) in world units; pass h=None (or a bare
        number for w) to take the height from the video's own aspect ratio.

        By default it behaves like a SCREEN: emissive=1.0 (it glows with the picture,
        and that light spills onto nearby surfaces, tinted by whatever is on screen)
        and unlit=True (the scene's lights don't wash the picture out). Use
        emissive=0, unlit=False for a plain "painted" video texture that is lit like
        any other surface.

        Playback: live view toggles all videos with the SPACE bar; video renders
        play them automatically from t=0 in step with the output timeline; stills use
        --video-time. Every video plane shares the same clock. The video is decoded
        at TextureLibrary.TEX_SIZE x TEX_SIZE (512 by default -- raise it, e.g.
        --texture-size 1024, for sharper video at the cost of GPU memory)."""
        tex_id = self.textures.get_video_id(video_path)
        src = self.textures.videos[tex_id]
        if isinstance(size, (int, float)):
            size = (float(size), None)
        w, h = size
        if h is None:
            aspect = (src.native_h / src.native_w) if src.native_w > 0 and src.native_h > 0 else 9.0 / 16.0
            h = float(w) * aspect
        size = (float(w), float(h))
        op = {
            'center': list(map(float, center)), 'size': list(size),
            'video_path': video_path, 'rotation': list(map(float, rotation)),
            'emissive': float(emissive), 'unlit': bool(unlit),
            'emissive_color': None if emissive_color is None else [int(c) for c in emissive_color[:3]],
            'roughness': float(roughness), 'transparency': float(transparency),
            'ior': float(ior), 'reflection_k': float(reflection_k),
            'tint': list(int(c) for c in tint[:3]),
        }
        self.ops.append({'method': 'add_video', 'kwargs': op})
        hw, hh = size[0] / 2.0, size[1] / 2.0
        local = np.array([(-hw, -hh, 0.0), (hw, -hh, 0.0),
                           (hw, hh, 0.0), (-hw, hh, 0.0)], dtype=float)
        world = local @ rotation_matrix(*rotation).T + np.array(center, dtype=float)
        uv = ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0))
        f = Face(world, tint, roughness, transparency, ior, reflection_k,
                 uv=uv, texture_id=tex_id,
                 emissive=emissive, emissive_color=emissive_color, unlit=unlit, two_sided=True)
        self.faces.append(f)
        avg = self.textures.average_color(tex_id)
        self.quads.append((world, tuple(int(np.clip(c, 0, 1) * 255) for c in avg)))
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
                                ior, reflection_k, rotation,
                                emissive=0.0, emissive_color=None, unlit=False):
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
        self.add_cube(pts, color, roughness, transparency, ior, reflection_k,
                      emissive=emissive, emissive_color=emissive_color, unlit=unlit)

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
                elif op['method'] == 'add_video':
                    _embed_asset(assets, op['kwargs'].get('video_path'))
                elif op['method'] == 'add_minecraft_character':
                    _embed_asset(assets, op['kwargs'].get('skin_path'))
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
            if op['method'] == 'add_video' and asset_dir is not None:
                kwargs['video_path'] = _resolve_asset_path(kwargs.get('video_path'), asset_dir)
            if op['method'] == 'add_minecraft_character' and asset_dir is not None:
                kwargs['skin_path'] = _resolve_asset_path(kwargs.get('skin_path'), asset_dir)
            method(**kwargs)
        return scene


def triangulate(faces):
    """Each quad Face (v0,v1,v2,v3) -> 2 triangles, carrying UV + texture_id."""
    v0s, v1s, v2s = [], [], []
    normals, colors = [], []
    rough, transp, iors, refl = [], [], [], []
    uv0s, uv1s, uv2s, texids = [], [], [], []
    emis, emis_col, unlit = [], [], []
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
            emis.append(f.emissive); emis_col.append(f.emissive_color)
            unlit.append((1 if f.unlit else 0) | (2 if f.two_sided else 0))   # bit0 unlit, bit1 two-sided emitter
    return dict(
        v0=np.array(v0s, dtype=np.float32), v1=np.array(v1s, dtype=np.float32),
        v2=np.array(v2s, dtype=np.float32), normal=np.array(normals, dtype=np.float32),
        color=np.array(colors, dtype=np.float32), roughness=np.array(rough, dtype=np.float32),
        transparency=np.array(transp, dtype=np.float32), ior=np.array(iors, dtype=np.float32),
        reflection_k=np.array(refl, dtype=np.float32),
        uv0=np.array(uv0s, dtype=np.float32), uv1=np.array(uv1s, dtype=np.float32),
        uv2=np.array(uv2s, dtype=np.float32), texture_id=np.array(texids, dtype=np.int32),
        emissive=np.array(emis, dtype=np.float32), emissive_color=np.array(emis_col, dtype=np.float32),
        unlit=np.array(unlit, dtype=np.int32),
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
        # Visibility toggles, independent of whether the sun/star params
        # exist at all: turning either OFF keeps its full configuration
        # (angle, color, density, seed, ...) so it can be turned back on
        # unchanged, rather than having to be re-described from scratch.
        # _repaint simply skips a disabled layer. See set_sun_enabled /
        # set_stars_enabled and RayTracer.toggle_sun / toggle_stars.
        self.sun_enabled = True
        self.stars_enabled = True
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
        if (self.stars is None or self._star_field is None or not self.stars_enabled
                or self.stars['twinkle'] <= 1e-4):
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
        if not self._star_field or not self.stars_enabled:
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

    def set_sun_enabled(self, enabled):
        """Shows/hides the sun/moon disc WITHOUT discarding its settings
        (contrast with set_sun(intensity=0), which overwrites the stored
        intensity and so can't be undone by a plain toggle). No-op if
        nothing would change."""
        enabled = bool(enabled)
        if enabled == self.sun_enabled:
            return self
        self.sun_enabled = enabled
        self._repaint()
        return self

    def set_stars_enabled(self, enabled):
        """Shows/hides the star field without discarding its settings --
        see set_sun_enabled."""
        enabled = bool(enabled)
        if enabled == self.stars_enabled:
            return self
        self.stars_enabled = enabled
        self._repaint()
        return self

    def _paint_sun(self):
        """Paints the current self.sun disc onto self.image (assumes the
        caller, _repaint, has already reset self.image to the clean
        base).

        Three layers, each softer/wider than the last: a limb-darkened
        disc (brighter at its center, dimming gently toward the edge --
        the way the real sun's photosphere looks, rather than a flat
        cutout), a tight bright corona right at its edge, and a much
        wider, faint atmospheric halo (real haze scatters sunlight across
        a big patch of sky, not just a thin ring). Every pixel that could
        be touched is supersampled on a small sub-pixel grid rather than
        tested once at its center, so the whole shape is anti-aliased and
        actually responds smoothly to angular_size_deg/glow_deg even when
        the disc itself is only a few pixels across (see
        _sample_background's docstring for why that matters at this
        texture's resolution)."""
        if self._sky_base_image is None or self.sun is None or not self.sun_enabled:
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
        halo_r = glow_r + max(disc_r, glow_r) * 1.5
        SS = 4  # sub-samples per axis per pixel (16 total) -- see docstring
        offs = [(k + 0.5) / SS - 0.5 for k in range(SS)]
        n_sub = float(SS * SS)
        pad = int(math.ceil(halo_r)) + 1
        y0, y1 = max(0, int(cy) - pad), min(h, int(cy) + pad + 1)
        for iy in range(y0, y1):
            for ixo in range(-pad, pad + 1):
                ix = (int(cx) + ixo) % w  # wraps around the horizon (azimuth 360 == 0)
                disc_acc = 0.0
                limb_acc = 0.0
                glow_acc = 0.0
                for soy in offs:
                    dyp = (iy + soy) - cy
                    for sox in offs:
                        dxp = (int(cx) + ixo + sox) - cx
                        dist = math.sqrt(dxp * dxp + dyp * dyp)
                        if dist <= disc_r:
                            disc_acc += 1.0
                            # Cheap limb-darkening approximation (Eddington
                            # -style): full brightness at the center,
                            # fading toward the rim.
                            r_norm = dist / disc_r
                            limb_acc += 1.0 - 0.45 * (1.0 - math.sqrt(max(0.0, 1.0 - r_norm * r_norm)))
                        elif dist <= glow_r:
                            t = 1.0 - (dist - disc_r) / max(1e-4, glow_r - disc_r)
                            glow_acc += t * t * (3.0 - 2.0 * t)  # smoothstep -- no seam at the disc edge
                        elif dist <= halo_r:
                            t = 1.0 - (dist - glow_r) / max(1e-4, halo_r - glow_r)
                            glow_acc += t * t * 0.18  # far fainter, wide atmospheric halo
                if disc_acc <= 0.0 and glow_acc <= 0.0:
                    continue
                disc_cov = disc_acc / n_sub
                glow_cov = min(1.0, glow_acc / n_sub)
                limb = (limb_acc / disc_acc) if disc_acc > 0.0 else 1.0
                cur = self.image[iy, ix]
                if disc_cov > 0.0:
                    cur = cur * (1.0 - disc_cov) + (col * limb) * disc_cov
                if glow_cov > 0.0:
                    cur = cur * (1.0 - glow_cov) + col * glow_cov
                self.image[iy, ix] = cur

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
        # Always written (not just when a sun/star field exists) so a
        # round-trip is exact even for a scene that has one configured but
        # currently hidden.
        d['sun_enabled'] = bool(self.sun_enabled)
        d['stars_enabled'] = bool(self.stars_enabled)
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
        # Default True so scenes written by older versions (which had no
        # concept of hiding either layer) load looking exactly as before.
        bg.sun_enabled = bool(d.get('sun_enabled', True))
        bg.stars_enabled = bool(d.get('stars_enabled', True))
        bg._repaint()
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
F_EMIS = F_EMIS_COL = F_UNLIT = None          # per-triangle self-illumination (see Face)
EM_TRI = EM_CDF = EM_PDF = EM_AREA = N_EM = EM_STRENGTH = None   # emissive-triangle light list
TEX_FIELD = None
N_MIN = N_MAX = N_LEFT = N_RIGHT = N_START = N_COUNT = None
BG_FIELD = None
LIGHT_POS = LIGHT_COLOR = LIGHT_BRIGHTNESS = LIGHT_VIS = LIGHT_SOFT = None
SPOT_POS = SPOT_DIR = SPOT_COLOR = SPOT_BRIGHTNESS = None
SPOT_COS_OUTER = SPOT_COS_INNER = SPOT_VIS = None
CLOUD_ENABLED = CLOUD_BASE = CLOUD_TOP = CLOUD_DENSITY = CLOUD_COVERAGE = None
CLOUD_SCALE = CLOUD_COLOR = CLOUD_WIND = CLOUD_STEPS = CLOUD_LIGHT_STEPS = CLOUD_SUN_DIR = None
CLOUD_TIME = None                           # "billow" offset -- see _cloud_density: shifts the
                                             # noise field's OWN vertical sampling axis over time,
                                             # so cross-sections morph/churn instead of just sliding
                                             # sideways the way CLOUD_WIND (a rigid xz translation) does.
PROBE_DEPTH = None
GC_CENTER = GC_RADIUS = GC_HALF = GC_RECV_Y = GC_VALID = N_GC = None
GLASS_ACC = GLASS_TMP = GLASS_MAP = GLASS_STRENGTH = None
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
    global F_EMIS, F_EMIS_COL, F_UNLIT
    global EM_TRI, EM_CDF, EM_PDF, EM_AREA, N_EM, EM_STRENGTH
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
    F_EMIS = ti.field(ti.f32, shape=n); F_EMIS.from_numpy(tri['emissive'][order])
    F_EMIS_COL = ti.Vector.field(3, ti.f32, shape=n); F_EMIS_COL.from_numpy(tri['emissive_color'][order])
    F_UNLIT = ti.field(ti.i32, shape=n); F_UNLIT.from_numpy(tri['unlit'][order])

    # Emissive triangles double as AREA LIGHTS. Sorted into BVH order like everything
    # else, then given a CDF proportional to (area x strength x brightness) so brighter/
    # bigger emitters are picked more often by the one light sample taken per shaded point.
    em_idx = np.nonzero(tri['emissive'][order] > 0.0)[0].astype(np.int32)
    n_em = int(len(em_idx))
    k = max(1, n_em)
    EM_TRI = ti.field(ti.i32, shape=k); EM_CDF = ti.field(ti.f32, shape=k)
    EM_PDF = ti.field(ti.f32, shape=k); EM_AREA = ti.field(ti.f32, shape=k)
    N_EM = ti.field(ti.i32, shape=()); EM_STRENGTH = ti.field(ti.f32, shape=())
    N_EM[None] = n_em
    EM_STRENGTH[None] = 1.0
    if n_em:
        v0 = tri['v0'][order][em_idx]; v1 = tri['v1'][order][em_idx]; v2 = tri['v2'][order][em_idx]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).astype(np.float32)
        ecol = tri['emissive_color'][order][em_idx]
        base = np.where((ecol[:, :1] >= 0.0), ecol, tri['color'][order][em_idx])
        lum = np.maximum(0.05, base @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
        w = np.maximum(1e-12, area * tri['emissive'][order][em_idx] * lum)
        pdf = (w / w.sum()).astype(np.float32)
        cdf = np.cumsum(pdf).astype(np.float32); cdf[-1] = 1.0
        EM_TRI.from_numpy(np.pad(em_idx, (0, k - n_em)))
        EM_CDF.from_numpy(np.pad(cdf, (0, k - n_em), constant_values=1.0))
        EM_PDF.from_numpy(np.pad(pdf, (0, k - n_em), constant_values=1.0))
        EM_AREA.from_numpy(np.pad(area, (0, k - n_em), constant_values=1.0))

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
    global LIGHT_POS, LIGHT_COLOR, LIGHT_BRIGHTNESS, LIGHT_VIS, LIGHT_SOFT
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
    LIGHT_SOFT = ti.field(ti.f32, shape=())   # light angular radius (tan) -> penumbra softness
    LIGHT_SOFT[None] = 0.0


def alloc_glass_fields():
    """Fields for the glass-caustic photon pass (called once, like the other
    alloc_* functions). N_GC = 0 means "no glass clusters", which makes the
    shading-side lookup a single field read."""
    global GC_CENTER, GC_RADIUS, GC_HALF, GC_RECV_Y, GC_VALID, N_GC
    global GLASS_ACC, GLASS_TMP, GLASS_MAP, GLASS_STRENGTH
    GC_CENTER = ti.Vector.field(3, ti.f32, shape=MAX_GLASS_CLUSTERS)
    GC_RADIUS = ti.field(ti.f32, shape=MAX_GLASS_CLUSTERS)
    GC_HALF = ti.field(ti.f32, shape=MAX_GLASS_CLUSTERS)      # half-extent of the receiver map
    GC_RECV_Y = ti.field(ti.f32, shape=MAX_GLASS_CLUSTERS)    # height of the surface under the glass
    GC_VALID = ti.field(ti.i32, shape=MAX_GLASS_CLUSTERS)
    N_GC = ti.field(ti.i32, shape=())
    shape3 = (MAX_GLASS_CLUSTERS, GLASS_CAUSTIC_RES, GLASS_CAUSTIC_RES)
    GLASS_ACC = ti.Vector.field(3, ti.f32, shape=shape3)      # raw splatted photon energy (signed)
    GLASS_TMP = ti.Vector.field(3, ti.f32, shape=shape3)
    GLASS_MAP = ti.Vector.field(3, ti.f32, shape=shape3)      # blurred result, read at render time
    GLASS_STRENGTH = ti.field(ti.f32, shape=())
    N_GC[None] = 0
    GLASS_STRENGTH[None] = 0.0


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
    global CLOUD_TIME
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
    CLOUD_TIME = ti.field(ti.f32, shape=())
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
    CLOUD_TIME[None] = 0.0

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
    # Bilinear (not nearest-pixel) lookup. The sky texture is only a few
    # hundred texels across (see Background.set_sky_gradient's default
    # 384x192), so nearest-neighbor sampling turns any small baked-in
    # feature -- most visibly the sun disc from Background._paint_sun --
    # into hard, blocky steps in the final render, and makes fine
    # adjustments to its angular_size_deg/glow_deg invisible unless they
    # happen to cross a whole-texel boundary. Bilinear filtering (wrapping
    # horizontally across the azimuth seam, clamping vertically at the
    # poles) turns that into a smooth gradient instead, so the sun/cloud
    # backdrop actually looks round and responds continuously to its
    # parameters.
    h = BG_FIELD.shape[0]; w = BG_FIELD.shape[1]
    dx = direction[0]; dy = direction[1]; dz = direction[2]
    u = (ti.atan2(dx, dz) / (2.0 * math.pi)) % 1.0
    v = 1.0 - (ti.asin(ti.max(-1.0, ti.min(1.0, dy))) / math.pi + 0.5)
    fx = u * w - 0.5
    fy = ti.max(0.0, ti.min(float(h - 1), v * (h - 1)))
    ix0 = int(ti.floor(fx))
    iy0 = int(ti.floor(fy))
    tx = fx - float(ix0)
    ty = fy - float(iy0)
    ix0w = ix0 % w
    if ix0w < 0:
        ix0w += w
    ix1w = (ix0w + 1) % w
    iy0c = int(ti.max(0.0, ti.min(float(h - 1), float(iy0))))
    iy1c = int(ti.max(0.0, ti.min(float(h - 1), float(iy0 + 1))))
    c00 = BG_FIELD[iy0c, ix0w]
    c10 = BG_FIELD[iy0c, ix1w]
    c01 = BG_FIELD[iy1c, ix0w]
    c11 = BG_FIELD[iy1c, ix1w]
    top = vec3(c00[0], c00[1], c00[2]) * (1.0 - tx) + vec3(c10[0], c10[1], c10[2]) * tx
    bot = vec3(c01[0], c01[1], c01[2]) * (1.0 - tx) + vec3(c11[0], c11[1], c11[2]) * tx
    return top * (1.0 - ty) + bot * ty


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
        warp_x = x + 0.6 * ti.sin(z * 0.07 + t * W0)
        warp_z = z + 0.6 * ti.cos(x * 0.065 - t * W0)

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

        d0x, d0z, r0x, r0z = _gerstner_wave(warp_x, warp_z, t,  0.80,  0.60, 0.35, 0.22, 2.0 * W0, 0.35)
        d1x, d1z, r1x, r1z = _gerstner_wave(warp_x, warp_z, t, -0.35,  0.94, 0.55, 0.14, 3.0 * W0, 0.30)
        d2x, d2z, r2x, r2z = _gerstner_wave(warp_x, warp_z, t,  0.94,  0.34, 0.95, 0.075, 4.0 * W0, 0.28)
        d3x, d3z, r3x, r3z = _gerstner_wave(warp_x, warp_z, t, -0.60, -0.80, 1.7, 0.04, 6.0 * W0, 0.22)
        d4x, d4z, r4x, r4z = _gerstner_wave(warp_x, warp_z, t,  0.20, -0.98, 3.1, 0.018, 8.0 * W0, 0.18)
        d5x, d5z, r5x, r5z = _gerstner_wave(warp_x, warp_z, t, -0.86,  0.51, 5.4, 0.008, 11.0 * W0, 0.12)

        dhdx = d0x + d1x + d2x + d3x + d4x + d5x
        dhdz = d0z + d1z + d2z + d3z + d4z + d5z

        # Tiny high-frequency jitter (not a traveling wave, just noise-like
        # texture) so dead-flat patches between the low-frequency swells
        # still show some sparkle/micro-ripple, like real water never being
        # perfectly glassy.
        jit_x = 0.006 * ti.sin(warp_x * 9.7 + warp_z * 6.1 + t * 16.0 * W0)
        jit_z = 0.006 * ti.cos(warp_z * 8.9 - warp_x * 7.3 + t * 13.0 * W0)

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
    # Zero outside [CLOUD_BASE, CLOUD_TOP]. Real cumulus-style clouds have
    # a noticeably flatter, SHARPER-edged base (moisture condenses at a
    # fairly consistent altitude, so the underside is comparatively crisp)
    # and a softer, rounder, more billowing TOP -- so, unlike a symmetric
    # fade, the bottom clears over a short band while the top eases out
    # over a much taller one. That asymmetry alone reads as "cloud" far
    # more than a slab with the same blur on both edges.
    base = CLOUD_BASE[None]
    top = CLOUD_TOP[None]
    thickness = ti.max(1e-3, top - base)
    bottom_edge = ti.min(thickness * 0.12, 4.0)
    top_edge = ti.min(thickness * 0.45, 14.0)
    y = p[1]
    vert = 0.0
    if base <= y <= top:
        bottom_fade = ti.min(1.0, (y - base) / ti.max(1e-3, bottom_edge))
        top_t = ti.min(1.0, (top - y) / ti.max(1e-3, top_edge))
        top_fade = top_t * top_t * (3.0 - 2.0 * top_t)  # smoothstep -- rounds the billowing top
        vert = bottom_fade * top_fade
    d = 0.0
    if vert > 0.0:
        scale = 1.0 / ti.max(1e-3, CLOUD_SCALE[None])
        wind = CLOUD_WIND[None]
        # CLOUD_WIND translates the xz sample position -- shapes slide
        # sideways but keep the exact same silhouette (a rigid drift).
        # CLOUD_TIME instead offsets the vertical sampling axis, which
        # feeds the SAME 3D noise volume -- so advancing it slides a
        # moving cross-section through that volume, and the cauliflower
        # shapes genuinely churn/morph over time ("billow") rather than
        # just translating. Combine both for clouds that drift AND billow.
        wp = vec3((p[0] + wind[0]) * scale, (y * 0.5 + CLOUD_TIME[None]) * scale, (p[2] + wind[1]) * scale)
        n = _cloud_fbm(wp)
        # `coverage` (0..1) raises/lowers how much of the noise has to
        # clear before it counts as any density at all -- low coverage =
        # mostly clear sky with a few dense puffs, high coverage = a
        # solid overcast sheet.
        #
        # Two things keep this from reading as ONE flat sheet (the old
        # `raw*raw*(3-2*raw)` ramp over a fixed threshold did):
        #  (a) The threshold RISES with height (hfrac 0 at the base ->
        #      1 at the top): near the base a lot of the noise counts as
        #      cloud, near the top only its highest peaks do, so each
        #      puff narrows into a rounded dome on a flat-ish base
        #      instead of being a full-thickness column of uniform
        #      density -- that vertical structure is what makes separate
        #      cumulus read as separate objects.
        #  (b) A steep gain on the remapped value (saturating at
        #      raw ~= 0.25) gives each puff a well-defined body with a
        #      soft but short edge, and clear sky between puffs, instead
        #      of a wide shallow fade that blends neighbours into haze.
        hfrac = (y - base) / thickness
        thr = (1.0 - CLOUD_COVERAGE[None]) - 0.10 + 0.32 * hfrac
        raw = ti.max(0.0, n - thr)
        g = ti.min(1.0, raw * 4.0)
        shaped = g * g * (3.0 - 2.0 * g)
        # A second, higher-frequency noise sample "erodes" fine detail
        # into the shape -- without this every puff is a perfectly smooth
        # blob, which is the single biggest tell that it's a raw noise
        # field rather than an actual cloud. Deliberately just one extra
        # octave (cheap) at a higher frequency, offset well away from the
        # base sample so it doesn't just look like the same shape zoomed
        # in.
        detail = _value_noise3(wp * 4.3 + vec3(19.1, 7.7, 53.3))
        # Erode the EDGES (low `shaped`) harder than the core: subtracting
        # a detail-driven amount frays the silhouette into cauliflower
        # lobes while dense interiors stay solid.
        shaped = ti.min(1.0, ti.max(0.0, shaped - 0.32 * (1.0 - detail)) / 0.86)
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
        span = t1 - t0
        # STEP SPACING. The slab's ray interval grows like 1/sin(elevation):
        # a ray at 3 degrees crosses a 60-unit slab over ~1100 units, and
        # the old uniform `span/steps` step was then 45+ units -- as large
        # as a whole noise feature -- so every step averaged over several
        # puffs and the layer collapsed into a smooth, featureless sheet
        # (worst toward the horizon, fine when looking up: hence "sometimes").
        # Now samples are spaced as t = t0 + span * s^pw (s in 0..1): pw = 1
        # is the old uniform spacing (used whenever the uniform step is
        # already fine relative to CLOUD_SCALE), and pw grows up to 3 as the
        # uniform step gets coarse, packing samples densely near the camera
        # -- where puffs are big on screen and detail matters -- and
        # spreading them out over the far part of the slab, where a puff is
        # a few pixels wide anyway.
        seg_uniform = span / steps
        pw = ti.max(1.0, ti.min(3.0, seg_uniform / ti.max(1.0, 0.25 * CLOUD_SCALE[None])))
        # One random jitter per ray (stratified: each step samples a random
        # spot inside its own stratum) breaks up the banding a fixed-step
        # march would leave, and averages out over samples.
        jit = ti.random(ti.f32)
        light_steps = CLOUD_LIGHT_STEPS[None]
        light_dir = CLOUD_SUN_DIR[None]
        cloud_col = CLOUD_COLOR[None]
        # How directly this ray looks toward the sun -- rd and light_dir
        # are both fixed for the whole march, so this is computed once
        # rather than inside the per-step loop below. Used for the
        # silver-lining boost: real clouds scatter light strongly FORWARD
        # (the Mie/Henyey-Greenstein forward lobe), which is why a cloud
        # sitting between you and the sun gets a bright glowing edge
        # instead of just reading as a dim silhouette.
        forward = ti.max(0.0, rd[0] * light_dir[0] + rd[1] * light_dir[1] + rd[2] * light_dir[2])
        fw2 = forward * forward
        fw8 = fw2 * fw2 * fw2 * fw2
        for i in range(steps):
            if transmittance > 1e-3:
                s_lo = ti.pow(ti.f32(i) / steps, pw)
                s_hi = ti.pow(ti.f32(i + 1) / steps, pw)
                s_mid = ti.pow((ti.f32(i) + jit) / steps, pw)
                seg = span * (s_hi - s_lo)   # this step's own length
                t = t0 + span * s_mid
                p = ro + rd * t
                dens = _cloud_density(p)
                if dens > 1e-4:
                    step_optical = dens * seg
                    step_transmit = ti.exp(-step_optical)
                    # Distance from THIS point to the slab's top along the
                    # light direction -- NOT a fixed CLOUD_TOP-CLOUD_BASE
                    # (that's only correct when the sun sits straight
                    # overhead; at any other elevation the true path length
                    # through the layer is longer, exactly like a shadow
                    # raymarch through a horizontal fog slab -- see
                    # _cloud_shadow_transmittance/_cloud_slab_interval,
                    # which get this right for the GROUND shadow. Without
                    # this, self-shadowing was badly underestimated for any
                    # non-vertical sun, so every puff came out nearly
                    # uniformly lit -- no top/bottom contrast -- which reads
                    # as a flat, evenly-toned sheet instead of a volumetric
                    # cloud. Clamp the vertical component so a
                    # near-horizontal or below-horizon light doesn't blow
                    # this up (or flip sign) -- it just gets the longest,
                    # darkest march instead, which is the physically right
                    # direction for a grazing sun angle.
                    light_dy = ti.max(0.05, light_dir[1])
                    light_seg = ti.max(1e-3, (CLOUD_TOP[None] - p[1]) / light_dy) / light_steps
                    light_optical = 0.0
                    lt = light_seg * 0.5
                    for _l in range(light_steps):
                        light_optical += _cloud_density(p + light_dir * lt) * light_seg
                        lt += light_seg
                    light_transmit = ti.exp(-light_optical)
                    # 0.35 floor -- even a fully self-shadowed cloud interior
                    # still gets some ambient sky-bounce light, real clouds
                    # are never lit purely from one direction.
                    # The 0.35 ambient term also gets brighter with height
                    # (sky light reaches a puff's top, not its underside),
                    # which keeps a top/bottom gradient even when the sun
                    # is behind the cloud or too low to self-shadow much.
                    hf = ti.min(1.0, ti.max(0.0, (p[1] - CLOUD_BASE[None]) / ti.max(1e-3, CLOUD_TOP[None] - CLOUD_BASE[None])))
                    lit_color = cloud_col * (0.30 + 0.22 * hf + 0.48 * light_transmit)
                    # Aerial perspective: very distant puffs (a few pixels
                    # wide, dozens of them stacked along a grazing ray)
                    # drift toward an even, bright haze instead of each
                    # showing its own dark underside, which otherwise turns
                    # the horizon into a grainy grey band.
                    haze = 1.0 - ti.exp(-t / ti.max(1.0, CLOUD_SCALE[None] * 40.0))
                    lit_color = lit_color * (1.0 - haze) + cloud_col * (0.9 * haze)
                    # Silver lining, scaled by light_transmit so it's
                    # strongest at thin, sunlit patches (little
                    # self-shadowing) rather than washing out a thick
                    # cloud's fully-shadowed core into a flat glow.
                    lit_color += cloud_col * (fw8 * light_transmit * 1.8)
                    scattered += transmittance * (1.0 - step_transmit) * lit_color
                    transmittance *= step_transmit
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
def _rand_in_ball() -> vec3:
    d = vec3(ti.randn(), ti.randn(), ti.randn()).normalized()
    return d * ti.pow(ti.random(ti.f32), 1.0 / 3.0)


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
    inside_start = 0.0   # path distance at which the ray last entered a glass solid

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

        # Absolute distance along the shadow ray to this hit. `travelled`
        # already includes every cutout step, so no seg_carry is needed.
        s_hit = travelled + t

        # Straight shadow rays through a glass solid. Two physical effects,
        # both of which the old code got wrong/omitted (hence "flat"):
        #  * Beer-Lambert absorption over the distance INSIDE the glass only.
        #    The old code applied it for every hit using `t`, which is the
        #    distance from the previous point (air!) -- so the shadow tint
        #    depended on how far the receiver was from the glass.
        #  * Fresnel loss at each interface: ~8% at normal incidence but rising
        #    sharply at grazing angles, so shadows get darker toward the
        #    silhouette/edges of the glass -- the main depth cue in a real one.
        geo_n = F_NORMAL[tid]
        cosg = ldir.dot(geo_n)
        cos_i = ti.min(1.0, ti.abs(cosg))
        gior = F_IOR[tid]
        if cosg < 0.0:
            inside_start = s_hit          # entering the solid
        else:
            throughput *= _beer_lambert_tint(col, transp, ti.max(0.0, s_hit - inside_start))
        # air->glass and glass->air interfaces of a slab have equal
        # transmittance for a given external angle.
        throughput *= (1.0 - _fresnel_schlick(cos_i, gior))

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
# 10a) Video textures + emissive area lights
# =============================================================================

@ti.kernel
def upload_texture_u8(tex_id: ti.i32, frame: ti.types.ndarray(dtype=ti.u8, ndim=3)):
    """Overwrites one TEX_FIELD slot with a uint8 RGBA frame (a video frame)."""
    for y, x in ti.ndrange(frame.shape[0], frame.shape[1]):
        TEX_FIELD[tex_id, y, x] = vec4(ti.cast(frame[y, x, 0], ti.f32), ti.cast(frame[y, x, 1], ti.f32),
                                       ti.cast(frame[y, x, 2], ti.f32), ti.cast(frame[y, x, 3], ti.f32)) / 255.0


@ti.func
def _emitted_radiance(tid: ti.i32, tex_rgb: vec3) -> vec3:
    """Radiance a face emits at a point whose (already texture-multiplied) albedo
    is `tex_rgb`: its explicit emissive_color if it has one, else its own colour."""
    ec = F_EMIS_COL[tid]
    base = tex_rgb
    if ec[0] >= 0.0:
        base = ec
    return base * F_EMIS[tid]


@ti.func
def _sample_emissive_light(p: vec3, n: vec3) -> vec3:
    """One-sample area-light estimate of the light that emissive faces (screens,
    lamps, glowing boxes) deliver to point p with normal n, BEFORE multiplying by
    the receiver's albedo. A face is picked by power (CDF), a point on it uniformly,
    then a shadow ray (glass tints it, opaque blocks it). Lambertian: albedo/pi * E."""
    res = vec3(0.0)
    nem = N_EM[None]
    if nem > 0:
        u = ti.random(ti.f32)
        lo = 0
        hi = nem - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if EM_CDF[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        tid = EM_TRI[lo]
        r1 = ti.sqrt(ti.random(ti.f32))
        r2 = ti.random(ti.f32)
        b0 = 1.0 - r1
        b1 = r1 * (1.0 - r2)
        b2 = r1 * r2
        q = F_V0[tid] * b0 + F_V1[tid] * b1 + F_V2[tid] * b2
        to = q - p
        d2 = ti.max(to.dot(to), 1e-2)
        wi = to / ti.sqrt(d2)
        cos_s = n.dot(wi)
        cos_l = -F_NORMAL[tid].dot(wi)             # light leaves through the face's outward side...
        if (F_UNLIT[tid] & 2) != 0:
            cos_l = ti.abs(cos_l)                  # ...except flat planes (images/videos), which emit both ways
        if cos_s > 1e-4 and cos_l > 1e-4:
            albedo = F_COLOR[tid]
            alpha = 1.0
            texid = F_TEXID[tid]
            if texid >= 0:
                uvp = F_UV0[tid] * b0 + F_UV1[tid] * b1 + F_UV2[tid] * b2
                texel = _sample_texture(texid, uvp[0], uvp[1])
                albedo = albedo * vec3(texel[0], texel[1], texel[2])
                alpha = texel[3]
            if alpha >= 0.1:
                le = _emitted_radiance(tid, albedo)
                tp = _shadow_throughput(p + n * 1e-3, q - wi * 0.01)
                res = le * tp * (cos_s * cos_l * EM_AREA[lo] / (d2 * EM_PDF[lo] * math.pi))
    return res


# =============================================================================
# 10b) Glass caustics (photon pass).
#
# The shadow rays in _shadow_throughput are straight, so on their own they
# can only make glass cast a uniformly tinted, slightly dimmed shadow. Real
# glass REDISTRIBUTES light: it shifts it sideways and bunches it up at
# bright streaks/edges, leaving darker bands next to them. This pass
# computes that redistribution once (not per frame):
#
#   * For every (glass cluster, light) pair, photons are fired at the cluster's
#     bounding disc as seen from the light.
#   * Each photon is traced TWICE through the real BVH: once refracting
#     (Snell + Fresnel + Beer-Lambert) -> where the light REALLY lands, and
#     once straight -> where the straight shadow ray in _shadow_throughput
#     already assumes it lands (with the same Fresnel/absorption).
#   * Energy is splatted into a 2D receiver map: +P at the real landing
#     point, -P at the straight one. So the map is the *difference* between
#     real and straight lighting: it adds to what the shadow rays already
#     deliver, never double-counts, and is energy-neutral (just moves light).
#   * The map is blurred a little, then render_sample adds `albedo * map`.
#
# Cost: the photon pass is a one-off per light/scene change. Per frame the
# only cost is one bilinear lookup per diffuse hit on an up-facing surface
# (and a single field read when the scene has no glass at all).
# Limits: receivers are the up-facing surface directly under each cluster
# (typically the floor); only transmitted light is traced (total internal
# reflection photons bounce up to GLASS_PHOTON_STEPS), and occluders between light and glass are
# tested for opaque blockers only.
# =============================================================================

@ti.func
def _gc_trace(ro_in: vec3, rd_in: vec3, straight: ti.i32):
    """Follows one photon until it lands on an opaque surface. straight=0
    refracts at every glass face; straight=1 goes in a straight line and
    applies the same Fresnel/absorption as _shadow_throughput. Returns
    (landed, landing_point, landing_normal, throughput, touched_glass)."""
    ro = ro_in
    rd = rd_in
    w = vec3(1.0)
    touched = 0
    landed = 0
    lp = vec3(0.0)
    ln = vec3(0.0, 1.0, 0.0)
    inside_start = 0.0
    travelled = 0.0
    alive = 1
    for _ in range(GLASS_PHOTON_STEPS):
        if alive != 0:
            tid, t, bu, bv = _bvh_closest_hit(ro, rd, 1e18)
            if tid < 0:
                alive = 0
            else:
                p = ro + rd * t
                transp = F_TRANSP[tid]
                texid = F_TEXID[tid]
                cutout = 0
                if texid >= 0:
                    uv0 = F_UV0[tid]; uv1 = F_UV1[tid]; uv2 = F_UV2[tid]
                    uvp = uv0 * (1.0 - bu - bv) + uv1 * bu + uv2 * bv
                    tex_rgba = _sample_texture(texid, uvp[0], uvp[1])
                    if tex_rgba[3] < 0.1:
                        cutout = 1
                if cutout != 0:
                    ro = p + rd * 1e-3
                    travelled += t + 1e-3
                elif transp <= 0.0:
                    lp = p
                    ln = F_NORMAL[tid]
                    landed = 1
                    alive = 0
                else:
                    touched = 1
                    col = F_COLOR[tid]
                    ior = F_IOR[tid]
                    gn = F_NORMAL[tid]
                    entering = rd.dot(gn) < 0.0
                    n = gn if entering else -gn
                    if straight != 0:
                        s_hit = travelled + t
                        cos_i = ti.min(1.0, ti.abs(rd.dot(gn)))
                        if entering:
                            inside_start = s_hit
                        else:
                            w *= _beer_lambert_tint(col, transp, ti.max(0.0, s_hit - inside_start))
                        w *= (1.0 - _fresnel_schlick(cos_i, ior))
                        ro = p + rd * 1e-3
                        travelled += t + 1e-3
                    else:
                        if not entering:
                            w *= _beer_lambert_tint(col, transp, t)
                        cosi = ti.max(1e-6, rd.dot(-n))
                        eta = (1.0 / ior) if entering else ior
                        sin2_t = eta * eta * (1.0 - cosi * cosi)
                        if sin2_t > 1.0:
                            # Total internal reflection: the photon stays inside and bounces on
                            # (no loss -- F = 1). It exits through another face, or runs out of
                            # steps. Dropping it instead (as a first version did) removed far
                            # too much light: a cube lit from the side TIRs on its bottom face.
                            rd = (rd - 2.0 * rd.dot(n) * n).normalized()
                            ro = p + n * 1e-4
                        else:
                            cos_s = cosi
                            if not entering:
                                cos_s = ti.sqrt(ti.max(0.0, 1.0 - sin2_t))
                            w *= (1.0 - _fresnel_schlick(cos_s, ior))
                            cost = ti.sqrt(ti.max(0.0, 1.0 - sin2_t))
                            rd = (eta * rd + (eta * cosi - cost) * n).normalized()
                            ro = p - n * 1e-4
    return landed, lp, ln, w, touched


@ti.func
def _gc_land_ok(ci: ti.i32, lp: vec3, ln: vec3) -> ti.i32:
    ok = 0
    if GC_VALID[ci] != 0 and ln[1] > 0.5 and ti.abs(lp[1] - GC_RECV_Y[ci]) < GLASS_Y_TOL:
        h = GC_HALF[ci]
        c = GC_CENTER[ci]
        if ti.abs(lp[0] - c[0]) < h and ti.abs(lp[2] - c[2]) < h:
            ok = 1
    return ok


@ti.func
def _gc_deposit(ci: ti.i32, lp: vec3, val: vec3):
    h = GC_HALF[ci]
    c = GC_CENTER[ci]
    res = float(GLASS_CAUSTIC_RES)
    fu = (lp[0] - (c[0] - h)) / (2.0 * h) * res - 0.5
    fv = (lp[2] - (c[2] - h)) / (2.0 * h) * res - 0.5
    ix = int(ti.floor(fu))
    iz = int(ti.floor(fv))
    tx = fu - float(ix)
    tz = fv - float(iz)
    cell = 2.0 * h / res
    v = val / (cell * cell)       # energy -> irradiance (power per receiver area)
    for dz in ti.static(range(2)):
        for dx in ti.static(range(2)):
            xx = ix + dx
            zz = iz + dz
            wx = (1.0 - tx) + (2.0 * tx - 1.0) * dx
            wz = (1.0 - tz) + (2.0 * tz - 1.0) * dz
            if 0 <= xx < GLASS_CAUSTIC_RES and 0 <= zz < GLASS_CAUSTIC_RES:
                GLASS_ACC[ci, zz, xx] += v * (wx * wz)


@ti.kernel
def glass_probe_receivers(n_clusters: ti.i32):
    """Finds the opaque surface straight below each cluster's centre (looking
    through any glass on the way) -- that height is the receiver plane."""
    for ci in range(n_clusters):
        ro = GC_CENTER[ci]
        down = vec3(0.0, -1.0, 0.0)
        state = 0
        y = 0.0
        for _ in range(GLASS_PHOTON_STEPS):
            if state == 0:
                tid, t, _bu, _bv = _bvh_closest_hit(ro, down, 1e18)
                if tid < 0:
                    state = -1
                else:
                    p = ro + down * t
                    if F_TRANSP[tid] <= 0.0:
                        state = 1
                        y = p[1]
                    else:
                        ro = p + down * 1e-3
        GC_VALID[ci] = 1 if state == 1 else 0
        GC_RECV_Y[ci] = y


@ti.kernel
def glass_caustics_kernel(n_clusters: ti.i32, n_lights: ti.i32, n_spots: ti.i32, n_photons: ti.i32):
    for ci, ei, k in ti.ndrange(n_clusters, n_lights + n_spots, n_photons):
        lpos = vec3(0.0)
        lpow = vec3(0.0)
        sdir = vec3(0.0, 0.0, 1.0)
        si = 0
        if ei < n_lights:
            lpos = LIGHT_POS[ei]
            lpow = LIGHT_COLOR[ei] * LIGHT_BRIGHTNESS[ei]
        else:
            si = ei - n_lights
            lpos = SPOT_POS[si]
            lpow = SPOT_COLOR[si] * SPOT_BRIGHTNESS[si]
            sdir = SPOT_DIR[si]
        if lpow.max() > 1e-6 and GC_VALID[ci] != 0:
            c = GC_CENTER[ci]
            r = GC_RADIUS[ci]
            to_c = c - lpos
            dist = to_c.norm()
            axis = to_c / ti.max(dist, 1e-6)
            helper = vec3(0.0, 1.0, 0.0)
            if ti.abs(axis[1]) > 0.95:
                helper = vec3(1.0, 0.0, 0.0)
            t1 = axis.cross(helper).normalized()
            t2 = axis.cross(t1)
            rr = r * ti.sqrt(ti.random(ti.f32))
            phi = 2.0 * math.pi * ti.random(ti.f32)
            q = c + t1 * (rr * ti.cos(phi)) + t2 * (rr * ti.sin(phi))
            d = (q - lpos).normalized()

            atten = 1.0
            ok = 1
            if dist < 2.5 * r:
                ok = 0                      # light inside/too close to the cluster: skip
            if ei >= n_lights:
                cos_theta = d.dot(sdir)
                cos_outer = SPOT_COS_OUTER[si]
                cos_inner = SPOT_COS_INNER[si]
                if cos_theta > cos_outer:
                    cone_t = ti.max(0.0, ti.min(1.0, (cos_theta - cos_outer) / ti.max(cos_inner - cos_outer, 1e-4)))
                    atten = cone_t * cone_t * (3.0 - 2.0 * cone_t)
                else:
                    ok = 0

            s0 = q - d * (2.0 * r)          # start just outside the cluster's bounding sphere
            if ok != 0:
                # An OPAQUE occluder between the light and the glass (e.g. a wall) blocks the photon.
                oh, ot, _ou, _ov = _bvh_closest_hit(lpos, d, dist - 2.0 * r - 1e-3)
                if oh >= 0 and F_TRANSP[oh] <= 0.0:
                    ok = 0

            if ok != 0:
                power = lpow * (math.pi * r * r / float(n_photons)) * atten
                landed_r, lp_r, ln_r, w_r, touched_r = _gc_trace(s0, d, 0)
                landed_s, lp_s, ln_s, w_s, touched_s = _gc_trace(s0, d, 1)
                if touched_s != 0:
                    if landed_s != 0 and _gc_land_ok(ci, lp_s, ln_s) != 0:
                        _gc_deposit(ci, lp_s, -power * w_s)
                    if landed_r != 0 and _gc_land_ok(ci, lp_r, ln_r) != 0:
                        _gc_deposit(ci, lp_r, power * w_r)


@ti.kernel
def glass_blur_h(n: ti.i32, src: ti.template(), dst: ti.template()):
    for ci, z, x in ti.ndrange(n, GLASS_CAUSTIC_RES, GLASS_CAUSTIC_RES):
        xm = ti.max(x - 1, 0)
        xp = ti.min(x + 1, GLASS_CAUSTIC_RES - 1)
        dst[ci, z, x] = (src[ci, z, xm] + 2.0 * src[ci, z, x] + src[ci, z, xp]) * 0.25


@ti.kernel
def glass_blur_v(n: ti.i32, src: ti.template(), dst: ti.template()):
    for ci, z, x in ti.ndrange(n, GLASS_CAUSTIC_RES, GLASS_CAUSTIC_RES):
        zm = ti.max(z - 1, 0)
        zp = ti.min(z + 1, GLASS_CAUSTIC_RES - 1)
        dst[ci, z, x] = (src[ci, zm, x] + 2.0 * src[ci, z, x] + src[ci, zp, x]) * 0.25


@ti.func
def _sample_glass_caustic(p: vec3, n: vec3) -> vec3:
    """Signed extra irradiance (rgb) at p from glass caustics -- zero unless
    p is an up-facing surface at a cluster's receiver height, inside its map."""
    res = vec3(0.0)
    if n[1] > 0.5:
        for ci in range(N_GC[None]):
            if GC_VALID[ci] != 0:
                c = GC_CENTER[ci]
                h = GC_HALF[ci]
                if (ti.abs(p[1] - GC_RECV_Y[ci]) < GLASS_Y_TOL
                        and ti.abs(p[0] - c[0]) < h and ti.abs(p[2] - c[2]) < h):
                    rs = float(GLASS_CAUSTIC_RES)
                    fx = ti.max(0.0, ti.min(rs - 1.0, (p[0] - (c[0] - h)) / (2.0 * h) * rs - 0.5))
                    fz = ti.max(0.0, ti.min(rs - 1.0, (p[2] - (c[2] - h)) / (2.0 * h) * rs - 0.5))
                    x0 = int(fx); z0 = int(fz)
                    x1 = ti.min(x0 + 1, GLASS_CAUSTIC_RES - 1)
                    z1 = ti.min(z0 + 1, GLASS_CAUSTIC_RES - 1)
                    tx = fx - float(x0); tz = fz - float(z0)
                    res += (GLASS_MAP[ci, z0, x0] * ((1.0 - tx) * (1.0 - tz))
                            + GLASS_MAP[ci, z0, x1] * (tx * (1.0 - tz))
                            + GLASS_MAP[ci, z1, x0] * ((1.0 - tx) * tz)
                            + GLASS_MAP[ci, z1, x1] * (tx * tz))
    return res


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
        real_bounce = 0         # counts actual optical events (reflect/refract) ONLY -- see
                                 # the tex_alpha branch below for why this has to be separate
                                 # from the `bounce` loop variable.
        # Distance already travelled since the ray's last REAL origin (camera,
        # or the last reflect/refract event) but "spent" skipping through
        # alpha-cutout pixels. Each cutout hit moves ray_o forward to the
        # sprite plane, so the next hit's `t` only measures from THAT plane;
        # without adding this back, Beer-Lambert absorption (which depends on
        # path length) and the depth used for DoF/clouds were computed from
        # the sprite plane instead of the true origin -- glass sitting behind
        # a see-through part of an RGBA image came out visibly paler/less
        # tinted than the same glass with nothing in front of it.
        carry_t = 0.0

        # The loop's own trip count is max_bounce PLUS a separate, generous
        # allowance for alpha-cutout pixels (see tex_alpha branch below): a
        # ray tunnelling through a transparent PNG's cutout regions (hair
        # strands, foliage, a sprite's edges) doesn't perform an optical
        # bounce and must NOT spend the real reflect/refract budget --
        # without this, a ray that has to pass through even a couple of
        # cutout pixels before reaching a glass/water surface behind them
        # could arrive with no real bounces left, making that glass/water
        # silently render as if it weren't there (see real_bounce checks
        # below, which gate on the SEPARATE real_bounce counter instead of
        # this loop's `bounce`).
        for bounce in range(max_bounce + 1 + CUTOUT_TUNNEL_BUDGET):
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
                carry_t += t + 1e-3   # remember the distance we just skipped over
                continue  # move on to the next bounce without stopping the ray here

            # A real (non-cutout) surface: the true path length since the last
            # real origin is the skipped cutout distance PLUS this segment.
            t = t + carry_t
            carry_t = 0.0

            if not depth_written:
                DEPTH_ACCUM[py, px] += t
                depth_written = True
                primary_t = t

            rough = F_ROUGH[tid]
            transp = F_TRANSP[tid]
            ior = F_IOR[tid]
            refl_k = F_REFL[tid]

            # Self-illumination: the face's own glow, added straight to the pixel
            # (through any reflection/refraction throughput that got us here).
            # `col` already includes the texture/video texel, so a video plane glows
            # with whatever frame is currently showing.
            if F_EMIS[tid] > 0.0:
                final_color += throughput * _emitted_radiance(tid, col)
            is_unlit = (F_UNLIT[tid] & 1) != 0

            cos_i_signed = -ray_dir.dot(geo_n)
            entering = cos_i_signed > 0.0
            n = geo_n if entering else -geo_n

            if transp > 0.0:
                # --- Dielectric material (glass): random Fresnel split --- (unchanged from v7/v8)
                if real_bounce >= max_bounce:
                    final_color += throughput * _sample_background(ray_dir) * bg_brightness
                    terminated = True
                    continue

                # Beer-Lambert absorption belongs to the distance travelled INSIDE
                # the medium. If this hit is from the inside (not `entering`), the
                # segment we just crossed was inside the glass -- tint it, whether
                # the ray then reflects (internal bounce) or refracts out. (The old
                # code tinted on every refraction instead, so a ray ENTERING glass
                # was tinted by its air distance to the surface, and internal
                # reflections were never absorbed at all.)
                if not entering:
                    throughput *= _beer_lambert_tint(col, transp, t)

                ns = n
                if ior > 1.30 and ior < 1.36:
                    ns = _get_water_normal(p, n)
                else:
                    ns = _perturb_in_cone(n, rough)
                cosi = ti.max(1e-6, ray_dir.dot(-ns))
                eta = (1.0 / ior) if entering else ior
                sin2_t = eta * eta * (1.0 - cosi * cosi)
                tir = sin2_t > 1.0

                # Fresnel reflectance. Schlick must be fed the cosine on the
                # DENSER side's... i.e. for glass->air use the transmitted angle's
                # cosine (not the incident one), otherwise exits are far too
                # transmissive. TIR is a perfect (F = 1) reflection.
                fresnel = 1.0
                if not tir:
                    cos_s = cosi
                    if not entering:
                        cos_s = ti.sqrt(ti.max(0.0, 1.0 - sin2_t))
                    fresnel = _fresnel_schlick(cos_s, ior)

                # Russian-roulette choice between reflect / refract. The clamp only
                # keeps variance sane; the weights below are what keep it unbiased.
                p_reflect = 1.0
                if not tir:
                    p_reflect = ti.max(0.05, ti.min(0.95, fresnel))

                if ti.random(ti.f32) < p_reflect:
                    rdir = (ray_dir - 2.0 * ray_dir.dot(ns) * ns).normalized()
                    ray_o = p + n * 1e-4
                    ray_dir = rdir
                    # estimator weight = F / p. (The old `/= p_reflect` forgot the F:
                    # a reflection that really carries ~4% of the light was counted
                    # at ~100% (x20 when p was clamped to 0.05) and TIR's 5% leftover
                    # branch got a 1/0.05 = x20 weight on a bogus refraction. That
                    # energy gain is the self-glow + the sparkle noise.)
                    throughput *= fresnel / p_reflect
                else:
                    cost = ti.sqrt(ti.max(0.0, 1.0 - sin2_t))
                    rdir = (eta * ray_dir + (eta * cosi - cost) * ns).normalized()
                    ray_o = p - n * 1e-4
                    ray_dir = rdir
                    # weight = (1 - F) / (1 - p); radiance scales by eta^2 when
                    # crossing media but cancels on the way back out, so omitted.
                    throughput *= (1.0 - fresnel) / ti.max(1e-4, 1.0 - p_reflect)

                real_bounce += 1
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
                if is_unlit:
                    ambient_term = vec3(0.0)
                direct = vec3(0.0)
                n_lights_eff = n_lights
                n_spots_eff = n_spots
                if is_unlit:
                    n_lights_eff = 0      # unlit faces (screens) ignore scene lighting
                    n_spots_eff = 0
                for li in range(n_lights_eff):
                    lpos = LIGHT_POS[li]
                    lcol = LIGHT_COLOR[li]
                    lbri = LIGHT_BRIGHTNESS[li]
                    # Area-light jitter: treat the light as a small sphere of
                    # angular radius LIGHT_SOFT (radius scales with distance, so
                    # a far "sun" and a near lamp both get a sensible penumbra).
                    # One random point per sample; accumulation averages it into
                    # a soft shadow edge.
                    ldist = (lpos - p).norm()
                    lpos = lpos + _rand_in_ball() * (ldist * LIGHT_SOFT[None])
                    light_dir = (lpos - p).normalized()
                    shadow_tp = _shadow_throughput(p + n * 1e-3, lpos)
                    diff = ti.max(0.0, n.dot(light_dir))
                    half_v = (light_dir + view_dir).normalized()
                    spec = specular_k * (ti.max(0.0, n.dot(half_v)) ** shininess)
                    direct += (diff * col + spec) * lcol * lbri * shadow_tp

                for si in range(n_spots_eff):
                    spos = SPOT_POS[si]
                    sdir = SPOT_DIR[si]
                    scol = SPOT_COLOR[si]
                    sbri = SPOT_BRIGHTNESS[si]
                    cos_outer = SPOT_COS_OUTER[si]
                    cos_inner = SPOT_COS_INNER[si]

                    sdist = (spos - p).norm()
                    spos = spos + _rand_in_ball() * (sdist * LIGHT_SOFT[None])
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
                glass_E = vec3(0.0)
                if N_GC[None] > 0 and GLASS_STRENGTH[None] != 0.0:
                    # Signed (real - straight) photon irradiance from glass -- see
                    # compute_glass_caustics. Albedo-tinted like any diffuse light.
                    glass_E = _sample_glass_caustic(p, n) * GLASS_STRENGTH[None]
                em_E = vec3(0.0)
                if N_EM[None] > 0 and EM_STRENGTH[None] != 0.0 and not is_unlit:
                    # Emissive faces light the scene too (see _sample_emissive_light).
                    em_E = _sample_emissive_light(p, n) * EM_STRENGTH[None]
                local = ambient_term + direct * caustic_mult + col * (glass_E + em_E)
                if is_unlit:
                    local = vec3(0.0)
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

                if refl_k > 0.0 and real_bounce < max_bounce:
                    final_color += throughput * (1.0 - refl_k) * local
                    throughput = throughput * col * refl_k
                    ns = _perturb_in_cone(n, rough)
                    rdir = (ray_dir - 2.0 * ray_dir.dot(ns) * ns).normalized()
                    ray_o = p + n * 1e-3
                    ray_dir = rdir
                    real_bounce += 1
                    if throughput.max() < 1e-3:
                        terminated = True
                else:
                    final_color += throughput * local
                    terminated = True

        if not terminated:
            # Practically never hit -- would need a ray to tunnel through
            # CUTOUT_TUNNEL_BUDGET alpha-cutout pixels in a row without ever
            # reaching real geometry. Falls back to background instead of
            # silently contributing black, the same as the tid < 0 case above.
            final_color += throughput * _sample_background(ray_dir) * bg_brightness

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
    __slots__ = ("pos", "yaw", "pitch", "speed", "duration", "zoom", "roll")

    def __init__(self, pos, yaw, pitch, speed=1.0, duration=None, zoom=1.0, roll=0.0):
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
        # Camera roll (radians) AT this keyframe. Older versions dropped it (the
        # camera always rolled back to level); it is now stored, saved to the
        # scene file and slerped between keyframes together with yaw/pitch.
        self.roll = float(roll)

    def to_dict(self):
        d = {'pos': [float(x) for x in self.pos], 'yaw': self.yaw,
             'pitch': self.pitch, 'speed': self.speed}
        if self.duration is not None:
            d['duration'] = self.duration
        if abs(self.zoom - 1.0) > 1e-6:
            d['zoom'] = self.zoom
        if abs(self.roll) > 1e-9:
            d['roll'] = self.roll
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
            return kf.pos.copy(), kf.yaw, kf.pitch, 0.0, 0.0, kf.roll
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
                # Orientation is interpolated as a single quaternion slerp
                # (yaw + pitch + the keyframe's roll together) rather than
                # yaw/pitch lerped independently -- see the quaternion
                # utilities near camera_matrix for why this is the
                # canonical way to blend camera orientation.
                qa = quat_from_euler(a.yaw, a.pitch, a.roll)
                qb = quat_from_euler(b.yaw, b.pitch, b.roll)
                yaw, pitch, roll = euler_from_quat(quat_slerp(qa, qb, local_t))
                phase_acc += (local_t * seg_len) / seg_stride if seg_stride > 1e-6 else 0.0
                return pos, float(yaw), float(pitch), phase_acc, seg_speed, float(roll)
            phase_acc += seg_len / seg_stride if seg_stride > 1e-6 else 0.0
            acc += d
        kf = self.keyframes[-1]
        return kf.pos.copy(), kf.yaw, kf.pitch, phase_acc, 0.0, kf.roll

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
                tuple(round(kf.roll, 6) for kf in self.keyframes),
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
        # Rotation is smoothed as a QUATERNION spring (see
        # _quat_spring_step), not independent yaw/pitch springs -- this is
        # the rotational analogue of the position spring just below, and
        # keeps the smoothing gimbal-lock-free and consistent with how
        # CameraPath/SensorCameraData interpolate everywhere else.
        quat = quat_from_euler(float(raw0[1]), float(raw0[2]), float(raw0[5]))
        vel_pos = np.zeros(3, dtype=np.float64)
        ang_vel = np.zeros(3, dtype=np.float64)

        grid_pos = np.empty((n_steps, 3), dtype=np.float32)
        grid_yaw = np.empty(n_steps, dtype=np.float32)
        grid_pitch = np.empty(n_steps, dtype=np.float32)
        grid_roll = np.empty(n_steps, dtype=np.float32)
        grid_phase = np.empty(n_steps, dtype=np.float32)
        grid_speed = np.empty(n_steps, dtype=np.float32)

        k = omega_n * omega_n
        c = 2.0 * zeta * omega_n
        # Footstep phase/speed are re-derived from the SPRING's own smoothed
        # velocity here (see below), not the raw path's declared per-segment
        # speed -- see the note on grid_phase/grid_speed just after the loop
        # for why: without this, footstep shake pops in intensity/timing at
        # every keyframe boundary even though the position itself is smooth.
        smoothed_phase = 0.0
        for i in range(n_steps):
            t = t0 + i * sim_dt
            t_query = max(0.0, min(total, t))
            r_pos, r_yaw, r_pitch, _raw_phase, _raw_speed, r_roll = self._raw_pose_at(t_query)
            r_pos = r_pos.astype(np.float64)
            r_quat = quat_from_euler(float(r_yaw), float(r_pitch), float(r_roll))

            acc_pos = k * (r_pos - pos) - c * vel_pos
            vel_pos = vel_pos + acc_pos * sim_dt
            pos = pos + vel_pos * sim_dt

            quat, ang_vel = _quat_spring_step(quat, ang_vel, r_quat, k, c, sim_dt)
            yaw, pitch, roll = euler_from_quat(quat)

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
            grid_roll[i] = roll
            grid_phase[i] = smoothed_phase
            grid_speed[i] = smoothed_speed

        self._inertia_cache = {
            'sig': self._inertia_signature(),
            't0': t0, 'dt': sim_dt, 'n': n_steps,
            'pos': grid_pos, 'yaw': grid_yaw, 'pitch': grid_pitch, 'roll': grid_roll,
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
        roll = cache['roll'][i0] + (cache['roll'][i1] - cache['roll'][i0]) * frac
        phase = cache['phase'][i0] + (cache['phase'][i1] - cache['phase'][i0]) * frac
        speed = cache['speed'][i0] + (cache['speed'][i1] - cache['speed'][i0]) * frac
        return pos.astype(np.float32), float(yaw), float(pitch), float(phase), float(speed), float(roll)

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

    def add(self, pos, yaw, pitch, speed=1.0, duration=None, zoom=1.0, roll=0.0):
        self.keyframes.append(CameraKeyframe(pos, yaw, pitch, speed, duration, zoom, roll))
        return self.keyframes[-1]

    def hold(self, duration=2.0, yaw=None, pitch=None, roll=None):
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
        new_roll = last.roll if roll is None else float(roll)
        return self.add(last.pos.copy(), new_yaw, new_pitch, speed=last.speed,
                         duration=duration, zoom=last.zoom, roll=new_roll)

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
        pos, yaw, pitch, phase_acc, seg_speed, kf_roll = self._sample_smoothed(t)
        # Handheld shake is evaluated at the UN-clamped t (not any
        # internally-clamped local time) so it keeps evolving smoothly even
        # if a caller samples slightly past the path's end -- only matters
        # at the boundary, keeps the wobble continuous there.
        pos_off, yaw_off, pitch_off, roll_off = self._handheld_offset(t)
        gpos, gyaw, gpitch, groll = self._gait_offset(t, seg_speed, phase_acc)
        return (pos + pos_off + gpos, yaw + yaw_off + gyaw,
                pitch + pitch_off + gpitch, kf_roll + roll_off + groll)

    def sample_quat(self, t):
        """Same pose as sample(), but returning (pos, quaternion) instead
        of (pos, yaw, pitch, roll) -- the gimbal-lock-free form callers
        should prefer (see the note on SensorCameraData.sample_quat, which
        is where it actually matters; a hand-placed keyframe path can't
        reach pitch=+-90deg, so for CameraPath the two agree exactly).
        Returns None for an empty path."""
        s = self.sample(t)
        if s is None:
            return None
        pos, yaw, pitch, roll = s
        return pos, quat_from_euler(yaw, pitch, roll)

    def to_list(self):
        return [kf.to_dict() for kf in self.keyframes]

    @staticmethod
    def from_list(items):
        cp = CameraPath()
        for kf in items:
            cp.add(kf['pos'], kf.get('yaw', 0.0), kf.get('pitch', 0.0),
                    kf.get('speed', 1.0), kf.get('duration'), kf.get('zoom', 1.0), kf.get('roll', 0.0))
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
    __slots__ = ("t", "pos", "yaw", "pitch", "roll", "quat")

    def __init__(self, t, pos, yaw, pitch, roll, quat=None):
        self.t = float(t)
        self.pos = pos  # np.float32[3]
        # yaw/pitch/roll are kept purely for display/readability (HUD,
        # "camera data" status line, camera_dict-style introspection);
        # `quat` is canonical and is what sample()/interpolation actually
        # uses -- see the quaternion utilities near camera_matrix. If not
        # given explicitly (e.g. by a loader that only has Euler angles),
        # it's derived from yaw/pitch/roll once here.
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.roll = float(roll)
        self.quat = quat_normalize(quat) if quat is not None else quat_from_euler(yaw, pitch, roll)


def _parse_locale_float(s):
    """Parses a float that may use ',' as the decimal separator (as seen
    in plain-text --camera-data exports like Map_Race_#01.txt) as well as
    the normal '.' separator. There is no thousands-grouping in these
    files -- fields are already split on ':' -- so a bare ',' is always a
    decimal point here."""
    s = s.strip()
    return float(s.replace(',', '.')) if s else 0.0


class SensorCameraData:
    """Camera path loaded from an external recording, driving the camera
    instead of hand-placed keyframes (--camera-data). Two file formats are
    understood (auto-detected by SensorCameraData.load):

    1) A sensor_record JSON (version 1: timestamp_unit "ms", position_unit
       "m", rotation_unit "deg", samples: [{timestamp, pos:{x,y,z},
       rot:{x,y,z}}, ...]). rot.x/.y/.z map to pitch/yaw/roll respectively
       (same axis convention as camera_matrix's Rx=pitch, Ry=yaw, Rz=roll).

    2) A plain-text quaternion track (e.g. 'Map_Race_#01.txt'): one sample
       per line, ':'-separated, `t:pos_x:pos_y:pos_z:qx:qy:qz:qw` (decimal
       comma or dot; an optional trailing 'END:<n>' line is ignored).
       Orientation here is ALREADY a quaternion -- no Euler conversion
       needed on the way in, which is exactly the canonical representation
       this class (and the rest of the core camera logic) uses internally.

    Orientation is stored and INTERPOLATED as a quaternion (see
    _SensorSample.quat / sample()'s use of quat_slerp) -- yaw/pitch/roll
    are only ever recovered from it for display."""

    def __init__(self, samples, multiplier=1.0, offset=(0.0, 0.0, 0.0)):
        self.samples = samples  # list[_SensorSample], t sorted ascending, t[0] == 0.0
        self.multiplier = float(multiplier)
        self.offset = np.array(offset, dtype=np.float32)
        # --- Inertia smoothing (see CameraPath.inertia's docstring) ------
        # Off (0.0) by default, matching CameraPath -- set from
        # --inertia/--inertia-bounce in main() after loading, same as it
        # is for keyframe paths. Uses the SAME spring math (position
        # spring + _quat_spring_step for rotation) and the same
        # precompute-a-dense-grid-up-front approach as
        # CameraPath._build_inertia_cache, for the same determinism
        # reasons (motion-blur sub-sampling / multi-GPU time blocks must
        # all agree on the smoothed trajectory regardless of call order).
        self.inertia = 0.0
        self.inertia_bounce = 0.35
        self._inertia_cache = None

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
    def load(path, multiplier=1.0, offset=(0.0, 0.0, 0.0), rot_offset_quat=None):
        """Loads either file format (see the class docstring), auto-
        detected: a '.json' extension (or content starting with '{') is
        parsed as the sensor_record JSON; anything else (e.g. '.txt', like
        'Map_Race_#01.txt') is parsed as the plain-text quaternion track.
        `rot_offset_quat`, if given, is applied (pre-multiplied) as an
        orientation offset on every sample -- see --camera-rot-offset /
        the rotation half of --camera-offset."""
        is_json = path.lower().endswith('.json')
        if not is_json:
            with open(path, 'r', encoding='utf-8') as f:
                head = f.read(256).lstrip()
            is_json = head.startswith('{')
        rot_decimals = None
        if is_json:
            samples = SensorCameraData._load_json_samples(path, multiplier, offset, rot_offset_quat)
        else:
            samples, rot_decimals = SensorCameraData._load_txt_samples(
                path, multiplier, offset, rot_offset_quat)
        if not samples:
            raise ValueError(f"'{path}' has no samples.")
        samples.sort(key=lambda s: s.t)
        data = SensorCameraData(samples, multiplier=multiplier, offset=offset)
        data.source_path = path
        data.rot_decimals = rot_decimals
        # A text track stores each quaternion component as a rounded
        # decimal. At 1 or 2 decimals that rounding is worth flagging: an
        # error of +-0.05 on a unit quaternion component is up to ~5.7deg
        # of orientation, so individual samples can be several degrees
        # off and the replay picks up a visible snap/twitch wherever the
        # rounding happens to move a component a whole step. This is a
        # property of the FILE, not of the interpolation -- slerp
        # reproduces faithfully whatever it is given. --inertia smooths it.
        if rot_decimals is not None and rot_decimals <= 2:
            worst_deg, worst_t = data.worst_rotation_step()
            print(f"Note: '{path}' stores rotation to {rot_decimals} decimal place"
                  f"{'' if rot_decimals == 1 else 's'} (~"
                  f"{math.degrees(2 * math.asin(min(1.0, 0.5 * 10 ** -rot_decimals))):.1f}deg of "
                  f"orientation resolution). Largest single-sample rotation step is "
                  f"{worst_deg:.1f}deg at t={worst_t:.2f}s; if that reads as a snap/flip, it is the "
                  f"recording's precision, not the replay -- try --inertia 0.3 (or higher) to "
                  f"smooth it.")
        return data

    def worst_rotation_step(self):
        """(largest rotation angle in degrees between two CONSECUTIVE
        recorded samples, the timestamp it happens at) -- a quick measure
        of how jumpy a recording is, used for the precision note in
        load(). Quaternion geometry, so it is meaningful even where a
        yaw/pitch/roll view of the same motion would be degenerate."""
        worst, worst_t = 0.0, 0.0
        for a, b in zip(self.samples, self.samples[1:]):
            dot = abs(float(np.dot(quat_normalize(a.quat), quat_normalize(b.quat))))
            ang = math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))
            if ang > worst:
                worst, worst_t = ang, b.t
        return worst, worst_t

    @staticmethod
    def _load_json_samples(path, multiplier, offset, rot_offset_quat):
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
            q = quat_from_euler(yaw, pitch, roll)
            if rot_offset_quat is not None:
                q = quat_normalize(quat_multiply(rot_offset_quat, q))
            yaw, pitch, roll = euler_from_quat(q)
            samples.append(_SensorSample(t, pos, yaw, pitch, roll, quat=q))
        return samples

    @staticmethod
    def _load_txt_samples(path, multiplier, offset, rot_offset_quat):
        """`t:pos_x:pos_y:pos_z:qx:qy:qz:qw` per line -- see class
        docstring. Blank lines and a trailing 'END:<n>' marker are
        skipped. Quaternion component order is assumed x,y,z,w (the
        common convention for this kind of exported track); each is
        re-normalized on load since text exports round each component to
        a fixed number of decimals, which drifts the norm slightly."""
        offset_arr = np.array(offset, dtype=np.float32)
        samples = []
        # Tracked so load() can warn about a coarsely-quantized recording
        # -- see the note there. Purely diagnostic; nothing is altered.
        max_rot_decimals = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.upper().startswith('END'):
                    continue
                parts = line.split(':')
                if len(parts) < 8:
                    continue
                t = _parse_locale_float(parts[0])
                px, py, pz = (_parse_locale_float(parts[i]) for i in (1, 2, 3))
                qx, qy, qz, qw = (_parse_locale_float(parts[i]) for i in (4, 5, 6, 7))
                pos = (np.array([px, py, pz], dtype=np.float32) * multiplier) + offset_arr
                for raw in parts[4:8]:
                    raw = raw.strip()
                    dec = len(raw.split(',')[1]) if ',' in raw else (
                        len(raw.split('.')[1]) if '.' in raw else 0)
                    max_rot_decimals = max(max_rot_decimals, dec)
                q = quat_normalize(np.array([qw, qx, qy, qz], dtype=np.float64))
                if rot_offset_quat is not None:
                    q = quat_normalize(quat_multiply(rot_offset_quat, q))
                yaw, pitch, roll = euler_from_quat(q)
                samples.append(_SensorSample(t, pos, yaw, pitch, roll, quat=q))
        if samples:
            t0 = samples[0].t
            for s in samples:
                s.t -= t0
        return samples, max_rot_decimals

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

    def _raw_sample_quat(self, t):
        """t (seconds from the first sample) -> (pos np.float32[3], unit
        quaternion), WITHOUT inertia smoothing -- LINEAR position
        interpolation and quaternion SLERP orientation interpolation
        between the two nearest samples.

        This, not _raw_sample, is the primitive: orientation never touches
        Euler angles on this path at all. That matters because a real
        recording CAN pass exactly through pitch = +-90deg, where the
        yaw/pitch/roll decomposition is singular (yaw and roll stop being
        separately determined) -- 'Map_Race_#01.txt' does exactly that at
        t ~= 7.4s, where the raw quaternion is a perfectly well-behaved
        (0.5, 0.5, -0.5, 0.5). Decomposing there and recomposing
        afterwards throws the orientation away and snaps the camera; the
        quaternion itself is continuous straight through it."""
        n = len(self.samples)
        if n == 0:
            return None
        if n == 1:
            s = self.samples[0]
            return s.pos.copy(), s.quat.copy()

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
        return pos, quat_slerp(a.quat, b.quat, local_t)

    def _raw_sample(self, t):
        """Euler view of _raw_sample_quat -- (pos, yaw, pitch, roll). Only
        for display/back-compat callers; anything that feeds the renderer
        should use sample_quat so it doesn't lose orientation at a
        gimbal-lock pose (see _raw_sample_quat)."""
        s = self._raw_sample_quat(t)
        if s is None:
            return None
        pos, q = s
        yaw, pitch, roll = euler_from_quat(q)
        return pos, float(yaw), float(pitch), float(roll)

    def _inertia_signature(self):
        return (id(self.samples), len(self.samples), self.total_duration(),
                self.inertia, self.inertia_bounce)

    def _build_inertia_cache(self):
        """Same approach as CameraPath._build_inertia_cache (see its
        docstring for the full rationale: fixed-grid precompute for
        determinism under motion-blur sub-sampling / multi-GPU time
        blocks) -- position spring + quaternion rotational spring
        (_quat_spring_step) chasing the raw recorded trajectory. The grid
        stores QUATERNIONS, not Euler angles, so neither the simulation
        nor the later lookup can be corrupted by a gimbal-lock pose
        anywhere along the recording."""
        n = len(self.samples)
        if n < 2 or self.inertia <= 1e-6:
            self._inertia_cache = None
            return
        total = self.total_duration()
        if total <= 1e-6:
            self._inertia_cache = None
            return
        sim_hz = 240.0
        sim_dt = 1.0 / sim_hz
        pre_roll = 1.5
        post_roll = 1.5
        t0 = -pre_roll
        t1 = total + post_roll
        n_steps = max(2, int(math.ceil((t1 - t0) / sim_dt)) + 1)

        time_constant = 0.08 + 0.6 * min(self.inertia, 3.0)
        omega_n = 1.0 / max(1e-3, time_constant)
        zeta = max(0.05, 1.0 - max(0.0, min(1.0, self.inertia_bounce)) * 0.85)
        k = omega_n * omega_n
        c = 2.0 * zeta * omega_n

        r0_pos, r0_quat = self._raw_sample_quat(max(0.0, t0))
        pos = r0_pos.astype(np.float64).copy()
        quat = r0_quat.copy()
        vel_pos = np.zeros(3, dtype=np.float64)
        ang_vel = np.zeros(3, dtype=np.float64)

        grid_pos = np.empty((n_steps, 3), dtype=np.float32)
        grid_quat = np.empty((n_steps, 4), dtype=np.float64)

        for i in range(n_steps):
            t = t0 + i * sim_dt
            t_query = max(0.0, min(total, t))
            r_pos, r_quat = self._raw_sample_quat(t_query)
            r_pos = r_pos.astype(np.float64)

            acc_pos = k * (r_pos - pos) - c * vel_pos
            vel_pos = vel_pos + acc_pos * sim_dt
            pos = pos + vel_pos * sim_dt

            quat, ang_vel = _quat_spring_step(quat, ang_vel, r_quat, k, c, sim_dt)

            grid_pos[i] = pos
            grid_quat[i] = quat

        # Hemisphere-align consecutive grid quaternions so the slerp done
        # at lookup time below always takes the short way round (q and -q
        # are the same rotation, but lerping between them is not).
        for i in range(1, n_steps):
            if float(np.dot(grid_quat[i - 1], grid_quat[i])) < 0.0:
                grid_quat[i] = -grid_quat[i]

        self._inertia_cache = {
            'sig': self._inertia_signature(),
            't0': t0, 'dt': sim_dt, 'n': n_steps,
            'pos': grid_pos, 'quat': grid_quat,
        }

    def sample_quat(self, t):
        """(pos, quaternion) at time t, with inertia smoothing applied
        when self.inertia > 0. This is the form the renderer should use --
        it is exactly what the recording says, with no Euler decomposition
        anywhere in the chain, so a recording that passes through
        pitch = +-90deg replays correctly instead of snapping (see
        _raw_sample_quat)."""
        if len(self.samples) < 2 or self.inertia <= 1e-6:
            return self._raw_sample_quat(t)
        if self._inertia_cache is None or self._inertia_cache['sig'] != self._inertia_signature():
            self._build_inertia_cache()
        cache = self._inertia_cache
        if cache is None:
            return self._raw_sample_quat(t)
        t0, dt, n = cache['t0'], cache['dt'], cache['n']
        f = (t - t0) / dt
        i0 = max(0, min(n - 1, int(math.floor(f))))
        i1 = max(0, min(n - 1, i0 + 1))
        local = max(0.0, min(1.0, f - i0))
        pos = cache['pos'][i0] + (cache['pos'][i1] - cache['pos'][i0]) * local
        quat = quat_slerp(cache['quat'][i0], cache['quat'][i1], local)
        return pos.astype(np.float32), quat

    def sample(self, t):
        """Euler view of sample_quat -- (pos, yaw, pitch, roll), for
        display and for back-compatibility with callers written against
        the old signature. Prefer sample_quat anywhere the result drives
        the camera."""
        s = self.sample_quat(t)
        if s is None:
            return None
        pos, q = s
        yaw, pitch, roll = euler_from_quat(q)
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

    def sample_quat(self, t=None):
        """(pos, quaternion) for the most recent pose -- the form the
        renderer should use. Like SensorCameraData.sample_quat, the blend
        between the last two packets is a quaternion SLERP rather than
        per-axis angle lerps, so a stream that passes through
        pitch = +-90deg doesn't snap there. Returns None until at least
        one packet has arrived."""
        self._drain()
        if self._latest is None:
            return None
        if self._prev is None or self.smoothing <= 0.0:
            s = self._latest
            return s.pos.copy(), s.quat.copy()
        a, b = self._prev, self._latest
        blend = 1.0 - self.smoothing
        pos = a.pos * (1.0 - blend) + b.pos * blend
        return pos, quat_slerp(a.quat, b.quat, blend)

    def sample(self, t=None):
        """Euler view of sample_quat -- (pos, yaw, pitch, roll), kept for
        interface-compatibility with the old signature. t is accepted (to
        match CameraPath/SensorCameraData) but ignored: a live stream has
        no timeline to seek, only a "most recent pose"."""
        s = self.sample_quat(t)
        if s is None:
            return None
        pos, q = s
        yaw, pitch, roll = euler_from_quat(q)
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


def merge_glass_spheres(spheres, max_clusters=MAX_GLASS_CLUSTERS, pad=1.05):
    """Merges overlapping bounding spheres (so photons aimed at neighbouring
    glass blocks aren't counted twice where their discs overlap) and then, if
    there are still more than max_clusters, the closest pairs until it fits."""
    sp = [(np.asarray(c, dtype=np.float64), float(r) * pad) for c, r in spheres]

    def enclose(a, b):
        (c1, r1), (c2, r2) = a, b
        d = float(np.linalg.norm(c2 - c1))
        if d + r2 <= r1:
            return a
        if d + r1 <= r2:
            return b
        R = (d + r1 + r2) / 2.0
        return (c1 + (c2 - c1) * ((R - r1) / d), R)

    def merge_pass(only_overlapping):
        best = None
        for i in range(len(sp)):
            for j in range(i + 1, len(sp)):
                gap = float(np.linalg.norm(sp[i][0] - sp[j][0])) - sp[i][1] - sp[j][1]
                if only_overlapping and gap >= 0.0:
                    continue
                if best is None or gap < best[0]:
                    best = (gap, i, j)
        if best is None:
            return False
        _, i, j = best
        merged = enclose(sp[i], sp[j])
        sp.pop(j); sp.pop(i)
        sp.append(merged)
        return True

    while len(sp) > 1 and merge_pass(True):
        pass
    while len(sp) > max_clusters and merge_pass(False):
        pass
    return sp


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
        # Canonical orientation storage -- see camera_quat/camera_rot/
        # camera_roll properties below. camera_rot/camera_roll (yaw/pitch,
        # roll) are still fully readable/writable exactly as before; they
        # just now go through this quaternion instead of being the
        # storage themselves.
        self._camera_quat = quat_identity()
        self.water_time = 0.0   # seconds -- 0.0 = water ripples stay still (still image/live
                                 # raytrace, same as before); render_video() updates this value
                                 # EVERY frame (or every sub-sample when motion blur is on) so
                                 # the water "moves".
        self.ambient = 0.12
        self.specular_k = 0.6
        self.shininess = 64.0
        self.sky_light_strength = 0.5   # hemispherical sky-fill strength (see render_sample)
        self.caustic_strength = 1.0     # runtime contrast dial on top of CAUSTIC_MAX_MULT
        self.light_softness = 0.012     # light angular radius (tan): ~0.7 deg, about the real sun's
                                        # size -> soft penumbrae. 0 = old razor-sharp point-light shadows.

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
        self.sun_light = None  # set by add_sun_light/add_sun -- lets set_sun_angle move it later

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
        # Glass caustics (photon pass for glass solids -- see compute_glass_caustics).
        # Emissive faces (screens/lamps) as area lights, and video playback state.
        self.emissive_lighting_enabled = True   # False = glow is visible but lights nothing else
        self.emissive_light_strength = 1.0      # runtime dial on the light emissive faces cast
        self.video_playing = True               # SPACE toggles this in the live view
        self.video_time = 0.0                   # shared playback clock for every video plane (s)
        self.glass_caustics_enabled = True
        self.glass_caustic_strength = 1.0
        self.glass_caustic_photons = 65536   # photons per (glass cluster, light); 5-10x for final renders
        self.glass_caustic_reach = 6.0       # receiver map half-size = reach * cluster radius
        self._glass_sig = None
        alloc_glass_fields()
        alloc_water_fields()
        self._sync_water_blocks()
        self.compute_caustics()

        self.clouds = None  # see set_clouds -- disabled by default
        # How often the interactive live view re-bakes the star field's
        # twinkle (seconds). 0 disables the update entirely, freezing the
        # field as baked -- see update_stars / set_star_update_interval.
        # Previously hard-coded to 1.0 in the interactive loop.
        self.star_update_interval = 0.0
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
        self.sun_light = light  # so set_sun_angle can find and move it later
        self.sync_lights()
        self.compute_glass_caustics()   # glass caustics need to know about the new light
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

    def set_sun_angle(self, azimuth_deg=None, elevation_deg=None):
        """Moves the sun/moon to a new azimuth/elevation -- updates the
        visual disc (Background.set_sun), the self-shadow direction
        clouds use (if any), AND, if add_sun_light/add_sun created a real
        light for this sun (tracked as self.sun_light), repositions that
        light too, at the SAME distance it was originally placed --
        without this, azimuth/elevation would be three separate things
        (disc, cloud shadows, actual light) to keep in sync by hand every
        time the sun moves. Pass just one of azimuth_deg/elevation_deg to
        leave the other as-is. No effect if no sun has been set."""
        if self.background.sun is None:
            return
        az = float(azimuth_deg) if azimuth_deg is not None else self.background.sun['azimuth_deg']
        el = float(elevation_deg) if elevation_deg is not None else self.background.sun['elevation_deg']
        self.background.set_sun(**{**self.background.sun, 'azimuth_deg': az, 'elevation_deg': el})
        if BG_FIELD is not None:
            BG_FIELD.from_numpy(self.background.image)
        if self.clouds is not None:
            self._sync_cloud_fields()
        if self.sun_light is not None:
            distance = float(np.linalg.norm(self.sun_light.position))
            azr, elr = math.radians(az), math.radians(el)
            dx, dz = math.sin(azr) * math.cos(elr), math.cos(azr) * math.cos(elr)
            dy = math.sin(elr)
            self.sun_light.position = np.array([dx, dy, dz], dtype=np.float64) * distance
            self.sync_lights()
            self.compute_glass_caustics()
        self.reset_accumulation()

    def set_sun_appearance(self, **kwargs):
        """Updates any of the sun/moon disc's other visual fields (color,
        angular_size_deg, glow_deg, intensity) in one call and re-uploads
        the sky texture -- for azimuth_deg/elevation_deg use set_sun_angle
        instead (those also need to move a real light, if one exists, not
        just the visual disc). No effect if no sun has been set."""
        if self.background.sun is None:
            return
        self.background.set_sun(**{**self.background.sun, **kwargs})
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

    def set_sun_enabled(self, enabled):
        """Shows/hides the sun/moon disc (keeping all its settings -- see
        Background.set_sun_enabled), re-uploads the sky texture and, if a
        real sun Light exists (add_sun_light/add_sun), mutes/unmutes that
        too, so the toggle covers both the visual disc and the light it
        casts rather than leaving a shadow-casting sun with no visible
        disc. The light's brightness is remembered while muted."""
        enabled = bool(enabled)
        self.background.set_sun_enabled(enabled)
        if self.sun_light is not None:
            if enabled:
                restored = getattr(self, '_sun_light_brightness', None)
                if restored is not None:
                    self.sun_light.brightness = restored
            else:
                self._sun_light_brightness = float(self.sun_light.brightness)
                self.sun_light.brightness = 0.0
            self.sync_lights()
        if BG_FIELD is not None:
            BG_FIELD.from_numpy(self.background.image)
        if self.clouds is not None:
            self._sync_cloud_fields()
        self.reset_accumulation()

    def toggle_sun(self):
        """Flips the sun/moon on/off -- see set_sun_enabled. Returns the
        new state."""
        self.set_sun_enabled(not self.background.sun_enabled)
        return self.background.sun_enabled

    def set_stars_enabled(self, enabled):
        """Shows/hides the star field (keeping all its settings -- see
        Background.set_stars_enabled) and re-uploads the sky texture."""
        self.background.set_stars_enabled(bool(enabled))
        if BG_FIELD is not None:
            BG_FIELD.from_numpy(self.background.image)
        self.reset_accumulation()

    def toggle_stars(self):
        """Flips the star field on/off -- see set_stars_enabled. Returns
        the new state."""
        self.set_stars_enabled(not self.background.stars_enabled)
        return self.background.stars_enabled

    def set_star_update_interval(self, seconds):
        """How often the interactive live view advances the star twinkle
        (seconds). 0 turns the update off entirely -- the field stays
        exactly as baked, which is also the cheapest option (each update
        re-bakes and re-uploads the sky texture, and resets live
        accumulation). Has no effect on offline video renders, which
        sample the twinkle at each frame's own timestamp instead."""
        self.star_update_interval = max(0.0, float(seconds))

    def update_stars(self, t):
        """Advances the star field's twinkle to time t (seconds) and
        re-uploads the sky texture -- see Background.update_stars. Cheap,
        but there's no reason to call this every single frame for an
        effect that's meant to read as occasional, not a strobe; the
        interactive live view throttles its own calls to roughly once a
        second (see the main loop) -- call it yourself at whatever
        cadence suits a script or video render. No effect if no star
        field has been set (or if the field is currently hidden -- see
        set_stars_enabled)."""
        if self.background.stars is None or not self.background.stars_enabled:
            return
        self.background.update_stars(t)
        if BG_FIELD is not None:
            BG_FIELD.from_numpy(self.background.image)
        self.reset_accumulation()

    def set_clouds(self, enabled=True, base=100.0, top=160.0, density=0.08, coverage=0.5,
                   scale=60.0, color=(255, 255, 255), steps=24, light_steps=4, wind=(0.0, 0.0),
                   billow=0.0, wind_speed=(0.0, 0.0), billow_speed=0.0):
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
          (e.g. from render_video) to drift the cloud shapes sideways
          without regenerating them (the noise itself doesn't change, just
          where in it you're sampling, so shapes stay self-consistent frame
          to frame instead of flickering).
        billow: offset added to the noise field's OWN vertical sampling
          axis (independent of world Y, which stays fixed per point) --
          see set_cloud_billow for animating this over time. Unlike wind
          (a rigid xz translation that slides shapes without changing
          them), advancing billow moves which cross-section of the noise
          volume gets sampled, so puffs slowly swell/split/merge instead
          of just sliding -- combine both for clouds that drift AND billow.
        wind_speed/billow_speed: (x, z) world-units/second and units/second
          -- continuous drift/billow speeds; see set_cloud_drift for the
          convenience call that sets these two without having to repeat
          every other cloud parameter. Stored on self.clouds (so they
          round-trip through save_scene_file/load_scene_file like every
          other cloud setting) rather than as separate attributes.

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
                       'wind': [float(wind[0]), float(wind[1])], 'billow': float(billow),
                       'wind_speed': [float(wind_speed[0]), float(wind_speed[1])],
                       'billow_speed': float(billow_speed)}
        self._sync_cloud_fields()
        # (Re-)anchor the drift/billow animation (see sample_cloud_drift) to
        # the wind/billow POSITION this call just set. set_clouds is the one
        # place every cloud change funnels through -- a fresh scene, a
        # reloaded one, or the F5 menu tweaking some unrelated cloud slider
        # (which re-spreads the whole tracer.clouds dict back through here)
        # -- so anchoring here, instead of only in set_cloud_drift, means a
        # reloaded scene resumes animating from its saved position instead
        # of snapping back to 0, and an unrelated menu edit doesn't reset
        # the animation's clock either.
        self._cloud_wind_base = tuple(self.clouds['wind'])
        self._cloud_billow_base = float(self.clouds['billow'])
        self._cloud_drift_t0 = None  # re-anchored on the next sample_cloud_drift call
        self.reset_accumulation()
        return self

    def set_cloud_wind(self, wind):
        """Updates just the wind offset (see set_clouds) without touching
        any other cloud parameter or resetting the whole config -- the
        cheap call to make from a per-frame animation loop / render_video
        for slowly drifting clouds sideways. No effect if set_clouds hasn't
        been called. See set_cloud_billow for the complementary "shapes
        actually change" axis, or set_cloud_drift for both at once."""
        if self.clouds is None:
            return
        self.clouds['wind'] = [float(wind[0]), float(wind[1])]
        CLOUD_WIND[None] = list(self.clouds['wind'])
        self.reset_accumulation()

    def set_cloud_billow(self, billow):
        """Updates just the billow offset (see set_clouds) without touching
        any other cloud parameter -- the cheap call to make from a
        per-frame animation loop / render_video to make cloud shapes churn/
        morph over time (as opposed to set_cloud_wind, which slides the
        same rigid shapes sideways). No effect if set_clouds hasn't been
        called."""
        if self.clouds is None:
            return
        self.clouds['billow'] = float(billow)
        CLOUD_TIME[None] = self.clouds['billow']
        self.reset_accumulation()

    def set_cloud_drift(self, wind_speed=(0.0, 0.0), billow_speed=0.0):
        """Configures continuous cloud animation: (x, z) wind_speed (world
        units/second) drifts the cloud shapes sideways over time; billow_speed
        (units/second) instead makes the puffs churn/swell/merge, rather than
        just translate (see set_clouds' docstring for why they look
        different). Stores both ON tracer.clouds -- via set_clouds, which
        also (re-)anchors the animation to the CURRENT wind/billow position
        -- so they save/reload with the rest of the cloud config instead of
        being lost as transient state. Call sample_cloud_drift(t) once per
        frame to actually apply it. A gentle real-world-ish look is usually
        a slow wind_speed (a few world units/second, in whatever direction
        you want the weather moving) plus a much smaller billow_speed
        (shapes evolving is a slower, subtler effect than drifting across
        the sky). No effect if set_clouds hasn't been called."""
        if self.clouds is None:
            return self
        return self.set_clouds(**{**self.clouds, 'wind_speed': wind_speed, 'billow_speed': billow_speed})

    def sample_cloud_drift(self, t, reset=True):
        """Applies the drift/billow speeds configured by set_cloud_drift (or
        loaded from a scene file -- see set_clouds) at absolute time `t`
        (seconds): wind = base + wind_speed * elapsed, billow = base +
        billow_speed * elapsed, where `base` is the wind/billow POSITION
        set_clouds last anchored to and `elapsed` is measured from whatever
        `t` was passed on the FIRST call after that anchor -- so the
        interactive view can pass wall-clock time.time() and render_video
        can pass each frame's exact (sub-frame-shutter-aware) sample
        timestamp, the same way spotlight animations and water-ripple time
        already work in this file, and either way the very first call after
        an anchor is a no-op offset rather than a jump.

        reset: whether to also call reset_accumulation() here. Leave this
        True for the interactive view (the default). Pass False when
        sampling repeatedly WITHIN one already-reset frame -- e.g.
        render_video's motion-blur sub-frame loop, which calls this once
        per shutter subsample and relies on those subsamples ACCUMULATING
        together (that's what makes the blur); resetting on every one of
        them would wipe out everything but the last subsample instead.

        Returns True if anything actually changed, so callers can OR it
        into their `any_input`/reset-accumulation flag -- same as any other
        animated element (spotlights, stars). No-op (returns False) if
        clouds are disabled or wind_speed/billow_speed are both zero."""
        if self.clouds is None or not self.clouds.get('enabled'):
            return False
        wx, wz = self.clouds.get('wind_speed', (0.0, 0.0))
        bs = self.clouds.get('billow_speed', 0.0)
        if wx == 0.0 and wz == 0.0 and bs == 0.0:
            return False
        if getattr(self, '_cloud_drift_t0', None) is None:
            self._cloud_drift_t0 = t
        elapsed = t - self._cloud_drift_t0
        wind_base = getattr(self, '_cloud_wind_base', tuple(self.clouds['wind']))
        billow_base = getattr(self, '_cloud_billow_base', self.clouds.get('billow', 0.0))
        self.clouds['wind'] = [float(wind_base[0] + wx * elapsed), float(wind_base[1] + wz * elapsed)]
        self.clouds['billow'] = float(billow_base + bs * elapsed)
        CLOUD_WIND[None] = list(self.clouds['wind'])
        CLOUD_TIME[None] = self.clouds['billow']
        if reset:
            self.reset_accumulation()
        return True

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
        CLOUD_TIME[None] = c.get('billow', 0.0)
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

    # --- Video playback ---------------------------------------------------
    @property
    def has_videos(self):
        return bool(self.scene.textures.videos)

    def set_video_time(self, t):
        """Shows each video at playback time t (seconds, looping). Decodes + uploads a
        texture ONLY for videos whose frame actually changed, so calling this every
        frame is cheap. Returns True if anything changed (callers should then restart
        progressive accumulation). Deterministic in t -- offline renders rely on that."""
        self.video_time = float(t)
        changed = False
        for tex_id, src in self.scene.textures.videos.items():
            idx = src.index_at(self.video_time)
            if idx == src.uploaded_idx:
                continue
            arr = src.frame(idx)
            if arr is None:
                continue
            upload_texture_u8(tex_id, arr)
            src.uploaded_idx = src.cur_idx      # after wrap-learning this may differ from idx
            changed = True
        return changed

    def update_videos(self, dt):
        """Live-view tick: advances the shared clock by dt while playing (paused =
        the clock and every frame stay put). True if a new frame was uploaded."""
        if not self.has_videos or not self.video_playing:
            return False
        return self.set_video_time(self.video_time + dt)

    def toggle_video_playback(self):
        if not self.has_videos:
            return None
        self.video_playing = not self.video_playing
        print(f"Video: {'playing' if self.video_playing else 'paused'} at {self.video_time:.2f}s")
        return self.video_playing

    def _glass_signature(self):
        r = lambda a: tuple(round(float(x), 3) for x in np.asarray(a).ravel())
        return (self.glass_caustic_photons, round(self.glass_caustic_reach, 3),
                len(self.scene.glass_blocks),
                tuple((r(l.position), r(l.color), round(l.brightness, 3)) for l in self.lights),
                tuple((r(s.position), r(s.direction), r(s.color), round(s.brightness, 3),
                       round(s.cone_angle, 2), round(s.softness, 3)) for s in self.spotlights))

    def compute_glass_caustics(self, force=False):
        """Photon pass for glass solids (see the comment block above
        _gc_trace). Cheap to call repeatedly: it returns immediately unless
        the lights/photon settings changed since the last run (or force).
        Prints its own timing so you can see the real cost on your hardware."""
        if not self.glass_caustics_enabled or not self.scene.glass_blocks:
            N_GC[None] = 0
            return
        sig = self._glass_signature()
        if not force and sig == self._glass_sig:
            return
        self._glass_sig = sig
        t0 = time.time()
        spheres = [(c, r) for c, r in merge_glass_spheres(self.scene.glass_blocks)
                   if r <= GLASS_MAX_CLUSTER_RADIUS]
        n = len(spheres)
        n_lights, n_spots = len(self.lights), len(self.spotlights)
        if n == 0 or (n_lights + n_spots) == 0:
            N_GC[None] = 0
            return
        centers = np.zeros((MAX_GLASS_CLUSTERS, 3), dtype=np.float32)
        radii = np.zeros(MAX_GLASS_CLUSTERS, dtype=np.float32)
        for i, (c, r) in enumerate(spheres):
            centers[i], radii[i] = c, r
        GC_CENTER.from_numpy(centers)
        GC_RADIUS.from_numpy(radii)
        GC_HALF.from_numpy((radii * float(self.glass_caustic_reach)).astype(np.float32))
        N_GC[None] = n
        glass_probe_receivers(n)
        GLASS_ACC.fill(0.0)
        self.sync_lights()
        glass_caustics_kernel(n, n_lights, n_spots, int(self.glass_caustic_photons))
        # two separable [1 2 1] blur passes: ACC -> TMP -> MAP -> TMP -> MAP
        glass_blur_h(n, GLASS_ACC, GLASS_TMP)
        glass_blur_v(n, GLASS_TMP, GLASS_MAP)
        glass_blur_h(n, GLASS_MAP, GLASS_TMP)
        glass_blur_v(n, GLASS_TMP, GLASS_MAP)
        ti.sync()
        print(f"Glass caustics: {n} cluster(s), {n_lights} light(s) + {n_spots} spot(s), "
              f"{int(self.glass_caustic_photons)} photons each -> {(time.time() - t0) * 1000:.0f} ms")

    def compute_caustics(self):
        """Recomputes the caustic map (photon pre-pass). Called once at
        startup, and again whenever a point Light's position changes (e.g.
        after pressing K) -- does NOT depend on image resolution so it does
        not need to be recomputed when switching between live/final resolution."""
        self.compute_glass_caustics()   # no-op unless lights/settings changed (see its docstring)
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
        # Straight from the canonical quaternion -- camera_rot/camera_roll
        # would decompose and recompose the same rotation on every call.
        R = quat_to_matrix(self.camera_quat)
        cp = self.camera_pos
        self.sync_lights()
        WATER_TIME[None] = float(self.water_time)
        LIGHT_SOFT[None] = float(self.light_softness)
        EM_STRENGTH[None] = float(self.emissive_light_strength) if self.emissive_lighting_enabled else 0.0
        GLASS_STRENGTH[None] = float(self.glass_caustic_strength) if self.glass_caustics_enabled else 0.0
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
        """Eye-adaptation metering -- see _measure_target_exposure_numpy for
        what is metered and why. Runs as a GPU row-sum reduction straight
        off the ACCUM field (the NumPy version below pulled the ENTIRE
        MAX_W x MAX_H x 3 float buffer to the host every frame); falls back
        to that NumPy version if GPU post-fx is off or the kernel fails."""
        global GPU_POSTFX
        if GPU_POSTFX and self.width >= 2 and self.height >= 2:
            try:
                n = max(1, int(SAMPLE_COUNT[None]))
                avg = gpu_measure_log_luma(self.width, self.height, n)
                if avg < 1e-5:
                    return self.exposure
                return float(np.clip(target_luma / avg, 0.05, 20.0))
            except Exception as e:
                print(f"\nWarning: GPU post-processing failed ({e!r}); "
                      f"falling back to the CPU/NumPy post-fx path.")
                GPU_POSTFX = False
        return self._measure_target_exposure_numpy(target_luma)

    def _measure_target_exposure_numpy(self, target_luma=0.2):
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

    def _resolve_gpu(self):
        """Meter (if eye adaptation is on) + resolve/tonemap ACCUM into
        OUTPUT/DEPTH_OUT. Nothing is read back to the host."""
        if self.eye_adapt_enabled:
            # Live-preview / still-image metering: adjust INSTANTLY (no lag)
            # -- see render_video for the version that lags on purpose.
            self.exposure = self.measure_target_exposure()
        resolve_output(self.width, self.height, float(self.exposure))

    def current_image_float(self):
        """CPU readback of the resolved frame (float colour + depth). Only
        needed by the CPU post-fx fallback now -- the GPU path never calls it."""
        self._resolve_gpu()
        color = np.clip(OUTPUT.to_numpy()[:self.height, :self.width], 0.0, 1.0)
        depth = DEPTH_OUT.to_numpy()[:self.height, :self.width]
        return color, depth

    def _present_gpu(self, post_fx, post_fx_on, frame_seed=0, flares=None, camera_pose=None,
                     motion_px=None, out_size=None, bgr=False, hud=None):
        """Resolve + the whole post-fx chain + 8-bit conversion, all on the
        GPU (see gpu_post_process). Returns a uint8 (h, w, 3) array, or None
        if GPU post-fx is disabled or just failed (it is then switched off
        for the rest of the run, and the caller takes the CPU path)."""
        global GPU_POSTFX
        if not GPU_POSTFX:
            return None
        try:
            self._resolve_gpu()
            return gpu_post_process(self.width, self.height, flares, post_fx,
                                    frame_seed=frame_seed, camera_pose=camera_pose,
                                    motion_px=motion_px, post_fx_on=post_fx_on,
                                    out_size=out_size, bgr=bgr, hud=hud)
        except Exception as e:
            print(f"\nWarning: GPU post-processing failed ({e!r}); "
                  f"falling back to the CPU/NumPy post-fx path for the rest of this run.")
            GPU_POSTFX = False
            return None

    def present_frame(self, post_fx, post_fx_on=True, frame_seed=0, flares=None,
                      camera_pose=None, motion_px=None, out_size=None, bgr=False, hud=None):
        """Finished 8-bit frame (uint8, RGB or BGR, optionally resized to
        out_size = (w, h)) from the current accumulation buffer: GPU chain
        if available, otherwise the original NumPy chain. `hud` is this frame's
        HudFrame (see make_hud_frame) or None."""
        img = self._present_gpu(post_fx, post_fx_on, frame_seed, flares, camera_pose,
                                motion_px, out_size, bgr, hud)
        if img is not None:
            return img
        color, depth = self.current_image_float()
        fx_active = bool(post_fx_on and post_fx)
        hud_fx = hud if (hud is not None and hud.fx and fx_active) else None
        overlay = hud if (hud is not None and hud_fx is None and hud.sprites) else None
        if post_fx_on:
            color = apply_post_processing(color, depth, flares, post_fx, frame_seed=frame_seed,
                                          camera_pose=camera_pose, motion_px=motion_px, hud=hud_fx)
        img = (np.clip(color, 0.0, 1.0) * 255).astype(np.uint8)
        final_size = tuple(out_size) if out_size is not None else (img.shape[1], img.shape[0])
        overlay_done = False
        if overlay is not None and (overlay.w, overlay.h) != final_size:
            overlay.blend_u8(img, 'rgb')          # laid out for the render size: draw before any resize
            overlay_done = True
        if bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if out_size is not None and (img.shape[1], img.shape[0]) != tuple(out_size):
            img = cv2.resize(img, tuple(out_size), interpolation=cv2.INTER_LINEAR)
        if overlay is not None and not overlay_done:
            overlay.blend_u8(img, 'bgr' if bgr else 'rgb')
        return img

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
        post_on = bool(post_fx and post_fx.get('enabled', True))
        flares = camera_pose = None
        if post_on:
            R = quat_to_matrix(self.camera_quat)
            flares = compute_flare_list(self, self.camera_pos, R)
            camera_pose = (self.camera_pos, R, self.half_tan, self.aspect)
        hud = make_hud_frame(self, post_fx, t=(self.video_time if self.has_videos else 0.0),
                             frame=0, fps=0.0) if post_fx else None
        img = self.present_frame(post_fx, post_on, 0, flares, camera_pose, None, hud=hud)
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
            3rd angle) -- CameraPath.sample() returns the keyframes' roll (0.0 unless a keyframe was captured while rolled).
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

        post_fx = post_fx if post_fx is not None else copy.deepcopy(DEFAULT_POST_FX)
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
        # Videos in the scene auto-play from t=0 along the OUTPUT timeline (looping), no
        # matter whether the live view had them paused. Remember the live state to restore.
        _video_state_before = (self.video_time, self.video_playing)

        sync_note = " (camera-sync: 1 frame/sample)" if camera_sync else ""
        n_render = len(frame_range)
        subset_note = f" [{n_render}/{n_frames} frames, this worker's block]" if frame_subset is not None else ""
        print(f"Rendering video {resolution[0]}x{resolution[1]}, {n_frames} frames "
              f"({total_time:.2f}s @ {fps}fps), {samples_per_frame} sample(s)/frame{sync_note}{subset_note} ...")
        t0 = time.time()
        # Previous frame's (yaw, pitch, roll), for apply_analog_camera's motion
        # smear -- None on this worker's very first rendered frame (no prior
        # pose to diff against yet, so that frame gets no smear).
        prev_pose_q = None
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
                        pos, q = camera_path.sample_quat(tt)
                        self.camera_pos = pos.astype(np.float32)
                        # Quaternion straight in -- no Euler decomposition
                        # in the chain, so a recording that passes through
                        # pitch = +-90deg replays correctly (see
                        # SensorCameraData._raw_sample_quat).
                        self.camera_quat = q
                        self.water_time = tt * WATER_WAVE_SPEED
                        if self.has_videos:
                            self.set_video_time(tt)   # uploads only if this sub-sample crossed a video frame
                        # Sample spotlight animations at this exact
                        # sub-frame shutter time too, so an animated
                        # spotlight motion-blurs consistently with the
                        # camera instead of jumping once per whole frame.
                        self.set_spotlight_animation_time(tt)
                        # Same idea for cloud drift/billow (see
                        # set_cloud_drift/sample_cloud_drift) -- sampled at
                        # the exact sub-frame timestamp `tt`, not wall-clock,
                        # so drifting/billowing clouds motion-blur correctly
                        # and the video is reproducible regardless of how
                        # fast the machine renders it.
                        if self.clouds is not None:
                            self.sample_cloud_drift(tt, reset=False)
                        self.add_samples(1)
                else:
                    pos, q = camera_path.sample_quat(t_center)
                    self.camera_pos = pos.astype(np.float32)
                    self.camera_quat = q
                    self.water_time = t_center * WATER_WAVE_SPEED
                    if self.has_videos:
                        self.set_video_time(t_center)
                    # Deterministic, frame-exact spotlight animation (see
                    # SpotLightAnimation) -- NOT wall-clock based like the
                    # interactive view, so re-rendering the same video is
                    # reproducible regardless of how fast the machine runs.
                    self.set_spotlight_animation_time(t_center)
                    # Same determinism story for cloud drift/billow. reset=
                    # False -- the per-frame reset_accumulation() already
                    # happened above, before this frame's block started.
                    if self.clouds is not None:
                        self.sample_cloud_drift(t_center, reset=False)
                    self.add_samples(samples_per_frame)

                if do_eye_adapt:
                    target_exposure = self.measure_target_exposure()
                    self.exposure += (target_exposure - self.exposure) * adapt_k

                if do_autofocus:
                    R_af = quat_to_matrix(self.camera_quat)
                    forward = R_af[:, 2]
                    probe_depth(float(self.camera_pos[0]), float(self.camera_pos[1]), float(self.camera_pos[2]),
                                float(forward[0]), float(forward[1]), float(forward[2]))
                    target_focus = float(np.clip(PROBE_DEPTH[None], 0.3, 2000.0))
                    current_focus += (target_focus - current_focus) * af_k
                    post_fx['dof_focus_distance'] = current_focus

                curr_pose_q = self.camera_quat.copy()
                post_fx_on = post_fx.get('enabled', True)
                if post_fx_on:
                    R = quat_to_matrix(self.camera_quat)
                    flares = compute_flare_list(self, self.camera_pos, R)
                    camera_pose = (self.camera_pos, R, self.half_tan, self.aspect)
                    motion_px = _camera_motion_px(prev_pose_q, curr_pose_q,
                                                   self.width, self.height, self.half_tan, self.aspect)
                else:
                    flares, camera_pose, motion_px = None, None, None
                prev_pose_q = curr_pose_q
                hud = None
                if post_fx.get('hud_enabled', False):
                    t_stamp = float(frame_times[fi]) if camera_sync else fi * dt
                    lag = float(post_fx.get('hud_lag', 0.0))
                    past_q = cur_q_hud = None
                    if lag > 1e-6:
                        # Deterministic lag: ask the camera path where it pointed `lag` s ago.
                        cur_q_hud = camera_path.sample_quat(t_center)[1]
                        past_q = camera_path.sample_quat(max(0.0, t_center - lag))[1]
                    hud = make_hud_frame(self, post_fx, t=t_stamp, frame=fi, fps=fps,
                                         real_time=HUD_RT_BASE + t_stamp,
                                         past_quat=past_q, cur_quat=cur_q_hud)
                # GPU path (default): the whole post-fx chain runs as kernels
                # right behind this frame's raytrace, only the finished 8-bit
                # frame comes back, and it goes straight to the (threaded)
                # ffmpeg writer -- no float readback, no CPU post-fx, and no
                # need for the pipeline thread. Frame order is trivially
                # preserved (everything is on this thread).
                gpu_img = self._present_gpu(post_fx, post_fx_on, fi, flares, camera_pose, motion_px, hud=hud)
                if gpu_img is not None:
                    writer.submit(gpu_img)
                else:
                    # CPU fallback (--postfx-backend cpu, or a GPU kernel
                    # failed): the heavy NumPy post-fx goes to the background
                    # pipeline thread -- submit() returns right away, so this
                    # loop can go straight on to frame fi+1's raytrace kernels
                    # instead of blocking on bloom/DoF/VHS/etc. first. Snapshot
                    # post_fx now (dict(post_fx)) since autofocus/eye-adapt
                    # will keep mutating the live dict for later frames while
                    # this one is still queued. (If the GPU path failed
                    # part-way through the run, every earlier frame was
                    # already written, so ordering still holds.)
                    color, depth = self.current_image_float()
                    pipeline.submit(color, depth, flares, dict(post_fx), post_fx_on, fi, camera_pose, motion_px, hud)

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
        if self.has_videos:
            self.video_playing = _video_state_before[1]
            self.set_video_time(_video_state_before[0])   # back to wherever the live view was
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

    # --- Camera orientation: quaternion is canonical, Euler is a view -----
    # camera_rot (yaw, pitch) and camera_roll are kept as properties, not
    # plain attributes, so every existing call site that reads/writes them
    # (HUD text, menus, keyframe capture, --camera-data replay, etc.) keeps
    # working unchanged while the REAL storage/composition is the
    # quaternion in self._camera_quat -- see the block comment near
    # camera_matrix for why.
    @property
    def camera_quat(self):
        return self._camera_quat

    @camera_quat.setter
    def camera_quat(self, q):
        self._camera_quat = quat_normalize(np.asarray(q, dtype=np.float64))

    @property
    def camera_rot(self):
        """(yaw, pitch) radians, DERIVED from camera_quat for readability.
        Assigning rebuilds the quaternion from Euler angles, preserving
        the current roll."""
        yaw, pitch, _roll = euler_from_quat(self._camera_quat)
        return np.array([yaw, pitch], dtype=np.float32)

    @camera_rot.setter
    def camera_rot(self, value):
        yaw, pitch = float(value[0]), float(value[1])
        _y, _p, roll = euler_from_quat(self._camera_quat)
        self._camera_quat = quat_from_euler(yaw, pitch, roll)

    @property
    def camera_roll(self):
        """Roll (radians), DERIVED from camera_quat -- see camera_rot."""
        return euler_from_quat(self._camera_quat)[2]

    @camera_roll.setter
    def camera_roll(self, value):
        yaw, pitch, _old_roll = euler_from_quat(self._camera_quat)
        self._camera_quat = quat_from_euler(yaw, pitch, float(value))

    # --- Serialization of the "shot" (camera + look/render params) --------
    def camera_dict(self):
        yaw, pitch, roll = euler_from_quat(self._camera_quat)
        return {'pos': [float(x) for x in self.camera_pos],
                # yaw/pitch/roll kept for readability/backward compatibility
                # with older scene files that only understand Euler angles;
                # 'quat' is the exact orientation (avoids any Euler
                # round-trip loss) and is preferred on load when present.
                'yaw': float(yaw), 'pitch': float(pitch), 'roll': float(roll),
                'quat': [float(c) for c in self._camera_quat]}


class PostProcessPipeline:
    """CPU-FALLBACK ONLY since post-fx moved to the GPU (section 13b): the
    default render_video path no longer queues anything here. It is still
    used when --postfx-backend cpu is given or a GPU post-fx kernel fails.

    Runs apply_post_processing (DoF/bloom/god-rays/VHS/chromatic
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
                color, depth, flares, post_fx_snapshot, post_fx_on, fi, camera_pose, motion_px, hud = job
                hud_fx = hud if (hud is not None and hud.fx and post_fx_on) else None
                if post_fx_on:
                    color = apply_post_processing(color, depth, flares, post_fx_snapshot,
                                                   frame_seed=fi, camera_pose=camera_pose,
                                                   motion_px=motion_px, hud=hud_fx)
                img = (np.clip(color, 0.0, 1.0) * 255).astype(np.uint8)
                if hud is not None and hud_fx is None and hud.sprites:
                    hud.blend_u8(img, 'rgb')
                self._writer.submit(img)
        except Exception as e:
            self._error = e

    def submit(self, color, depth, flares, post_fx_snapshot, post_fx_on, fi, camera_pose, motion_px, hud=None):
        """color/depth: this frame's arrays from current_image_float() --
        fresh allocations each call (to_numpy()), so no aliasing risk with
        the next frame's arrays. post_fx_snapshot MUST be a fresh dict copy
        (not the live, still-being-mutated post_fx dict) -- autofocus/eye
        adaptation update post_fx's shared dict every frame in the main
        thread, and without a snapshot this thread could read a LATER
        frame's focus distance while processing an EARLIER frame."""
        if self._error is not None:
            raise RuntimeError(f"Post-processing pipeline failed: {self._error}")
        self._queue.put((color, depth, flares, post_fx_snapshot, post_fx_on, fi, camera_pose, motion_px, hud))

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
    env['RT_HUD_EPOCH'] = repr(HUD_RT_BASE)   # same {rh}{rm}{rs} origin in every worker
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
    applies post-processing every frame (on the GPU -- see section 13b). Uses tracer.live_max_bounce
    (a low bounce cap) instead of the scene's full-quality max_bounce --
    see item 5 in the module docstring."""

    def __init__(self, tracer: RayTracer, samples_per_frame=1, post_fx=None):
        self.tracer = tracer
        self.samples_per_frame = samples_per_frame
        self.active = False
        self.post_fx = post_fx if post_fx is not None else copy.deepcopy(DEFAULT_POST_FX)
        self._frame_seed = 0
        self._prev_pose_q = None  # camera quaternion -- see apply_analog_camera's motion smear

    def start(self):
        self.tracer.reset_accumulation()
        self.active = True

    def step(self, any_input, window_size, hud_t=0.0, hud_past_quat=None, hud_real_time=None):
        """Renders one live-preview frame and returns it as a BGR uint8 numpy
        array sized `window_size` = (w, h), ready for cv2.imshow."""
        if any_input:
            self.tracer.reset_accumulation()
        self.tracer.add_samples(self.samples_per_frame, max_bounce_override=self.tracer.live_max_bounce)

        R = quat_to_matrix(self.tracer.camera_quat)
        flares = compute_flare_list(self.tracer, self.tracer.camera_pos, R)
        self._frame_seed += 1
        camera_pose = (self.tracer.camera_pos, R, self.tracer.half_tan, self.tracer.aspect)
        curr_pose_q = self.tracer.camera_quat.copy()
        motion_px = _camera_motion_px(self._prev_pose_q, curr_pose_q,
                                       self.tracer.width, self.tracer.height,
                                       self.tracer.half_tan, self.tracer.aspect)
        self._prev_pose_q = curr_pose_q
        # Resolve + post-fx + BGR + resize-to-window all happen on the GPU
        # (falls back to the NumPy chain automatically) -- see present_frame.
        w, h = window_size
        hud = make_hud_frame(self.tracer, self.post_fx, t=hud_t, frame=self._frame_seed,
                             real_time=hud_real_time, past_quat=hud_past_quat, out_size=(w, h))
        return self.tracer.present_frame(self.post_fx, True, self._frame_seed, flares,
                                         camera_pose, motion_px, out_size=(w, h), bgr=True, hud=hud)

# =============================================================================
# 13) Post-processing -- lens flare, DoF, chromatic aberration,
#     fog, god rays, bloom, VHS, analog-camera film look. All parameters
#     default in DEFAULT_POST_FX, editable directly in code or at runtime
#     via the post_fx dict.
#     The functions in THIS section are the CPU / numpy implementation: the
#     reference + automatic fallback. The default path is the GPU version in
#     section 13b (gpu_post_process), which implements the same chain.
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
    'flare_ghosts': 0.8,          # aperture-shaped internal-reflection ghosts along the lens axis (0 = off)
    'flare_starburst': 0.35,      # diffraction spikes from the aperture blades (0 = off)
    'flare_blades': 6,            # aperture blade count (3..12; odd counts give twice as many spikes)
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
    'godrays_enabled': False,      # screen-space crepuscular rays / light shafts from visible lights through sky
    'godrays_intensity': 0.35,    # additive strength of the ray contribution
    'godrays_decay': 0.97,        # per-sample falloff along each ray (closer to 1 = rays reach further)
    'godrays_density': 0.9,       # sample step size as a fraction of the full source->pixel distance
    'godrays_samples': 24,        # radial samples per pixel per light -- higher = smoother rays, slower
    'godrays_sky_depth': 5000.0,  # depth (world units) above which a pixel counts as unoccluded "sky"
                                   # and can act as a ray source (matches the renderer's 1e4 miss sentinel)
    'vhs_enabled': False,         # VHS tape effect -- works for BOTH stills and video
    'vhs_strength': 1.0,          # overall VHS effect intensity (0..~2)
    'crt_enabled': False,         # CRT monitor look (applied after VHS) -- see apply_crt / _gp_crt
    'crt_curvature': 0.45,        # curved-glass warp, black outside the tube (0 = flat)
    'crt_scanlines': 0.55,        # scanline depth (0..1); the beam widens with pixel brightness
    'crt_lines': 240,             # virtual scanline count (capped at half the frame height)
    'crt_mask': 0.35,             # RGB phosphor-stripe strength
    'crt_glow': 0.35,             # phosphor glow around bright areas
    'crt_convergence': 0.8,       # R/B misconvergence toward the edges (px at 360p)
    'crt_vignette': 0.45,         # edge darkening
    'crt_flicker': 0.25,          # mains flicker + slow rolling bright bar
    'crt_brightness': 1.15,       # gain compensating the scanline/mask darkening
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
    'analog_grain': 0.00,         # grain amount (0..~0.15), stronger in shadows/midtones than highlights
    'analog_grain_size': 1.6,     # grain "clump" size in px -- 1 = fine per-pixel noise, >1 = coarser blobs
                                   # closer to real CCD/tape noise than flat per-pixel grain
    'analog_halation': 0.3,       # warm red-orange bleed around bright/blown-out highlights (0 = off)
    'analog_vignette': 0.2,       # soft optical vignette strength (0..~1)
    'analog_warmth': 0.00,        # warm-highlight/cool-shadow split-tone strength (0 = neutral)
    'analog_saturation': 1.0,     # chroma multiplier (1.0 = neutral, scaled by analog_strength). Deliberately
                                   # NOT capped at 1: values >1 push colours past the sensor's range so the
                                   # saturated channels clip -- hues lose gradation and slam toward the
                                   # primaries, the cramped dynamic range of a vintage camcorder. <1 washes out.
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
    'hud_enabled': True,          # render HUD (section 13c) -- F7 menu
    'hud_fx': False,              # True = HUD text goes through the FX chain, False = clean overlay on the finished frame
    'hud_lag': 0.0,               # seconds the HUD trails behind camera rotation (0 = fixed to the screen)
    'hud_elements': [dict(_e) for _e in (
        {'text': "{hh}:{mm}:{ss}\nZOOM x{zoom}", 'size': 14.0, 'x': 2.0, 'y': 3.0,
         'anchor': 'top-left', 'color': [255, 255, 255], 'opacity': 1.0},
        {'text': "+", 'size': 26.0, 'x': 50.0, 'y': 50.0,
         'anchor': 'center', 'color': [255, 255, 255], 'opacity': 1.0},
        {'text': "Noir", 'size': 18.0, 'x': 98.0, 'y': 97.0,
         'anchor': 'bottom-right', 'color': [255, 255, 255], 'opacity': 1.0},
    )],
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
    """Radial R/B channel split. Two fixes vs. the old version:
      1) BILINEAR resampling (cv2.remap) instead of nearest-neighbor integer
         indexing -- the old `.astype(np.int32)` truncation gave every
         shifted channel a crisp, un-blended copy of itself, so small bright
         features (a star, a rim-light highlight) didn't get a soft color
         fringe, they tore into fully separate red/green/blue dots, and
         every silhouette edge got a hard double line instead of gentle
         fringing. That's independent of sample count -- it happens to the
         final resolved image regardless of how many samples produced it,
         which is why it doesn't go away no matter how high you push
         --samples.
      2) The shift now eases in with an r^2 falloff from screen center
         instead of scaling linearly across the whole frame. Real lenses
         show essentially no fringing at the optical center and increasing
         separation toward the edges/corners; the old linear version gave
         every pixel a shift proportional to `strength` even a few hundred
         pixels from center, which is why content near the middle of frame
         (like a subject's face) showed just as much splitting as the
         corners."""
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r_max = math.sqrt(cx * cx + cy * cy)
    r_norm = np.clip(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(r_max, 1e-6), 0.0, 1.0)
    falloff = r_norm * r_norm  # 0 at center, 1 at the corners

    def sample(channel, sign):
        local_strength = strength * sign * falloff
        sx = (cx + (xx - cx) * (1.0 + local_strength)).astype(np.float32)
        sy = (cy + (yy - cy) * (1.0 + local_strength)).astype(np.float32)
        return cv2.remap(channel.astype(np.float32), sx, sy,
                          interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    r = sample(img[..., 0], 1.0)
    g = img[..., 1]
    b = sample(img[..., 2], -1.0)
    return np.stack([r, g, b], axis=-1)


# Lens-flare optics tables, shared by apply_lens_flare (NumPy) and _gp_lens_flare (GPU).
# Ghost: (position on the axis through the image centre as a multiple of the light's offset
# from the centre [1 = at the light, 0 = centre, negative = mirrored through it], radius as a
# multiple of the flare size, coating tint, relative strength).
_FLARE_GHOSTS = (
    (0.62, 0.55, (0.55, 0.80, 1.00), 0.55),
    (0.34, 0.80, (0.60, 1.00, 0.75), 0.40),
    (-0.28, 0.60, (1.00, 0.75, 0.45), 0.60),
    (-0.62, 1.30, (0.85, 0.55, 1.00), 0.45),
    (-1.00, 0.90, (0.50, 0.90, 1.00), 0.55),
    (1.45, 0.70, (1.00, 0.55, 0.75), 0.35),
)
_FLARE_DISP = (1.04, 1.0, 0.96)          # per-channel ghost size (R, G, B) -> coloured fringes
_FLARE_HALO_DISP = (0.94, 1.0, 1.06)     # per-channel halo-ring radius


def _smoothstep_np(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def apply_lens_flare(img, flares, flare_size, flare_intensity, anamorphic=0.0, halo=0.0,
                     ghosts=0.0, starburst=0.0, blades=6):
    """flares: list of (screen_x, screen_y, color[0-1] tuple, brightness).
    Optical model (the GPU kernel _gp_lens_flare mirrors this): a glow with a long veiling
    tail; diffraction spikes from the aperture blades (`starburst`); the anamorphic streak;
    a chromatically dispersed halo ring; and `ghosts` -- internal reflections on the axis
    through the image centre, drawn as aperture-shaped polygons with a bright rim, coating
    tints and per-channel dispersion, growing stronger as the light moves off-axis. All of
    it fades out smoothly as the light leaves the frame."""
    if not flares or flare_intensity <= 0:
        return img
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = img.copy()
    cx0, cy0 = (w - 1) / 2.0, (h - 1) / 2.0
    hd = 0.5 * math.hypot(w, h)
    R = max(2.0, flare_size * min(w, h) / 360.0)
    blades = int(max(3, min(16, round(blades))))
    seg = 2.0 * math.pi / blades
    star_m = blades if blades % 2 == 0 else blades * 2
    star_step = 2.0 * math.pi / star_m

    for (sx, sy, color, brightness) in flares:
        C = np.array(color, dtype=np.float32)
        e_edge = max(abs(sx - cx0) / (0.5 * w), abs(sy - cy0) / (0.5 * h))
        amp = flare_intensity * brightness * (1.0 - float(_smoothstep_np(1.0, 1.25, e_edge)))
        if amp <= 1e-5:
            continue
        dx, dy = xx - sx, yy - sy
        d2 = dx * dx + dy * dy
        dd = np.sqrt(d2)

        glow = (0.68 * np.exp(-d2 / (2.0 * R * R)) + 0.20 * np.exp(-d2 / (2.0 * (2.6 * R) ** 2))
                + 0.10 / (1.0 + d2 / (36.0 * R * R)))
        out += glow[..., None] * C * amp

        if starburst > 0.0:
            ang = np.arctan2(dy, dx) - math.pi / 2.0
            kk = np.floor(ang / star_step + 0.5)
            perp = dd * np.abs(np.sin(ang - kk * star_step))
            sig = max(0.55, R * 0.035)
            ray = np.exp(-(perp * perp) / (2.0 * sig * sig))
            along = np.exp(-dd / (R * 7.0)) / (1.0 + dd / (R * 1.2))
            ph = (dd / (R * 12.0))[..., None] + np.array([0.0, 0.33333, 0.66667], dtype=np.float32)
            rainbow = 0.5 + 0.5 * np.cos(2.0 * math.pi * ph)
            out += (ray * along)[..., None] * (C * 0.55 + rainbow * 0.45) * (amp * starburst * 0.9)

        if anamorphic > 0.0:
            streak_col = C * 0.5 + np.array([0.6, 0.75, 1.0], dtype=np.float32) * 0.5
            s1, s2 = R * 0.12, R * 0.55
            streak = (np.exp(-(dy * dy) / (2.0 * s1 * s1)) * np.exp(-np.abs(dx) / (R * 7.0))
                      + 0.22 * np.exp(-(dy * dy) / (2.0 * s2 * s2)) * np.exp(-np.abs(dx) / (R * 4.0)))
            out += streak[..., None] * streak_col * (amp * anamorphic)

        if halo > 0.0:
            rw = R * 0.32
            ring = np.stack([np.exp(-((dd - R * 2.4 * s) ** 2) / (2.0 * rw * rw))
                             for s in _FLARE_HALO_DISP], axis=-1)
            out += ring * (C * 0.6 + 0.4) * (amp * halo * 0.5)

        if ghosts > 0.0:
            axv, ayv = cx0 - sx, cy0 - sy
            r_off = math.hypot(axv, ayv) / hd
            gain = 0.25 + 0.75 * min(1.0, r_off * 1.6)
            rot = math.atan2(ayv, axv)
            for (gf, gsz, gtint, gst) in _FLARE_GHOSTS:
                gx, gy = cx0 + (sx - cx0) * gf, cy0 + (sy - cy0) * gf
                gr = R * gsz * 1.6 * (0.9 + 0.5 * r_off)
                ext = gr * 1.2
                x0, x1 = max(0, int(math.floor(gx - ext))), min(w, int(math.ceil(gx + ext)) + 1)
                y0, y1 = max(0, int(math.floor(gy - ext))), min(h, int(math.ceil(gy + ext)) + 1)
                if x0 >= x1 or y0 >= y1:
                    continue
                gdx, gdy = xx[y0:y1, x0:x1] - gx, yy[y0:y1, x0:x1] - gy
                rr = np.sqrt(gdx * gdx + gdy * gdy)
                a = np.arctan2(gdy, gdx) - rot
                am = a - seg * np.floor(a / seg)
                q = rr / (gr * math.cos(math.pi / blades) / np.cos(am - seg * 0.5))
                chans = []
                for dsp in _FLARE_DISP:
                    qc = q * dsp
                    chans.append((1.0 - _smoothstep_np(0.90, 1.10, qc))
                                 * (0.32 + 0.68 * _smoothstep_np(0.45, 1.0, qc)))
                val = np.stack(chans, axis=-1)
                out[y0:y1, x0:x1] += val * (C * np.array(gtint, dtype=np.float32)) * (amp * ghosts * gst * gain * 0.75)
    return out


def _flare_brightness_curve(b):
    """Soft-compresses a light's raw `brightness` before it drives the
    lens-flare/god-ray post-effects. Those effects are ADDITIVE on top of
    the already-resolved, already-ACES-tonemapped image (see
    apply_post_processing -> apply_lens_flare/apply_god_rays, both called
    after resolve_output), so unlike a surface's own shading -- which gets
    a graceful filmic highlight rolloff from render_sample/resolve_output --
    a raw, uncompressed brightness fed straight into the flare's glow/ghost/
    anamorphic-streak/halo intensities scales them linearly and without any
    ceiling. A sun/moon light in particular tends to carry a large
    brightness on purpose (point lights here have no distance falloff, see
    add_sun_light's docstring, so it needs enough of it to actually
    illuminate the scene from far away) -- fed raw into the flare, that
    turns into a huge, hard-edged, "supernova" blowout instead of a
    tasteful glow, for any strong light, sun/moon or otherwise. This caps
    how large a flare's contribution can ever get (saturating toward 3.0x)
    while staying close to linear for more modest brightness values, so
    ordinary point/spot lights still look proportionate to how bright they
    actually are."""
    b = max(0.0, b)
    return 3.0 * b / (3.0 + b)


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
            flares.append((sx, sy, tuple(lt.color), _flare_brightness_curve(lt.brightness)))

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
                flares.append((sx, sy, tuple(sl.color), _flare_brightness_curve(sl.brightness * 0.8)))
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


def apply_crt(img, post_fx, frame_seed=0):
    """CRT monitor look (the GPU kernel _gp_crt mirrors this): curved-glass warp with black
    outside the tube, R/B convergence error that grows toward the edges, phosphor glow,
    scanlines whose beam widens with brightness (so highlights fill their line), RGB
    phosphor stripes, vignette/bezel falloff, and mains flicker + a slow rolling bright bar
    (both a pure function of frame_seed, so video stays reproducible)."""
    if not post_fx.get('crt_enabled', False):
        return img
    h, w = img.shape[:2]
    curv = float(post_fx.get('crt_curvature', 0.0))
    scan = float(post_fx.get('crt_scanlines', 0.0))
    lines = float(min(max(10.0, float(post_fx.get('crt_lines', 240))), h * 0.5))
    mask = float(post_fx.get('crt_mask', 0.0))
    glow_amt = float(post_fx.get('crt_glow', 0.0))
    scale = min(w, h) / 360.0
    conv_px = float(post_fx.get('crt_convergence', 0.0)) * scale
    vig = float(post_fx.get('crt_vignette', 0.0))
    flick = float(post_fx.get('crt_flicker', 0.0))
    bright = float(post_fx.get('crt_brightness', 1.0))
    rng = np.random.default_rng(7000 + int(frame_seed))
    flick_gain = float(np.clip(1.0 + flick * 0.04 * rng.normal(), 0.85, 1.15))
    band_phase = float((int(frame_seed) * 0.0113) % 1.0)

    src = img.astype(np.float32)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ux = 2.0 * (xs + 0.5) / w - 1.0
    uy = 2.0 * (ys + 0.5) / h - 1.0
    sx = ux * (1.0 + curv * 0.10 * uy * uy)
    sy = uy * (1.0 + curv * 0.14 * ux * ux)
    if curv > 0.0:
        cov = np.clip(np.minimum((1.0 - np.abs(sx)) * 0.5 * w, (1.0 - np.abs(sy)) * 0.5 * h) + 0.5, 0.0, 1.0)
    else:
        cov = np.ones_like(sx)
    fx = ((sx * 0.5 + 0.5) * w - 0.5).astype(np.float32)
    fy = ((sy * 0.5 + 0.5) * h - 0.5).astype(np.float32)

    def samp(a, mx, my):
        return cv2.remap(a, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    col = samp(src, fx, fy)
    if conv_px != 0.0:
        off = conv_px * ux
        col[..., 0] = samp(src[..., 0], fx - off, fy)
        col[..., 2] = samp(src[..., 2], fx + off, fy)
    if glow_amt > 0.0:
        luma = src[..., 0] * 0.2126 + src[..., 1] * 0.7152 + src[..., 2] * 0.0722
        t = np.clip((luma - (0.55 - 0.2)) / 0.4, 0.0, 1.0)
        bp = src * (t * t * (3.0 - 2.0 * t))[..., None]
        r_g = max(1, int(round(5 * scale)))
        g = _box_blur(_box_blur(bp, r_g), r_g)
        col = col + samp(g.astype(np.float32), fx, fy) * glow_amt
    gain = bright
    v = sy * 0.5 + 0.5
    if scan > 0.0:
        l = np.clip(col[..., 0] * 0.2126 + col[..., 1] * 0.7152 + col[..., 2] * 0.0722, 0.0, 1.0)
        ph = v * lines
        dd = ph - np.floor(ph) - 0.5
        sg = 0.22 + 0.30 * l
        prof = np.exp(-(dd * dd) / (2.0 * sg * sg))
        col = col * (1.0 - scan * (1.0 - prof))[..., None]
        gain *= 1.0 + 0.45 * scan
    if mask > 0.0:
        tw = max(3.0, 2.25 * scale)
        xp = (xs + 0.5) / tw
        m3 = np.stack([0.5 + 0.5 * np.cos(2.0 * math.pi * (xp - o)) for o in (0.0, 0.33333, 0.66667)], axis=-1)
        col = col * ((1.0 - mask) + mask * 2.0 * m3)
    d2 = sx * sx + sy * sy
    vv = np.maximum(0.0, 1.0 - vig * 0.55 * d2)
    edge = np.maximum(np.abs(sx), np.abs(sy))
    vv = vv * (1.0 - vig * 0.35 * _smoothstep_np(0.7, 1.0, edge))
    band = (0.5 + 0.5 * np.cos(2.0 * math.pi * (v - band_phase))) ** 6
    fl = flick_gain * (1.0 + flick * 0.08 * band)
    return col * (gain * vv * fl * cov)[..., None]


def _fisheye_geometry(w, h, strength):
    """Shared fisheye parameters (CPU path, GPU kernel and the flare-position
    helper all use these so they can never disagree).

    Returns (a, rc): `a` is the lens "bend" angle in radians (0 = rectilinear
    / no distortion) and `rc` is the radius, in PIXELS, of the visible image
    circle. Everything farther than `rc` from the frame center is outside the
    lens' image circle and is rendered black (see _fisheye_circle_mask).

    The mapping (destination -> source, u = dst_radius / rc in [0, 1]) is
        src_radius = rc * tan(a * u) / tan(a)
    -- tan() is convex, so src_radius <= dst_radius everywhere inside the
    circle: the CENTER of the picture is magnified (a small source patch is
    spread over many output pixels) while the PERIPHERY is squeezed (a lot of
    source is packed into few output pixels). That is a true barrel / fisheye
    look. (The previous version used src = dst**(1/(1+k)), whose slope is
    zero at the center -- i.e. it MINIFIED the center and stretched the
    edges, the inverse of a fisheye.) Because src_radius <= dst_radius the
    source sample can never fall outside the rendered frame, so there are no
    replicated/smeared border pixels: the only region with "no image" is the
    area beyond the circle, which the mask blacks out.

    strength 0 -> rc = half the frame diagonal and a = 0 (identity);
    strength 1 -> rc = half the frame height (a full circle, like the
    reference fisheye photo) and a strong bend. In between it blends."""
    t = min(max(float(strength), 0.0), 1.0)
    half_h = h / 2.0
    half_diag = math.hypot(w / 2.0, half_h)
    rc = half_diag * (1.0 - t) + half_h * t
    a = 1.45 * t
    return a, rc


def _fisheye_circle_mask(h, w, rc):
    """(h, w) float32 mask: 1 inside the fisheye image circle, 0 outside, with
    a ~1px anti-aliased rim."""
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip(rc - r + 0.5, 0.0, 1.0).astype(np.float32)


def apply_fisheye(img, depth, strength):
    """True fisheye (barrel) distortion: magnified center, compressed
    periphery, circular image with the area outside the lens circle left to
    be blacked out (apply_post_processing re-applies the circle mask at the
    very end of the chain so later effects -- fog, bloom, grain -- can't
    paint over the black). See _fisheye_geometry for the maths.
    Returns (img, depth); depth is warped identically so depth-aware effects
    downstream (fog, god rays) stay aligned with the distorted image."""
    h, w = img.shape[:2]
    a, rc = _fisheye_geometry(w, h, strength)
    if a <= 1e-4:
        return img, depth
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx * dx + dy * dy)
    k = np.ones_like(r)
    inside = (r > 1e-6) & (r < rc)
    u = r[inside] / rc
    k[inside] = (rc * np.tan(a * u) / math.tan(a)) / r[inside]
    src_x = (cx + dx * k).astype(np.float32)
    src_y = (cy + dy * k).astype(np.float32)
    out_img = cv2.remap(img.astype(np.float32), src_x, src_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    out_depth = cv2.remap(depth.astype(np.float32), src_x, src_y, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    return out_img, out_depth


def _fisheye_warp_point(sx, sy, w, h, strength):
    """Maps a point's RECTILINEAR screen position to where it ends up after
    the fisheye warp (so lens-flare / god-ray sources still land on the light
    itself). Exact inverse of the destination->source mapping in
    _fisheye_geometry:  dst_radius = rc * atan(tan(a) * src_radius / rc) / a."""
    a, rc = _fisheye_geometry(w, h, strength)
    if a <= 1e-4:
        return sx, sy
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    dx, dy = sx - cx, sy - cy
    r = math.hypot(dx, dy)
    if r < 1e-6:
        return sx, sy
    r_dst = rc * math.atan(math.tan(a) * r / rc) / a
    k = r_dst / r
    return cx + dx * k, cy + dy * k


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


def apply_analog_motion_smear(img, motion_px, strength, max_px, text_mask=None):
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
    # text_mask (HUD text alpha, h x w): text pixels are neither smeared nor smeared INTO their
    # surroundings -- taps that land on text are dropped from the average (weights renormalised).
    if text_mask is not None:
        text_mask = np.clip(text_mask * 3.0, 0.0, 1.0)      # antialiased edge pixels count as text too
    wsrc = src if text_mask is None else src * (1.0 - text_mask)[..., None]
    wgt = None if text_mask is None else (1.0 - text_mask).astype(np.float32)
    acc = np.zeros_like(src)
    wacc = None if text_mask is None else np.zeros((h, w), dtype=np.float32)
    for i in range(n_taps):
        f = (i / (n_taps - 1)) - 0.5  # -0.5 .. +0.5 along the motion vector
        ox, oy = ux * length * f, uy * length * f
        M = np.float32([[1.0, 0.0, ox], [0.0, 1.0, oy]])
        acc += cv2.warpAffine(wsrc, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        if wacc is not None:
            wacc += cv2.warpAffine(wgt, M, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
    if text_mask is None:
        return acc / n_taps
    ok = wacc > 1e-4
    smeared = np.where(ok[..., None], acc / np.maximum(wacc, 1e-4)[..., None], src)
    return smeared * (1.0 - text_mask)[..., None] + src * text_mask[..., None]


def _camera_motion_px(prev_pose, curr_pose, w, h, half_tan, aspect):
    """Approximates the on-screen pixel shift a point near the center of
    frame would show between two camera poses -- used only to size
    apply_analog_motion_smear's streak, not for anything physically exact
    (a true per-pixel motion vector would also depend on scene depth and
    camera translation, which this ignores: rotation dominates the blur in
    handheld/camcorder footage, and that's the dominant term here).

    prev_pose/curr_pose are QUATERNIONS, and the shift is derived by
    rotating the previous frame's forward axis into the current frame and
    reading off where it lands, rather than by differencing yaw/pitch.
    That difference matters at a gimbal-lock pose (pitch = +-90deg, which
    real --camera-data recordings do pass through): there, yaw and roll
    are not separately determined, so a yaw-difference reads as a huge
    swing across a frame where the camera barely moved, and the analog
    smear would briefly streak the whole image. prev_pose is None for "no
    previous frame" (returns zero motion)."""
    if prev_pose is None:
        return 0.0, 0.0
    # Forward (+Z) of the previous pose, expressed in the CURRENT camera's
    # frame: R_curr^T @ R_prev @ [0,0,1].
    R_prev = quat_to_matrix(prev_pose)
    R_curr = quat_to_matrix(curr_pose)
    v = R_curr.T @ (R_prev @ np.array([0.0, 0.0, 1.0]))
    vz = float(v[2])
    if vz <= 1e-4:
        return 0.0, 0.0  # rotated past 90deg in one frame -- no sensible streak
    focal_px_x = (w / 2.0) / max(1e-4, half_tan * aspect)
    focal_px_y = (h / 2.0) / max(1e-4, half_tan)
    return float(v[0]) / vz * focal_px_x, -float(v[1]) / vz * focal_px_y


def apply_analog_camera(img, post_fx, frame_seed=0, motion_px=None, text_mask=None):
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
      4b) Saturation: scales chroma around luma by analog_saturation
         (1.0 = neutral; deliberately allowed ABOVE 1). Nothing is clamped
         here -- channels pushed past 1.0 (or under 0) are left to clip at
         the end of the chain, so saturated colours run out of range and
         flatten into hard primaries: the reduced dynamic range of a cheap
         vintage camcorder sensor.
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

    # 4b) Saturation -- may exceed 1.0; the overshoot is clipped by the final
    #     clip, which is what squeezes the dynamic range
    sat_eff = max(0.0, 1.0 + (post_fx.get('analog_saturation', 1.0) - 1.0) * strength)
    if abs(sat_eff - 1.0) > 1e-6:
        l_s = _luma(out)[..., None]
        out = l_s + (out - l_s) * sat_eff

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
        out = apply_analog_motion_smear(out, motion_px, smear, post_fx.get('analog_smear_max_px', 40),
                                        text_mask=text_mask)

    return out


def apply_post_processing(img, depth, flares, post_fx, frame_seed=0, camera_pose=None, motion_px=None, hud=None):
    """`hud`: an FX-mode HudFrame (hud.fx True) to composite right after DoF -- see section 13c."""
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
    text_mask = None
    if hud is not None and hud.sprites:
        out, depth = hud.blend_float(out, depth)
        if (post_fx.get('analog_enabled', False) and post_fx.get('analog_smear', 0.0) > 0.0
                and motion_px is not None):
            text_mask = hud.alpha_float(*out.shape[:2])
    fisheye_rc = None
    if post_fx.get('fisheye_enabled', False):
        fh, fw = out.shape[:2]
        strength = post_fx.get('fisheye_strength', 0.32)
        out, depth = apply_fisheye(out, depth, strength)
        _fa, _frc = _fisheye_geometry(fw, fh, strength)
        if _fa > 1e-4:
            fisheye_rc = _frc
            if text_mask is not None:
                text_mask = apply_fisheye(text_mask, text_mask, strength)[0]
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
                                halo=post_fx.get('flare_halo', 0.0),
                                ghosts=post_fx.get('flare_ghosts', 0.0),
                                starburst=post_fx.get('flare_starburst', 0.0),
                                blades=post_fx.get('flare_blades', 6))
    out = apply_analog_camera(out, post_fx, frame_seed=frame_seed, motion_px=motion_px, text_mask=text_mask)
    if post_fx.get('chroma_enabled', False):
        out = apply_chromatic_aberration(out, post_fx['chroma_strength'])
    if post_fx.get('vhs_enabled', False):
        out = apply_vhs(out, post_fx.get('vhs_strength', 1.0), frame_seed=frame_seed)
    if post_fx.get('crt_enabled', False):
        out = apply_crt(out, post_fx, frame_seed=frame_seed)
    if fisheye_rc is not None:
        # Black out everything outside the fisheye image circle (done LAST so
        # fog/bloom/grain/VHS can't tint the "no image here" area).
        fh, fw = out.shape[:2]
        out = out * _fisheye_circle_mask(fh, fw, fisheye_rc)[..., None]
    return np.clip(out, 0.0, 1.0)


# =============================================================================
# 13b) GPU post-processing (Taichi) -- the same effect chain as
#      apply_post_processing above, executed entirely on the GPU.
#
#      WHY: the NumPy/OpenCV chain above forces, EVERY frame, (a) a readback of
#      the whole float32 OUTPUT + DEPTH_OUT buffers (MAX_W x MAX_H -- 99 MB +
#      33 MB at the default 3840x2160 ceiling, even when the live view is only
#      96x54!), (b) a pile of full-frame float32 CPU passes (up to 16 box
#      blurs for DoF alone), and (c) a float->uint8 conversion. Here the
#      resolved image never leaves the GPU until it is a finished, packed
#      8-bit frame, so only ~1/4 of the bytes cross the bus -- and only for the
#      active (width x height) region, not the MAX_W x MAX_H field.
#
#      WHAT RUNS WHERE NOW:
#        GPU : DoF, zoom focus-hunt blur, fisheye, fog, god rays, bloom, lens
#              flare, the whole analog-camera look (range squash, blowout,
#              halation, split-tone, grain, vignette, motion smear), chromatic
#              aberration, VHS, clip + float->8-bit (+ RGB->BGR + resize-to-
#              window for the live view), and eye-adaptation metering.
#        CPU : only tiny per-frame scalar work -- projecting the (<=12) light
#              flares to screen space, the camera-motion vector for the smear,
#              and picking each effect's parameters.
#
#      The NumPy implementations above are kept unchanged as (1) the reference
#      the GPU kernels were checked against, and (2) an automatic fallback: if
#      a GPU kernel ever fails to compile/run on some backend, a warning is
#      printed and the renderer silently drops back to the CPU chain for the
#      rest of the run. Force the CPU chain explicitly with
#      --postfx-backend cpu (or RT_POSTFX_BACKEND=cpu).
#
#      Numerical notes (vs. the NumPy chain):
#        - DoF / bloom / halation box blurs use per-row / per-column prefix
#          sums in float32, like _box_blur did; DoF's per-pixel variable-radius
#          window is evaluated directly from those prefix sums.
#        - Gaussian weights come from cv2.getGaussianKernel, so the zoom
#          focus-hunt blur uses OpenCV's exact kernel (incl. its fixed small-
#          kernel tables) and BORDER_REFLECT_101 edges.
#        - Bilinear remaps (fisheye / chromatic aberration / smear) use the
#          same pixel-center convention + edge-replicate as cv2.remap /
#          cv2.warpAffine (OpenCV quantizes sample coordinates to 1/32 px, so
#          results can differ by a fraction of an 8-bit level there).
#        - VHS / analog-grain noise is generated on the GPU from a stateless
#          integer hash of (frame_seed, pixel/cell index) instead of NumPy's
#          default_rng. It is still deterministic per frame_seed (video stays
#          reproducible and still re-rolls every frame), but it is a different
#          random stream than the NumPy chain, so grain/VHS noise is
#          statistically equivalent, not bit-identical.
# =============================================================================

GPU_POSTFX = os.environ.get('RT_POSTFX_BACKEND', 'gpu').strip().lower() not in ('cpu', 'numpy', 'off')
_GP_MAX_FLARES = 32      # flares beyond this (8 lights + 4 spotlights max today) are dropped

_gp_nd3 = ti.types.ndarray(dtype=ti.f32, ndim=3)    # (h, w, 3) float image
_gp_nd2 = ti.types.ndarray(dtype=ti.f32, ndim=2)    # (h, w) float plane
_gp_nd1 = ti.types.ndarray(dtype=ti.f32, ndim=1)
_gp_ndi2 = ti.types.ndarray(dtype=ti.i32, ndim=2)   # (h, w) packed 8-bit RGB(A) pixels


# --- tiny helpers -----------------------------------------------------------

@ti.func
def _gp_ld3(a: ti.template(), y, x) -> vec3:
    return vec3(a[y, x, 0], a[y, x, 1], a[y, x, 2])


@ti.func
def _gp_st3(a: ti.template(), y, x, v: vec3):
    a[y, x, 0] = v[0]
    a[y, x, 1] = v[1]
    a[y, x, 2] = v[2]


@ti.func
def _gp_smoothstep(e0: ti.f32, e1: ti.f32, x: ti.f32) -> ti.f32:
    t = _gp_clamp((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@ti.func
def _gp_luma(v: vec3) -> ti.f32:
    return v[0] * 0.2126 + v[1] * 0.7152 + v[2] * 0.0722


@ti.func
def _gp_clamp(v: ti.f32, lo: ti.f32, hi: ti.f32) -> ti.f32:
    return ti.min(ti.max(v, lo), hi)


@ti.func
def _gp_bilerp3(a: ti.template(), sx: ti.f32, sy: ti.f32, w: ti.i32, h: ti.i32) -> vec3:
    """Bilinear sample at float pixel-center coordinates, edge-replicate --
    the sampling convention of cv2.remap(INTER_LINEAR, BORDER_REPLICATE)."""
    x0f = ti.floor(sx)
    y0f = ti.floor(sy)
    fx = sx - x0f
    fy = sy - y0f
    x0 = ti.cast(x0f, ti.i32)
    y0 = ti.cast(y0f, ti.i32)
    xa = ti.min(ti.max(x0, 0), w - 1)
    xb = ti.min(ti.max(x0 + 1, 0), w - 1)
    ya = ti.min(ti.max(y0, 0), h - 1)
    yb = ti.min(ti.max(y0 + 1, 0), h - 1)
    top = _gp_ld3(a, ya, xa) * (1.0 - fx) + _gp_ld3(a, ya, xb) * fx
    bot = _gp_ld3(a, yb, xa) * (1.0 - fx) + _gp_ld3(a, yb, xb) * fx
    return top * (1.0 - fy) + bot * fy


@ti.func
def _gp_bilerp1(a: ti.template(), sx: ti.f32, sy: ti.f32, w: ti.i32, h: ti.i32) -> ti.f32:
    x0f = ti.floor(sx)
    y0f = ti.floor(sy)
    fx = sx - x0f
    fy = sy - y0f
    x0 = ti.cast(x0f, ti.i32)
    y0 = ti.cast(y0f, ti.i32)
    xa = ti.min(ti.max(x0, 0), w - 1)
    xb = ti.min(ti.max(x0 + 1, 0), w - 1)
    ya = ti.min(ti.max(y0, 0), h - 1)
    yb = ti.min(ti.max(y0 + 1, 0), h - 1)
    top = a[ya, xa] * (1.0 - fx) + a[ya, xb] * fx
    bot = a[yb, xa] * (1.0 - fx) + a[yb, xb] * fx
    return top * (1.0 - fy) + bot * fy


@ti.func
def _gp_hash_u32(v: ti.u32) -> ti.u32:
    # Integer finalizer (constants kept < 2^31 so they are plain i32 literals
    # on every Taichi backend). u32 arithmetic wraps, which is what we want.
    v = ((v >> ti.u32(16)) ^ v) * ti.u32(73244475)
    v = ((v >> ti.u32(16)) ^ v) * ti.u32(73244475)
    v = (v >> ti.u32(16)) ^ v
    return v


@ti.func
def _gp_hash_normal(seed: ti.i32, idx: ti.i32) -> ti.f32:
    """Stateless N(0,1) sample for (seed, idx) -- Box-Muller on two hashed
    uniforms. Same (seed, idx) -> same value, which is what makes the GPU
    VHS/grain noise reproducible per frame_seed without any RNG state."""
    base = _gp_hash_u32(ti.cast(seed, ti.u32) * ti.u32(374761393) + ti.cast(idx, ti.u32) * ti.u32(668265263) + ti.u32(1013904223))
    h2 = _gp_hash_u32(base + ti.u32(1442695041))
    u1 = (ti.cast(base >> ti.u32(8), ti.f32) + 0.5) / 16777216.0
    u2 = (ti.cast(h2 >> ti.u32(8), ti.f32) + 0.5) / 16777216.0
    return ti.sqrt(-2.0 * ti.log(u1)) * ti.cos(6.2831853 * u2)


@ti.func
def _gp_reflect101(i: ti.i32, n: ti.i32) -> ti.i32:
    """cv2.BORDER_REFLECT_101 (gfedcb|abcdefgh|gfedcba), made safe for
    kernels wider than the image."""
    res = i
    if n == 1:
        res = 0
    else:
        if res < 0:
            res = -res
        if res > n - 1:
            res = 2 * (n - 1) - res
        res = ti.min(ti.max(res, 0), n - 1)
    return res


# --- load / present ---------------------------------------------------------

@ti.kernel
def _gp_load(col: _gp_nd3, dep: _gp_nd2, w: ti.i32, h: ti.i32):
    """Copies the resolved (tonemapped) frame + depth out of the big
    MAX_W x MAX_H fields into right-sized scratch arrays -- GPU-to-GPU, no
    host involvement."""
    for y, x in ti.ndrange(h, w):
        c = OUTPUT[y, x]
        col[y, x, 0] = c[0]
        col[y, x, 1] = c[1]
        col[y, x, 2] = c[2]
        dep[y, x] = DEPTH_OUT[y, x]


@ti.kernel
def _gp_present(src: _gp_nd3, out: _gp_ndi2, w: ti.i32, h: ti.i32,
                ow: ti.i32, oh: ti.i32, bgr: ti.i32):
    """Final stage: clip to [0,1], truncate to 8-bit exactly like
    (np.clip(c,0,1)*255).astype(uint8), optionally bilinear-resize to the
    window size (cv2.resize INTER_LINEAR convention) and pack the 3 bytes
    (RGB, or BGR for cv2.imshow) into one i32 per pixel."""
    sxs = ti.cast(w, ti.f32) / ti.cast(ow, ti.f32)
    sys_ = ti.cast(h, ti.f32) / ti.cast(oh, ti.f32)
    for y, x in ti.ndrange(oh, ow):
        c = vec3(0.0)
        if ow == w and oh == h:
            c = _gp_ld3(src, y, x)
        else:
            sx = (ti.cast(x, ti.f32) + 0.5) * sxs - 0.5
            sy = (ti.cast(y, ti.f32) + 0.5) * sys_ - 0.5
            c = _gp_bilerp3(src, sx, sy, w, h)
        r = ti.cast(_gp_clamp(c[0], 0.0, 1.0) * 255.0, ti.i32)
        g = ti.cast(_gp_clamp(c[1], 0.0, 1.0) * 255.0, ti.i32)
        b = ti.cast(_gp_clamp(c[2], 0.0, 1.0) * 255.0, ti.i32)
        if bgr != 0:
            out[y, x] = b | (g << 8) | (r << 16)
        else:
            out[y, x] = r | (g << 8) | (b << 16)


# --- box blur (prefix sums) -------------------------------------------------

@ti.kernel
def _gp_prefix_rows(src: _gp_nd3, pre: _gp_nd3, w: ti.i32, h: ti.i32):
    # pre[y, k] = sum of src[y, 0..k-1]  (so pre[y, 0] == 0, shape (h, w+1, 3))
    for y in range(h):
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        pre[y, 0, 0] = 0.0
        pre[y, 0, 1] = 0.0
        pre[y, 0, 2] = 0.0
        for x in range(w):
            s0 += src[y, x, 0]
            s1 += src[y, x, 1]
            s2 += src[y, x, 2]
            pre[y, x + 1, 0] = s0
            pre[y, x + 1, 1] = s1
            pre[y, x + 1, 2] = s2


@ti.kernel
def _gp_prefix_cols(src: _gp_nd3, pre: _gp_nd3, w: ti.i32, h: ti.i32):
    # pre[k, x] = sum of src[0..k-1, x]  (shape (h+1, w, 3))
    for x in range(w):
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        pre[0, x, 0] = 0.0
        pre[0, x, 1] = 0.0
        pre[0, x, 2] = 0.0
        for y in range(h):
            s0 += src[y, x, 0]
            s1 += src[y, x, 1]
            s2 += src[y, x, 2]
            pre[y + 1, x, 0] = s0
            pre[y + 1, x, 1] = s1
            pre[y + 1, x, 2] = s2


@ti.func
def _gp_row_window(p: ti.template(), y: ti.i32, x: ti.i32, r: ti.i32, w: ti.i32) -> vec3:
    """Sum of the 2r+1 pixels centred on x in row y, with edge-REPLICATE
    (out-of-range taps repeat the first/last pixel), from the row prefix
    array p. Uses only p, so the array the prefix was built from can safely
    be overwritten while this runs."""
    lo = x - r
    hi = x + r
    nl = ti.max(0, -lo)
    nr = ti.max(0, hi - (w - 1))
    lc = ti.max(lo, 0)
    hc = ti.min(hi, w - 1)
    inside = _gp_ld3(p, y, hc + 1) - _gp_ld3(p, y, lc)
    first = _gp_ld3(p, y, 1)
    last = _gp_ld3(p, y, w) - _gp_ld3(p, y, w - 1)
    return inside + first * ti.cast(nl, ti.f32) + last * ti.cast(nr, ti.f32)


@ti.func
def _gp_col_window(p: ti.template(), y: ti.i32, x: ti.i32, r: ti.i32, h: ti.i32) -> vec3:
    lo = y - r
    hi = y + r
    nl = ti.max(0, -lo)
    nr = ti.max(0, hi - (h - 1))
    lc = ti.max(lo, 0)
    hc = ti.min(hi, h - 1)
    inside = _gp_ld3(p, hc + 1, x) - _gp_ld3(p, lc, x)
    first = _gp_ld3(p, 1, x)
    last = _gp_ld3(p, h, x) - _gp_ld3(p, h - 1, x)
    return inside + first * ti.cast(nl, ti.f32) + last * ti.cast(nr, ti.f32)


@ti.kernel
def _gp_box_h(pre: _gp_nd3, dst: _gp_nd3, w: ti.i32, h: ti.i32, r: ti.i32):
    inv = 1.0 / ti.cast(2 * r + 1, ti.f32)
    for y, x in ti.ndrange(h, w):
        _gp_st3(dst, y, x, _gp_row_window(pre, y, x, r, w) * inv)


@ti.kernel
def _gp_box_v(pre: _gp_nd3, dst: _gp_nd3, w: ti.i32, h: ti.i32, r: ti.i32):
    inv = 1.0 / ti.cast(2 * r + 1, ti.f32)
    for y, x in ti.ndrange(h, w):
        _gp_st3(dst, y, x, _gp_col_window(pre, y, x, r, h) * inv)


@ti.kernel
def _gp_copy(src: _gp_nd3, dst: _gp_nd3, w: ti.i32, h: ti.i32):
    for y, x in ti.ndrange(h, w):
        _gp_st3(dst, y, x, _gp_ld3(src, y, x))


# --- depth of field ---------------------------------------------------------

@ti.func
def _gp_box2d(src: ti.template(), prx: ti.template(), y: ti.i32, x: ti.i32,
              r: ti.i32, w: ti.i32, h: ti.i32) -> vec3:
    """(2r+1)x(2r+1) edge-replicate box average around (y, x) -- the same
    value levels[r] holds at that pixel in apply_depth_of_field, but
    evaluated per pixel at ITS OWN radius, so no stack of full-frame blurred
    copies is needed."""
    res = _gp_ld3(src, y, x)
    if r > 0:
        acc = vec3(0.0)
        for yy in range(y - r, y + r + 1):
            yc = ti.min(ti.max(yy, 0), h - 1)
            acc += _gp_row_window(prx, yc, x, r, w)
        res = acc / ti.cast((2 * r + 1) * (2 * r + 1), ti.f32)
    return res


@ti.kernel
def _gp_dof(src: _gp_nd3, prx: _gp_nd3, depth: _gp_nd2, dst: _gp_nd3,
            w: ti.i32, h: ti.i32, focus: ti.f32, strength: ti.f32, max_r: ti.i32):
    max_rf = ti.cast(max_r, ti.f32)
    for y, x in ti.ndrange(h, w):
        coc = _gp_clamp(ti.abs(depth[y, x] - focus) * strength, 0.0, max_rf)
        lo = ti.min(ti.max(ti.cast(ti.floor(coc), ti.i32), 0), max_r)
        hi = ti.min(lo + 1, max_r)
        frac = coc - ti.cast(lo, ti.f32)
        res = _gp_box2d(src, prx, y, x, lo, w, h)
        if frac > 1e-6 and hi != lo:
            res = res * (1.0 - frac) + _gp_box2d(src, prx, y, x, hi, w, h) * frac
        _gp_st3(dst, y, x, res)


# --- zoom focus-hunt (Gaussian) --------------------------------------------

@ti.kernel
def _gp_gauss_h(src: _gp_nd3, dst: _gp_nd3, wts: _gp_nd1, w: ti.i32, h: ti.i32, rad: ti.i32):
    for y, x in ti.ndrange(h, w):
        acc = vec3(0.0)
        for k in range(-rad, rad + 1):
            acc += _gp_ld3(src, y, _gp_reflect101(x + k, w)) * wts[k + rad]
        _gp_st3(dst, y, x, acc)


@ti.kernel
def _gp_gauss_v_mix(tmp: _gp_nd3, orig: _gp_nd3, dst: _gp_nd3, wts: _gp_nd1,
                    w: ti.i32, h: ti.i32, rad: ti.i32, amount: ti.f32):
    # vertical Gaussian pass fused with apply_zoom_focal_shift's final blend
    # (img*(1-a) + blurred*a)
    for y, x in ti.ndrange(h, w):
        acc = vec3(0.0)
        for k in range(-rad, rad + 1):
            acc += _gp_ld3(tmp, _gp_reflect101(y + k, h), x) * wts[k + rad]
        _gp_st3(dst, y, x, _gp_ld3(orig, y, x) * (1.0 - amount) + acc * amount)


# --- fisheye ----------------------------------------------------------------

@ti.kernel
def _gp_fisheye(src: _gp_nd3, dep: _gp_nd2, dst: _gp_nd3, dst_dep: _gp_nd2,
                w: ti.i32, h: ti.i32, a: ti.f32, rc: ti.f32):
    # True fisheye: src_radius = rc * tan(a*u) / tan(a), u = dst_radius / rc
    # (see _fisheye_geometry). Outside the circle (r >= rc) we just sample 1:1;
    # that area is blacked out by _gp_circle_mask at the end of the chain.
    cx = ti.cast(w - 1, ti.f32) * 0.5
    cy = ti.cast(h - 1, ti.f32) * 0.5
    ta = ti.tan(a)
    for y, x in ti.ndrange(h, w):
        dx = ti.cast(x, ti.f32) - cx
        dy = ti.cast(y, ti.f32) - cy
        r = ti.sqrt(dx * dx + dy * dy)
        k = 1.0
        if r > 1e-6 and r < rc:
            k = (rc * ti.tan(a * (r / rc)) / ta) / r
        sx = cx + dx * k
        sy = cy + dy * k
        _gp_st3(dst, y, x, _gp_bilerp3(src, sx, sy, w, h))
        dst_dep[y, x] = _gp_bilerp1(dep, sx, sy, w, h)


@ti.kernel
def _gp_circle_mask(img: _gp_nd3, w: ti.i32, h: ti.i32, rc: ti.f32):
    # In-place: darkens everything outside the fisheye image circle (1px AA rim).
    cx = ti.cast(w - 1, ti.f32) * 0.5
    cy = ti.cast(h - 1, ti.f32) * 0.5
    for y, x in ti.ndrange(h, w):
        dx = ti.cast(x, ti.f32) - cx
        dy = ti.cast(y, ti.f32) - cy
        m = _gp_clamp(rc - ti.sqrt(dx * dx + dy * dy) + 0.5, 0.0, 1.0)
        if m < 1.0:
            _gp_st3(img, y, x, _gp_ld3(img, y, x) * m)


# --- fog --------------------------------------------------------------------

@ti.kernel
def _gp_fog(src: _gp_nd3, dep: _gp_nd2, dst: _gp_nd3, w: ti.i32, h: ti.i32,
            density: ti.f32, max_depth: ti.f32, falloff: ti.f32, base_h: ti.f32,
            cam_y: ti.f32, r0: ti.f32, r1: ti.f32, r2: ti.f32,
            half_tan: ti.f32, aspect: ti.f32,
            fr: ti.f32, fg: ti.f32, fb: ti.f32, use_height: ti.i32):
    fog_col = vec3(fr, fg, fb)
    wm1 = ti.cast(ti.max(w - 1, 1), ti.f32)
    hm1 = ti.cast(ti.max(h - 1, 1), ti.f32)
    for y, x in ti.ndrange(h, w):
        d = _gp_clamp(dep[y, x], 0.0, max_depth)
        fog = 1.0 - ti.exp(-d * density)
        if use_height != 0:
            xn = (ti.cast(x, ti.f32) / wm1) * 2.0 - 1.0
            yn = 1.0 - (ti.cast(y, ti.f32) / hm1) * 2.0
            dc = vec3(xn * half_tan * aspect, yn * half_tan, 1.0).normalized()
            world_y = cam_y + (dc[0] * r0 + dc[1] * r1 + dc[2] * r2) * d
            fog = fog * ti.exp(-ti.max(world_y - base_h, 0.0) * falloff)
        fog = _gp_clamp(fog, 0.0, 1.0)
        _gp_st3(dst, y, x, _gp_ld3(src, y, x) * (1.0 - fog) + fog_col * fog)


# --- god rays ---------------------------------------------------------------

@ti.kernel
def _gp_godray_src(img: _gp_nd3, dep: _gp_nd2, s: _gp_nd2, w: ti.i32, h: ti.i32, sky_depth: ti.f32):
    for y, x in ti.ndrange(h, w):
        v = 0.0
        if dep[y, x] > sky_depth:
            v = _gp_clamp(_gp_luma(_gp_ld3(img, y, x)), 0.0, 4.0)
        s[y, x] = v


@ti.kernel
def _gp_godrays(img: _gp_nd3, s: _gp_nd2, fl: _gp_nd2, nf: ti.i32, dst: _gp_nd3,
                w: ti.i32, h: ti.i32, n_samples: ti.i32, density: ti.f32,
                decay: ti.f32, intensity: ti.f32):
    wm1 = ti.cast(w - 1, ti.f32)
    hm1 = ti.cast(h - 1, ti.f32)
    inv_n = 1.0 / ti.cast(n_samples, ti.f32)
    for y, x in ti.ndrange(h, w):
        xf = ti.cast(x, ti.f32)
        yf = ti.cast(y, ti.f32)
        total = vec3(0.0)
        for k in range(nf):
            sx = fl[k, 0]
            sy = fl[k, 1]
            step_x = (sx - xf) * (density * inv_n)
            step_y = (sy - yf) * (density * inv_n)
            cur_x = xf
            cur_y = yf
            accum = 0.0
            fall = 1.0
            for _i in range(n_samples):
                cur_x += step_x
                cur_y += step_y
                ix = ti.cast(_gp_clamp(cur_x, 0.0, wm1), ti.i32)
                iy = ti.cast(_gp_clamp(cur_y, 0.0, hm1), ti.i32)
                accum += s[iy, ix] * fall
                fall *= decay
            accum *= inv_n
            col = vec3(fl[k, 2], fl[k, 3], fl[k, 4])
            total += accum * col * intensity * (0.3 + 0.7 * ti.min(1.0, fl[k, 5]))
        _gp_st3(dst, y, x, _gp_ld3(img, y, x) + total)


# --- bloom ------------------------------------------------------------------

@ti.kernel
def _gp_bloom_bright(img: _gp_nd3, dst: _gp_nd3, w: ti.i32, h: ti.i32, threshold: ti.f32):
    knee = 0.2
    for y, x in ti.ndrange(h, w):
        c = _gp_ld3(img, y, x)
        t = _gp_clamp((_gp_luma(c) - (threshold - knee)) / ti.max(1e-4, 2.0 * knee), 0.0, 1.0)
        soft = t * t * (3.0 - 2.0 * t)
        _gp_st3(dst, y, x, c * soft)


@ti.kernel
def _gp_bloom_combine(img: _gp_nd3, g_small: _gp_nd3, g_large: _gp_nd3, dst: _gp_nd3,
                      w: ti.i32, h: ti.i32, intensity: ti.f32):
    for y, x in ti.ndrange(h, w):
        c = _gp_ld3(img, y, x)
        glow = (_gp_ld3(g_small, y, x) * 0.6 + _gp_ld3(g_large, y, x) * 0.4) * intensity
        glow = vec3(_gp_clamp(glow[0], 0.0, 1.0), _gp_clamp(glow[1], 0.0, 1.0), _gp_clamp(glow[2], 0.0, 1.0))
        _gp_st3(dst, y, x, vec3(1.0) - (vec3(1.0) - c) * (vec3(1.0) - glow))   # screen blend


# --- lens flare -------------------------------------------------------------
# Optical model (mirrored by apply_lens_flare in section 13 -- keep them in step):
#   glow      core + wide halo + a long Lorentzian tail (veiling glare), not a bare Gaussian
#   starburst diffraction spikes from the aperture blades (odd blade counts give 2x spikes),
#             thin lines with a slow rainbow drift along their length
#   anamorphic  the cool horizontal streak, plus a faint wide band under it
#   halo      ring around the light with per-channel dispersion (red inside, blue outside)
#   ghosts    internal reflections on the axis through the image centre: aperture-shaped
#             polygons (rotated with the axis), brighter rim, coating tint, per-channel
#             size dispersion; they strengthen as the light moves off-axis
#   fade      everything fades out as the light leaves the frame instead of popping off

@ti.kernel
def _gp_lens_flare(img: _gp_nd3, fl: _gp_nd2, nf: ti.i32, dst: _gp_nd3,
                   w: ti.i32, h: ti.i32, radius: ti.f32, intensity: ti.f32,
                   anamorphic: ti.f32, halo: ti.f32, ghosts: ti.f32, starburst: ti.f32,
                   blades: ti.i32):
    cx0 = ti.cast(w - 1, ti.f32) * 0.5
    cy0 = ti.cast(h - 1, ti.f32) * 0.5
    wf = ti.cast(w, ti.f32)
    hf = ti.cast(h, ti.f32)
    hd = 0.5 * ti.sqrt(wf * wf + hf * hf)
    R = radius
    nb = ti.cast(blades, ti.f32)
    seg = 6.2831853 / nb
    cos_pn = ti.cos(3.14159265 / nb)
    star_m = blades
    if blades % 2 == 1:
        star_m = blades * 2
    star_step = 6.2831853 / ti.cast(star_m, ti.f32)
    for y, x in ti.ndrange(h, w):
        px = ti.cast(x, ti.f32)
        py = ti.cast(y, ti.f32)
        o = _gp_ld3(img, y, x)
        for k in range(nf):
            sx = fl[k, 0]
            sy = fl[k, 1]
            color = vec3(fl[k, 2], fl[k, 3], fl[k, 4])
            bright = fl[k, 5]
            e_edge = ti.max(ti.abs(sx - cx0) / (0.5 * wf), ti.abs(sy - cy0) / (0.5 * hf))
            amp = intensity * bright * (1.0 - _gp_smoothstep(1.0, 1.25, e_edge))
            if amp > 1e-5:
                dx = px - sx
                dy = py - sy
                d2 = dx * dx + dy * dy
                dd = ti.sqrt(d2)

                glow = (0.68 * ti.exp(-d2 / (2.0 * R * R))
                        + 0.20 * ti.exp(-d2 / (2.0 * (2.6 * R) * (2.6 * R)))
                        + 0.10 / (1.0 + d2 / (36.0 * R * R)))
                o += glow * color * amp

                if starburst > 0.0:
                    ang = ti.atan2(dy, dx) - 1.5707963
                    kk = ti.floor(ang / star_step + 0.5)
                    delta = ang - kk * star_step
                    perp = dd * ti.abs(ti.sin(delta))
                    sig = ti.max(0.55, R * 0.035)
                    ray = ti.exp(-(perp * perp) / (2.0 * sig * sig))
                    along = ti.exp(-dd / (R * 7.0)) / (1.0 + dd / (R * 1.2))
                    ph = dd / (R * 12.0)
                    rainbow = vec3(0.5 + 0.5 * ti.cos(6.2831853 * ph),
                                   0.5 + 0.5 * ti.cos(6.2831853 * (ph + 0.33333)),
                                   0.5 + 0.5 * ti.cos(6.2831853 * (ph + 0.66667)))
                    o += ray * along * (color * 0.55 + rainbow * 0.45) * (amp * starburst * 0.9)

                if anamorphic > 0.0:
                    streak_col = color * 0.5 + vec3(0.6, 0.75, 1.0) * 0.5
                    s1 = R * 0.12
                    s2 = R * 0.55
                    streak = (ti.exp(-(dy * dy) / (2.0 * s1 * s1)) * ti.exp(-ti.abs(dx) / (R * 7.0))
                              + 0.22 * ti.exp(-(dy * dy) / (2.0 * s2 * s2)) * ti.exp(-ti.abs(dx) / (R * 4.0)))
                    o += streak * streak_col * (amp * anamorphic)

                if halo > 0.0:
                    rw = R * 0.32
                    ring = vec3(0.0)
                    for c in ti.static(range(3)):
                        rr_ = R * 2.4 * _FLARE_HALO_DISP[c]
                        ring[c] = ti.exp(-((dd - rr_) * (dd - rr_)) / (2.0 * rw * rw))
                    o += ring * (color * 0.6 + vec3(0.4)) * (amp * halo * 0.5)

                if ghosts > 0.0:
                    axv = cx0 - sx
                    ayv = cy0 - sy
                    r_off = ti.sqrt(axv * axv + ayv * ayv) / hd
                    gain = 0.25 + 0.75 * ti.min(1.0, r_off * 1.6)
                    rot = ti.atan2(ayv, axv)
                    for gi in ti.static(range(len(_FLARE_GHOSTS))):
                        gf = _FLARE_GHOSTS[gi][0]
                        gsz = _FLARE_GHOSTS[gi][1]
                        gtint = vec3(_FLARE_GHOSTS[gi][2][0], _FLARE_GHOSTS[gi][2][1], _FLARE_GHOSTS[gi][2][2])
                        gst = _FLARE_GHOSTS[gi][3]
                        gx = cx0 + (sx - cx0) * gf
                        gy = cy0 + (sy - cy0) * gf
                        gr = R * gsz * 1.6 * (0.9 + 0.5 * r_off)
                        gdx = px - gx
                        gdy = py - gy
                        rr2 = gdx * gdx + gdy * gdy
                        if rr2 < gr * gr * 1.44:
                            rr = ti.sqrt(rr2)
                            a = ti.atan2(gdy, gdx) - rot
                            am = a - seg * ti.floor(a / seg)
                            rpoly = gr * cos_pn / ti.cos(am - seg * 0.5)
                            q = rr / rpoly
                            val = vec3(0.0)
                            for c in ti.static(range(3)):
                                qc = q * _FLARE_DISP[c]
                                fill = 1.0 - _gp_smoothstep(0.90, 1.10, qc)
                                val[c] = fill * (0.32 + 0.68 * _gp_smoothstep(0.45, 1.0, qc))
                            o += val * color * gtint * (amp * ghosts * gst * gain * 0.75)
        _gp_st3(dst, y, x, o)


# --- analog camera ----------------------------------------------------------

@ti.kernel
def _gp_analog_a(img: _gp_nd3, dst: _gp_nd3, hot: _gp_nd3, w: ti.i32, h: ti.i32,
                 dr: ti.f32, hc: ti.f32, strength: ti.f32, do_hal: ti.i32):
    """Steps 1+2 (range squash, highlight blowout) fused, plus the
    halation source (out * hot-mask) so step 3's blur has its input."""
    for y, x in ti.ndrange(h, w):
        o = _gp_ld3(img, y, x)
        if dr > 0.0:
            black_lift = 0.14 * dr
            o = o * (1.0 - black_lift) + black_lift
            knee = 0.55
            for c in ti.static(range(3)):
                v = o[c]
                if v > knee:
                    over = v - knee
                    o[c] = knee + over / (1.0 + over * (4.0 * dr))
        if hc > 0.0:
            luma = _gp_clamp(_gp_luma(o), 0.0, 4.0)
            thresh = ti.max(0.02, 1.0 - hc * 0.9)
            blow = ti.pow(_gp_clamp((luma - thresh) / ti.max(1e-4, 1.0 - thresh), 0.0, 1.0), 0.35)
            o = o + (vec3(1.15) - o) * (blow * strength)
        _gp_st3(dst, y, x, o)
        if do_hal != 0:
            m = _gp_clamp((_gp_luma(o) - 0.7) / 0.3, 0.0, 1.0)
            _gp_st3(hot, y, x, o * (m * m))


@ti.kernel
def _gp_noise_grid(noise: _gp_nd3, gw: ti.i32, gh: ti.i32, seed: ti.i32):
    """Fills the (gh x gw) low-res grain grid: channel 0 = luma noise,
    1..3 = per-channel chroma noise (all N(0,1))."""
    for y, x in ti.ndrange(gh, gw):
        base = (y * gw + x) * 4
        for k in ti.static(range(4)):
            noise[y, x, k] = _gp_hash_normal(seed, base + k)


@ti.func
def _gp_grain_at(noise: ti.template(), x: ti.i32, y: ti.i32, w: ti.i32, h: ti.i32,
                 gw: ti.i32, gh: ti.i32, upsample: ti.i32) -> vec4:
    """Grain value at output pixel (x, y). upsample == 0: the grid IS the
    image. Otherwise bilinear-upsample the (gh x gw) grid with cv2.resize's
    half-pixel-center mapping + edge replicate."""
    res = vec4(noise[y, x, 0], noise[y, x, 1], noise[y, x, 2], noise[y, x, 3])
    if upsample != 0:
        sxf = (ti.cast(x, ti.f32) + 0.5) * (ti.cast(gw, ti.f32) / ti.cast(w, ti.f32)) - 0.5
        syf = (ti.cast(y, ti.f32) + 0.5) * (ti.cast(gh, ti.f32) / ti.cast(h, ti.f32)) - 0.5
        x0 = ti.cast(ti.floor(sxf), ti.i32)
        y0 = ti.cast(ti.floor(syf), ti.i32)
        fx = sxf - ti.floor(sxf)
        fy = syf - ti.floor(syf)
        if x0 < 0:
            x0 = 0
            fx = 0.0
        if x0 >= gw - 1:
            x0 = gw - 1
            fx = 0.0
        if y0 < 0:
            y0 = 0
            fy = 0.0
        if y0 >= gh - 1:
            y0 = gh - 1
            fy = 0.0
        x1 = ti.min(x0 + 1, gw - 1)
        y1 = ti.min(y0 + 1, gh - 1)
        res = vec4(0.0)
        for c in ti.static(range(4)):
            top = noise[y0, x0, c] * (1.0 - fx) + noise[y0, x1, c] * fx
            bot = noise[y1, x0, c] * (1.0 - fx) + noise[y1, x1, c] * fx
            res[c] = top * (1.0 - fy) + bot * fy
    return res


@ti.kernel
def _gp_analog_b(img: _gp_nd3, glow: _gp_nd3, noise: _gp_nd3, dst: _gp_nd3,
                 w: ti.i32, h: ti.i32, strength: ti.f32,
                 halation: ti.f32, do_hal: ti.i32, warmth: ti.f32,
                 grain: ti.f32, gw: ti.i32, gh: ti.i32, upsample: ti.i32,
                 vignette: ti.f32, sat_eff: ti.f32):
    """Steps 3(add)+4+4b+5+6 fused: halation add, split-tone, saturation,
    grain, vignette."""
    cx = ti.cast(w - 1, ti.f32) * 0.5
    cy = ti.cast(h - 1, ti.f32) * 0.5
    hal_tint = vec3(1.0, 0.35, 0.15)
    shadow_tint = vec3(-0.02, -0.01, 0.03)
    highlight_tint = vec3(0.05, 0.02, -0.03)
    for y, x in ti.ndrange(h, w):
        o = _gp_ld3(img, y, x)
        if do_hal != 0:
            o = o + _gp_ld3(glow, y, x) * hal_tint * halation * strength
        if warmth > 0.0:
            l = _gp_clamp(_gp_luma(o), 0.0, 1.0)
            o = o + (shadow_tint * (1.0 - l) + highlight_tint * l) * warmth * strength
        if ti.abs(sat_eff - 1.0) > 1e-6:
            # unclamped on purpose (see apply_analog_camera step 4b)
            ls = _gp_luma(o)
            o = vec3(ls) + (o - vec3(ls)) * sat_eff
        if grain > 0.0:
            gwt = 1.0 - 0.6 * _gp_clamp(_gp_luma(o), 0.0, 1.0)
            nz = _gp_grain_at(noise, x, y, w, h, gw, gh, upsample)
            o = o + vec3(nz[0]) * (grain * gwt * strength)
            o = o + vec3(nz[1], nz[2], nz[3]) * (grain * 0.35 * gwt * strength)
        if vignette > 0.0:
            ddx = (ti.cast(x, ti.f32) - cx) / (ti.cast(w, ti.f32) * 0.5)
            ddy = (ti.cast(y, ti.f32) - cy) / (ti.cast(h, ti.f32) * 0.5)
            d = _gp_clamp(ti.sqrt(ddx * ddx + ddy * ddy), 0.0, 1.4)
            o = o * (1.0 - vignette * strength * 0.5 * d * d)
        _gp_st3(dst, y, x, o)


@ti.kernel
def _gp_smear(img: _gp_nd3, dst: _gp_nd3, w: ti.i32, h: ti.i32,
              ux: ti.f32, uy: ti.f32, length: ti.f32, n_taps: ti.i32,
              tmask: _gp_nd2, use_mask: ti.i32):
    """apply_analog_motion_smear: mean of n_taps copies of the frame shifted
    along (ux, uy) by length * f, f spanning -0.5..+0.5 (warpAffine forward
    translation == sampling the source at (x - ox, y - oy))."""
    for y, x in ti.ndrange(h, w):
        acc = vec3(0.0)
        wacc = 0.0
        for i in range(n_taps):
            f = ti.cast(i, ti.f32) / ti.cast(n_taps - 1, ti.f32) - 0.5
            ox = ux * length * f
            oy = uy * length * f
            tx = ti.cast(x, ti.f32) - ox
            ty = ti.cast(y, ti.f32) - oy
            if use_mask != 0:
                # HUD text (tmask = its alpha) must not be smeared, nor smear into its
                # surroundings: weight each tap by (1 - text alpha) and renormalise.
                wt = 1.0 - ti.min(1.0, 3.0 * _gp_bilerp1(tmask, tx, ty, w, h))
                acc += _gp_bilerp3(img, tx, ty, w, h) * wt
                wacc += wt
            else:
                acc += _gp_bilerp3(img, tx, ty, w, h)
        centre = _gp_ld3(img, y, x)
        res = acc / ti.cast(n_taps, ti.f32)
        if use_mask != 0:
            m0 = ti.min(1.0, 3.0 * tmask[y, x])
            sm = centre
            if wacc > 1e-4:
                sm = acc / wacc
            res = sm * (1.0 - m0) + centre * m0
        _gp_st3(dst, y, x, res)


# --- chromatic aberration ---------------------------------------------------

@ti.kernel
def _gp_chroma(img: _gp_nd3, dst: _gp_nd3, w: ti.i32, h: ti.i32, strength: ti.f32):
    cx = ti.cast(w - 1, ti.f32) * 0.5
    cy = ti.cast(h - 1, ti.f32) * 0.5
    r_max = ti.sqrt(cx * cx + cy * cy)
    for y, x in ti.ndrange(h, w):
        xf = ti.cast(x, ti.f32)
        yf = ti.cast(y, ti.f32)
        r_norm = _gp_clamp(ti.sqrt((xf - cx) * (xf - cx) + (yf - cy) * (yf - cy)) / ti.max(r_max, 1e-6), 0.0, 1.0)
        fall = r_norm * r_norm
        ls_r = strength * fall
        ls_b = -strength * fall
        cr = _gp_bilerp3(img, cx + (xf - cx) * (1.0 + ls_r), cy + (yf - cy) * (1.0 + ls_r), w, h)
        cb = _gp_bilerp3(img, cx + (xf - cx) * (1.0 + ls_b), cy + (yf - cy) * (1.0 + ls_b), w, h)
        g = _gp_ld3(img, y, x)
        _gp_st3(dst, y, x, vec3(cr[0], g[1], cb[2]))


# --- VHS --------------------------------------------------------------------

@ti.kernel
def _gp_vhs(img: _gp_nd3, dst: _gp_nd3, w: ti.i32, h: ti.i32,
            strength: ti.f32, shift: ti.i32, seed: ti.i32):
    """apply_vhs, fused into one gather: R/B crosstalk shift + per-row
    tracking wobble (both horizontal wrap-around rolls) -> scanlines ->
    noise -> vignette -> desaturation -> clip."""
    cx = ti.cast(w - 1, ti.f32) * 0.5
    cy = ti.cast(h - 1, ti.f32) * 0.5
    hm1 = ti.cast(ti.max(h - 1, 1), ti.f32)
    for y, x in ti.ndrange(h, w):
        yf = ti.cast(y, ti.f32)
        edge_w = _gp_clamp(1.0 - ti.abs(yf / hm1 - 0.5) * 2.0, 0.0, 1.0)
        wob = ti.cast(ti.floor(_gp_hash_normal(seed, y) * (1.0 - edge_w * 0.6) * 1.5 * strength + 0.5), ti.i32)
        xs = x - wob
        xr = ((xs + shift) % w + w) % w
        xg = (xs % w + w) % w
        xb = ((xs - shift) % w + w) % w
        o = vec3(img[y, xr, 0], img[y, xg, 1], img[y, xb, 2])
        o = o * (1.0 - 0.12 * strength * ti.cast(y % 2, ti.f32))
        o = o + vec3(_gp_hash_normal(seed + 1, y * w + x) * 0.035 * strength)
        ddx = (ti.cast(x, ti.f32) - cx) / (ti.cast(w, ti.f32) * 0.5)
        ddy = (yf - cy) / (ti.cast(h, ti.f32) * 0.5)
        d = ti.sqrt(ddx * ddx + ddy * ddy)
        o = o * (1.0 - 0.25 * strength * _gp_clamp(d - 0.6, 0.0, 1.0))
        gray = (o[0] + o[1] + o[2]) / 3.0
        o = o * (1.0 - 0.15 * strength) + vec3(gray) * (0.15 * strength)
        _gp_st3(dst, y, x, vec3(_gp_clamp(o[0], 0.0, 1.0), _gp_clamp(o[1], 0.0, 1.0), _gp_clamp(o[2], 0.0, 1.0)))


# --- HUD text (sprite blit) -----------------------------------------------------

@ti.kernel
def _gp_clear1(plane: _gp_nd2, w: ti.i32, h: ti.i32):
    for y, x in ti.ndrange(h, w):
        plane[y, x] = 0.0


@ti.kernel
def _gp_hud_alpha(plane: _gp_nd2, spr: _gp_nd2, sw: ti.i32, sh: ti.i32,
                  ix0: ti.i32, iy0: ti.i32, fx: ti.f32, fy: ti.f32, w: ti.i32, h: ti.i32):
    """Max-blends a text sprite's coverage into an (h, w) plane (same sampling as _gp_hud_blit,
    opacity ignored) -- the 'where is HUD text' mask the analog smear respects."""
    for j, i in ti.ndrange(sh + 1, sw + 1):
        x = ix0 + i
        y = iy0 + j
        if x >= 0 and x < w and y >= 0 and y < h:
            u = ti.cast(i, ti.f32) - fx
            v = ti.cast(j, ti.f32) - fy
            u0f = ti.floor(u)
            v0f = ti.floor(v)
            tu = u - u0f
            tv = v - v0f
            u0 = ti.cast(u0f, ti.i32)
            v0 = ti.cast(v0f, ti.i32)
            a = ((_gp_spr_at(spr, v0, u0, sw, sh) * (1.0 - tu) + _gp_spr_at(spr, v0, u0 + 1, sw, sh) * tu) * (1.0 - tv)
                 + (_gp_spr_at(spr, v0 + 1, u0, sw, sh) * (1.0 - tu) + _gp_spr_at(spr, v0 + 1, u0 + 1, sw, sh) * tu) * tv)
            plane[y, x] = ti.max(plane[y, x], a)


@ti.func
def _gp_spr_at(spr: ti.template(), v: ti.i32, u: ti.i32, sw: ti.i32, sh: ti.i32) -> ti.f32:
    r = 0.0
    if u >= 0 and u < sw and v >= 0 and v < sh:
        r = spr[v, u]
    return r


@ti.kernel
def _gp_hud_blit(img: _gp_nd3, dep: _gp_nd2, spr: _gp_nd2, sw: ti.i32, sh: ti.i32,
                 ix0: ti.i32, iy0: ti.i32, fx: ti.f32, fy: ti.f32, w: ti.i32, h: ti.i32,
                 cr: ti.f32, cg: ti.f32, cb: ti.f32, opacity: ti.f32, use_depth: ti.i32):
    """Alpha-blends one text sprite (float alpha mask) onto `img` at (ix0 + fx, iy0 + fy),
    bilinear for the fractional part. With use_depth the depth plane is pulled toward 0
    under the text so fog / god rays treat the HUD as being in front of the scene."""
    for j, i in ti.ndrange(sh + 1, sw + 1):
        x = ix0 + i
        y = iy0 + j
        if x >= 0 and x < w and y >= 0 and y < h:
            u = ti.cast(i, ti.f32) - fx
            v = ti.cast(j, ti.f32) - fy
            u0f = ti.floor(u)
            v0f = ti.floor(v)
            tu = u - u0f
            tv = v - v0f
            u0 = ti.cast(u0f, ti.i32)
            v0 = ti.cast(v0f, ti.i32)
            a00 = _gp_spr_at(spr, v0, u0, sw, sh)
            a10 = _gp_spr_at(spr, v0, u0 + 1, sw, sh)
            a01 = _gp_spr_at(spr, v0 + 1, u0, sw, sh)
            a11 = _gp_spr_at(spr, v0 + 1, u0 + 1, sw, sh)
            a = ((a00 * (1.0 - tu) + a10 * tu) * (1.0 - tv)
                 + (a01 * (1.0 - tu) + a11 * tu) * tv) * opacity
            if a > 0.0:
                c = _gp_ld3(img, y, x)
                _gp_st3(img, y, x, c * (1.0 - a) + vec3(cr, cg, cb) * a)
                if use_depth != 0:
                    dep[y, x] = dep[y, x] * (1.0 - a)


@ti.kernel
def _gp_resize3(src: _gp_nd3, dst: _gp_nd3, w: ti.i32, h: ti.i32, ow: ti.i32, oh: ti.i32):
    """Bilinear float resize (same sampling convention as _gp_present's resize) -- used so a
    clean HUD overlay can be drawn at the OUTPUT size instead of on the low-res render."""
    sxs = ti.cast(w, ti.f32) / ti.cast(ow, ti.f32)
    sys_ = ti.cast(h, ti.f32) / ti.cast(oh, ti.f32)
    for y, x in ti.ndrange(oh, ow):
        sx = (ti.cast(x, ti.f32) + 0.5) * sxs - 0.5
        sy = (ti.cast(y, ti.f32) + 0.5) * sys_ - 0.5
        _gp_st3(dst, y, x, _gp_bilerp3(src, sx, sy, w, h))


# --- CRT monitor ------------------------------------------------------------------

@ti.kernel
def _gp_crt(img: _gp_nd3, glow: _gp_nd3, dst: _gp_nd3, w: ti.i32, h: ti.i32,
            curv: ti.f32, scan: ti.f32, lines: ti.f32, mask: ti.f32, glow_amt: ti.f32,
            conv_px: ti.f32, vig: ti.f32, flick_gain: ti.f32, flick_amt: ti.f32,
            band_phase: ti.f32, bright: ti.f32, scale: ti.f32):
    """apply_crt, fused into one gather pass: curved-glass warp (+ black outside the tube, AA'd
    edge) -> R/B convergence error -> phosphor glow -> beam-width-aware scanlines -> RGB
    phosphor stripes -> vignette / bezel falloff -> flicker + rolling bright bar. `glow` is
    the pre-blurred bright-pass of the input (any buffer when glow_amt == 0)."""
    wf = ti.cast(w, ti.f32)
    hf = ti.cast(h, ti.f32)
    tw = ti.max(3.0, 2.25 * scale)
    for y, x in ti.ndrange(h, w):
        ux = 2.0 * (ti.cast(x, ti.f32) + 0.5) / wf - 1.0
        uy = 2.0 * (ti.cast(y, ti.f32) + 0.5) / hf - 1.0
        sx = ux * (1.0 + curv * 0.10 * uy * uy)
        sy = uy * (1.0 + curv * 0.14 * ux * ux)
        cov = 1.0
        if curv > 0.0:
            ex = (1.0 - ti.abs(sx)) * 0.5 * wf
            ey = (1.0 - ti.abs(sy)) * 0.5 * hf
            cov = _gp_clamp(ti.min(ex, ey) + 0.5, 0.0, 1.0)
        out = vec3(0.0)
        if cov > 0.0:
            fx = (sx * 0.5 + 0.5) * wf - 0.5
            fy = (sy * 0.5 + 0.5) * hf - 0.5
            col = _gp_bilerp3(img, fx, fy, w, h)
            if conv_px != 0.0:
                off = conv_px * ux
                cr = _gp_bilerp3(img, fx - off, fy, w, h)
                cb = _gp_bilerp3(img, fx + off, fy, w, h)
                col = vec3(cr[0], col[1], cb[2])
            if glow_amt > 0.0:
                col += _gp_bilerp3(glow, fx, fy, w, h) * glow_amt
            gain = bright
            v = sy * 0.5 + 0.5
            if scan > 0.0:
                l = _gp_clamp(_gp_luma(col), 0.0, 1.0)
                ph = v * lines
                dd = ph - ti.floor(ph) - 0.5
                sg = 0.22 + 0.30 * l
                prof = ti.exp(-(dd * dd) / (2.0 * sg * sg))
                col = col * (1.0 - scan * (1.0 - prof))
                gain *= 1.0 + 0.45 * scan
            if mask > 0.0:
                xp = (ti.cast(x, ti.f32) + 0.5) / tw
                mr = 0.5 + 0.5 * ti.cos(6.2831853 * xp)
                mg = 0.5 + 0.5 * ti.cos(6.2831853 * (xp - 0.33333))
                mb = 0.5 + 0.5 * ti.cos(6.2831853 * (xp - 0.66667))
                col = col * vec3((1.0 - mask) + mask * 2.0 * mr,
                                 (1.0 - mask) + mask * 2.0 * mg,
                                 (1.0 - mask) + mask * 2.0 * mb)
            d2 = sx * sx + sy * sy
            vv = ti.max(0.0, 1.0 - vig * 0.55 * d2)
            edge = ti.max(ti.abs(sx), ti.abs(sy))
            vv *= 1.0 - vig * 0.35 * _gp_smoothstep(0.7, 1.0, edge)
            band = 0.5 + 0.5 * ti.cos(6.2831853 * (v - band_phase))
            band = ti.pow(band, 6.0)
            flick = flick_gain * (1.0 + flick_amt * 0.08 * band)
            out = col * (gain * vv * flick * cov)
        _gp_st3(dst, y, x, out)


# --- eye-adaptation metering -----------------------------------------------

@ti.kernel
def _gp_meter_rows(rows: _gp_nd2, w: ti.i32, h: ti.i32, n: ti.i32):
    """Per-row partial sums of the centre-weighted log-luminance metering
    (see RayTracer.measure_target_exposure) read straight from ACCUM -- the
    host only has to add up 2*h floats instead of pulling the whole
    MAX_W x MAX_H x 3 float buffer across the bus every frame."""
    n_f = ti.cast(n, ti.f32)
    cx = ti.cast(w - 1, ti.f32) * 0.5
    cy = ti.cast(h - 1, ti.f32) * 0.5
    for y in range(h):
        sl = 0.0
        sw = 0.0
        ddy = (ti.cast(y, ti.f32) - cy) / (ti.cast(h, ti.f32) * 0.5)
        for x in range(w):
            a = ACCUM[y, x]
            luma = (a[0] * 0.2126 + a[1] * 0.7152 + a[2] * 0.0722) / n_f
            ddx = (ti.cast(x, ti.f32) - cx) / (ti.cast(w, ti.f32) * 0.5)
            d = ti.sqrt(ddx * ddx + ddy * ddy)
            wt = 1.0 - 0.65 * _gp_clamp(d, 0.0, 1.0)
            sl += wt * ti.log(ti.max(luma, 1e-4))
            sw += wt
        rows[0, y] = sl
        rows[1, y] = sw


# --- host-side orchestration ------------------------------------------------

class _GpuPostBuffers:
    """Right-sized scratch arrays for one (width, height). Everything is
    allocated lazily the first time an effect needs it, so a frame with only
    DoF enabled doesn't pay for the analog-camera noise grid, etc. A new
    instance (and so freed old arrays) is created whenever the render
    resolution changes."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self._arrs = {}
        self._out = {}

    def _get(self, name, dtype, shape):
        a = self._arrs.get(name)
        if a is None:
            a = ti.ndarray(dtype, shape=shape)
            self._arrs[name] = a
        return a

    a = property(lambda s: s._get('a', ti.f32, (s.h, s.w, 3)))
    b = property(lambda s: s._get('b', ti.f32, (s.h, s.w, 3)))
    t1 = property(lambda s: s._get('t1', ti.f32, (s.h, s.w, 3)))
    t2 = property(lambda s: s._get('t2', ti.f32, (s.h, s.w, 3)))
    prx = property(lambda s: s._get('prx', ti.f32, (s.h, s.w + 1, 3)))
    pry = property(lambda s: s._get('pry', ti.f32, (s.h + 1, s.w, 3)))
    d0 = property(lambda s: s._get('d0', ti.f32, (s.h, s.w)))
    d1 = property(lambda s: s._get('d1', ti.f32, (s.h, s.w)))
    sky = property(lambda s: s._get('sky', ti.f32, (s.h, s.w)))
    hm0 = property(lambda s: s._get('hm0', ti.f32, (s.h, s.w)))
    hm1 = property(lambda s: s._get('hm1', ti.f32, (s.h, s.w)))
    noise = property(lambda s: s._get('noise', ti.f32, (s.h, s.w, 4)))
    noise_dummy = property(lambda s: s._get('noise_dummy', ti.f32, (1, 1, 4)))
    flares = property(lambda s: s._get('flares', ti.f32, (_GP_MAX_FLARES, 8)))
    gauss = property(lambda s: s._get('gauss', ti.f32, (32,)))
    meter = property(lambda s: s._get('meter', ti.f32, (2, s.h)))

    def up(self, ow, oh):
        """Float RGB buffer at the OUTPUT size (clean HUD overlay on an upscaled live view)."""
        return self._get(f'up{ow}x{oh}', ti.f32, (oh, ow, 3))

    def out(self, ow, oh):
        a = self._out.get((ow, oh))
        if a is None:
            a = ti.ndarray(ti.i32, shape=(oh, ow))
            self._out[(ow, oh)] = a
        return a


_GP_BUFFERS = None


def _gp_buffers(w, h):
    global _GP_BUFFERS
    if _GP_BUFFERS is None or (_GP_BUFFERS.w, _GP_BUFFERS.h) != (w, h):
        _GP_BUFFERS = None            # drop the old arrays before allocating new ones
        _GP_BUFFERS = _GpuPostBuffers(w, h)
    return _GP_BUFFERS


def _gp_box_blur(buf, src, dst, r, w, h):
    """GPU _box_blur (separable, edge-replicate). `dst` may be `src` (the
    blur only reads the prefix-sum arrays once they're built)."""
    if r <= 0:
        if dst is not src:
            _gp_copy(src, dst, w, h)
        return
    _gp_prefix_rows(src, buf.prx, w, h)
    _gp_box_h(buf.prx, dst, w, h, int(r))
    _gp_prefix_cols(dst, buf.pry, w, h)
    _gp_box_v(buf.pry, dst, w, h, int(r))


def _gp_upload_flares(buf, flares):
    arr = np.zeros((_GP_MAX_FLARES, 8), dtype=np.float32)
    n = min(len(flares), _GP_MAX_FLARES)
    for i in range(n):
        sx, sy, color, brightness = flares[i]
        arr[i, 0] = sx
        arr[i, 1] = sy
        arr[i, 2], arr[i, 3], arr[i, 4] = color[0], color[1], color[2]
        arr[i, 5] = brightness
    buf.flares.from_numpy(arr)
    return n


def gpu_post_process(width, height, flares, post_fx, frame_seed=0, camera_pose=None,
                     motion_px=None, post_fx_on=True, out_size=None, bgr=False, hud=None):
    """GPU equivalent of
        (np.clip(apply_post_processing(...), 0, 1) * 255).astype(np.uint8)
    (+ optional RGB->BGR and resize to `out_size`) -- same effects, same
    order, same parameter defaults. Expects resolve_output() to have already
    filled OUTPUT/DEPTH_OUT for the active (width x height) region (see
    RayTracer._resolve_gpu). Returns a C-contiguous uint8 (oh, ow, 3) array."""
    w, h = int(width), int(height)
    ow, oh = (w, h) if out_size is None else (int(out_size[0]), int(out_size[1]))
    buf = _gp_buffers(w, h)
    cur, oth = buf.a, buf.b
    depth = buf.d0
    _gp_load(cur, depth, w, h)

    fx_active = bool(post_fx_on and post_fx)
    hud_in_fx = hud if (hud is not None and hud.fx and fx_active and hud.sprites) else None
    hud_overlay = hud if (hud is not None and hud_in_fx is None and hud.sprites) else None

    if fx_active:
        flares = list(flares or [])

        # 1) Depth of field
        if post_fx.get('dof_enabled', False):
            max_radius = max(1, int(round(post_fx['dof_max_radius'])))
            blur_strength = float(post_fx['dof_blur_strength'])
            if blur_strength > 0:
                _gp_prefix_rows(cur, buf.prx, w, h)
                _gp_dof(cur, buf.prx, depth, oth, w, h,
                        float(post_fx['dof_focus_distance']), blur_strength, max_radius)
                cur, oth = oth, cur

        # 2) Zoom focus-hunt (video only -- _zoom_focal_kick is only set by render_video)
        zoom_kick = float(post_fx.get('_zoom_focal_kick', 0.0))
        if zoom_kick > 1e-4:
            a = min(1.0, zoom_kick)
            radius = max(1, int(round(2 + 5 * a)))
            ksize = radius * 2 + 1
            kern = cv2.getGaussianKernel(ksize, 0, ktype=cv2.CV_32F).ravel().astype(np.float32)
            wbuf = np.zeros(32, dtype=np.float32)
            wbuf[:ksize] = kern
            buf.gauss.from_numpy(wbuf)
            _gp_gauss_h(cur, buf.t1, buf.gauss, w, h, radius)
            _gp_gauss_v_mix(buf.t1, cur, oth, buf.gauss, w, h, radius, float(a))
            cur, oth = oth, cur

        # 2b) HUD text, FX mode: composited here so everything from fisheye on (bloom, flare,
        #     analog, chroma, VHS, CRT) acts on it, but AFTER DoF so it isn't depth-blurred;
        #     its depth is pulled to 0 so fog doesn't haze it.
        text_mask = None
        if hud_in_fx is not None:
            hud_in_fx.apply_gpu(cur, depth, w, h)
            if (post_fx.get('analog_enabled', False) and post_fx.get('analog_strength', 0.0) > 0
                    and post_fx.get('analog_smear', 0.0) > 0.0 and motion_px is not None):
                hud_in_fx.alpha_gpu(buf.hm0, w, h)     # where the text is, so the smear can skip it
                text_mask = buf.hm0

        # 3) Fisheye (warps depth too, and the flare positions)
        fisheye_rc = None
        if post_fx.get('fisheye_enabled', False):
            strength = post_fx.get('fisheye_strength', 0.32)
            _fa, _frc = _fisheye_geometry(w, h, strength)
            if _fa > 1e-4:
                fisheye_rc = float(_frc)
                _gp_fisheye(cur, depth, oth, buf.d1, w, h, float(_fa), fisheye_rc)
                if text_mask is not None:
                    # warp the text mask exactly like the image (scratch colour output is unused)
                    _gp_fisheye(cur, text_mask, buf.t1, buf.hm1, w, h, float(_fa), fisheye_rc)
                    text_mask = buf.hm1
                cur, oth = oth, cur
                depth = buf.d1
                flares = [(*_fisheye_warp_point(sx, sy, w, h, strength), color, brightness)
                          for (sx, sy, color, brightness) in flares]

        # 4) Fog
        density = post_fx.get('fog_density', 0.0)
        if post_fx.get('fog_enabled', False) and density > 0:
            max_depth = max(1e-3, post_fx.get('fog_max_depth', 260.0))
            falloff = post_fx.get('fog_height_falloff', 0.0)
            use_height = falloff > 0.0 and camera_pose is not None
            cam_y, r0, r1, r2, ht, asp = 0.0, 0.0, 1.0, 0.0, 1.0, 1.0
            if use_height:
                cpos, Rm, ht, asp = camera_pose
                cam_y = float(cpos[1])
                r0, r1, r2 = float(Rm[1, 0]), float(Rm[1, 1]), float(Rm[1, 2])
            fc = post_fx.get('fog_color', (0.6, 0.65, 0.75))
            _gp_fog(cur, depth, oth, w, h, float(density), float(max_depth), float(falloff),
                    float(post_fx.get('fog_base_height', 0.0)), cam_y, r0, r1, r2,
                    float(ht), float(asp), float(fc[0]), float(fc[1]), float(fc[2]), int(use_height))
            cur, oth = oth, cur

        # 5) God rays
        gr_intensity = post_fx.get('godrays_intensity', 0.0)
        if post_fx.get('godrays_enabled', False) and flares and gr_intensity > 0:
            on_screen = [f for f in flares
                         if (-0.3 * w <= f[0] <= 1.3 * w and -0.3 * h <= f[1] <= 1.3 * h)]
            if on_screen:
                nf = _gp_upload_flares(buf, on_screen)
                _gp_godray_src(cur, depth, buf.sky, w, h, float(post_fx.get('godrays_sky_depth', 5000.0)))
                _gp_godrays(cur, buf.sky, buf.flares, nf, oth, w, h,
                            max(1, int(post_fx.get('godrays_samples', 24))),
                            float(post_fx.get('godrays_density', 0.9)),
                            float(post_fx.get('godrays_decay', 0.97)), float(gr_intensity))
                cur, oth = oth, cur

        # 6) Bloom
        if post_fx.get('bloom_enabled', False):
            b_int = post_fx.get('bloom_intensity', 0.45)
            if b_int > 0:
                scale = min(w, h) / 360.0
                b_rad = post_fx.get('bloom_radius', 22)
                r_small = max(1, int(round(b_rad * 0.35 * scale)))
                r_large = max(1, int(round(b_rad * scale)))
                _gp_bloom_bright(cur, buf.t1, w, h, float(post_fx.get('bloom_threshold', 0.62)))
                _gp_box_blur(buf, buf.t1, buf.t2, r_small, w, h)       # glow_small
                _gp_box_blur(buf, buf.t1, buf.t1, r_large, w, h)       # glow_large (2 passes)
                _gp_box_blur(buf, buf.t1, buf.t1, r_large, w, h)
                _gp_bloom_combine(cur, buf.t2, buf.t1, oth, w, h, float(b_int))
                cur, oth = oth, cur

        # 7) Lens flare
        if post_fx.get('flare_enabled', False) and flares and post_fx['flare_intensity'] > 0:
            nf = _gp_upload_flares(buf, flares)
            scale = min(w, h) / 360.0
            radius = max(2.0, post_fx['flare_size'] * scale)
            _gp_lens_flare(cur, buf.flares, nf, oth, w, h, float(radius),
                           float(post_fx['flare_intensity']),
                           float(post_fx.get('flare_anamorphic', 0.0)),
                           float(post_fx.get('flare_halo', 0.0)),
                           float(post_fx.get('flare_ghosts', 0.0)),
                           float(post_fx.get('flare_starburst', 0.0)),
                           int(max(3, min(16, round(post_fx.get('flare_blades', 6))))))
            cur, oth = oth, cur

        # 8) Analog camera
        a_strength = post_fx.get('analog_strength', 0.0)
        if post_fx.get('analog_enabled', False) and a_strength > 0:
            dr = post_fx.get('analog_contrast', 0.0) * a_strength
            hc = post_fx.get('analog_highlight_clip', 0.0)
            halation = post_fx.get('analog_halation', 0.0)
            warmth = post_fx.get('analog_warmth', 0.0)
            grain = post_fx.get('analog_grain', 0.0)
            vignette = post_fx.get('analog_vignette', 0.0)
            sat_eff = max(0.0, 1.0 + (post_fx.get('analog_saturation', 1.0) - 1.0) * a_strength)
            do_hal = halation > 0.0

            _gp_analog_a(cur, oth, buf.t1, w, h, float(dr), float(hc), float(a_strength), int(do_hal))
            cur, oth = oth, cur                       # cur = steps 1+2 applied
            glow = cur
            if do_hal:
                r_h = max(2, int(round(16 * (min(w, h) / 360.0))))
                _gp_box_blur(buf, buf.t1, buf.t1, r_h, w, h)
                _gp_box_blur(buf, buf.t1, buf.t1, r_h, w, h)
                glow = buf.t1
            noise, gw_, gh_, upsample = buf.noise_dummy, 1, 1, 0
            if grain > 0.0:
                gsz = max(1.0, post_fx.get('analog_grain_size', 1.0))
                gh_, gw_ = max(1, int(round(h / gsz))), max(1, int(round(w / gsz)))
                noise = buf.noise
                _gp_noise_grid(noise, gw_, gh_, 3000 + int(frame_seed))
                upsample = int(gsz > 1.0001)
            _gp_analog_b(cur, glow, noise, oth, w, h, float(a_strength), float(halation), int(do_hal),
                         float(warmth), float(grain), int(gw_), int(gh_), int(upsample), float(vignette),
                         float(sat_eff))
            cur, oth = oth, cur

            smear = post_fx.get('analog_smear', 0.0) * a_strength
            max_px = post_fx.get('analog_smear_max_px', 40)
            if smear > 0.0 and max_px > 0 and motion_px is not None:
                dx, dy = motion_px
                mag = math.hypot(dx, dy)
                if mag >= 0.05:
                    length = min(max_px, mag * smear)
                    if length >= 0.4:
                        n_taps = max(3, min(16, int(round(length)) + 3))
                        _gp_smear(cur, oth, w, h, float(dx / mag), float(dy / mag), float(length), int(n_taps),
                                  text_mask if text_mask is not None else buf.d0,
                                  1 if text_mask is not None else 0)
                        cur, oth = oth, cur

        # 9) Chromatic aberration
        if post_fx.get('chroma_enabled', False) and post_fx['chroma_strength'] > 0:
            _gp_chroma(cur, oth, w, h, float(post_fx['chroma_strength']))
            cur, oth = oth, cur

        # 10) VHS
        if post_fx.get('vhs_enabled', False) and post_fx.get('vhs_strength', 1.0) > 0:
            v_str = float(post_fx.get('vhs_strength', 1.0))
            _gp_vhs(cur, oth, w, h, v_str, max(1, int(round(2 * v_str))), 1000 + int(frame_seed))
            cur, oth = oth, cur

        # 10b) CRT monitor
        if post_fx.get('crt_enabled', False):
            scale_c = min(w, h) / 360.0
            glow_amt = float(post_fx.get('crt_glow', 0.0))
            glow_buf = cur
            if glow_amt > 0.0:
                _gp_bloom_bright(cur, buf.t1, w, h, 0.55)
                r_g = max(1, int(round(5 * scale_c)))
                _gp_box_blur(buf, buf.t1, buf.t1, r_g, w, h)
                _gp_box_blur(buf, buf.t1, buf.t1, r_g, w, h)
                glow_buf = buf.t1
            flick = float(post_fx.get('crt_flicker', 0.0))
            rng_c = np.random.default_rng(7000 + int(frame_seed))
            flick_gain = float(np.clip(1.0 + flick * 0.04 * rng_c.normal(), 0.85, 1.15))
            band_phase = float((int(frame_seed) * 0.0113) % 1.0)
            lines_eff = float(min(max(10.0, float(post_fx.get('crt_lines', 240))), h * 0.5))
            _gp_crt(cur, glow_buf, oth, w, h,
                    float(post_fx.get('crt_curvature', 0.0)), float(post_fx.get('crt_scanlines', 0.0)),
                    lines_eff, float(post_fx.get('crt_mask', 0.0)), glow_amt,
                    float(post_fx.get('crt_convergence', 0.0)) * scale_c,
                    float(post_fx.get('crt_vignette', 0.0)), flick_gain, flick, band_phase,
                    float(post_fx.get('crt_brightness', 1.0)), float(scale_c))
            cur, oth = oth, cur

        # 11) Fisheye image-circle mask: black out everything the lens doesn't
        #     cover (last, so no later effect paints over the black).
        if fisheye_rc is not None:
            _gp_circle_mask(cur, w, h, fisheye_rc)

    out = buf.out(ow, oh)
    if hud_overlay is not None:
        # Clean HUD: on the FINISHED frame (no FX touched it). When the window is bigger than
        # the render (live view), upscale first and draw at the output size so the text stays
        # sharp; the layout size tells us which case we are in.
        if (hud_overlay.w, hud_overlay.h) == (ow, oh) and (ow, oh) != (w, h):
            up = buf.up(ow, oh)
            _gp_resize3(cur, up, w, h, ow, oh)
            hud_overlay.apply_gpu(up, None, ow, oh)
            _gp_present(up, out, ow, oh, ow, oh, int(bool(bgr)))
        else:
            hud_overlay.apply_gpu(cur, None, w, h)
            _gp_present(cur, out, w, h, ow, oh, int(bool(bgr)))
    else:
        _gp_present(cur, out, w, h, ow, oh, int(bool(bgr)))
    packed = out.to_numpy()                                   # (oh, ow) int32 -- the ONLY readback
    rgba = packed.view(np.uint8).reshape(oh, ow, 4)           # little-endian: [c0, c1, c2, 0]
    return np.ascontiguousarray(rgba[..., :3])


def gpu_measure_log_luma(width, height, n_samples):
    """Centre-weighted log-average luminance of the accumulation buffer
    (the same quantity measure_target_exposure computes), via the GPU row-sum
    kernel. Returns the metered average luminance."""
    buf = _gp_buffers(int(width), int(height))
    rows = buf.meter
    _gp_meter_rows(rows, int(width), int(height), int(max(1, n_samples)))
    r = rows.to_numpy().astype(np.float64)
    return float(math.exp(r[0].sum() / max(r[1].sum(), 1e-12)))


# =============================================================================
# 13c) Render HUD -- text elements that are part of the RENDER (live view,
#      stills and video), as opposed to the editor overlay of section 14c.
#
#      Config lives in the post_fx dict (so it is saved/loaded with the scene
#      and copied to --multi-gpu workers like every other FX setting):
#        hud_enabled   -- master switch
#        hud_fx        -- True: the text goes through the FX chain (it is
#                         composited right after DoF, so fisheye / bloom /
#                         flare / analog / chroma / VHS / CRT all act on it);
#                         False: a clean overlay drawn on the FINISHED frame
#                         (at the output size, so it stays crisp in the
#                         low-res live view too).
#        hud_lag       -- seconds the HUD trails behind camera rotation
#                         (0 = glued to the screen). The HUD sits where the
#                         camera was looking `hud_lag` seconds ago, so it
#                         drifts opposite to a turn and settles back.
#        hud_elements  -- list of dicts: text, size, x, y, anchor, color,
#                         opacity (see HUD_DEFAULT_ELEMENTS).
#
#      Text is rasterised on the CPU with Pillow into small alpha sprites
#      (cached while the text is unchanged) and blended on the GPU by
#      _gp_hud_blit; the NumPy blend (HudFrame.blend_float / blend_u8) is the
#      fallback and what the non-raytraced preview uses.
# =============================================================================

HUD_ANCHORS = ('top-left', 'top-center', 'top-right', 'left', 'center', 'right',
               'bottom-left', 'bottom-center', 'bottom-right')
HUD_COLOR_PRESETS = (
    ('white', (255, 255, 255)), ('black', (0, 0, 0)), ('gray', (160, 160, 160)),
    ('red', (255, 60, 60)), ('orange', (255, 160, 40)), ('yellow', (255, 230, 60)),
    ('green', (80, 255, 100)), ('cyan', (60, 230, 255)), ('blue', (80, 120, 255)),
    ('magenta', (255, 80, 220)),
)
HUD_DEFAULT_ELEMENTS = (
    {'text': "{hh}:{mm}:{ss}\nZOOM x{zoom}", 'size': 14.0, 'x': 2.0, 'y': 3.0,
     'anchor': 'top-left', 'color': [255, 255, 255], 'opacity': 1.0},
    {'text': "+", 'size': 26.0, 'x': 50.0, 'y': 50.0,
     'anchor': 'center', 'color': [255, 255, 255], 'opacity': 1.0},
    {'text': "Noir", 'size': 18.0, 'x': 98.0, 'y': 97.0,
     'anchor': 'bottom-right', 'color': [255, 255, 255], 'opacity': 1.0},
)
HUD_DIST_FAR = 9999       # {dist} when nothing is hit / the hit is this far or farther
HUD_MIN_PX = 6            # smallest rasterised font height (px) -- keeps tiny live frames legible
HUD_RT_BASE = float(os.environ.get('RT_HUD_EPOCH') or time.time())
# ^ wall-clock origin of {rh}/{rm}/{rs} for video renders (frame time + this), so the clock
#   runs at video speed and every --multi-gpu worker (which inherits it) agrees on it.


def hud_normalize_anchor(name):
    """Accepts the canonical names plus common spellings ('middle-left', 'centre',
    'top', 'bottom_right', ...); unknown strings fall back to 'top-left'."""
    toks = [t for t in re.split(r'[\s_\-]+', str(name).strip().lower()) if t]
    known = {'top', 'bottom', 'left', 'right', 'center', 'centre', 'middle', 'mid'}
    if not toks or any(t not in known for t in toks):
        return 'top-left'
    v = 'top' if 'top' in toks else ('bottom' if 'bottom' in toks else 'middle')
    hz = 'left' if 'left' in toks else ('right' if 'right' in toks else 'center')
    if v == 'middle':
        return 'center' if hz == 'center' else hz
    return f"{v}-{hz}"


def hud_default_element():
    return {'text': "Text", 'size': 14.0, 'x': 50.0, 'y': 50.0, 'anchor': 'center',
            'color': [255, 255, 255], 'opacity': 1.0}


def _hud_norm_element(e):
    """Fills in / clamps every field, so hand-edited scene JSON can't crash the renderer."""
    def num(key, default, lo, hi):
        try:
            v = float(e.get(key, default))
        except (TypeError, ValueError):
            v = default
        if not math.isfinite(v):
            v = default
        return max(lo, min(hi, v))
    col = e.get('color', (255, 255, 255))
    try:
        col = [max(0, min(255, int(round(float(c))))) for c in list(col)[:3]]
        while len(col) < 3:
            col.append(255)
    except (TypeError, ValueError):
        col = [255, 255, 255]
    return {'text': str(e.get('text', '')), 'size': num('size', 14.0, 1.0, 400.0),
            'x': num('x', 0.0, -100.0, 200.0), 'y': num('y', 0.0, -100.0, 200.0),
            'anchor': hud_normalize_anchor(e.get('anchor', 'top-left')),
            'color': col, 'opacity': num('opacity', 1.0, 0.0, 1.0)}


# --- fonts -------------------------------------------------------------------

_HUD_MONO_NAMES = ("DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "UbuntuMono-R.ttf",
                   "NotoSansMono-Regular.ttf", "FreeMono.ttf", "consola.ttf", "lucon.ttf",
                   "Menlo.ttc", "Monaco.ttf", "Courier New.ttf", "cour.ttf",
                   "DroidSansMono.ttf", "CutiveMono.ttf")
_HUD_MONO_PATHS = ("/system/fonts/DroidSansMono.ttf", "/system/fonts/CutiveMono.ttf",
                   "/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Courier.ttc",
                   "C:\\Windows\\Fonts\\consola.ttf", "C:\\Windows\\Fonts\\cour.ttf")
_HUD_ANY_NAMES = ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "NotoSans-Regular.ttf",
                  "Roboto-Regular.ttf", "DroidSans.ttf", "Arial.ttf", "arial.ttf",
                  "Helvetica.ttc", "segoeui.ttf", "FreeSans.ttf")


def _hud_font_ok(path_or_name):
    try:
        ImageFont.truetype(path_or_name, 12)
        return True
    except Exception:
        return False


def _hud_fc_match(pattern):
    """Asks fontconfig (if installed) for a font file; None when unavailable."""
    try:
        r = subprocess.run(['fc-match', '-f', '%{file}', pattern], capture_output=True,
                           text=True, timeout=3)
        p = r.stdout.strip()
        if r.returncode == 0 and p and os.path.isfile(p):
            return p
    except Exception:
        pass
    return None


class HudFontManager:
    """Picks the HUD font once, in this order:
         1) --hud-font PATH (if given and loadable)
         2) ./font.ttf in the current directory
         3) a monospace system font
         4) any system font, and finally Pillow's built-in default font."""

    def __init__(self):
        self.user_path = None
        self._path = None
        self._source = ''
        self._resolved = False
        self._cache = {}

    def set_user_path(self, path):
        self.user_path = path or None
        self._resolved = False
        self._cache.clear()

    def resolve(self):
        if self._resolved:
            return self._path
        self._resolved = True
        if self.user_path:
            if os.path.isfile(self.user_path) and _hud_font_ok(self.user_path):
                self._path, self._source = self.user_path, "--hud-font"
            else:
                print(f"[HUD] --hud-font '{self.user_path}' was not found or is not a usable "
                      f"font file -- trying the fallbacks.")
        if self._path is None:
            local = os.path.join(os.getcwd(), 'font.ttf')
            if os.path.isfile(local) and _hud_font_ok(local):
                self._path, self._source = local, "font.ttf in the current directory"
        if self._path is None:
            cands = [_hud_fc_match('monospace')] + list(_HUD_MONO_PATHS) + list(_HUD_MONO_NAMES)
            for c in cands:
                if c and _hud_font_ok(c):
                    self._path, self._source = c, "system monospace font"
                    break
        if self._path is None:
            cands = [_hud_fc_match('sans-serif'), _hud_fc_match('')] + list(_HUD_ANY_NAMES)
            for c in cands:
                if c and _hud_font_ok(c):
                    self._path, self._source = c, "system font"
                    break
        if self._path is None:
            self._source = "Pillow built-in default font"
        print(f"[HUD] font: {self._path or 'Pillow default'} ({self._source})")
        return self._path

    def get(self, px):
        px = max(1, int(px))
        f = self._cache.get(px)
        if f is None:
            path = self.resolve()
            if path is not None:
                f = ImageFont.truetype(path, px)
            else:
                try:
                    f = ImageFont.load_default(size=px)      # Pillow >= 10.1: scalable
                except TypeError:
                    f = ImageFont.load_default()
            self._cache[px] = f
        return f


# --- text variables ------------------------------------------------------------

_HUD_VAR_RE = re.compile(r'\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_]*)(?::([^{}]*))?\}')
_HUD_DEFAULT_SPEC = {'hh': '02d', 'mm': '02d', 'ss': '02d', 'rh': '02d', 'rm': '02d', 'rs': '02d',
                     't': '.2f', 'zoom': '.2f', 'z': '.2f', 'fov': '.1f', 'fps': '.2f',
                     'px': '.2f', 'py': '.2f', 'pz': '.2f',
                     'yaw': '.1f', 'pitch': '.1f', 'roll': '.1f'}


def hud_format_text(template, variables):
    """Substitutes {name} / {name:format} (Python format specs, e.g. {zoom:.1f}).
    '{{' / '}}' give literal braces, a typed backslash-n is a line break, and an
    unknown {name} is left exactly as typed."""
    template = str(template).replace('\\n', '\n')

    def sub(m):
        tok = m.group(0)
        if tok == '{{':
            return '{'
        if tok == '}}':
            return '}'
        name, spec = m.group(1), m.group(2)
        if name not in variables:
            return tok
        v = variables[name]
        if not spec:
            spec = _HUD_DEFAULT_SPEC.get(name, '')
            if name == 'dist' and isinstance(v, float):
                spec = '.1f'
        try:
            return format(v, spec)
        except (ValueError, TypeError):
            return str(v)
    return _HUD_VAR_RE.sub(sub, template)


def _hud_probe_dist(tracer):
    """Distance from the camera to whatever is under the screen centre (one BVH ray).
    Returns the int 9999 for sky / misses / anything at or beyond 9999."""
    try:
        R = quat_to_matrix(tracer.camera_quat)
        fwd = R[:, 2]
        probe_depth(float(tracer.camera_pos[0]), float(tracer.camera_pos[1]), float(tracer.camera_pos[2]),
                    float(fwd[0]), float(fwd[1]), float(fwd[2]))
        d = float(PROBE_DEPTH[None])
    except Exception:
        return HUD_DIST_FAR
    if not math.isfinite(d) or d >= HUD_DIST_FAR:
        return HUD_DIST_FAR
    return max(0.0, d)


def _hud_variables(tracer, elements, t, frame, fps, real_time):
    """Every variable usable in HUD text (see hud_format_text). `dist` costs one ray
    per frame, so it is only measured when some element actually mentions it."""
    t = max(0.0, float(t))
    rt = time.localtime(time.time() if real_time is None else real_time)
    w, h = int(tracer.width), int(tracer.height)
    yaw, pitch, roll = euler_from_quat(tracer.camera_quat)
    zoom = float(tracer.zoom)
    v = {
        'hh': int(t // 3600), 'mm': int((t % 3600) // 60), 'ss': int(t % 60),
        'rh': rt.tm_hour, 'rm': rt.tm_min, 'rs': rt.tm_sec,
        't': t, 'zoom': zoom, 'z': zoom,
        'w': w, 'h': h, 'width': w, 'height': h,
        'dist': HUD_DIST_FAR,
        # extras
        'frame': int(frame), 'fps': float(fps), 'fov': float(tracer.fov_deg),
        'px': float(tracer.camera_pos[0]), 'py': float(tracer.camera_pos[1]),
        'pz': float(tracer.camera_pos[2]),
        'yaw': math.degrees(yaw), 'pitch': math.degrees(pitch), 'roll': math.degrees(roll),
    }
    if any('{dist' in str(e.get('text', '')) for e in elements):
        v['dist'] = _hud_probe_dist(tracer)
    return v


# --- camera lag ----------------------------------------------------------------

def _hud_lag_offset(past_q, cur_q, half_tan, aspect, w, h, limit=0.35):
    """Pixel offset of the point the camera was looking at `lag` seconds ago, as seen
    from the current camera -- i.e. how far the HUD trails behind the camera's turn."""
    Rp = quat_to_matrix(past_q)
    Rc = quat_to_matrix(cur_q)
    v = Rc.T @ Rp[:, 2]
    zc = max(float(v[2]), 0.2)
    dx = float(v[0]) / (zc * half_tan * aspect) * (w / 2.0)
    dy = -float(v[1]) / (zc * half_tan) * (h / 2.0)
    return (max(-limit * w, min(limit * w, dx)), max(-limit * h, min(limit * h, dy)))


class HudLagHistory:
    """Recent (time, camera quaternion) samples for the live view, so the HUD can ask
    'where was the camera pointing N seconds ago'. (Video renders sample the camera
    path directly instead, which keeps them deterministic.)"""

    def __init__(self, keep=4.0):
        self.keep = keep
        self.items = []

    def push(self, t, q):
        self.items.append((t, np.array(q, dtype=np.float64)))
        cut = t - self.keep
        while len(self.items) > 2 and self.items[0][0] < cut:
            self.items.pop(0)

    def at(self, tq):
        it = self.items
        if not it:
            return None
        if tq <= it[0][0]:
            return it[0][1]
        if tq >= it[-1][0]:
            return it[-1][1]
        for i in range(len(it) - 1, 0, -1):
            if it[i - 1][0] <= tq <= it[i][0]:
                (t0, q0), (t1, q1) = it[i - 1], it[i]
                return quat_slerp(q0, q1, (tq - t0) / max(t1 - t0, 1e-9))
        return it[-1][1]


# --- compositing helpers (NumPy / OpenCV) -----------------------------------------

def _hud_alpha_roi(mask, x0, y0, W, H):
    """Sprite alpha placed at float position (x0, y0) on a W x H frame, bilinear for
    the fractional part (same sampling as _gp_hud_blit). Returns (xa, ya, alpha_roi)
    clipped to the frame, or None when it is entirely off-screen."""
    sh, sw = mask.shape
    ix, iy = int(math.floor(x0)), int(math.floor(y0))
    fx, fy = x0 - ix, y0 - iy
    if fx == 0.0 and fy == 0.0:
        roi, rw, rh = mask, sw, sh
    else:
        rw, rh = sw + 1, sh + 1
        M = np.array([[1.0, 0.0, fx], [0.0, 1.0, fy]], dtype=np.float32)
        roi = cv2.warpAffine(mask, M, (rw, rh), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    xa, xb = max(ix, 0), min(ix + rw, W)
    ya, yb = max(iy, 0), min(iy + rh, H)
    if xa >= xb or ya >= yb:
        return None
    return xa, ya, roi[ya - iy:yb - iy, xa - ix:xb - ix]


class HudFrame:
    """One frame's worth of laid-out HUD text, ready to composite. `sprites` is a list of
    dicts: mask (float32 alpha, h x w), x0/y0 (frame position of the sprite's corner,
    already including the lag offset), color (r, g, b in 0..1) and opacity.
    (w, h) is the frame size the layout was computed for; `fx` says which stage of the
    FX chain it belongs to (see the section comment)."""

    def __init__(self, w, h, fx):
        self.w, self.h, self.fx = int(w), int(h), bool(fx)
        self.sprites = []

    def blend_float(self, img, depth=None):
        """NumPy composite onto a float RGB image (0..1). When `depth` is given, text
        pixels pull it toward 0 so fog / god rays treat the HUD as 'in front of everything'.
        Returns (img, depth); inputs are not modified."""
        img = img.copy()
        if depth is not None:
            depth = depth.copy()
        H, W = img.shape[:2]
        for s in self.sprites:
            r = _hud_alpha_roi(s['mask'], s['x0'], s['y0'], W, H)
            if r is None:
                continue
            xa, ya, a = r
            a = (a * s['opacity'])[..., None]
            reg = img[ya:ya + a.shape[0], xa:xa + a.shape[1]]
            reg[:] = reg * (1.0 - a) + np.array(s['color'], dtype=np.float32) * a
            if depth is not None:
                dreg = depth[ya:ya + a.shape[0], xa:xa + a.shape[1]]
                dreg[:] = dreg * (1.0 - a[..., 0])
        return img, depth

    def alpha_float(self, H, W):
        """Coverage of all text (0..1, opacity ignored) as an (H, W) float32 plane -- used to
        keep HUD text out of the analog motion smear."""
        m = np.zeros((H, W), dtype=np.float32)
        for s in self.sprites:
            r = _hud_alpha_roi(s['mask'], s['x0'], s['y0'], W, H)
            if r is None:
                continue
            xa, ya, a = r
            reg = m[ya:ya + a.shape[0], xa:xa + a.shape[1]]
            np.maximum(reg, a, out=reg)
        return m

    def alpha_gpu(self, plane, w, h):
        """GPU version of alpha_float: clears `plane` (h, w) and max-blends every sprite in."""
        _gp_clear1(plane, int(w), int(h))
        for s in self.sprites:
            mask = s['mask']
            sh, sw = mask.shape
            ix, iy = int(math.floor(s['x0'])), int(math.floor(s['y0']))
            if ix >= w or iy >= h or ix + sw + 1 <= 0 or iy + sh + 1 <= 0:
                continue
            _gp_hud_alpha(plane, mask, int(sw), int(sh), ix, iy,
                          float(s['x0'] - ix), float(s['y0'] - iy), int(w), int(h))

    def blend_u8(self, canvas, order='rgb'):
        """In-place composite onto a uint8 (h, w, 3) image; order is 'rgb' or 'bgr'."""
        H, W = canvas.shape[:2]
        for s in self.sprites:
            r = _hud_alpha_roi(s['mask'], s['x0'], s['y0'], W, H)
            if r is None:
                continue
            xa, ya, a = r
            a = (a * s['opacity'])[..., None]
            col = np.array(s['color'], dtype=np.float32) * 255.0
            if order == 'bgr':
                col = col[::-1]
            reg = canvas[ya:ya + a.shape[0], xa:xa + a.shape[1]].astype(np.float32)
            canvas[ya:ya + a.shape[0], xa:xa + a.shape[1]] = np.clip(
                reg * (1.0 - a) + col * a + 0.5, 0, 255).astype(np.uint8)
        return canvas

    def apply_gpu(self, img, depth, w, h):
        """GPU composite in place on a (h, w, 3) Taichi ndarray (and, if `depth` is not
        None, on its (h, w) depth plane)."""
        use_depth = 0 if depth is None else 1
        dep = depth
        if dep is None:
            dep = _gp_buffers(w, h).d0       # any (h, w) plane -- unused when use_depth == 0
        for s in self.sprites:
            mask = s['mask']
            sh, sw = mask.shape
            ix, iy = int(math.floor(s['x0'])), int(math.floor(s['y0']))
            if ix >= w or iy >= h or ix + sw + 1 <= 0 or iy + sh + 1 <= 0:
                continue
            cr, cg, cb = s['color']
            _gp_hud_blit(img, dep, mask, int(sw), int(sh), ix, iy,
                         float(s['x0'] - ix), float(s['y0'] - iy), int(w), int(h),
                         float(cr), float(cg), float(cb), float(s['opacity']), use_depth)


class HudRenderer:
    """Turns the element list + variables into a HudFrame (text -> cached alpha sprites)."""

    def __init__(self):
        self.fonts = HudFontManager()
        self._masks = {}

    def _mask(self, text, px, align):
        key = (text, px, align)
        hit = self._masks.get(key)
        if hit is not None:
            return hit
        font = self.fonts.get(px)
        spacing = max(1, int(round(px * 0.2)))
        probe = ImageDraw.Draw(Image.new('L', (8, 8)))
        l, t, r, b = probe.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align=align)
        bw, bh = max(1, int(math.ceil(r - l))), max(1, int(math.ceil(b - t)))
        pad = 2
        im = Image.new('L', (bw + 2 * pad, bh + 2 * pad), 0)
        ImageDraw.Draw(im).multiline_text((pad - l, pad - t), text, font=font, fill=255,
                                          spacing=spacing, align=align)
        arr = np.ascontiguousarray(np.asarray(im, dtype=np.float32) / 255.0)
        if len(self._masks) > 256:
            self._masks.clear()
        out = (arr, pad, bw, bh)
        self._masks[key] = out
        return out

    def build(self, elements, variables, w, h, offset=(0.0, 0.0), fx=False):
        frame = HudFrame(w, h, fx)
        scale = min(w, h) / 360.0
        moving = abs(offset[0]) > 1e-6 or abs(offset[1]) > 1e-6
        for raw in elements:
            e = _hud_norm_element(raw)
            text = hud_format_text(e['text'], variables)
            if not text.strip():
                continue
            px = int(round(max(float(HUD_MIN_PX), e['size'] * scale)))
            anchor = e['anchor']
            align = 'left' if anchor.endswith('left') or anchor == 'left' else \
                    ('right' if anchor.endswith('right') or anchor == 'right' else 'center')
            try:
                arr, pad, bw, bh = self._mask(text, px, align)
            except Exception as ex:           # a font that can't draw this text must not kill a render
                print(f"[HUD] could not draw {text!r}: {ex}")
                continue
            ax, ay = e['x'] / 100.0 * w, e['y'] / 100.0 * h
            x0 = ax if align == 'left' else (ax - bw / 2.0 if align == 'center' else ax - bw)
            vert = 'top' if anchor.startswith('top') else ('bottom' if anchor.startswith('bottom') else 'middle')
            y0 = ay if vert == 'top' else (ay - bh / 2.0 if vert == 'middle' else ay - bh)
            x0 += offset[0] - pad
            y0 += offset[1] - pad
            if not moving:                    # keep glyph edges pixel-sharp when nothing is lagging
                x0, y0 = float(round(x0)), float(round(y0))
            frame.sprites.append({'mask': arr, 'x0': x0, 'y0': y0,
                                  'color': tuple(c / 255.0 for c in e['color']),
                                  'opacity': e['opacity']})
        return frame


HUD_RENDERER = HudRenderer()


def make_hud_frame(tracer, post_fx, t=0.0, frame=0, fps=0.0, real_time=None,
                   past_quat=None, cur_quat=None, out_size=None, overlay_only=False):
    """Builds this frame's HudFrame, or None when the HUD is off / empty.
      t / frame / fps / real_time -- feed {hh,mm,ss,t}, {frame}, {fps}, {rh,rm,rs}
      past_quat / cur_quat        -- camera orientation `hud_lag` seconds ago / now (lag)
      out_size                    -- final image size (the live window): a non-FX HUD is laid
                                     out at THIS size so it is not blurred by the upscale
      overlay_only                -- force the clean-overlay mode (non-raytraced preview)"""
    if not post_fx or not post_fx.get('hud_enabled', False):
        return None
    elements = post_fx.get('hud_elements') or []
    if not elements:
        return None
    fx = bool(post_fx.get('hud_fx', False)) and bool(post_fx.get('enabled', True)) and not overlay_only
    if fx or out_size is None:
        lw, lh = int(tracer.width), int(tracer.height)
    else:
        lw, lh = int(out_size[0]), int(out_size[1])
    variables = _hud_variables(tracer, elements, t, frame, fps, real_time)
    off = (0.0, 0.0)
    lag = float(post_fx.get('hud_lag', 0.0))
    if lag > 1e-6 and past_quat is not None:
        cq = tracer.camera_quat if cur_quat is None else cur_quat
        off = _hud_lag_offset(past_quat, cq, tracer.half_tan, tracer.aspect, lw, lh)
    return HUD_RENDERER.build(elements, variables, lw, lh, off, fx)


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


# ----------------------------------------------------------------------------
# Z-buffer quick preview (alternative to the painter's algorithm above).
#
# WHY: draw_preview() sorts faces by their MEAN depth and paints back-to-front.
# Mean depth is only a heuristic: a big floor/wall quad has a mean depth that is
# often "nearer" than the boxes sitting on it, so it gets painted OVER them and
# faces seem to vanish depending on the viewing angle. Fixing it by sorting
# harder (per-box, per-pixel, BSP...) is slow, and splitting faces gets worse
# when the camera is inside a box.
#
# HOW (here): a real per-pixel depth test, done on the GPU/Taichi backend:
#   1. numpy projects every face once (vectorised; only the few faces crossing
#      the near plane go through a small Python clip) and splits quads into
#      triangles. NO sorting at all.
#   2. pass 1 : every covered pixel does atomic_min of the (perspective-correct)
#      depth -- stored as int32 bits (positive floats order like their bits).
#   3. pass 2 : every covered pixel whose depth equals the winning depth writes
#      its face colour (and a 1px black edge if `lines` is on, depth-correct).
# Work is split per scanline-of-triangle so large faces (floors) are spread
# over many GPU threads. Cost does not depend on draw order or on whether the
# camera is inside an object. Toggle with the 7 key; the painter's version
# stays untouched as the fallback.
# ----------------------------------------------------------------------------

_zb_i2 = ti.types.ndarray(dtype=ti.i32, ndim=2)
_zb_i1 = ti.types.ndarray(dtype=ti.i32, ndim=1)
_zb_f2 = ti.types.ndarray(dtype=ti.f32, ndim=2)
_ZB_FAR_BITS = 0x7F7FFFFF   # bit pattern of the largest finite float32


@ti.func
def _zb_eval(tv: ti.template(), t, px: ti.f32, py: ti.f32, want_edge: ti.i32, flags: ti.i32):
    x0 = tv[t, 0]; y0 = tv[t, 1]; z0 = tv[t, 2]
    x1 = tv[t, 3]; y1 = tv[t, 4]; z1 = tv[t, 5]
    x2 = tv[t, 6]; y2 = tv[t, 7]; z2 = tv[t, 8]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    covered = 0
    zbits = _ZB_FAR_BITS
    edge = 0
    if ti.abs(area) > 1e-6:
        inv = 1.0 / area
        w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) * inv
        w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) * inv
        w2 = 1.0 - w0 - w1
        eps = -1e-4   # inclusive edges: adjacent triangles leave no cracks
        if w0 >= eps and w1 >= eps and w2 >= eps:
            covered = 1
            iz = w0 / z0 + w1 / z1 + w2 / z2      # perspective-correct: 1/z is linear on screen
            zbits = ti.bit_cast(1.0 / iz, ti.i32)
            if want_edge != 0 and flags != 0:
                aa = ti.abs(area)
                d_a = 1e9; d_b = 1e9; d_c = 1e9
                if (flags & 1) != 0:   # edge v0-v1 (opposite v2)
                    d_a = w2 * aa / ti.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + 1e-12)
                if (flags & 2) != 0:   # edge v1-v2 (opposite v0)
                    d_b = w0 * aa / ti.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + 1e-12)
                if (flags & 4) != 0:   # edge v2-v0 (opposite v1)
                    d_c = w1 * aa / ti.sqrt((x0 - x2) ** 2 + (y0 - y2) ** 2 + 1e-12)
                if ti.min(d_a, ti.min(d_b, d_c)) < 1.0:
                    edge = 1
    return covered, zbits, edge


@ti.kernel
def _zb_clear(zbuf: _zb_i2, img: _zb_i2, w: ti.i32, h: ti.i32, bg: ti.i32):
    for y, x in ti.ndrange(h, w):
        zbuf[y, x] = _ZB_FAR_BITS
        img[y, x] = bg


@ti.kernel
def _zb_pass_depth(tv: _zb_f2, flags: _zb_i1, r_tri: _zb_i1, r_y: _zb_i1,
                   r_x0: _zb_i1, r_x1: _zb_i1, n_rows: ti.i32, zbuf: _zb_i2):
    for r in range(n_rows):
        t = r_tri[r]
        y = r_y[r]
        for x in range(r_x0[r], r_x1[r] + 1):
            covered, zb, edge = _zb_eval(tv, t, ti.cast(x, ti.f32), ti.cast(y, ti.f32), 0, 0)
            if covered != 0:
                ti.atomic_min(zbuf[y, x], zb)


@ti.kernel
def _zb_pass_color(tv: _zb_f2, flags: _zb_i1, tcol: _zb_i1, r_tri: _zb_i1, r_y: _zb_i1,
                   r_x0: _zb_i1, r_x1: _zb_i1, n_rows: ti.i32, zbuf: _zb_i2, img: _zb_i2,
                   want_edge: ti.i32):
    for r in range(n_rows):
        t = r_tri[r]
        y = r_y[r]
        for x in range(r_x0[r], r_x1[r] + 1):
            covered, zb, edge = _zb_eval(tv, t, ti.cast(x, ti.f32), ti.cast(y, ti.f32),
                                         want_edge, flags[t])
            if covered != 0 and zb <= zbuf[y, x] + 2:   # winner (2-ulp tolerance)
                if edge != 0:
                    img[y, x] = 0
                else:
                    img[y, x] = tcol[t]


_ZB_CACHE = {}


def _zb_static(geo):
    """Per-scene face list (world space), built once and cached."""
    c = _ZB_CACHE.get('geo')
    if c is not None and c[0] is geo:
        return c[1]
    cube_verts, cube_colors, quad_verts, quad_colors = geo
    faces, cols = [], []
    if len(cube_colors) > 0:
        fw = cube_verts[:, _CUBE_FACE_IDX, :]                    # (N,6,4,3)
        faces.append(fw.reshape(-1, 4, 3))
        cols.extend([c_ for c_ in cube_colors for _ in range(6)])
    if len(quad_colors) > 0:
        faces.append(np.asarray(quad_verts, dtype=float).reshape(-1, 4, 3))
        cols.extend(list(quad_colors))
    if not faces:
        data = (np.zeros((0, 4, 3)), np.zeros((0,), dtype=np.int32))
    else:
        packed = np.array([int(np.clip(c_[2], 0, 255))
                           | (int(np.clip(c_[1], 0, 255)) << 8)
                           | (int(np.clip(c_[0], 0, 255)) << 16) for c_ in cols], dtype=np.int32)
        data = (np.concatenate(faces, axis=0), packed)
    _ZB_CACHE['geo'] = (geo, data)
    return data


def _zb_clip_near(face_cam, near):
    """Sutherland-Hodgman against z = near; returns the clipped polygon in camera space."""
    out = []
    n = len(face_cam)
    for i in range(n):
        curr, prev = face_cam[i], face_cam[i - 1]
        ci, pi = curr[2] > near, prev[2] > near
        if ci != pi:
            d = curr[2] - prev[2]
            t = (near - prev[2]) / d if abs(d) > 1e-9 else 0.0
            out.append(prev + t * (curr - prev))
        if ci:
            out.append(curr)
    return out


def draw_preview_zbuffer(canvas, geo, camera_pos, R, half_tan, aspect, sw, sh, lines=False):
    """Per-pixel depth-tested version of draw_preview (same signature). Raises
    on any GPU/backend problem so the caller can fall back to the painter."""
    faces_w, face_col = _zb_static(geo)
    h, w = int(sh), int(sw)
    st = _ZB_CACHE.get('buf')
    if st is None or st[0] != (h, w):
        st = ((h, w), ti.ndarray(ti.i32, shape=(h, w)), ti.ndarray(ti.i32, shape=(h, w)))
        _ZB_CACHE['buf'] = st
    zbuf, img = st[1], st[2]
    bg = 30 | (25 << 8) | (20 << 16)   # canvas background (30,25,20) in BGR, packed b | g<<8 | r<<16
    _zb_clear(zbuf, img, w, h, bg)
    tvs, tcs, tfs = [], [], []
    if len(faces_w) > 0:
        near = _PREVIEW_NEAR
        hw, hh = sw / 2.0, sh / 2.0
        cam = (faces_w - camera_pos) @ R                          # == R.T @ rel per vertex
        z = cam[:, :, 2]
        front = (z > near).all(axis=1)
        back = (z <= near).all(axis=1)
        # ---- fully-in-front faces: fully vectorised
        fi = np.nonzero(front)[0]
        if len(fi):
            c = cam[fi]
            zz = c[:, :, 2]
            sx = (c[:, :, 0] / (zz * half_tan * aspect)) * hw + hw
            sy = (-c[:, :, 1] / (zz * half_tan)) * hh + hh
            V = np.stack([sx, sy, zz], axis=2)                    # (F,4,3)
            tvs.append(V[:, [0, 1, 2], :].reshape(-1, 9)); tfs.append(np.full(len(fi), 3, np.int32))
            tvs.append(V[:, [0, 2, 3], :].reshape(-1, 9)); tfs.append(np.full(len(fi), 6, np.int32))
            tcs.append(face_col[fi]); tcs.append(face_col[fi])
        # ---- faces crossing the near plane: clip (few), fan-triangulate
        pi_ = np.nonzero(~front & ~back)[0]
        for idx in pi_:
            poly = _zb_clip_near(cam[idx], near)
            n = len(poly)
            if n < 3:
                continue
            P = np.asarray(poly)
            sx = (P[:, 0] / (P[:, 2] * half_tan * aspect)) * hw + hw
            sy = (-P[:, 1] / (P[:, 2] * half_tan)) * hh + hh
            V = np.stack([sx, sy, P[:, 2]], axis=1)
            for k in range(1, n - 1):
                fl = (1 if k == 1 else 0) | 2 | (4 if k == n - 2 else 0)
                tvs.append(V[[0, k, k + 1], :].reshape(1, 9)); tfs.append(np.array([fl], np.int32))
                tcs.append(face_col[idx:idx + 1])
    if tvs:
        tv = np.ascontiguousarray(np.concatenate(tvs, axis=0), dtype=np.float32)
        tf = np.concatenate(tfs).astype(np.int32)
        tc = np.concatenate(tcs).astype(np.int32)
        xs, ys = tv[:, 0::3], tv[:, 1::3]
        mnx, mxx, mny, mxy = xs.min(1), xs.max(1), ys.min(1), ys.max(1)
        ok = (mxx >= 0) & (mnx <= w - 1) & (mxy >= 0) & (mny <= h - 1) & np.isfinite(mnx + mxx + mny + mxy)
        if ok.any():
            tv, tf, tc = tv[ok], tf[ok], tc[ok]
            x0 = np.clip(np.floor(mnx[ok]), 0, w - 1).astype(np.int32)
            x1 = np.clip(np.ceil(mxx[ok]), 0, w - 1).astype(np.int32)
            y0 = np.clip(np.floor(mny[ok]), 0, h - 1).astype(np.int32)
            y1 = np.clip(np.ceil(mxy[ok]), 0, h - 1).astype(np.int32)
            cnt = (y1 - y0 + 1).astype(np.int64)
            n_rows = int(cnt.sum())
            r_tri = np.repeat(np.arange(len(tv), dtype=np.int32), cnt)
            starts = np.cumsum(cnt) - cnt
            r_y = (y0[r_tri] + (np.arange(n_rows) - np.repeat(starts, cnt))).astype(np.int32)
            r_x0, r_x1 = x0[r_tri], x1[r_tri]
            tv = np.ascontiguousarray(tv)
            _zb_pass_depth(tv, tf, r_tri, r_y, r_x0, r_x1, n_rows, zbuf)
            _zb_pass_color(tv, tf, tc, r_tri, r_y, r_x0, r_x1, n_rows, zbuf, img, 1 if lines else 0)
    packed = img.to_numpy()
    canvas[:] = packed.view(np.uint8).reshape(h, w, 4)[..., :3]



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


def _hud_line_width(line):
    """A HUD line is either a plain string or a list of (text, (R,G,B))
    segments (lets one line mix colours, e.g. the FX status rows)."""
    if isinstance(line, str):
        return _text_width(line)
    return sum(_text_width(t) for t, _ in line)


def _hud_draw_line(canvas, line, x, y):
    if isinstance(line, str):
        _draw_text_shadow(canvas, line, (x, y))
        return
    for text, color in line:
        _draw_text_shadow(canvas, text, (x, y), color=color)
        x += _text_width(text)


def draw_hud_text(canvas, sw, sh, tl_lines, tr_lines, bl_lines, br_lines):
    """Draws 4 groups of info in the 4 screen corners, no background (just
    text + a dark shadow so it stays legible over a bright background).
    Each line may be a str or a list of (text, (R,G,B)) colour segments."""
    pad = 12
    line_h = _HUD_LINE_H
    # cv2.putText anchors at the text BASELINE (bottom-left), unlike pygame's
    # top-left anchor -- offset y by one line height so top-anchored lines
    # still read top-to-bottom in the same visual order as before.

    for i, line in enumerate(tl_lines):
        _hud_draw_line(canvas, line, pad, pad + (i + 1) * line_h - 4)

    for i, line in enumerate(bl_lines):
        y = sh - pad - (len(bl_lines) - i - 1) * line_h
        _hud_draw_line(canvas, line, pad, y)

    for i, line in enumerate(tr_lines):
        w = _hud_line_width(line)
        _hud_draw_line(canvas, line, sw - pad - w, pad + (i + 1) * line_h - 4)

    for i, line in enumerate(br_lines):
        w = _hud_line_width(line)
        y = sh - pad - (len(br_lines) - i - 1) * line_h
        _hud_draw_line(canvas, line, sw - pad - w, y)


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
    if len(data) > 25 * 1024 * 1024:
        print(f"Note: embedding '{path}' ({len(data) / 1048576:.0f} MB) into the scene file -- "
              f"use --no-embed-assets to reference it by path instead.")
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
    """Writes a full scene to a single JSON file at `path`: geometry,
    camera (as a quaternion plus a readable yaw/pitch/roll), lights,
    spotlights, background (including the procedural sky gradient, the
    sun/moon disc and the star field with their enable toggles), the
    volumetric cloud layer, optical-zoom state, shading/exposure/caustic
    settings, post-FX, bounce counts and the camera keyframe path with its
    shake/inertia settings. Anything a scene can currently be configured
    with round-trips through this file -- see load_scene_file for the
    matching reader, which defaults every field added after a given
    version so older scene files still load. Image assets are embedded as
    base64 by default (embed_assets=True) so the file is portable to
    another machine."""
    assets = {} if embed_assets else None
    # Which entry in lights[] is the sun's own distant Light (added by
    # add_sun_light/add_sun), so a reload can restore tracer.sun_light and
    # keep set_sun_angle/set_sun_enabled able to move and mute it. Stored
    # as an index rather than duplicating the Light itself.
    sun_light_index = None
    if getattr(tracer, 'sun_light', None) is not None:
        for i, lt in enumerate(tracer.lights):
            if lt is tracer.sun_light:
                sun_light_index = i
                break
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
        # Volumetric cloud layer (see RayTracer.set_clouds) -- None when
        # no cloud layer has ever been configured.
        'clouds': dict(tracer.clouds) if tracer.clouds is not None else None,
        # Optical zoom state (see set_zoom): base_fov_deg is the unzoomed
        # FOV, 'fov' above is the CURRENT (zoomed) one, so both are needed
        # to restore the shot exactly.
        'base_fov_deg': float(tracer.base_fov_deg),
        'zoom': float(tracer.zoom),
        'zoom_target': float(tracer.zoom_target),
        'zoom_speed': float(tracer.zoom_speed),
        'zoom_min': float(tracer.zoom_min),
        'zoom_max': float(tracer.zoom_max),
        # Star-twinkle cadence for the interactive view (see
        # set_star_update_interval); 0 = frozen field.
        'star_update_interval': float(tracer.star_update_interval),
        'sun_light_index': sun_light_index,
        'lights': [lt.to_dict() for lt in tracer.lights],
        'spotlights': [sl.to_dict() for sl in tracer.spotlights],
        'post_fx': dict(post_fx) if post_fx is not None else copy.deepcopy(DEFAULT_POST_FX),
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
    post_fx = copy.deepcopy(DEFAULT_POST_FX)
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
        # Fields added after the first version of this format -- each
        # defaults to what the renderer used before it existed, so a scene
        # file written by an older version still loads and looks the same.
        'clouds': data.get('clouds'),
        'base_fov_deg': data.get('base_fov_deg', data.get('fov', 65)),
        'zoom': float(data.get('zoom', 1.0)),
        'zoom_target': float(data.get('zoom_target', data.get('zoom', 1.0))),
        'zoom_speed': float(data.get('zoom_speed', 1.0)),
        'zoom_min': float(data.get('zoom_min', 1.0)),
        'zoom_max': float(data.get('zoom_max', 8.0)),
        'star_update_interval': float(data.get('star_update_interval', 0.0)),
        'sun_light_index': data.get('sun_light_index'),
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
    'red': (255, 0, 0),
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
    scene.add_box((0, -1.1, 0), (60, 1, 60), (200, 200, 205), roughness=0.35)
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

    # MC skin for no reason
    scene.add_minecraft_character(
    "skin.png", position=(4, 6, -6), rotation=(0, 0, 0), degrees=True,
    pose={'head': (20, 10, 0),        # yaw, pitch, roll
          'right_arm': (0, 0, -150),  # raised outward
          'left_arm': 35,             # a bare number is a pitch
          'left_leg': (0, -25, 0)},
    scale=1/4
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
    
    #scene.add_box((0, -1, 0), (60, 1, 60), (200, 200, 205), roughness=0.35)
    #scene.add_water_surface(0)
    #scene.add_image((0, 2.4, 0), (15.5, 5), "larp.png", emissive=4)
    #scene.add_image((8.5, 0.9, -0.1), (0.81, 2), "miku_wonder.png")
    
    #scene.add_water_surface(0)
    #scene.add_box((0, 2.48, 5), (10, 5, 1), (255, 255, 255))
    #scene.add_image((0, 2.48, 4.49), (None, 5), "screenshot.png")
    #scene.add_image((-5.2, 0.9, 4), (None, 1.8), "osage_cheer.png")
    #scene.add_image((4.6, 0.9, 4), (None, 1.8), "osage_cheer.png")

    scene.add_box((0, -40.5, 0), (240, 1, 240), (200, 200, 205), roughness=0.35)
    scene.add_box((-6, 4, 0), (6, 8, 6), (235, 245, 255), roughness=0.0,
                  transparency=0.95, ior=1.5)
    scene.add_box((4, 3, -6), (5, 6, 5), (180, 60, 60), roughness=0.0,
                  transparency=0.9, ior=1.45)
    scene.add_box((8, 3, 4), (6, 6, 6), (230, 200, 90), roughness=0.02,
                  reflection_k=0.9)
    scene.add_box((-8, 2.5, 8), (5, 5, 5), (210, 210, 220), roughness=0.15,
                  reflection_k=0.8)
    scene.add_box((0, 2, -10), (4, 4, 4), (200, 60, 60), roughness=0.9)

    scene.add_box((-2, 3, 14), (4, 4, 4), (90, 140, 220), roughness=0.25,
                  rotation=(math.radians(30), math.radians(15), 0))

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

    scene.add_water(
        center=(0, -19, 0),
        size=(240, 40, 240),
        color=(120, 220, 240),
        transparency=0.02
    )

    scene.add_minecraft_character(
    "skin.png", position=(4, 6, -6), rotation=(0, 0, 0), degrees=True,
    pose={'head': (20, 10, 0),      
          'right_arm': (0, 0, -150),
          'left_arm': 35,           
          'left_leg': (0, -25, 0)},
    scale=1/4
    )
    scene.add_box((0, 1.5, -2), (2, 4, 2), (220, 50, 50), roughness=0.5)


    return scene

# Default pose for --skin-demo (degrees). Shows the pose API: a walking stride, a raised
# waving arm and a turned head. See Scene.add_minecraft_character for the axis conventions.
SKIN_DEMO_POSE_DEG = {
    'head': (25.0, 8.0, 0.0),          # yaw 25 toward the character's right, tip up 8
    'body': (0.0, -4.0, 0.0),          # bow forward slightly (negative pitch)
    'right_arm': (0.0, 0.0, -150.0),   # negative roll raises the RIGHT arm outward/up
    'left_arm': (0.0, 30.0, 0.0),      # swing forward
    'right_leg': (0.0, 28.0, 0.0),
    'left_leg': (0.0, -28.0, 0.0),
}


def build_skin_demo_scene(skin_path, position=(0.0, 0.0, 0.0), yaw_deg=0.0, pose_deg=None,
                           slim=None, scale=1.0 / 16.0, overlay=True):
    """Minimal showcase scene: a floor plus one Minecraft character, posed in code.

    HOW TO ROTATE JOINTS FROM CODE -- the scene is immutable once the RayTracer exists
    (the BVH is built and uploaded a single time), so a pose is part of scene
    CONSTRUCTION. Pass a {joint: (yaw, pitch, roll)} dict to add_minecraft_character:

        scene.add_minecraft_character(
            "skin.png", position=(0, 0, 0), rotation=(0, 0, 0), degrees=True,
            pose={'head': (20, 10, 0),          # turn right 20, look up 10
                  'right_arm': (0, 0, -150),    # raise right arm (negative roll = outward)
                  'left_arm': 35,               # a bare number is a pitch: swing forward 35
                  'left_leg': (0, -25, 0)})

    The same call can be repeated with different poses to place several copies, and a
    pose can be computed -- e.g. one scene per animation frame, rendered with
    --headless, using pose={'left_leg': 30 * math.sin(t), ...}. Returned world-space
    joint pivots let you attach props to a hand, etc."""
    scene = Scene()
    scene.add_floor(y=0.0, color=(95, 105, 95), roughness=0.85, visibility=80.0 * max(scale * 16.0, 1.0))
    scene.add_minecraft_character(
        skin_path, position=position, rotation=(yaw_deg, 0.0, 0.0), degrees=True, scale=scale,
        pose=SKIN_DEMO_POSE_DEG if pose_deg is None else pose_deg,
        slim=slim, overlay_layer=overlay)
    return scene


def _parse_skin_pose(s):
    """'head=20,10,0;right_arm=0,0,-150;left_leg=-25' -> {'head': (20,10,0), ...} (degrees).
    Each value is 'yaw,pitch,roll' or a single number (= pitch)."""
    pose = {}
    for item in s.replace('|', ';').split(';'):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(f"Bad --skin-pose entry {item!r}: expected joint=value (joints: {', '.join(MC_JOINTS)})")
        name, val = item.split('=', 1)
        nums = [float(x) for x in val.split(',')]
        if len(nums) not in (1, 3):
            raise ValueError(f"Bad --skin-pose value for {name.strip()!r}: give 'pitch' or 'yaw,pitch,roll'")
        pose[_mc_joint_name(name)] = nums[0] if len(nums) == 1 else tuple(nums)
    return pose


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
    'f1': {65470},
    'f5': {65474},
    'f6': {65475},
    'f7': {65476},
    'f9': {65478},
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

def _kf_roll_text(kf):
    return f"  --  roll {math.degrees(kf.roll):.1f} deg" if abs(kf.roll) > 1e-6 else ""


def keyframe_options_menu_cv(canvas, sw, sh, kf, idx):
    """Per-keyframe options (digits choose). A TIMED keyframe (one with an explicit
    duration -- a J "hold", or any keyframe given a duration) shows/edits its
    DURATION in seconds instead of a speed (its speed is only used for the leg
    that LEAVES it, so it moves to option 6). A normal keyframe edits speed;
    option 6 lets it be given a fixed duration too (0 clears it again).
    Option 5 teleports the live camera to this keyframe's pose and closes."""
    timed = kf.duration is not None
    if timed:
        labels = ["1) Duration +0.5 s", "2) Duration -0.5 s",
                  "3) Enter an exact duration (terminal; 0 = back to speed-based)",
                  "4) Set zoom (terminal)", "5) Set camera pose to this keyframe",
                  "6) Speed leaving this keyframe (terminal)",
                  "7) Delete this keyframe", "8) Close (Esc)"]
    else:
        labels = ["1) Speed +0.5 units/s", "2) Speed -0.5 units/s",
                  "3) Enter an exact speed (terminal)",
                  "4) Set zoom (terminal)", "5) Set camera pose to this keyframe",
                  "6) Give this keyframe a fixed duration (terminal)",
                  "7) Delete this keyframe", "8) Close (Esc)"]
    result = ('cancel', None)
    choosing = True
    while choosing:
        canvas[:] = (22, 18, 18)
        if timed:
            title = (f"Camera keyframe #{idx + 1}  --  hold/duration: {kf.duration:.2f} s"
                     f"  --  zoom: {kf.zoom:.2f}x{_kf_roll_text(kf)}")
        else:
            title = (f"Camera keyframe #{idx + 1}  --  speed: {kf.speed:.2f} units/second"
                     f"  --  zoom: {kf.zoom:.2f}x{_kf_roll_text(kf)}")
        tw = _text_width(title)
        _draw_text_shadow(canvas, title, (sw // 2 - tw // 2, sh // 2 - 90))
        for i, label in enumerate(labels):
            lw = _text_width(label)
            y = sh // 2 - 40 + i * 28
            _draw_text_shadow(canvas, label, (sw // 2 - lw // 2, y))
        hint = "Press 1-8 to choose -- Esc to close"
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
            result, choosing = (('duration_delta', 0.5) if timed else ('speed_delta', 0.5)), False
        elif ascii_code == ord('2'):
            result, choosing = (('duration_delta', -0.5) if timed else ('speed_delta', -0.5)), False
        elif ascii_code == ord('3'):
            result, choosing = (('duration_prompt', None) if timed else ('speed_prompt', None)), False
        elif ascii_code == ord('4'):
            result, choosing = ('zoom_prompt', None), False
        elif ascii_code == ord('5'):
            result, choosing = ('set_pose', None), False
        elif ascii_code == ord('6'):
            result, choosing = (('speed_prompt', None) if timed else ('duration_prompt', None)), False
        elif ascii_code == ord('7'):
            result, choosing = ('delete', None), False
        elif ascii_code == ord('8'):
            result, choosing = ('cancel', None), False

    if result[0] == 'speed_prompt':
        try:
            raw_in = input(f"Enter a new speed for keyframe #{idx + 1} "
                            f"(world units/second, currently {kf.speed:.2f}): ")
            result = ('set_speed', max(0.01, float(raw_in)))
        except Exception:
            print("Invalid value -- speed left unchanged.")
            result = ('cancel', None)
    elif result[0] == 'duration_prompt':
        try:
            cur = "none" if kf.duration is None else f"{kf.duration:.2f} s"
            raw_in = input(f"Enter a new duration for keyframe #{idx + 1} in seconds "
                            f"(currently {cur}; 0 = no fixed duration, use speed): ")
            val = float(raw_in)
            result = ('set_duration', None if val <= 0 else max(0.05, val))
        except Exception:
            print("Invalid value -- duration left unchanged.")
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


def keyframe_list_menu_cv(canvas, sw, sh, camera_path, open_keyframe_fn, set_pose_fn=None):
    """Blocking modal menu (F10): Up/Down selects a keyframe from the FULL
    list, Enter opens ITS options menu (delegates straight to
    open_keyframe_fn -- the SAME _open_keyframe_menu closure the crosshair/
    O-key path already uses, so editing or deleting a keyframe from here
    behaves identically, nothing is reimplemented twice), Esc closes.
    Exists so a keyframe can be picked by name/index instead of needing
    the camera physically aimed near one first -- handy once a path has
    more keyframes than are easy to eyeball from a single viewpoint. Re-
    reads camera_path.keyframes fresh every frame (rather than a cached
    snapshot) so a delete from the sub-menu is reflected immediately, and
    clamps the selection back into range if the list just got shorter.
    T sets the live camera to the highlighted keyframe's pose (via
    set_pose_fn) and closes the list so the result is visible at once;
    open_keyframe_fn may likewise return True (its "Set camera pose"
    option) to close the list as well."""
    sel = 0
    row_h = 24
    top_y = 68
    open_ = True
    while open_:
        n = len(camera_path.keyframes)
        if n == 0:
            break  # last keyframe was deleted from the sub-menu -- nothing left to list
        sel = max(0, min(sel, n - 1))
        canvas[:] = (16, 16, 20)
        title = "Camera keyframes  --  Up/Down select, Enter options, T go to keyframe, Esc close"
        _draw_text_shadow(canvas, title, (24, 32))
        viewport = max(1, (sh - top_y - 14) // row_h)
        top = max(0, min(sel - viewport // 2, max(0, n - viewport)))
        bottom = min(n, top + viewport)
        for row, i in enumerate(range(top, bottom)):
            kf = camera_path.keyframes[i]
            y = top_y + row * row_h
            is_sel = (i == sel)
            color = (255, 255, 130) if is_sel else (195, 195, 195)
            marker = ">" if is_sel else " "
            timing = (f"hold {kf.duration:.2f}s" if kf.duration is not None
                      else f"speed {kf.speed:.2f}")
            label = (f"{marker} Keyframe #{i + 1}  --  pos {kf.pos[0]:.1f}, {kf.pos[1]:.1f}, "
                     f"{kf.pos[2]:.1f}  --  {timing}  --  zoom {kf.zoom:.2f}x{_kf_roll_text(kf)}")
            _draw_text_shadow(canvas, label, (24, y), color=color, scale=_HUD_SCALE_SMALL)
        cv2.imshow(WINDOW_NAME, canvas)
        raw = cv2.waitKeyEx(30)
        kind, val = classify_key(raw)
        if kind == 'arrow':
            if val == 'up':
                sel = (sel - 1) % n
            elif val == 'down':
                sel = (sel + 1) % n
        elif kind == 'char':
            if val == 'enter':
                if open_keyframe_fn(sel):
                    open_ = False
            elif val == 't' and set_pose_fn is not None:
                set_pose_fn(sel)
                open_ = False
            elif val == 'esc':
                open_ = False


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


def light_list_menu_cv(canvas, sw, sh, tracer, open_light_fn, start_sel=0):
    """Blocking modal menu (F9): the light-source counterpart of
    keyframe_list_menu_cv (F10). Up/Down selects from the FULL combined
    list [point lights..., spotlights...] (the same ordering Tab/
    selected_light use), Enter opens THAT light's properties menu via
    open_light_fn(combined_index) -- the same closure the mouse-click path
    uses, so editing from here behaves identically and nothing is
    reimplemented twice -- and Esc closes. Exists so a light can be picked
    by number instead of needing it on screen under the cursor (handy when
    a light is behind the camera, off-screen, or buried among many).
    Re-reads tracer.lights/tracer.spotlights fresh every frame so edits
    made in the sub-menu (brightness, color, position, animation) show up
    in the list immediately. Each row has a small swatch of the light's
    color."""
    sel = start_sel
    row_h = 24
    top_y = 68
    open_ = True
    while open_:
        n_pt = len(tracer.lights)
        n = n_pt + len(tracer.spotlights)
        if n == 0:
            break
        sel = max(0, min(sel, n - 1))
        canvas[:] = (16, 16, 20)
        title = "Lights  --  Up/Down select, Enter opens its options, Esc close"
        _draw_text_shadow(canvas, title, (24, 32))
        viewport = max(1, (sh - top_y - 14) // row_h)
        top = max(0, min(sel - viewport // 2, max(0, n - viewport)))
        bottom = min(n, top + viewport)
        for row, i in enumerate(range(top, bottom)):
            is_spot = i >= n_pt
            lt = tracer.spotlights[i - n_pt] if is_spot else tracer.lights[i]
            y = top_y + row * row_h
            is_sel = (i == sel)
            color = (255, 255, 130) if is_sel else (195, 195, 195)
            marker = ">" if is_sel else " "
            kind_name = "Spotlight" if is_spot else "Light"
            num = (i - n_pt + 1) if is_spot else (i + 1)
            label = (f"{marker} {kind_name} #{num}  --  pos {lt.position[0]:.1f}, "
                     f"{lt.position[1]:.1f}, {lt.position[2]:.1f}  --  brightness {lt.brightness:.2f}")
            if is_spot:
                label += f"  --  cone {lt.cone_angle:.0f} deg  --  soft {lt.softness:.2f}"
                if lt.animation is not None:
                    label += f"  --  anim [{lt.animation.kind}]"
            # Color swatch (light.color is 0-1 RGB; cv2 canvas is BGR).
            sw_bgr = tuple(int(max(0.0, min(1.0, float(c))) * 255) for c in lt.color[::-1])
            cv2.rectangle(canvas, (24, y - 12), (38, y + 2), sw_bgr, -1)
            cv2.rectangle(canvas, (24, y - 12), (38, y + 2), (120, 120, 120), 1)
            _draw_text_shadow(canvas, label, (46, y), color=color, scale=_HUD_SCALE_SMALL)
        cv2.imshow(WINDOW_NAME, canvas)
        raw = cv2.waitKeyEx(30)
        kind, val = classify_key(raw)
        if kind == 'arrow':
            if val == 'up':
                sel = (sel - 1) % n
            elif val == 'down':
                sel = (sel + 1) % n
        elif kind == 'char':
            if val == 'enter':
                open_light_fn(sel)
            elif val == 'esc':
                open_ = False


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
        # Sun/moon gets its own tree, no longer hanging off "Sky light" --
        # it is a separate object, not a property of the sky fill.
        add("[Sun/moon] Enabled", lambda: tracer.background.sun_enabled,
            lambda v: tracer.set_sun_enabled(v), 'bool')
        add("  Intensity", lambda: tracer.background.sun['intensity'],
            lambda v: tracer.set_sun_intensity(v), 'float', 0.05, 0.0, 3.0)
        add("  Azimuth (deg)", lambda: tracer.background.sun['azimuth_deg'],
            lambda v: tracer.set_sun_angle(azimuth_deg=v), 'float', 2.0, 0.0, 360.0)
        add("  Elevation (deg)", lambda: tracer.background.sun['elevation_deg'],
            lambda v: tracer.set_sun_angle(elevation_deg=v), 'float', 2.0, -90.0, 90.0)
        add("  Angular size (deg)", lambda: tracer.background.sun['angular_size_deg'],
            lambda v: tracer.set_sun_appearance(angular_size_deg=v), 'float', 0.2, 0.1, 20.0)
        add("  Glow (deg)", lambda: tracer.background.sun['glow_deg'],
            lambda v: tracer.set_sun_appearance(glow_deg=v), 'float', 1.0, 0.0, 30.0)
        add("  Color R", lambda: tracer.background.sun['color'][0],
            lambda v: tracer.set_sun_appearance(color=(int(v), tracer.background.sun['color'][1],
                                                        tracer.background.sun['color'][2])),
            'int', 4, 0, 255)
        add("  Color G", lambda: tracer.background.sun['color'][1],
            lambda v: tracer.set_sun_appearance(color=(tracer.background.sun['color'][0], int(v),
                                                        tracer.background.sun['color'][2])),
            'int', 4, 0, 255)
        add("  Color B", lambda: tracer.background.sun['color'][2],
            lambda v: tracer.set_sun_appearance(color=(tracer.background.sun['color'][0],
                                                        tracer.background.sun['color'][1], int(v))),
            'int', 4, 0, 255)
    if tracer.background.stars is not None:
        # Stars are their own top-level tree too -- previously these rows
        # were indented under "Sky light" as if they were part of it.
        add("[Stars] Enabled", lambda: tracer.background.stars_enabled,
            lambda v: tracer.set_stars_enabled(v), 'bool')
        add("  Brightness", lambda: tracer.background.stars['brightness'],
            lambda v: tracer.add_stars(**{**tracer.background.stars, 'brightness': v}),
            'float', 0.05, 0.0, 3.0)
        add("  Twinkle", lambda: tracer.background.stars['twinkle'],
            lambda v: tracer.add_stars(**{**tracer.background.stars, 'twinkle': v}),
            'float', 0.05, 0.0, 1.0)
        add("  Twinkle update interval (s, 0=off)", lambda: tracer.star_update_interval,
            lambda v: tracer.set_star_update_interval(v), 'float', 0.25, 0.0, 30.0)
        add("  Density", lambda: tracer.background.stars['density'],
            lambda v: tracer.add_stars(**{**tracer.background.stars, 'density': v}),
            'float', 0.0005, 0.0, 0.02)
        add("  Min size (px)", lambda: tracer.background.stars['min_size'],
            lambda v: tracer.add_stars(**{**tracer.background.stars, 'min_size': v}),
            'float', 0.1, 0.1, 5.0)
        add("  Max size (px)", lambda: tracer.background.stars['max_size'],
            lambda v: tracer.add_stars(**{**tracer.background.stars, 'max_size': v}),
            'float', 0.1, 0.1, 8.0)
        add("  Color variation", lambda: tracer.background.stars['color_variation'],
            lambda v: tracer.add_stars(**{**tracer.background.stars, 'color_variation': v}),
            'float', 0.05, 0.0, 1.0)
        add("  Above horizon only", lambda: tracer.background.stars['above_horizon_only'],
            lambda v: tracer.add_stars(**{**tracer.background.stars, 'above_horizon_only': v}),
            'bool')

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
        # Drift/billow speeds (see set_cloud_drift/sample_cloud_drift) --
        # wind_speed slides the whole cloud layer sideways, billow_speed
        # instead makes the puffs themselves churn/morph over time. These
        # now live on tracer.clouds itself (alongside 'wind'/'billow', the
        # CURRENT drifted position) so they save/reload with the rest of
        # the cloud config -- see set_clouds.
        add("  Wind speed X", lambda: tracer.clouds.get('wind_speed', (0.0, 0.0))[0],
            lambda v: tracer.set_cloud_drift(
                wind_speed=(v, tracer.clouds.get('wind_speed', (0.0, 0.0))[1]),
                billow_speed=tracer.clouds.get('billow_speed', 0.0)),
            'float', 0.2, -50.0, 50.0)
        add("  Wind speed Z", lambda: tracer.clouds.get('wind_speed', (0.0, 0.0))[1],
            lambda v: tracer.set_cloud_drift(
                wind_speed=(tracer.clouds.get('wind_speed', (0.0, 0.0))[0], v),
                billow_speed=tracer.clouds.get('billow_speed', 0.0)),
            'float', 0.2, -50.0, 50.0)
        add("  Billow speed", lambda: tracer.clouds.get('billow_speed', 0.0),
            lambda v: tracer.set_cloud_drift(
                wind_speed=tracer.clouds.get('wind_speed', (0.0, 0.0)), billow_speed=v),
            'float', 0.05, -10.0, 10.0)

    add("[C] Caustics enabled", lambda: tracer.caustics_enabled,
        lambda v: setattr(tracer, 'caustics_enabled', v), 'bool')
    add("  Emissive light strength", lambda: tracer.emissive_light_strength,
        lambda v: setattr(tracer, 'emissive_light_strength', max(0.0, v)), 'float', 0.1, 0.0, 10.0)
    if tracer.has_videos:
        add("  Video playing (SPACE)", lambda: bool(tracer.video_playing),
            lambda v: setattr(tracer, 'video_playing', bool(v)), 'bool')
    add("  Glass caustic strength", lambda: tracer.glass_caustic_strength,
        lambda v: setattr(tracer, 'glass_caustic_strength', v), 'float', 0.1, 0.0, 5.0)
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


def _build_camera_params(tracer, post_fx, camera_path, has_camera_data, res_getset, sample_getset,
                          camera_data=None):
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
    elif camera_data is not None and hasattr(camera_data, 'inertia'):
        # Handheld/footstep shake stays off for recorded camera data (it
        # already carries real motion -- see --handheld-shake's help), but
        # inertia smoothing DOES apply to it: it smooths the recording's
        # own jitter/abrupt direction changes rather than inventing
        # motion. Editable live here, same as for keyframe paths.
        add("Camera-data inertia (smoothing)", lambda: camera_data.inertia,
            lambda v: setattr(camera_data, 'inertia', v), 'float', 0.05, 0.0, 3.0)
        add("  Bounce (overshoot)", lambda: camera_data.inertia_bounce,
            lambda v: setattr(camera_data, 'inertia_bounce', v), 'float', 0.05, 0.0, 1.0)

    add("Live resolution width (px)", res_getset['live_w_get'], res_getset['live_w_set'],
        'int', 16, 16, 1920)
    add("  height (px)", res_getset['live_h_get'], res_getset['live_h_set'], 'int', 9, 9, 1080)
    add("Final render width (px)", res_getset['final_w_get'], res_getset['final_w_set'],
        'int', 64, 64, 7680)
    add("  height (px)", res_getset['final_h_get'], res_getset['final_h_set'], 'int', 36, 36, 4320)

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
    add("  Ghosts", pf('flare_ghosts'), spf('flare_ghosts'), 'float', 0.05, 0.0, 2.0)
    add("  Starburst", pf('flare_starburst'), spf('flare_starburst'), 'float', 0.02, 0.0, 1.0)
    add("  Aperture blades", pf('flare_blades'), spf('flare_blades'), 'int', 1, 3, 12)

    add("[F] Fisheye lens", pf('fisheye_enabled'), spf('fisheye_enabled'), 'bool')
    add("  Strength", pf('fisheye_strength'), spf('fisheye_strength'), 'float', 0.02, 0.0, 1.0)

    add("[B] Bloom", pf('bloom_enabled'), spf('bloom_enabled'), 'bool')
    add("  Threshold", pf('bloom_threshold'), spf('bloom_threshold'), 'float', 0.02, 0.0, 1.0)
    add("  Intensity", pf('bloom_intensity'), spf('bloom_intensity'), 'float', 0.02, 0.0, 3.0)
    add("  Radius (px)", pf('bloom_radius'), spf('bloom_radius'), 'int', 1, 1, 100)

    add("[V] VHS effect", pf('vhs_enabled'), spf('vhs_enabled'), 'bool')
    add("  Strength", pf('vhs_strength'), spf('vhs_strength'), 'float', 0.05, 0.0, 3.0)

    add("[6] CRT monitor", pf('crt_enabled'), spf('crt_enabled'), 'bool')
    add("  Curvature", pf('crt_curvature'), spf('crt_curvature'), 'float', 0.02, 0.0, 1.0)
    add("  Scanlines", pf('crt_scanlines'), spf('crt_scanlines'), 'float', 0.02, 0.0, 1.0)
    add("  Scanline count", pf('crt_lines'), spf('crt_lines'), 'int', 10, 60, 1080)
    add("  Phosphor mask", pf('crt_mask'), spf('crt_mask'), 'float', 0.02, 0.0, 1.0)
    add("  Phosphor glow", pf('crt_glow'), spf('crt_glow'), 'float', 0.02, 0.0, 2.0)
    add("  Convergence (px)", pf('crt_convergence'), spf('crt_convergence'), 'float', 0.1, 0.0, 6.0)
    add("  Vignette", pf('crt_vignette'), spf('crt_vignette'), 'float', 0.02, 0.0, 1.5)
    add("  Flicker", pf('crt_flicker'), spf('crt_flicker'), 'float', 0.02, 0.0, 1.0)
    add("  Brightness", pf('crt_brightness'), spf('crt_brightness'), 'float', 0.02, 0.2, 3.0)

    add("[5] Analog camera (camcorder)", pf('analog_enabled'), spf('analog_enabled'), 'bool')
    add("  Strength", pf('analog_strength'), spf('analog_strength'), 'float', 0.05, 0.0, 2.0)
    add("  Dynamic range crush", pf('analog_contrast'), spf('analog_contrast'), 'float', 0.02, 0.0, 1.0)
    add("  Highlight clip", pf('analog_highlight_clip'), spf('analog_highlight_clip'), 'float', 0.02, 0.0, 1.0)
    add("  Grain", pf('analog_grain'), spf('analog_grain'), 'float', 0.005, 0.0, 0.2)
    add("  Grain size", pf('analog_grain_size'), spf('analog_grain_size'), 'float', 0.1, 1.0, 6.0)
    add("  Halation", pf('analog_halation'), spf('analog_halation'), 'float', 0.02, 0.0, 1.0)
    add("  Vignette", pf('analog_vignette'), spf('analog_vignette'), 'float', 0.02, 0.0, 1.5)
    add("  Warmth", pf('analog_warmth'), spf('analog_warmth'), 'float', 0.01, 0.0, 1.0)
    add("  Saturation (>1 clips)", pf('analog_saturation'), spf('analog_saturation'), 'float', 0.05, 0.0, 4.0)
    add("  Motion smear", pf('analog_smear'), spf('analog_smear'), 'float', 0.05, 0.0, 2.0)

    add("[M] Motion blur (video only)", pf('motion_blur_enabled'), spf('motion_blur_enabled'), 'bool')
    add("  Shutter", pf('motion_blur_shutter'), spf('motion_blur_shutter'), 'float', 0.05, 0.0, 1.0)

    return params


def _postfx_row_geometry(params):
    """Derives tree info for each row from the label-indentation convention (2 leading
    spaces per level: no indent = root, 2 = child, 4 = grandchild) so the menu can draw
    tree-style branch lines without a separate data structure. One dict per row:
      'depth'     -- 0 for a root row, 1 / 2 for nested rows.
      'text'      -- the label with its indent stripped (the tree lines carry the nesting).
      'is_last'   -- this row is the last child of its parent (an "L" corner, not a "T").
      'last'      -- {level: bool}: for each ancestor level 1..depth, whether that ancestor
                     is the last of its siblings (if not, its trunk keeps running past
                     this row).
      'has_child' -- the next row is nested deeper than this one.
    """
    n = len(params)
    depths, texts = [], []
    for p in params:
        label = p['label']
        stripped = label.lstrip(' ')
        depths.append(min(3, (len(label) - len(stripped)) // 2))
        texts.append(stripped)
    geom = []
    for i in range(n):
        d = depths[i]
        last = {}
        for k in range(1, d + 1):
            is_last = True
            for j in range(i + 1, n):
                if depths[j] < k:
                    break
                if depths[j] == k:
                    is_last = False
                    break
            last[k] = is_last
        geom.append({'depth': d, 'text': texts[i], 'is_last': last.get(d, True),
                     'last': last, 'has_child': (i + 1 < n and depths[i + 1] > d)})
    return geom


def _menu_value_text(p):
    val = p['get']()
    k = p['kind']
    if k == 'bool':
        return "ON" if val else "off"
    if k == 'int':
        return f"{int(val)}"
    if k in ('enum', 'header'):
        return str(val)
    if k == 'color':
        return "#%02X%02X%02X" % tuple(int(c) for c in val[:3])
    if k == 'text':
        return "[Enter] edit"
    if k == 'action':
        return "[Enter]"
    if p.get('fmt'):
        return p['fmt'](val)
    return f"{val:.3f}"


def menu_terminal_prompt(canvas, message, prompt="> "):
    """Typed input for the menus (cv2 has no text field): shows a hint on the canvas, then
    reads one line from the terminal. Returns None on EOF/Ctrl-C."""
    ch, cw = canvas.shape[:2]
    cv2.rectangle(canvas, (0, ch // 2 - 22), (cw, ch // 2 + 22), (30, 30, 40), -1)
    _draw_text_shadow(canvas, message, (16, ch // 2 + 4), color=(255, 255, 130), scale=_HUD_SCALE_SMALL)
    try:
        cv2.imshow(WINDOW_NAME, canvas)
        cv2.waitKey(1)
    except Exception:
        pass
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        return None


def postfx_menu_cv(canvas, sw, sh, params, title=None, subtitle=None, rebuild=None):
    """Blocking modal menu (shared by F5 / F6 / F7 / `` ` `` -- see the _build_*_params
    functions for what each passes): Up/Down selects a row, Left/Right adjusts it, Enter
    flips a bool row or runs the row's `on_enter` (typed values, actions), Esc closes.
    Every row edits the live tracer/post_fx/camera_path object directly -- closing the menu
    doesn't "apply" anything, it already IS applied.

    Row kinds: bool, int, float (optional 'fmt' callable), enum ('options'; Left/Right
    cycle), color ([r, g, b] 0-255; Left/Right cycle presets, Enter types a value), text /
    action (Enter only), header (display only). `on_enter(canvas)` may return 'rebuild' or
    ('rebuild', tag): the menu then calls `rebuild()` for a fresh row list (rows added /
    removed / relabelled) and selects the row whose 'tag' matches.

    Layout: root rows sit at a fixed left margin, nested rows are indented and joined to
    their parent by tree-style branch lines; each row's value is right-aligned and joined
    to the label by a dark-gray leader line.
    """
    if title is None:
        title = "Post-FX / renderer properties  --  Up/Down select, Left/Right adjust, Esc close"
    if subtitle is None:
        subtitle = ("[key] before a row = its own hotkey outside this menu, e.g. [Y] toggles "
                    "the same eye-adaptation flag as pressing Y. A few other hotkeys don't have "
                    "a matching row here: [T] focuses instantly (one-shot, not a persistent "
                    "setting), [H] cycles handheld shake and [2] types in an exact camera "
                    "position/angle (both are on CameraPath, not post-fx).")
    sub_lines = list(subtitle) if isinstance(subtitle, (list, tuple)) else [subtitle]
    header_h = 68 + 13 * (len(sub_lines) - 1)
    footer_h = 14
    sel = 0
    n = len(params)
    geom = _postfx_row_geometry(params)

    def _row_h():
        # Derived from the actual window height so every row stays on-screen (down to a
        # legible minimum) no matter the window size.
        return max(14, min(24, (sh - header_h - footer_h) // max(1, min(len(params), 18))))

    row_h = _row_h()
    viewport = max(1, (sh - header_h - footer_h) // row_h)

    tree_x = 42
    root_text_x = 42
    child_indent = 26
    value_right_x = sw - 28
    leader_color = (90, 90, 95)
    trunk_color = (80, 80, 88)

    def trunk_x(level):
        return tree_x + 6 + (level - 1) * child_indent

    open_ = True
    while open_ and n > 0:
        canvas[:] = (16, 16, 20)
        _draw_text_shadow(canvas, title, (24, 32))
        for li, line in enumerate(sub_lines):
            _draw_text_shadow(canvas, line, (24, 50 + 13 * li), color=(150, 150, 150), scale=_HUD_SCALE_SMALL)
        top = max(0, min(sel - viewport // 2, max(0, n - viewport)))
        bottom = min(n, top + viewport)

        for row, i in enumerate(range(top, bottom)):
            p = params[i]
            g = geom[i]
            d = g['depth']
            y = header_h + row * row_h
            y_mid = y - 5
            y_up = y_mid - row_h // 2 + 2          # top boundary of this row's slot
            y_dn = y_mid + row_h // 2 + 2          # bottom boundary
            is_sel = (i == sel)
            color = (255, 255, 130) if is_sel else (195, 195, 195)
            marker = ">" if is_sel else " "

            # --- tree lines ---
            for lvl in range(1, d):                # ancestor trunks that keep running past this row
                if not g['last'].get(lvl, True):
                    cv2.line(canvas, (trunk_x(lvl), y_up), (trunk_x(lvl), y_dn), trunk_color, 1, cv2.LINE_AA)
            if d >= 1:
                tx = trunk_x(d)
                cv2.line(canvas, (tx, y_up), (tx, y_mid if g['is_last'] else y_dn), trunk_color, 1, cv2.LINE_AA)
                cv2.line(canvas, (tx, y_mid), (tx + 14, y_mid), trunk_color, 1, cv2.LINE_AA)
            if g['has_child']:
                cv2.line(canvas, (trunk_x(d + 1), y_mid), (trunk_x(d + 1), y_dn), trunk_color, 1, cv2.LINE_AA)

            if d == 0 or is_sel:
                _draw_text_shadow(canvas, marker, (tree_x - 16, y), color=color, scale=_HUD_SCALE_SMALL)
            text_x = root_text_x + d * child_indent
            _draw_text_shadow(canvas, g['text'], (text_x, y), color=color, scale=_HUD_SCALE_SMALL)

            # --- leader line + right-aligned value ---
            val_s = _menu_value_text(p)
            label_end_x = text_x + _text_width(g['text'], scale=_HUD_SCALE_SMALL)
            val_w = _text_width(val_s, scale=_HUD_SCALE_SMALL)
            val_x = value_right_x - val_w
            leader_x0 = label_end_x + 6
            leader_x1 = val_x - 6 - (16 if p['kind'] == 'color' else 0)
            if leader_x1 > leader_x0:
                cv2.line(canvas, (leader_x0, y_mid), (leader_x1, y_mid), leader_color, 1, cv2.LINE_AA)
            if p['kind'] == 'color':
                c = p['get']()
                cv2.rectangle(canvas, (val_x - 16, y - 9), (val_x - 5, y + 1),
                              (int(c[2]), int(c[1]), int(c[0])), -1)
                cv2.rectangle(canvas, (val_x - 16, y - 9), (val_x - 5, y + 1), (120, 120, 120), 1)
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
                k = p['kind']
                direction = 1 if val == 'right' else -1
                if k == 'bool':
                    p['set'](not p['get']())
                elif k == 'enum':
                    opts = list(p['options'])
                    cur = p['get']()
                    idx = opts.index(cur) if cur in opts else 0
                    p['set'](opts[(idx + direction) % len(opts)])
                elif k == 'color':
                    cur = [int(c) for c in p['get']()[:3]]
                    cols = [list(c) for _, c in HUD_COLOR_PRESETS]
                    idx = cols.index(cur) if cur in cols else (-1 if direction > 0 else 0)
                    p['set'](list(cols[(idx + direction) % len(cols)]))
                elif k in ('header', 'text', 'action'):
                    pass
                else:
                    newv = p['get']() + p['step'] * direction
                    if p['lo'] is not None:
                        newv = max(p['lo'], newv)
                    if p['hi'] is not None:
                        newv = min(p['hi'], newv)
                    if k == 'int':
                        newv = int(round(newv))
                    p['set'](newv)
        elif kind == 'char':
            if val == 'enter':
                p = params[sel]
                if p['kind'] == 'bool':
                    p['set'](not p['get']())
                elif p.get('on_enter') is not None:
                    res = p['on_enter'](canvas)
                    tag = None
                    do_rebuild = False
                    if res == 'rebuild':
                        do_rebuild = True
                    elif isinstance(res, tuple) and res and res[0] == 'rebuild':
                        do_rebuild, tag = True, (res[1] if len(res) > 1 else None)
                    if do_rebuild and rebuild is not None:
                        params = rebuild()
                        n = len(params)
                        geom = _postfx_row_geometry(params)
                        row_h = _row_h()
                        viewport = max(1, (sh - header_h - footer_h) // row_h)
                        new_sel = min(sel, max(0, n - 1))
                        if tag is not None:
                            for ri, rp in enumerate(params):
                                if rp.get('tag') == tag:
                                    new_sel = ri
                                    break
                        sel = new_sel
            elif val == 'esc':
                open_ = False


# =============================================================================
# 19c) HUD menu (F7) -- the HUD's own FX-menu-style tree:
#        [HUD] HUD overlay                (main switch)
#          |- Affected by FX
#          `- Camera lag (s)
#        HUD text
#          |- [1] <content>               (Enter edits the text)
#          |   |- Size / X (%) / Y (%) / Anchor / Color / Opacity / Delete
#          `- + Add text element
# =============================================================================

HUD_MENU_SUBTITLE = (
    "Variables: {hh}{mm}{ss} {rh}{rm}{rs} {t} {zoom}/{z} {w}{h}/{width}{height} {dist}",
    "Extras: {frame} {fps} {fov} {yaw} {pitch} {roll} {px} {py} {pz}   Format: {zoom:.1f}",
    "New line: \\n   Literal braces: {{ }}   X/Y = % of frame; anchor = point of the text at X/Y",
)


def _hud_parse_color(s):
    """'#RRGGBB', 'RRGGBB', 'r,g,b' (0-255) or a preset name -> [r, g, b], else None."""
    s = s.strip().lower()
    if not s:
        return None
    for name, c in HUD_COLOR_PRESETS:
        if s == name:
            return list(c)
    if ',' in s or ' ' in s:
        try:
            parts = [float(x) for x in re.split(r'[,\s]+', s) if x]
            if len(parts) == 3:
                return [max(0, min(255, int(round(v)))) for v in parts]
        except ValueError:
            return None
        return None
    h = s.lstrip('#')
    if len(h) == 6:
        try:
            return [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)]
        except ValueError:
            return None
    return None


def _build_hud_params(post_fx):
    """Rows for the F7 menu. Element rows are tagged so the menu can re-select them after
    an add / delete / retype (see postfx_menu_cv's `rebuild`)."""
    params = []
    elems = post_fx.setdefault('hud_elements', [])

    def add(label, get, set_, kind, step=0.0, lo=None, hi=None, **extra):
        d = {'label': label, 'get': get, 'set': set_, 'kind': kind, 'step': step, 'lo': lo, 'hi': hi}
        d.update(extra)
        params.append(d)

    def pf(key, default):
        return lambda: post_fx.get(key, default)

    def spf(key):
        def setter(v):
            post_fx[key] = v
        return setter

    add("[HUD] HUD overlay", pf('hud_enabled', True), spf('hud_enabled'), 'bool')
    add("  Affected by FX", pf('hud_fx', False), spf('hud_fx'), 'bool')
    add("  Camera lag (s)", pf('hud_lag', 0.0), spf('hud_lag'), 'float', 0.02, 0.0, 2.0)
    add("HUD text", lambda: f"{len(elems)} element(s)", None, 'header', tag=('hud_text',))

    def typed_number(e, key, lo, hi, as_int=False):
        def on_enter(canvas):
            s = menu_terminal_prompt(canvas, f"Type a value for {key} in the terminal ({lo}..{hi}) ...",
                                     f"{key} [{e[key]}]: ")
            if s is None or not s.strip():
                return None
            try:
                v = float(s.replace(',', '.'))
            except ValueError:
                print(f"'{s}' is not a number.")
                return None
            v = max(lo, min(hi, v))
            e[key] = int(round(v)) if as_int else v
            return None
        return on_enter

    for i, e in enumerate(elems):
        shown = str(e.get('text', '')).replace('\n', '\\n')
        if len(shown) > 26:
            shown = shown[:25] + "~"
        shown = shown.encode('ascii', 'replace').decode('ascii')

        def edit_text(canvas, e=e):
            print("\nCurrent text:", repr(e.get('text', '')))
            print("Variables: {hh} {mm} {ss} {rh} {rm} {rs} {t} {zoom} {z} {w} {h} {width} {height} {dist} "
                  "(+ {frame} {fps} {fov} {yaw} {pitch} {roll} {px} {py} {pz}); {zoom:.1f} = number format; "
                  "type \\n for a new line.")
            s = menu_terminal_prompt(canvas, "Type the new text in the terminal (empty = keep) ...", "text: ")
            if s is None or s == "":
                return None
            e['text'] = s.replace('\\n', '\n')
            return ('rebuild', ('elem', id(e)))

        add(f"  [{i + 1}] {shown}", lambda e=e: e.get('text', ''), None, 'text',
            on_enter=edit_text, tag=('elem', id(e)))
        add("    Size (pt @360p)", lambda e=e: e.get('size', 14), lambda v, e=e: e.__setitem__('size', v),
            'int', 1, 4, 200, on_enter=typed_number(e, 'size', 4, 200, True))
        add("    X (%)", lambda e=e: e.get('x', 0.0), lambda v, e=e: e.__setitem__('x', v),
            'float', 0.5, 0.0, 100.0, fmt=lambda v: f"{v:.2f}", on_enter=typed_number(e, 'x', 0.0, 100.0))
        add("    Y (%)", lambda e=e: e.get('y', 0.0), lambda v, e=e: e.__setitem__('y', v),
            'float', 0.5, 0.0, 100.0, fmt=lambda v: f"{v:.2f}", on_enter=typed_number(e, 'y', 0.0, 100.0))
        add("    Anchor", lambda e=e: hud_normalize_anchor(e.get('anchor', 'top-left')),
            lambda v, e=e: e.__setitem__('anchor', v), 'enum', options=HUD_ANCHORS)

        def edit_color(canvas, e=e):
            s = menu_terminal_prompt(canvas, "Type a colour: #RRGGBB, r,g,b (0-255) or a name ...",
                                     "color [" + ",".join(str(c) for c in e.get('color', [255, 255, 255])) + "]: ")
            if s is None or not s.strip():
                return None
            c = _hud_parse_color(s)
            if c is None:
                print(f"Could not read '{s}' as a colour.")
            else:
                e['color'] = c
            return None

        add("    Color", lambda e=e: list(e.get('color', [255, 255, 255])),
            lambda v, e=e: e.__setitem__('color', list(v)), 'color', on_enter=edit_color)
        add("    Opacity", lambda e=e: e.get('opacity', 1.0), lambda v, e=e: e.__setitem__('opacity', v),
            'float', 0.05, 0.0, 1.0, fmt=lambda v: f"{v:.2f}",
            on_enter=typed_number(e, 'opacity', 0.0, 1.0))

        def delete_elem(canvas, e=e):
            if e in elems:
                elems.remove(e)
            return ('rebuild', ('hud_text',))

        add("    Delete this element", lambda: "", None, 'action', on_enter=delete_elem)

    def add_elem(canvas):
        ne = hud_default_element()
        elems.append(ne)
        return ('rebuild', ('elem', id(ne)))

    add("  + Add text element", lambda: "", None, 'action', on_enter=add_elem)
    return params


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
    p.add_argument('--hud-font', type=str, default=None,
                    help="Path to a font file (.ttf/.otf/.ttc) for the render HUD text. If omitted, "
                         "font.ttf in the current directory is tried, then a monospace system font, "
                         "then any available font.")
    p.add_argument('--no-hud', action='store_true',
                    help="Turn the render HUD off for this run (F7 in the interactive window edits it).")
    p.add_argument('--print-keys', action='store_true',
                    help="Debug: print the raw code of every key pressed in the interactive window.")
    p.add_argument('--camera-data', type=str, default=None,
                    help="Load a camera recording and drive the camera position/rotation from it "
                         "instead of the scene's camera keyframes (mutually exclusive with camera "
                         "keyframes -- any existing keyframes on the loaded/built scene are dropped). "
                         "Two file formats are auto-detected: a sensor_record JSON (position x/y/z "
                         "in m, rotation x/y/z in deg mapped to pitch/yaw/roll), or a plain-text "
                         "quaternion track like 'Map_Race_#01.txt' (one ':'-separated "
                         "t:x:y:z:qx:qy:qz:qw sample per line -- see SensorCameraData's docstring). "
                         "Orientation is carried/interpolated as a quaternion either way -- see "
                         "--camera-multiplier / --camera-offset / --camera-rot-offset / --camera-sync "
                         "/ --inertia.")
    p.add_argument('--camera-multiplier', type=float, default=1.0,
                    help="With --camera-data: multiplies every sample's position (x/y/z) by this "
                         "factor before applying --camera-offset.")
    p.add_argument('--camera-offset', type=str, default=None,
                    help="With --camera-data: adds a position (and optionally rotation) offset to "
                         "every sample, applied AFTER --camera-multiplier. Either 3 values "
                         "'x,y,z' (position only, world units -- the original behaviour), or 6 "
                         "'x,y,z,r,p,y' / 'x,y,z/r,p,y' values adding a r,p,y rotation offset (degrees) "
                         "on top -- e.g. --camera-offset 0,5,0,0,0,90. Prefer the separate "
                         "--camera-rot-offset flag for the rotation part; if both are given, "
                         "--camera-rot-offset wins.")
    p.add_argument('--camera-rot-offset', type=str, default=None,
                    help="With --camera-data: rotation offset 'r,p,y' (degrees -- roll,pitch,yaw) "
                         "applied to every sample's orientation, composed as a quaternion (see the "
                         "quaternion utilities near camera_matrix) so it stacks cleanly regardless "
                         "of the recording's own orientation. Preferred over the 6-value form of "
                         "--camera-offset; takes precedence if both are given.")
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
                    help="Adds 'momentum' to camera motion: instead of instantly snapping onto each "
                         "new direction, the camera eases/overshoots through direction changes like "
                         "a real handheld camera can't instantly reverse. Applies BOTH to camera "
                         "KEYFRAME paths (P/O) and, when > 0, to --camera-data replay, where it "
                         "smooths the recording's own jitter and abrupt direction changes (position "
                         "spring + quaternion rotational spring -- see SensorCameraData. "
                         "_build_inertia_cache). 0 = off (exact linear/slerp interpolation of the "
                         "raw data), higher = more momentum/lag. Distinct from --handheld-shake "
                         "(that's noise on top of the path; this changes how the path itself is "
                         "followed) -- the two stack fine together.")
    p.add_argument('--inertia-bounce', type=float, default=None,
                    help="Damping for --inertia's spring: 0 = critically damped (smooth ease, no "
                         "overshoot), 1 (default 0.35) = strongly underdamped (visibly bounces past "
                         "the target before settling). Has no effect if --inertia is 0.")
    p.add_argument('--no-sun', action='store_true',
                    help="Hide the sun/moon disc (and mute the real sun light it casts, if any) "
                         "without discarding its settings -- the scene keeps its angle/color/size, "
                         "so it can be switched back on from the F5 world menu. Also toggleable "
                         "there at any time.")
    p.add_argument('--no-glass-caustics', action='store_true',
                    help="Disable the glass-caustic photon pass (light bunching/shifting through glass "
                         "solids). Per-frame cost is tiny either way; this skips the one-off pass.")
    p.add_argument('--glass-photons', type=int, default=None,
                    help="Photons per (glass cluster, light) for the glass-caustic pass (default 65536). "
                         "Raise (e.g. 500000) for cleaner final stills; lower for faster startup.")
    p.add_argument('--video-time', type=float, default=0.0,
                    help="Playback time (seconds) the scene's video planes start at -- the frame shown "
                         "in stills (--headless without --video) and when the interactive view opens. "
                         "Loops past the video's end. Video renders always start from 0 and play along "
                         "the output timeline.")
    p.add_argument('--video-start-paused', action='store_true',
                    help="Open the interactive view with video planes paused (SPACE plays/pauses).")
    p.add_argument('--skin', type=str, default=None,
                    help="Add a Minecraft player model textured with this skin PNG (64x64, legacy "
                         "64x32, or HD) to the scene -- works on top of the demo scene, --mc-schem or "
                         "--scene. See --skin-pos/--skin-yaw/--skin-pose/--skin-scale/--skin-model.")
    p.add_argument('--skin-pos', type=str, default="0,0,0",
                    help="With --skin: world position 'x,y,z' of the point on the ground between the character's feet.")
    p.add_argument('--skin-yaw', type=float, default=0.0,
                    help="With --skin: turn the whole character about the vertical axis, degrees. 0 faces -Z "
                         "(toward the default camera); positive turns toward the character's right.")
    p.add_argument('--skin-pose', type=str, default=None,
                    help="With --skin: joint rotations in DEGREES, e.g. "
                         "'head=20,10,0;right_arm=0,0,-150;left_leg=-25'. Joints: body, head, right_arm, "
                         "left_arm, right_leg, left_leg. A value is 'yaw,pitch,roll' or one number (= pitch; "
                         "+ swings limbs forward / tips the head up). Positive roll moves a limb toward the "
                         "character's LEFT, so raise the right arm outward with a negative roll.")
    p.add_argument('--skin-scale', type=float, default=1.0 / 16.0,
                    help="With --skin: world units per skin pixel (1/16 = one Minecraft block per world unit, "
                         "i.e. a 2-unit-tall character).")
    p.add_argument('--skin-model', type=str, default='auto', choices=['auto', 'classic', 'slim'],
                    help="With --skin: arm width. 'auto' guesses from the skin's transparency.")
    p.add_argument('--no-skin-overlay', action='store_true',
                    help="With --skin: leave out the overlay layer (hat/jacket/sleeves/pants).")
    p.add_argument('--skin-demo', action='store_true',
                    help="With --skin (and no --scene/--mc-schem): use a minimal showcase scene -- a floor and "
                         "just this character, with the camera framed on it -- instead of the default demo scene. "
                         "Uses a built-in walking/waving pose unless --skin-pose is given.")
    p.add_argument('--texture-size', type=int, default=None,
                    help="Resolution (pixels per side, default 512) every texture and video frame is "
                         "resized to. Raise (e.g. 1024) for sharper video planes at the cost of GPU memory.")
    p.add_argument('--no-emissive-light', action='store_true',
                    help="Emissive faces/video screens still glow, but don't light other surfaces.")
    p.add_argument('--no-stars', action='store_true',
                    help="Hide the star field without discarding its settings (density/seed/sizes "
                         "are kept) -- toggleable again from the F5 world menu.")
    p.add_argument('--star-update-interval', type=float, default=None,
                    help="How often (seconds) the interactive view advances the star twinkle; "
                         "0 freezes the field exactly as baked, which is also the cheapest option "
                         "(each update re-bakes and re-uploads the sky texture). Default 1.0. No "
                         "effect on offline video renders, which sample the twinkle per frame.")
    p.add_argument('--zoom', type=float, default=1.0,
                    help="Starting optical zoom multiplier (see the ;/' keys), 1.0 = unzoomed. Also "
                         "used as the constant zoom for a --headless render/video unless the camera "
                         "path's keyframes carry their own zoom (F10/O menu's 'Set zoom').")
    p.add_argument('--zoom-speed', type=float, default=1.0,
                    help="How fast the optical zoom can rack, in zoom-multiplier/second -- both for "
                         "the interactive ;/' keys and for render_video's per-frame zoom ramp (also "
                         "editable live from the ` post-fx menu). Higher = snappier zoom, lower = "
                         "slower/more cinematic racking.")
    p.add_argument('--postfx-backend', choices=['gpu', 'cpu'], default=None,
                    help="Where the post-processing chain (DoF, bloom, fog, god rays, lens flare, "
                         "analog camera, fisheye, chromatic aberration, VHS) runs. 'gpu' (default): "
                         "as Taichi kernels right after the raytrace, only the finished 8-bit frame "
                         "is read back. 'cpu': the original NumPy/OpenCV chain (reference "
                         "implementation; also what the GPU path falls back to automatically if a "
                         "kernel fails). Can also be set with the RT_POSTFX_BACKEND env var.")
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


def _parse_rot_offset(s):
    """Parses a 'r,p,y' rotation offset (degrees -- roll,pitch,yaw) into an
    offset quaternion (see --camera-rot-offset), composed with
    quat_from_euler using the SAME yaw/pitch/roll convention as
    camera_matrix so it stacks predictably with a sample's own
    orientation (see SensorCameraData.load's rot_offset_quat)."""
    parts = [float(x) for x in s.split(',')]
    if len(parts) != 3:
        raise ValueError(f"Expected 'r,p,y' (degrees), got: {s!r}")
    roll, pitch, yaw = (math.radians(p) for p in parts)
    return quat_from_euler(yaw, pitch, roll)


def _parse_camera_offset(s):
    """Parses --camera-offset, accepting either the original 3-value
    'x,y,z' position-only form, or a 6-value position+rotation form:
    'x,y,z,r,p,y' (6 comma-separated values) or 'x,y,z/r,p,y' (position
    triplet, '/', rotation triplet). Returns (pos_offset_xyz,
    rot_offset_quat_or_None)."""
    s = s.strip()
    if '/' in s:
        pos_part, _, rot_part = s.partition('/')
        pos = _parse_xyz(pos_part)
        rot_q = _parse_rot_offset(rot_part)
        return pos, rot_q
    parts = [p.strip() for p in s.split(',')]
    if len(parts) == 3:
        return tuple(float(p) for p in parts), None
    if len(parts) == 6:
        pos = tuple(float(p) for p in parts[:3])
        rot_q = _parse_rot_offset(','.join(parts[3:]))
        return pos, rot_q
    raise ValueError(f"Expected 'x,y,z' or 'x,y,z,r,p,y' (or 'x,y,z/r,p,y'), got: {s!r}")


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

    global PREVIEW_RES, LIVE_RENDER_RES, FINAL_RENDER_RES, MAX_W, MAX_H, GPU_POSTFX
    if args.postfx_backend is not None:
        GPU_POSTFX = (args.postfx_backend == 'gpu')
    if args.preview_res:
        PREVIEW_RES = _parse_wh(args.preview_res)
    if args.live_res:
        LIVE_RENDER_RES = _parse_wh(args.live_res)
    if args.resolution:
        FINAL_RENDER_RES = _parse_wh(args.resolution)
    MAX_W = max(MAX_W, FINAL_RENDER_RES[0], PREVIEW_RES[0], LIVE_RENDER_RES[0])
    MAX_H = max(MAX_H, FINAL_RENDER_RES[1], PREVIEW_RES[1], LIVE_RENDER_RES[1])

    if args.texture_size:
        TextureLibrary.TEX_SIZE = max(64, int(args.texture_size))   # before any scene loads textures

    loaded = None
    if args.scene:
        loaded = load_scene_file(args.scene)
        scene = loaded['scene']
    elif args.mc_schem:
        scene = build_mc_scene(schem_path=args.mc_schem)
        #scene = build_custom_scene()
    else:
        scene = build_custom_scene()
        #scene = build_demo_scene()

    # --- Minecraft skin (--skin): either a standalone showcase scene or one more object in
    #     whatever scene was just built/loaded. A freshly built scene is the only case where
    #     the showcase may replace it -- never throw away a scene the user asked for.
    skin_showcase = False
    if args.skin:
        skin_pose = _parse_skin_pose(args.skin_pose) if args.skin_pose else None
        skin_slim = {'auto': None, 'classic': False, 'slim': True}[args.skin_model]
        skin_pos = _parse_xyz(args.skin_pos)
        if args.skin_demo and loaded is None and not args.mc_schem:
            skin_showcase = True
            scene = build_skin_demo_scene(args.skin, position=skin_pos, yaw_deg=args.skin_yaw,
                                          pose_deg=skin_pose, slim=skin_slim, scale=args.skin_scale,
                                          overlay=not args.no_skin_overlay)
        else:
            scene.add_minecraft_character(args.skin, position=skin_pos, rotation=(args.skin_yaw, 0.0, 0.0),
                                          degrees=True, scale=args.skin_scale, pose=skin_pose, slim=skin_slim,
                                          overlay_layer=not args.no_skin_overlay)
    elif args.skin_demo:
        print("--skin-demo needs --skin <file.png> -- ignoring it.")
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
        
        bg = Background(color=(20, 25, 35), image_path="sky2.png", brightness=1.0)
        #bg = Background(color=(20, 25, 35), brightness=1.0)
        if skin_showcase and not bg.has_image:
            # The showcase scene shouldn't depend on a sky photo being next to the script
            # (add_sun/add_stars below need a sky image to paint on): use the built-in gradient.
            bg.set_sky_gradient(resolution=(1024, 512))

        #bg.set_sky_gradient(zenith_color=(5, 12, 30), horizon_color=(35, 55, 85), ground_color=(10, 12, 18), curve=0.4, resolution=(600, 400))
        #bg.set_sky_gradient(curve=0.2, resolution=(2000, 1000))
        lights = [
            #Light((-18.41, 29.48, -12.73), (255, 251, 235), 1.1),   # key light, slightly warm
            #Light((-1.47, -4.69, 6.94), (120, 160, 200), 1.0),      # fill light, cooler blue
            Light((-5.88, 16.79, -13.09), (255, 251, 235), 0.0),
        ]
        spotlights = [
            #SpotLight((0.0, 20.0, -5.0), (0.15, -1.0, 0.25), color=(16, 255, 255), brightness=0.3, cone_angle=14.0, softness=0.04),
            #SpotLight((0.0, 20.0, -5.0), (0.15, -1.0, 0.25), color=(255, 16, 255), brightness=0.3, cone_angle=14.0, softness=0.04),
            #SpotLight((0.0, 20.0, -5.0), (0.15, -1.0, 0.25), color=(255, 255, 16), brightness=0.3, cone_angle=14.0, softness=0.04),
            #SpotLight((0.0, 20.0, -5.0), (0.15, -1.0, 0.25), color=(255, 255, 255), brightness=0.3, cone_angle=14.0, softness=0.04),
        ]
        fov = 65
        max_bounce = DEFAULT_MAX_BOUNCE
        live_max_bounce = LIVE_MAX_BOUNCE
        cam_cfg = {'pos': [0.0, 6.0, -30.0], 'yaw': 0.0, 'pitch': 0.0}
        post_fx = copy.deepcopy(DEFAULT_POST_FX)
        camera_path = CameraPath()
        ambient, specular_k, shininess = 0.12, 0.6, 64.0
        caustics_enabled = True
        sky_light_strength, caustic_strength = 0.5, 1.0
        exposure, eye_adapt_enabled, eye_adapt_speed = 1.0, False, 1.4

    if skin_showcase:
        # Frame the character: its 2-unit-tall model (times scale*16) centred in view, from the front.
        h = 2.0 * args.skin_scale * 16.0
        cam_cfg = {'pos': [skin_pos[0], skin_pos[1] + h * 0.55, skin_pos[2] - h * 2.1], 'yaw': 0.0, 'pitch': 0.0}
        fov = 50

    HUD_RENDERER.fonts.set_user_path(args.hud_font)
    if args.no_hud:
        post_fx['hud_enabled'] = False

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
    if 'quat' in cam_cfg:
        # Scenes saved by this version carry the exact quaternion --
        # use it directly rather than round-tripping through Euler.
        tracer.camera_quat = np.array(cam_cfg['quat'], dtype=np.float64)
    else:
        # Backward compatibility with older scene files, which only ever
        # had yaw/pitch (and never a 'roll' or 'quat' field at all):
        # convert their Euler angles to a quaternion on load, same as
        # setting camera_rot/camera_roll always does under the hood.
        tracer.camera_rot = np.array([cam_cfg.get('yaw', 0.0), cam_cfg.get('pitch', 0.0)], dtype=np.float32)
        tracer.camera_roll = float(cam_cfg.get('roll', 0.0))
    tracer.zoom_speed = max(1e-4, float(args.zoom_speed))

    if loaded is not None:
        # A loaded scene already carries its own background (sky gradient,
        # sun/moon disc, star field, and whether each is enabled -- see
        # Background.from_dict) and its own lights list, which ALREADY
        # includes the sun's distant Light if it had one. Calling add_sun/
        # add_stars here would overwrite the loaded sun with the default
        # one and append a second sun Light every time the scene was
        # reloaded, so the defaults below are for a freshly-built scene
        # only. Re-point tracer.sun_light at the light that was exported
        # as the sun, so set_sun_angle/set_sun_enabled can still move and
        # mute it.
        sun_idx = loaded.get('sun_light_index')
        if sun_idx is not None and 0 <= sun_idx < len(tracer.lights):
            tracer.sun_light = tracer.lights[sun_idx]
        clouds_cfg = loaded.get('clouds')
        if clouds_cfg:
            tracer.set_clouds(**clouds_cfg)
        tracer.set_star_update_interval(loaded.get('star_update_interval', 0.0))
        tracer.base_fov_deg = float(loaded.get('base_fov_deg', tracer.base_fov_deg))
        tracer.zoom_min = float(loaded.get('zoom_min', tracer.zoom_min))
        tracer.zoom_max = float(loaded.get('zoom_max', tracer.zoom_max))
        tracer.zoom_speed = max(1e-4, float(loaded.get('zoom_speed', tracer.zoom_speed)))
        tracer.set_zoom(float(loaded.get('zoom', 1.0)))
        tracer.zoom_target = float(loaded.get('zoom_target', tracer.zoom))
        # The sky texture now reflects the loaded sun/stars + their enable
        # flags -- push it to the GPU.
        if BG_FIELD is not None:
            BG_FIELD.from_numpy(tracer.background.image)
        tracer.reset_accumulation()
    else:
        tracer.add_sun()
        tracer.add_stars()

    # Explicit CLI flags win over whatever the scene carried. --zoom-speed
    # and --zoom both have non-None defaults, so "explicit" here means
    # "differs from the default" -- otherwise simply loading a scene would
    # silently reset its stored zoom to 1.0x at its default speed.
    if abs(float(args.zoom_speed) - 1.0) > 1e-9:
        tracer.zoom_speed = max(1e-4, float(args.zoom_speed))
    if loaded is None or abs(float(args.zoom) - 1.0) > 1e-6:
        tracer.set_zoom(float(args.zoom))
    if args.star_update_interval is not None:
        tracer.set_star_update_interval(args.star_update_interval)
    if args.no_sun:
        tracer.set_sun_enabled(False)
    if args.no_emissive_light:
        tracer.emissive_lighting_enabled = False
    if tracer.has_videos:
        tracer.set_video_time(args.video_time)
        if args.video_start_paused:
            tracer.video_playing = False
    if args.no_glass_caustics:
        tracer.glass_caustics_enabled = False
    if args.glass_photons is not None:
        tracer.glass_caustic_photons = max(1024, int(args.glass_photons))
    tracer.compute_glass_caustics()
    if args.no_stars:
        tracer.set_stars_enabled(False)

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
        # --camera-offset takes either 'x,y,z' (position only, as before) or
        # 'x,y,z,r,p,y' / 'x,y,z/r,p,y' (position + rotation, degrees) --
        # see _parse_camera_offset. The separate --camera-rot-offset flag is
        # the preferred way to give the rotation half, and wins if both are
        # supplied.
        cam_offset = (0.0, 0.0, 0.0)
        cam_rot_quat = None
        if args.camera_offset:
            cam_offset, cam_rot_quat = _parse_camera_offset(args.camera_offset)
        if args.camera_rot_offset:
            rot_q = _parse_rot_offset(args.camera_rot_offset)
            if cam_rot_quat is not None:
                print("Both --camera-offset's rotation part and --camera-rot-offset given -- "
                      "using --camera-rot-offset.")
            cam_rot_quat = rot_q
        sensor_data = SensorCameraData.load(args.camera_data, multiplier=cam_multiplier,
                                             offset=cam_offset, rot_offset_quat=cam_rot_quat)
        # --inertia/--inertia-bounce apply to camera-data replay too, not
        # just to keyframe paths: camera_path is discarded just below, so
        # setting them only on it (as the code used to) silently did
        # nothing whenever --camera-data was in play. SensorCameraData runs
        # the same spring (position + quaternion rotation) over the
        # recorded trajectory -- see its _build_inertia_cache.
        if args.inertia:
            sensor_data.inertia = float(args.inertia)
        if args.inertia_bounce is not None:
            sensor_data.inertia_bounce = float(args.inertia_bounce)
        if len(camera_path.keyframes) > 0:
            print(f"--camera-data given -- dropping {len(camera_path.keyframes)} existing camera "
                  f"keyframe(s) from the scene in favor of the sensor recording.")
        camera_path = CameraPath()
        print(f"Loaded {len(sensor_data.samples)} camera samples from '{args.camera_data}' "
              f"({sensor_data.total_duration():.2f}s)")
        if cam_rot_quat is not None:
            y_off, p_off, r_off = euler_from_quat(cam_rot_quat)
            print(f"Camera rotation offset applied: roll/pitch/yaw = "
                  f"{math.degrees(r_off):.1f} / {math.degrees(p_off):.1f} / "
                  f"{math.degrees(y_off):.1f} deg")
        if sensor_data.inertia > 0.0:
            print(f"Camera-data inertia smoothing: {sensor_data.inertia:.2f} "
                  f"(bounce {sensor_data.inertia_bounce:.2f})")
        if args.camera_get_fps:
            avg_fps = sensor_data.average_fps()
            print(f"Sensor recording average fps: {avg_fps:.3f}")
            if avg_fps > 0:
                video_fps = avg_fps
        pos0, yaw0, pitch0, roll0 = sensor_data.sample(0.0)
        tracer.camera_pos = pos0.astype(np.float32)
        # Set the quaternion directly rather than round-tripping the pose
        # back through the Euler setters (camera_rot/camera_roll), which
        # would recompose what the recording already gave us as rotation.
        tracer.camera_quat = quat_from_euler(yaw0, pitch0, roll0)
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
    # While a path/recording is replaying, the pose comes in as a
    # quaternion and THAT is what gets rendered -- camera_rot/camera_roll
    # are refreshed from it for the HUD only. None means "not replaying",
    # i.e. free-look, where the Euler locals (arrow-key deltas) are the
    # source of truth and get composed into a quaternion instead. See
    # _current_camera_quat just below.
    replay_quat = None

    def _current_camera_quat():
        """The orientation actually being rendered this frame: the replay
        quaternion if one is active (gimbal-lock-safe -- a recording can
        legitimately pass through pitch = +-90deg, where yaw/roll stop
        being separately determined), else one composed from the
        free-look Euler locals."""
        if replay_quat is not None:
            return replay_quat
        return quat_from_euler(camera_rot[0], camera_rot[1], camera_roll)
    roll_speed = 0.015   # radians per "held" tick (,/. keys) -- matches look_speed
    move_speed = 0.6
    look_speed = 0.015   # radians per "held" tick (arrow keys) -- also used as a base for mouse-drag
    lines_mode = True
    preview_zbuffer = False     # quick-preview renderer: False = painter's algorithm (default, fast), True = z-buffer ([7] toggles)
    zb_ok = True                # flips to False if the z-buffer path ever fails (auto-fallback)
    live_render = False
    selected_light = 0          # index into the COMBINED list [lights..., spotlights...]
    autofocus_flash = 0.0        # countdown (seconds) for the focus-ring flash effect

    fps_smooth = 0.0
    last_frame_time = time.time()
    _last_zoom_tick = [time.time()]  # mutable cell for the ;/' zoom ramp below (see there)
    _star_clock_start = time.time()
    debug_hud = True              # F1 toggles the editor overlay (text, gizmo, viewfinder)
    hud_hist = HudLagHistory()    # camera orientation history for the HUD's lag option
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
    print("F9: list every light/spotlight (Up/Down + Enter to open one's properties menu)")
    print("  -- in a spotlight's menu: A) configure a looping/keyframed animation, Z) clear it")
    print("[ / ]: narrow / widen a spotlight's cone (when a spotlight is selected)")
    print("T: autofocus DoF on whatever is under the crosshair | - / =: nudge DoF focus distance")
    print("1: toggle wireframe outline | C: toggle caustics")
    print("U: DoF | R: chromatic aberration | N: lens flare | V: VHS | M: motion blur (video only)")
    print("F: fisheye lens | B: bloom | Y: eye adaptation/auto exposure | G: continuous autofocus (video only)")
    print("3: fog | 4: god rays | 5: analog camera (Y2K camcorder: LDR/blowout/grain/smear)")
    print("H: cycle handheld camera shake (keyframe paths only)")
    print("2: type in an exact camera position/angle | `: open the post-FX properties menu")
    print("F1: show/hide the editor overlay (debug text, gizmo, viewfinder)")
    print("F7: HUD settings (text elements, FX / lag options) | 6: CRT monitor effect")
    print("F5: world/atmosphere settings (fog, clouds, sky, sun, stars) | "
          "F6: camera/renderer settings (exposure, zoom, DoF, resolution, samples)")
    print("7: toggle quick-preview algorithm (z-buffer / painter's)")
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
                live_pose = active_path.sample_quat(None)
                if live_pose is not None:
                    rp_pos, rp_quat = live_pose
                    camera_pos[:] = rp_pos
                    # The quaternion is the pose that actually gets
                    # rendered; camera_rot/camera_roll are refreshed from
                    # it only so the HUD readout stays truthful. Going the
                    # other way (Euler -> render) would lose the
                    # orientation at pitch = +-90deg.
                    replay_quat = rp_quat
                    camera_rot[0], camera_rot[1], camera_roll = euler_from_quat(rp_quat)
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
                    replay_quat = None
                    camera_roll = 0.0  # reset roll after a --camera-data replay ends (see module notes)
                    print("Path replay finished.")
                else:
                    rp_pos, rp_quat = active_path.sample_quat(elapsed_replay)
                    camera_pos[:] = rp_pos
                    replay_quat = rp_quat
                    camera_rot[0], camera_rot[1], camera_roll = euler_from_quat(rp_quat)
                    any_input = True
        else:
            replay_quat = None  # free-look: the Euler locals drive the camera again
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

        R = quat_to_matrix(_current_camera_quat())

        n_pt_lights = len(tracer.lights)
        n_all_lights = n_pt_lights + len(tracer.spotlights)
        targeted_kf_idx = find_targeted_keyframe(camera_path, camera_pos, R, tracer.half_tan,
                                                  tracer.aspect, WIN_W, WIN_H)

        def _goto_keyframe(kf_idx):
            # Put the live camera exactly on keyframe `kf_idx` (position, yaw,
            # pitch AND roll). Stops a running replay so it doesn't snatch the
            # camera straight back.
            nonlocal camera_roll, replaying
            kf = camera_path.keyframes[kf_idx]
            camera_pos[:] = kf.pos
            camera_rot[0], camera_rot[1] = kf.yaw, kf.pitch
            camera_roll = kf.roll
            replaying = False
            print(f"Camera set to keyframe #{kf_idx + 1}.")

        def _open_keyframe_menu(kf_idx):
            # Returns True when the pose was applied (callers then close their menu too).
            kf = camera_path.keyframes[kf_idx]
            action, value = keyframe_options_menu_cv(canvas, WIN_W, WIN_H, kf, kf_idx)
            if action == 'speed_delta':
                kf.speed = max(0.01, kf.speed + value)
            elif action == 'set_speed':
                kf.speed = value
            elif action == 'duration_delta':
                kf.duration = max(0.05, (kf.duration or 0.05) + value)
            elif action == 'set_duration':
                kf.duration = value
            elif action == 'set_zoom':
                kf.zoom = value
            elif action == 'set_pose':
                _goto_keyframe(kf_idx)
                return True
            elif action == 'delete':
                camera_path.remove(kf_idx)
                print(f"Keyframe #{kf_idx + 1} deleted.")
            return False

        def _open_light_menu(light_idx):
            # Shared by the mouse-click path and the F9 list menu. `light_idx`
            # indexes the COMBINED [lights..., spotlights...] list; opening
            # a light's menu also makes it the Tab-selected one.
            nonlocal selected_light
            selected_light = light_idx
            n_pt = len(tracer.lights)
            is_spot = light_idx >= n_pt
            lt = tracer.spotlights[light_idx - n_pt] if is_spot else tracer.lights[light_idx]
            if light_options_menu_cv(canvas, WIN_W, WIN_H, lt, light_idx, camera_pos,
                                      camera_rot, camera_roll, is_spot=is_spot):
                tracer.sync_lights()
                tracer.compute_caustics()

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
                _open_light_menu(hit_light)
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
            elif ch == ' ' and inp.one_shot('video_toggle'):
                # Play/pause EVERY video plane at once (they share one clock).
                tracer.toggle_video_playback()
            elif ch == '1' and inp.one_shot('lines'):
                lines_mode = not lines_mode
            elif ch == '7' and inp.one_shot('preview_algo'):
                preview_zbuffer = not preview_zbuffer
                if preview_zbuffer:
                    zb_ok = True   # let the user retry after an automatic fallback
                print(f"Quick preview: {'z-buffer (per-pixel depth)' if preview_zbuffer else 'painter (sorted faces)'}")
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
            elif ch == '6' and inp.one_shot('crt_toggle'):
                post_fx['crt_enabled'] = not post_fx.get('crt_enabled', False)
                print(f"CRT monitor: {post_fx['crt_enabled']}")
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
                postfx_menu_cv(canvas, WIN_W, WIN_H, params,
                               title="Post-FX properties  --  Up/Down select, Left/Right adjust, Esc close")
            elif ch == 'p' and inp.one_shot('add_keyframe'):
                if has_camera_data:
                    print("Camera keyframes are disabled while --camera-data is active.")
                else:
                    camera_path.add(camera_pos.copy(), camera_rot[0], camera_rot[1], speed=5.0,
                                     zoom=tracer.zoom, roll=camera_roll)
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
                    camera_path.hold(duration=max(0.05, dur), yaw=camera_rot[0], pitch=camera_rot[1],
                                     roll=camera_roll)
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
                        # One quaternion set rather than camera_rot-then-
                        # camera_roll: same result, but composes the whole
                        # orientation in one go instead of round-tripping
                        # through Euler twice.
                        tracer.camera_quat = _current_camera_quat()
                        save_scene_file(out_path, scene, tracer, post_fx=post_fx,
                                         camera_path=camera_path, embed_assets=True)
                except Exception as e:
                    print(f"Export failed: {e}")
        elif kind == 'func':
            # F10: keyframe LIST menu -- lets you pick a keyframe by name/
            # index (Up/Down + Enter) instead of needing the camera aimed
            # near one first. O (see near targeted_kf_idx below) is still
            # the quick "edit whichever keyframe I'm looking at" shortcut;
            # this is the "browse all of them" one.
            if val == 'f10' and inp.one_shot('kf_menu'):
                if has_camera_data:
                    print("Camera keyframes are disabled while --camera-data is active.")
                elif not camera_path.keyframes:
                    print("No camera keyframes yet -- press P to add one first.")
                else:
                    keyframe_list_menu_cv(canvas, WIN_W, WIN_H, camera_path, _open_keyframe_menu,
                                          set_pose_fn=_goto_keyframe)
            elif val == 'f9' and inp.one_shot('light_menu'):
                # F9: light LIST menu -- same idea as F10 for keyframes, but
                # for point lights + spotlights. Pick one by number (Up/Down
                # + Enter) instead of having to click its on-screen marker.
                if n_all_lights == 0:
                    print("No lights in the scene.")
                else:
                    light_list_menu_cv(canvas, WIN_W, WIN_H, tracer, _open_light_menu,
                                        start_sel=selected_light)
            elif val == 'f5' and inp.one_shot('world_menu', debounce=0.3):
                # World/atmosphere/lighting settings (fog, clouds, sky, sun,
                # stars, caustics, god rays) -- see _build_world_params.
                params = _build_world_params(tracer, post_fx, camera_path, has_camera_data)
                missing = []
                if tracer.background.sun is None:
                    missing.append("sun/moon (tracer.add_sun(...) to add one)")
                if tracer.background.stars is None:
                    missing.append("stars (tracer.add_stars(...) to add them)")
                world_subtitle = ("World / atmosphere / lighting settings.")
                if missing:
                    world_subtitle += "  Not configured yet, so no rows for: " + "; ".join(missing) + "."
                postfx_menu_cv(canvas, WIN_W, WIN_H, params,
                               title="World settings (F5)  --  Up/Down select, Left/Right adjust, Esc close",
                               subtitle=world_subtitle)
            elif val == 'f1' and inp.one_shot('debug_hud_toggle', debounce=0.3):
                debug_hud = not debug_hud
                print(f"Editor overlay (debug HUD): {'on' if debug_hud else 'off'}")
            elif val == 'f7' and inp.one_shot('hud_menu', debounce=0.3):
                # Render-HUD settings -- see _build_hud_params / section 13c.
                _hud_build = lambda: _build_hud_params(post_fx)
                postfx_menu_cv(canvas, WIN_W, WIN_H, _hud_build(),
                               title="HUD settings (F7)  --  Up/Down select, Left/Right adjust, "
                                     "Enter edit/type, Esc close",
                               subtitle=HUD_MENU_SUBTITLE, rebuild=_hud_build)
            elif val == 'f6' and inp.one_shot('camera_menu', debounce=0.3):
                # Camera/renderer settings (exposure, zoom, DoF, camera
                # shake, resolution, samples-per-mode) -- see
                # _build_camera_params.
                params = _build_camera_params(tracer, post_fx, camera_path, has_camera_data,
                                              res_getset, sample_getset,
                                              camera_data=sensor_data)
                postfx_menu_cv(canvas, WIN_W, WIN_H, params,
                               title="Camera / renderer settings (F6)  --  Up/Down select, "
                                     "Left/Right adjust, Esc close")

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

        # Video planes: advance the shared playback clock by this frame's wall-clock delta
        # (no-op while paused or when the scene has no video) and, if a new video frame was
        # uploaded, restart progressive accumulation like any other animated change. While
        # paused nothing triggers, so the accumulated image keeps converging as usual.
        if live_render and tracer.update_videos(frame_dt):
            any_input = True

        # Cloud drift/billow (see set_cloud_drift/sample_cloud_drift):
        # wall-clock-driven here, same as spotlight animations above --
        # a no-op unless set_cloud_drift was called with a nonzero speed.
        if tracer.clouds is not None and tracer.sample_cloud_drift(now_t):
            any_input = True

        # Star twinkle: throttled (see RayTracer.update_stars' docstring
        # for why) rather than advanced every frame -- "occasionally shift
        # slightly", not a strobe. The interval is
        # tracer.star_update_interval (default 0s/off, adjustable in the F5
        # menu / --star-update-interval); 0 turns the update off and
        # leaves the field frozen as baked. Skipped entirely while the
        # star field is hidden.
        if (tracer.background.stars is not None and tracer.background.stars_enabled
                and tracer.star_update_interval > 0.0 and now_t >= _next_star_update):
            tracer.update_stars(now_t - _star_clock_start)
            _next_star_update = now_t + tracer.star_update_interval
            any_input = True

        # --- Draw the frame ---
        if live_render:
            tracer.camera_pos = camera_pos.astype(np.float32)
            # Single quaternion composition per frame (see the note at the
            # X-key export above) -- the interactive loop keeps yaw/pitch/
            # roll as plain locals because arrow-key look IS an Euler
            # delta, and converts once, here, at that boundary.
            tracer.camera_quat = _current_camera_quat()
            _hud_now = (time.time() - replay_start_time) if replaying else (now_t - _star_clock_start)
            _hud_lag = float(post_fx.get('hud_lag', 0.0))
            hud_hist.push(now_t, tracer.camera_quat)
            canvas = prog.step(any_input, (WIN_W, WIN_H), hud_t=_hud_now,
                               hud_past_quat=(hud_hist.at(now_t - _hud_lag) if _hud_lag > 1e-6 else None))
        else:
            canvas[:] = (30, 25, 20)
            _drawn = False
            if preview_zbuffer and zb_ok:
                try:
                    draw_preview_zbuffer(canvas, preview_geo, camera_pos, R, tracer.half_tan,
                                         tracer.aspect, WIN_W, WIN_H, lines=lines_mode)
                    _drawn = True
                except Exception as _e:
                    zb_ok = False
                    print(f"Z-buffer preview unavailable ({_e}); falling back to the painter's algorithm.")
            if not _drawn:
                canvas[:] = (30, 25, 20)
                draw_preview(canvas, preview_geo, camera_pos, R, tracer.half_tan, tracer.aspect,
                             WIN_W, WIN_H, lines=lines_mode)
            draw_light_indicator(canvas, tracer.lights, selected_light, camera_pos, R,
                                  tracer.half_tan, tracer.aspect, WIN_W, WIN_H,
                                  spotlights=tracer.spotlights)
        if not live_render and post_fx.get('hud_enabled', False):
            # Non-raytraced preview: the HUD as a clean overlay (no FX run in this mode).
            tracer.camera_pos = camera_pos.astype(np.float32)
            tracer.camera_quat = _current_camera_quat()
            _hud_now = (time.time() - replay_start_time) if replaying else (now_t - _star_clock_start)
            _hud_lag = float(post_fx.get('hud_lag', 0.0))
            hud_hist.push(now_t, tracer.camera_quat)
            _hud_prev = make_hud_frame(tracer, post_fx, t=_hud_now, frame=0,
                                       past_quat=(hud_hist.at(now_t - _hud_lag) if _hud_lag > 1e-6 else None),
                                       out_size=(WIN_W, WIN_H), overlay_only=True)
            if _hud_prev is not None:
                _hud_prev.blend_u8(canvas, 'bgr')
        draw_camera_path(canvas, camera_path, targeted_kf_idx, camera_pos, R,
                          tracer.half_tan, tracer.aspect, WIN_W, WIN_H)
        if debug_hud:
            draw_camera_viewfinder(canvas, WIN_W, WIN_H, WIN_W // 2, WIN_H // 2,
                                    focused=(autofocus_flash > 0.0))
        # Small orientation compass, placed just under the pos/yaw-pitch-roll
        # text (3 lines incl. zoom) in the top-left corner.
        if debug_hud:
            draw_orientation_gizmo(canvas, 12 + 45, 12 + 3 * _HUD_LINE_H + 49, R, radius=28)

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
            f"post-fx: {'GPU' if GPU_POSTFX else 'CPU'}",
        ]
        if has_camera_data:
            cam_line = f"camera data: {_camera_data_count_str()}" + (" (following)" if is_live_stream and replaying else " (replaying)" if replaying else "")
        else:
            cam_line = f"keyframes: {len(camera_path.keyframes)}" + (" (replaying)" if replaying else "")
        bl_lines = [
            f"selected {sel_kind} #{sel_num}" if n_all_lights else "no lights in scene",
            cam_line,
        ]
        if not live_render:
            bl_lines.append(f"preview: {'z-buffer' if (preview_zbuffer and zb_ok) else 'painter'} [7]")
        # FX status: enabled effects are YELLOW, disabled ones WHITE. With
        # continuous autofocus active, the "DoF" label itself turns CYAN.
        _FX_ON, _FX_OFF, _FX_AF = (255, 220, 40), (255, 255, 255), (0, 255, 255)
        def _fx(label, on, extra=""):
            return [(f"{label} {'on' if on else 'off'}{extra}", _FX_ON if on else _FX_OFF)]
        _gap = [("  ", _FX_OFF)]
        _dof_on = bool(post_fx['dof_enabled'])
        _af_on = bool(post_fx.get('autofocus_enabled', False))
        br_lines = [
            [("DoF", _FX_AF if _af_on else (_FX_ON if _dof_on else _FX_OFF)),
             (f" {'on' if _dof_on else 'off'}"
              + (f" @ {post_fx['dof_focus_distance']:.1f}" if _dof_on else "")
              + (" (AF)" if _af_on else ""), _FX_ON if _dof_on else _FX_OFF)],
            _fx("VHS", post_fx['vhs_enabled']) + _gap + _fx("flare", post_fx['flare_enabled'])
            + _gap + _fx("caustics", tracer.caustics_enabled),
            _fx("fisheye", post_fx.get('fisheye_enabled', False)) + _gap
            + _fx("bloom", post_fx.get('bloom_enabled', False)) + _gap
            + [(f"exposure {tracer.exposure:.2f}{'(auto)' if tracer.eye_adapt_enabled else ''}",
                _FX_ON if tracer.eye_adapt_enabled else _FX_OFF)],
            _fx("fog", post_fx.get('fog_enabled', False)) + _gap
            + _fx("god rays", post_fx.get('godrays_enabled', False)) + _gap
            + _fx("analog", post_fx.get('analog_enabled', False)),
            _fx("CRT", post_fx.get('crt_enabled', False)) + _gap
            + _fx("HUD", post_fx.get('hud_enabled', False)),
        ]
        if debug_hud:
            draw_hud_text(canvas, WIN_W, WIN_H, tl_lines, tr_lines, bl_lines, br_lines)

        cv2.imshow(WINDOW_NAME, canvas)
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            running = False

    cv2.destroyAllWindows()
    if stream_data is not None:
        stream_data.close()


if __name__ == "__main__":
    main()
