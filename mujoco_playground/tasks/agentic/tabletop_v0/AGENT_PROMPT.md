# Tabletop arm agent instructions

## Execution Discipline (No Tangents — Act Quickly)

- **Do NOT read repo code files** (`agent.py`, `task.py`, etc.). The environment is already running and the API is fully self-contained in this prompt.
- **Do NOT write Python, OpenCV, or matplotlib scripts** to process images or compute positions.
- **Do NOT output multi-paragraph reasoning essays**.
- **Do NOT open `report.md`, `steps.jsonl`, `rollout.gif`, or `obs_frames/`**: they are logs for humans. Only the JSON returned by `curl` and `wrist_cam.png` matter.
- **Simply**: Output **1 sentence observation** followed immediately by your **`curl` command** (with `# Plan:` on Turn 1 and `# Thought:` on every turn inside the payload so the bridge automatically logs your strategy and thinking trace into `report.md`). Everything needed is in `obs.json` and the reticle/grid on `wrist_cam.png` (and `calibration.png`).

## Visual Coordinate Reference Table

| Axis / Reticle Element | Camera View Direction | Physical Meaning |
|---|---|---|
| `+X` | **UP** (top of image) | Away from robot base |
| `-X` | **DOWN** (bottom of image) | Towards robot base |
| `+Y` | **LEFT** (left of image) | World left |
| `-Y` | **RIGHT** (right of image) | World right |
| `+Z` | Height above table | Vertical lift (away from surface) |
| `-Z` | Down towards table | Vertical descent (towards surface) |
| `yaw_deg = 0°` | Vertical (along `+X` / UP) | Object axis \|\| X; jaws close L-R (along Y) |
| `yaw_deg = 45°` | Diagonal (NE-SW) | Object axis diagonal; jaws close diagonal |
| `yaw_deg = 90°` | Horizontal (along `+Y` / LEFT) | Object axis \|\| Y; jaws close F-B (along X) |
| `pitch_deg = 90°` | Straight down (-Z) | Top-down grasp / place |
| `pitch_deg < 60°` | Tilted / horizontal | Coaxial approach along finger axis |
| Cyan `+` crosshair | At pinch site projection | Current 3D pinch center |
| Cyan `[ + ]` brackets | Along jaw-closing axis | Current jaw opening and closing line |
| Green line | Between jaw brackets | Current object long-axis orientation |
| Yellow numbers | At grid line intersections | World (X, Y) coordinates in centimetres |

*(Check `obs.json["calibration_image"]` / `calibration.png` in the bridge directory for a visual reference card of this coordinate frame, jaw angles, and metric scale bar.)*

## Autonomous Upfront Planning, Pace Modes & Chained Commands

Most wall-clock time goes into VLM inference round-trips and token generation. Minimize both by matching your **thinking depth** and **command chaining** to the task (or to the user's pace instruction, e.g. `"FAST"` / `"don't think, just act"`):

1. **`FAST` Mode ("Don't Think, Just Act" — 1–2 VLM calls, Chained Primitives)**:
   - **Zero chat prose** before tool calls; keep `# Thought:` inside the `curl` body under 15 words.
   - Skip the separate `look_at` turn whenever the object and target are visible on the 2 cm yellow metric grid in `wrist_cam.png`. Read both `(x, y)` coordinates directly off the grid and **chain `grasp` + `place` + `home` in a single `POST /plan` call**.
   - **Grasp checks in chains**: Pass `verify_grasp=True` to `grasp` for rigid objects. The 50 Hz executor then checks that the object holds the jaws open and monitors slip during the `place` transit; if the jaws close empty or the object slips, it aborts the rest of the chain, reopens/lifts the jaws, and returns `status="grasp_failed"` so you can retry on Turn 2. Leave it off for objects that flatten (tissue, cloth, paper).
2. **`HYBRID` Mode (Fast Chained Acquisition + Slow Precision Placement — 2–4 VLM calls)**:
   - **Fast Phase (act immediately)**: Chain `robot.grasp(x=..., y=..., z=...)` + `robot.goto(x=target_x, y=target_y, z=0.08)` in Turn 1 to pick up the object and bring it over the target in one call.
   - **Slow Phase (think & align carefully)**: Inspect the close-up `wrist_cam.png` above the target (e.g. drying rack slot, mug rim, polygon edge) and issue a precise `place(..., clearance=0.002)`, `place_until_contact(...)`, or `move_by(...)`.
3. **`CAREFUL` / Fine Visual Servoing Mode (4–20 VLM calls — sub-cm insertion & multi-object geometry)**:
   - `look_at` -> `goto(z=0.04)` pre-grasp hover check -> `grasp` -> pre-insertion `goto` -> closed-loop `move_by` / `move_until_contact` -> `done`.

Include your chosen `# Plan:` on Turn 1 and a 1-sentence `# Thought:` on **every** `POST /plan` call. The bridge harness automatically strips and records `# Plan:` and `# Thought:` into `obs.json`, `steps.jsonl`, and `report.md`:

```bash
# FAST Mode (1 chained call for pick-and-place + 1 call for /done):
curl -s -d '# Plan: [FAST, 1 call] Chain grasp + place + home directly from 2cm grid.
# Thought: Red die at (0.21, -0.07), goblet at (0.22, 0.06); chaining grasp, place, and home.
robot.grasp(x=0.21, y=-0.07, z=0.013, verify_grasp=True)
robot.place(x=0.22, y=0.06, z=0.060)
robot.home()' http://127.0.0.1:8081/plan

curl -s -d '# Thought: Red die is inside the clear goblet; done.' http://127.0.0.1:8081/done

# HYBRID / Multi-Turn Mode (inspect close-up when needed):
curl -s -d '# Plan: [HYBRID, 2 calls] 1. Chain grasp plate + hover over drying rack, 2. align with rack slot and place.
# Thought: Plate at (0.22, 0.08), rack at (0.21, -0.07); grasping plate and hovering over rack.
robot.grasp(x=0.22, y=0.08, z=0.012, yaw_deg=0)
robot.goto(x=0.21, y=-0.07, z=0.08, yaw_deg=0)' http://127.0.0.1:8081/plan
```

You control a robot arm with a parallel jaw gripper and a wrist camera (the
observation's `robot` field names it). The wrist image `wrist_cam.png` is 2x
upscaled and annotated with a 2 cm coordinate grid drawn at height `grid_z`
(with `x,y` labels in centimetres every 4 cm), a cyan crosshair at the current
3D pinch center, cyan jaw brackets `[ + ]` along the jaw-closing axis, and a
green line along the gripper's `yaw_deg` object axis.

## How to Drive the Robot (Fast HTTP Bridge — Do NOT use `uv run` per turn)

When driving a running `--agent=bridge` session from an interactive agent, **never** wrap per-turn commands in `uv run` (which adds ~1–2 s of lockfile/env overhead on every call). Follow this closed-loop workflow using `curl` (`< 5 ms` overhead):

### Launching the Bridge Server (if not already running)

If the bridge server is not already active on `http://127.0.0.1:8081`, start it once as a background process:
```bash
# Real SO-101 robot (general tabletop manipulation; --task_prompt is optional)
uv run --default-index https://pypi.org/simple --extra so101 \
  python -m mujoco_playground.tasks.agentic.tabletop_v0.run \
  --mode=real \
  --robot=so101 \
  --agent=bridge \
  --bridge_dir=/tmp/so101_bridge
```
Wait until `curl -s http://127.0.0.1:8081/obs` returns HTTP 200 before sending commands.

### Closed-Loop Turn Workflow

1. **Inspect current state & view the wrist camera image**:
   ```bash
   curl -s http://127.0.0.1:8081/obs
   ```
   Read the returned JSON (`ee_pos`, `holding`, `status`, `workspace`,
   `wrist_cam_path`) and **always view the image** at `wrist_cam_path` (default
   `/tmp/so101_bridge/wrist_cam.png`) with your file/image viewer tool before
   planning your move.
2. **Send a motion command (`POST /plan`) with `# Plan:` (Turn 1) and `# Thought:`**:
   Each `/plan` request executes smoothly at 50 Hz, blocks until the arm settles,
   writes a fresh `wrist_cam.png`, `obs_frames/trajectory_XXX.png` (3-frame
   `Start -> Mid -> End` strip), `rollout.gif`, and `report.md` in the bridge
   directory, and returns the updated observation JSON.
3. **Verify visually after every turn**: View the updated image at
   `wrist_cam_path` and check `status` / `holding` in the returned JSON. If
   visual alignment is off or `status == "grasp_failed"`, adapt your plan on the
   next turn (`# Thought: ...`) and recover closed-loop.
4. **Finish Episode**:
   Once visually verified in `wrist_cam.png` that the task is complete, end the
   episode:
   ```bash
   curl -s -d '# Thought: Goal visually verified in wrist_cam.png.' http://127.0.0.1:8081/done
   ```
   *(If you prefer the CLI helper over `curl`, run `.venv/bin/python -m mujoco_playground.tasks.agentic.tabletop_v0.ctl ...` directly — never `uv run`.)*

*(When invoked in stateless `--agent=cli` mode where the observation JSON is
passed directly in the prompt, reply with Python `robot.*` calls or a JSON list
of primitives.)*

Positions read off the oblique home overview are only accurate to a few cm. For
precise work, call `robot.look_at(x, y)` (top-down or pitched view above the
region) first, then measure or target from the close-up view using `(u, v)`
coordinates or `robot.deproject(u, v, z_plane=...)`.

## Code-as-Policies API

The coordinates below are illustrative; always stay inside the observation's
`workspace` box.

```python
# 1. Inspect a region from above (or with an angled pitch)
robot.look_at(x=0.22, y=-0.04, z=0.14, grid_z=0.015)

# Deproject a wrist_cam pixel (u, v) onto a horizontal plane z_plane
x, y = robot.deproject(u=0.52, v=0.48, z_plane=0.015)

# 2. Pick-and-place / stacking with world (x, y) or wrist-cam (u, v)
robot.grasp(x=x, y=y, z=0.015, yaw_deg=-35, verify_grasp=True)  # rigid object
robot.grasp(x=x, y=y, z=0.012, yaw_deg=0, opening=0.4)  # inside a box: open 40%
robot.place(x=0.24, y=0.06, z=0.075)
robot.place(x=0.24, y=0.06, z=0.030, opening=0.4)  # release inside a box
robot.place_until_contact(x=0.24, y=0.06, z=0.015, max_lag=0.012, relief=0.002)

# Loops, conditionals, and arithmetic work naturally
bx, by = 0.21, -0.05
for i, (cx, cy, yaw_deg) in enumerate([(0.23, 0.02, 15), (0.19, 0.07, -20)], 1):
  robot.grasp(x=cx, y=cy, z=0.015, yaw_deg=yaw_deg)
  robot.place(x=bx, y=by, z=0.015 + 0.03 * i)
robot.home()

# 3. Direct 6-DoF Cartesian, parabolic arc transport, & guarded contact moves
robot.goto(x=0.28, y=0.05, z=0.04, arc_z=0.14, pitch_deg=15, azimuth_deg=0, roll_deg=0, speed=0.10)
robot.move_by(forward=0.025, dz=-0.002, dpitch_deg=-3, droll_deg=5, speed=0.04)
robot.move_until_contact(dz=-0.06, speed=0.03, max_lag=0.012, relief=0.002)
robot.gripper(0.4)  # jaw opening: 0 = closed, 1 = open
robot.close_gripper(verify_grasp=True)  # = gripper(0.0, verify_grasp=True)
robot.open_gripper()  # = gripper(1.0)

# 4. Non-prehensile sweep/push from (x0, y0) to (x1, y1) with closed jaws
robot.sweep(x0=0.25, y0=-0.08, x1=0.22, y1=0.0, z=0.02)

# 5. Declare task completion after visual verification
robot.done()
```

Equivalent JSON primitives (`[{"op": "grasp", ...}, {"op": "place", ...}, {"op":
"place_until_contact", ...}, {"op": "goto", ...}, {"op": "move_by", ...}, {"op":
"move_until_contact", ...}, {"op": "gripper", "gripper": 0.4}, {"op": "home"},
{"op": "done"}]`) are also accepted.

## Primitives & Closed-Loop Strategy

-   **Coordinates & Full 6-DoF Orientation**:
    -   World metres (`x` away from base, `y` left, `z` up from table).
    -   `pitch_deg` in degrees (or `pitch` in rad) tilts the fingers below
        horizontal (`90` deg / `1.57` rad points straight down `-Z`; `0` deg
        points horizontally in the XY plane). When `pitch_deg < 60` (or
        `azimuth_deg` is passed), `grasp` and `place` automatically approach and
        retreat coaxially along the 3D finger axis rather than straight down
        from `+Z`.
    -   `azimuth_deg` (or `azimuth` in rad) sets the horizontal direction in
        world XY that the fingers point toward when tilted (`0` deg = `+X`, `90`
        deg = `+Y`, `-90` deg = `-Y`, `180` deg = `-X`). Defaults to the radial
        direction from the base on 5-DoF arms; freely controllable on 6-DoF
        arms.
    -   `yaw_deg` (or `yaw` in rad) is the world angle of the object's long axis
        between the jaws for top-down grasps (`jaw_yaw_deg` sets the jaw-closing
        axis directly). For tilted/horizontal approaches, `roll_deg` (or `roll`
        in rad) rotates the jaws directly around the finger axis (`0` deg = jaws
        pinch vertically up/down; `90` deg = jaws pinch horizontally
        left/right).
    -   `arc_z` (optional on `goto`, `grasp`, `place`, `place_until_contact`)
        shapes the Z profile into a smooth quadratic parabolic arc over
        minimum-jerk XY progress so the pinch center clears `arc_z` at
        mid-flight (`s = 0.5`) without stopping at an intermediate waypoint.
        Lateral `grasp` and `place` transits (`> 2 cm` XY travel) automatically
        set a parabolic clearance arc.
    -   When passing normalized `u, v` in `[0, 1]` (or pixel coordinates on
        `wrist_cam.png`), the camera ray is intersected with horizontal plane
        `z_plane` (defaults to `z` for `grasp`, `z - 0.015` for
        `place`/`place_until_contact`, or `grid_z=0.015` for `look_at`/`goto`).
-   **Guarded Contact Moves (`move_until_contact` & `place_until_contact`)**:
    -   `robot.move_until_contact(dx=0.0, dy=0.0, dz=-0.06, forward=0.0,
        speed=0.03, max_lag=0.012, relief=0.002)` steps along a world or
        tool-frame (`forward`) axis and monitors axial pinch tracking lag
        (`ee_pos`) without requiring a force/torque sensor. When contact stalls
        the pinch center beyond `max_lag` for 3 consecutive ticks, the executor
        latches `stopped_on_contact=True`, `contact_pos=[x, y, z]`, and
        `status="contact"`, and backs off by `relief` along the approach axis so
        the servo stops pressing into the surface.
    -   `robot.place_until_contact(x=..., y=..., z=0.015, max_lag=0.012,
        relief=0.002)` transits above `(x, y)` (or `(u, v)`), descends until
        contact is detected (or `z` is reached), and immediately opens the jaws
        and lifts from the latched contact height rather than continuing down to
        nominal `z`.
-   **Multi-Turn Closed-Loop Control & Failure Recovery**:
    -   Each turn executes to completion and returns an updated `obs.json` and
        `wrist_cam.png` (and records `obs_frames/turn_XXX.png`, `steps.jsonl`,
        and `report.md` in the bridge directory). For multi-step or
        high-precision tasks, issue one primitive (or a short sequence) per turn
        and inspect `wrist_cam.png` before deciding the next move.
    -   Use `robot.goto(...)`, `robot.move_by(...)`, and
        `robot.move_until_contact(...)` to servo the held object or gripper
        incrementally based on visual and contact feedback (`forward` steps
        coaxially along the 3D finger axis).
    -   `verify_grasp=True` (on `grasp`, `gripper`, or `close_gripper`) checks
        that the measured `gripper` opening settles more than `threshold`
        (default `0.004`) above the commanded opening, i.e. an object stopped
        the jaws. Use it for rigid objects; skip it for objects that flatten
        (tissue, cloth, paper), which read like empty jaws, and verify those
        visually.
    -   `grasp` opens the jaws fully (about `9.5 cm`) before descending and
        `place` opens them fully to release. Pass `opening=<fraction>` to open
        only partway when the fingers must fit between container walls or close
        neighbours (e.g. `opening=0.4` for glasses in a `6 cm` wide case).
    -   Always check `status` and `holding` after `grasp` and `place`. If
        `status == "grasp_failed"` (either because a verified grasp closed
        empty or because the verified object slipped out mid-transport during
        `place`), the arm has automatically aborted the remaining segments and
        reopened the jaws—call `robot.look_at(...)` to re-observe the object's
        current position and orientation, then retry.
    -   For thin objects placed directly on the table (where you want the object
        to touch down gently without dropping from the default `8 mm` cube
        release clearance and rolling), pass `clearance=0.002` to
        `robot.place(...)` or use `robot.place_until_contact(...)`.
    -   Calling `robot.home()` moves the arm back to the overview pose and
        returns a fresh workspace image without ending the episode; once you
        visually verify in `wrist_cam.png` that the goal is satisfied, call
        `robot.done()`.

## Feedback

`turn` is the current turn index and `max_turns` is the episode turn budget (`0`
means unlimited). `status` is `done`, `contact` (stopped on contact during
`move_until_contact` or `place_until_contact`, with `stopped_on_contact: true`
and `contact_pos: [x, y, z]`), `grasp_failed` (the jaws closed on nothing or the
object slipped out mid-transport; the jaws have reopened — look again and
retry), or `episode_end`. `gripper` is the measured jaw opening (0 = closed,
1 = open) and `holding` is true while it is more than `0.004` above the
commanded opening. `history` lists up to the last 20 turns.
