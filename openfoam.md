# OpenFOAM on MilliCloud

How to submit, run, watch, and open CFD jobs. This matches the current worker
pipeline — not a full `Allrun` / DAKOTA / multi-region runner.

---

## What it is

MilliCloud treats OpenFOAM as a **second engine** next to Blender:

1. You zip a case and create a job on the **website** (`/jobs/new` → OpenFOAM CFD).
2. A registered **desktop / CLI worker** claims the job (one chunk, whole case).
3. The worker downloads the zip, meshes, runs one solver, uploads `openfoam-results.zip`.
4. You download results from the job page and open them in ParaView (or Glance).

The worker is **not** `./Allrun`. It does a fixed sequence (below). Cases that
need extra pre-steps (`setFields`, DAKOTA, AMI, `flattenMesh`, `chtMultiRegion`)
will not run as-is.

On Apple Silicon the worker runs OpenFOAM in Docker:

`opencfd/openfoam-default:2412` with `--platform linux/amd64`

Native `simpleFoam` / `blockMesh` on `PATH` is used instead if present.

Use **Trusted host** mode on the Mac. Secure sandbox is for Blender sidecars,
not this OpenFOAM path.

---

## Prerequisites

| Piece | Why |
|---|---|
| API + web UI running | Create jobs, watch logs, download zip |
| MilliCloud Desktop signed in and **this computer registered** | Worker token bound to the machine |
| Agent **started** (Agent tab) | Polls `/agent/poll` and claims queued OpenFOAM jobs |
| Docker Desktop running (Mac) | Image pull + `docker run` for each mesh/solver step |
| Disk space | motorBike-class cases write a large mesh + `openfoam-results.zip` (~100–200 MB) |

First OpenFOAM job on a machine pulls the ~multi-GB OpenFOAM image. That can
look like “nothing is happening” for several minutes. After that, jobs start faster.

The agent control panel listens on `127.0.0.1:8787` by default. If something
else already bound that port, the panel fails soft — the worker still runs.

---

## Create a job

1. Open the website → **New render**.
2. Engine: **OpenFOAM CFD** (not Blender).
3. Upload a **`.zip` only** (not `.tar.gz`, not a folder).
4. Set options, submit. File goes to R2; confirm-upload queues the job.
5. Keep the desktop agent running. It picks the job on the next poll (~10 s).
   No launch command is required for a registered always-on worker.

You cannot create OpenFOAM jobs from the desktop Jobs tab yet. Desktop only
lists cloud jobs and runs the agent.

### Job options (website)

| Field | Default | Meaning |
|---|---|---|
| **Solver** | `auto` | Reads `application` from `system/controlDict`. Or force `icoFoam`, `simpleFoam`, `pimpleFoam`, `pisoFoam`, `interFoam`, `rhoSimpleFoam`. API also allows `rhoPimpleFoam`, `buoyantSimpleFoam`, `buoyantPimpleFoam`, `potentialFoam`, `scalarTransportFoam`. |
| **Local MPI ranks** | `1` | `1` = serial. `>1` runs `decomposePar` → `mpirun -np N solver -parallel` → `reconstructPar -latestTime`. Only if that host has MPI + OpenFOAM that match. On Mac Docker this is usually leave at `1`. |
| **endTime override** | empty | Patches `system/controlDict` `endTime`. Use a small value (e.g. `50`) for a smoke test. |
| **snappyHexMesh** | on | If `system/snappyHexMeshDict` is in the zip, run snappy. Turn **off** if you already meshed and do not want remesh. |

`run_block_mesh` defaults to **on** whenever `system/blockMeshDict` exists (not shown in the form).

OpenFOAM jobs are always **one chunk** (`frame_start = frame_end = 1`). They do
not split across workers.

---

## Zip format

The archive must be a **zip of the case directory contents** (or a single
top-level folder that contains them). The worker finds the case by searching
for `system/controlDict`.

Required:

```
0/                  # or 0.orig/  (not 0.org/)
constant/           # transport, turbulence, and geometry
system/controlDict
system/fvSchemes
system/fvSolution
```

If you use snappy, also include:

```
system/blockMeshDict
system/snappyHexMeshDict
system/surfaceFeatureExtractDict   # if the tutorial has it
constant/triSurface/*.obj[.gz]     # or constant/geometry/
0.orig/                            # motorBike stores fields here, not in 0/
```

### Zip commands

**Wrong** (zip of the parent folder name only, or missing `0.orig`):

```bash
zip motorbike-case.zip motorBike          # often missing hidden / 0.orig
zip -r motorbike-case.zip *               # misses 0.orig if you are not inside the case
```

**Right** — you are *inside* the case:

```bash
cd /path/to/motorBike
zip -r ~/Downloads/motorbike-case.zip 0 0.orig constant system
# add these if present:
#   system/surfaceFeatureExtractDict constant/triSurface constant/geometry
```

Check before upload:

```bash
unzip -l ~/Downloads/motorbike-case.zip | grep -E 'controlDict|0.orig/p|0/p|triSurface|blockMeshDict'
```

You want `system/controlDict` and **either** `0/p` **or** `0.orig/p`.

`0.org/` (Holzmann) is **not** copied. Rename or copy to `0.orig/` or `0/`.

Do not zip `processor*`, existing time folders (`100/`, `500/`), or
`postProcessing/` unless you intend the solver to restart from them. A clean
case is smaller and clearer.

---

## What the worker actually runs

After download + unzip + locate `system/controlDict`:

1. If `0.orig/` exists and `0/p` does not → copy `0.orig` → `0`.
2. Fail if there is still no `0/p` and no `0.orig/p`.
3. Optionally patch `endTime`.
4. Then, **only if the file exists**:

   | Step | When |
   |---|---|
   | `blockMesh` | `run_block_mesh` (default on) and `system/blockMeshDict` |
   | `surfaceFeatureExtract` | `system/surfaceFeatureExtractDict` |
   | `snappyHexMesh -overwrite` | `run_snappy` on and `system/snappyHexMeshDict` |
   | `topoSet` | `system/topoSetDict` |
   | `createPatch -overwrite` | `system/createPatchDict` |
   | solver (`simpleFoam`, …) | always |
   | MPI extras | only if `nprocs > 1` |

5. Pack `openfoam-results.zip` and upload.

**Not run:** `Allrun`, `setFields`, `foamDictionary`, `dakota`, `flattenMesh`,
`extrudeMesh`, `renumberMesh`, `mapFields`, `splitMeshRegions`,
`changeDictionary`, `createBaffles`, `foamToVTK`.

On Docker, the case is mounted at `/work` inside the container. Host
`/var/folders/...` paths are never passed as `-case` (that caused
`cannot open root directory` on Mac).

---

## Watch the job

Open the **job page** (`/jobs/<id>`), not docker-compose and not only the
worker “host agent log”.

| What you see | What it is |
|---|---|
| Status `queued` | Waiting for a worker poll |
| Status `claimed` / `rendering` | Worker has the chunk |
| Progress % | Last `Time = …` / `endTime` from the **solver** log (not mesh time) |
| **Render log** | `[agent]` + `[openfoam]` lines shipped to the API |
| Host page “Host agent log” | Lifecycle only (poll / claim). Not solver stdout |
| `docker compose logs` on the API | HTTP access (`GET /jobs`, `POST /agent/poll`). **Not** OpenFOAM |

Solver lines can take a minute to appear: image pull, download, `blockMesh`,
then snappy (long, chatty). Progress stays at 0% until the solver prints
`Time = …`.

Desktop **Logs** tab is local agent stdout (`[agent]`, `[openfoam]`). Same
run, different window.

Abort: stop/abort from the worker page / desktop if the host supports it. The
worker checks abort between steps and while reading solver stdout.

---

## Results zip

On success the job page lists **`openfoam-results.zip`**. Download it.

Packed contents (current packer):

- **Last time directory only** (e.g. `500/`) — field files `U`, `p`, `k`, …
- **`postProcessing/`** — function objects (VTP, EnSight, SVGs, reports)
- **`log.<solver>`**
- **`system/controlDict`**

**Not packed:** `constant/polyMesh`, most of `system/`, `0/`.  
So **File → Open → `case.foam`** will fail on this zip. Use EnSight / VTP
(below), or merge the last time folder back into your original case.

Retention follows the same output expiry as Blender jobs. Expired jobs lose
the download.

---

## Open in ParaView

ParaView 5/6 on the Mac (`/Applications/ParaView-*.app` or `brew`).

### Full 3D + time (motorBike-style)

If the case wrote EnSight (motorBike does):

```
File → Open → openfoam-results/postProcessing/ensightWrite/work.case → Apply
```

Color by `U` or `p`. Time slider = write intervals (e.g. 100 … 500).

### Slices and streamlines

```
…/postProcessing/cuttingPlane/500/yNormal.vtp
…/postProcessing/sets/streamLines/500/track0.vtp
…/postProcessing/sets/wallBoundedStreamLines/500/track0.vtp
```

### Charts without ParaView

```
…/postProcessing/residualGraph1/500/residualGraph1.svg
…/postProcessing/forceCoeffsGraph1/500/forceCoeffsGraph1.svg
…/postProcessing/foamReport1/500/foamReport1.md
```

### Native OpenFOAM reader (optional)

Only if you still have the **original case** with `constant/polyMesh`:

```bash
cp -R ~/Downloads/openfoam-results/500 /path/to/original/motorBike/
cd /path/to/original/motorBike
touch case.foam
open -a ParaView case.foam
```

### Browser glance (not in MilliCloud yet)

[Kitware Glance](https://kitware.github.io/paraview-glance/app/) — drag a
`.vtp` onto the page. No in-app 3D viewer on the job page today.

---

## Ready-made test cases

Copy tutorials out of the **same image** the worker uses:

```bash
mkdir -p ~/foam-cases && cd ~/foam-cases

docker run --rm --platform linux/amd64 \
  -v "$PWD":/out opencfd/openfoam-default:2412 \
  bash -lc 'cp -a $FOAM_TUTORIALS/incompressible/simpleFoam/pitzDaily /out/'

cd pitzDaily
zip -r ../pitzDaily.zip 0 system constant
```

| Case | Path in the image | Good for | Notes |
|---|---|---|---|
| **cavity** | `incompressible/icoFoam/cavity/cavity` | Smoke test (seconds) | Solver `icoFoam`. Zip `0 system constant`. |
| **pitzDaily** | `incompressible/simpleFoam/pitzDaily` | simpleFoam, no snappy | Easy second job. |
| **airFoil2D** | `incompressible/simpleFoam/airFoil2D` | Force coeffs | blockMesh only. |
| **motorBike** | `incompressible/simpleFoam/motorBike` | Full path | Must include **`0.orig/`** and `constant/triSurface`. First run is slow. |
| **windAroundBuildings** | `incompressible/simpleFoam/windAroundBuildings` | Second snappy case | Same zip rules as motorBike. |

Official trees:

- ESI / OpenCFD (matches `2412`): [tutorials on develop.openfoam.com](https://develop.openfoam.com/Development/openfoam/-/tree/master/tutorials)
- Foundation v13 (paths differ): [OpenFOAM-13/tutorials](https://github.com/OpenFOAM/OpenFOAM-13/tree/master/tutorials)
- Extra geometries: [Holzmann training cases](https://holzmann-cfd.com/community/training-cases), [Wolf Dynamics](https://www.wolfdynamics.com/tutorials.html)

For motorBike from the image:

```bash
docker run --rm --platform linux/amd64 \
  -v "$HOME/foam-cases":/out opencfd/openfoam-default:2412 \
  bash -lc 'cp -a $FOAM_TUTORIALS/incompressible/simpleFoam/motorBike /out/'

cd ~/foam-cases/motorBike
zip -r ~/Downloads/motorbike-case.zip 0.orig constant system
unzip -l ~/Downloads/motorbike-case.zip | grep -E '0.orig/p|controlDict|motorBike.obj'
```

Shorten a long run: on the new-job form set **endTime override** to `50` or
`100` (motorBike tutorial default is `500`).

---

## Cases that will not run as-is

| Kind | Why |
|---|---|
| **`dakotaGeometricVariation`** (Holzmann) | Needs `dakota` + `dprepro` + `dakota.sh` (geometry loop). Worker never starts DAKOTA. Mesh script also uses `flattenMesh` / `extrudeMesh` / `0.org`. |
| Other DAKOTA / optimisation wrappers | Same. |
| **`damBreak`** / most VOF | Needs `setFields` before `interFoam`. Run `blockMesh && setFields` locally, then zip the resulting `0/`. |
| AMI / ACMI / 6-DoF / sliding mesh | Extra mesh utilities, not in the pipeline. |
| Multi-region CHT (`chtMultiRegionFoam`) | No `splitMeshRegions` / region loop. |
| Pre-meshed case with `blockMeshDict` + `snappyHexMeshDict` still in `system/` | Worker **will remesh**. Delete those dicts if you want to solve an existing `constant/polyMesh`. |
| OpenFOAM 4.x / old Holzmann dicts | Function objects (`surfaceRegion`, old `scalarTransport`) often fail on v2412. |
| `.tar.gz` upload | Job create requires `.zip`. |

One frozen geometry from a DAKOTA case is possible only if you mesh it
yourself, zip `0/` + `constant/polyMesh` + `system/` **without**
`blockMeshDict` / `snappyHexMeshDict`, and the dicts are v2412-compatible.

---

## Common errors

| Symptom | Cause | Fix |
|---|---|---|
| `cannot open root directory /var/folders/...` | Old agent passed the Mac temp path into Docker | Current agent mounts `/work`. Restart the desktop agent. |
| `FOAM FATAL ERROR: cannot find file 0/p` | Zip missing `0/` and `0.orig/` | Re-zip including `0.orig` (motorBike) or `0`. |
| `No 0/p after meshing` | snappy ran; fields only in `0.orig` | Include `0.orig/`. Agent copies it again before the solver. |
| `snappyHexMeshDict present but no constant/triSurface` | OBJ/STL left out of the zip | Add `constant/triSurface` or `constant/geometry`. |
| `blockMesh` / `simpleFoam` exit code 1 | Bad case, old dicts, or missing includes | Read the **job** Render log, not API compose logs. |
| Job `rendering`, log empty, worker idle | Previous agent died; chunk still leased (~5 min) | Wait for lease requeue, or retry after the chunk is `queued` again. |
| API screen filled with `GET /jobs` / `POST /agent/poll` | Those are HTTP logs | Use the job page Render log. |
| Host page log stuck at a few lines | That stream is lifecycle only | Open `/jobs/<id>` for solver stdout. |
| First job sits in `rendering` a long time at 0% | Docker pulling `openfoam-default:2412` | `docker images` / Docker Desktop. Later jobs skip this. |
| `simpleFoam` then success but ParaView cannot open a `.foam` file | Results zip has no mesh | Use EnSight `work.case` or VTP, or copy the last time into the original case. |

---

## Worker / Mac notes

- Register the host once (website token → desktop **Register this computer**).
- Start the agent before or after creating the job; poll interval is ~10 s.
- Apple M-series: every OpenFOAM step is `docker run --platform linux/amd64`.
  Keep Docker Desktop running.
- Trusted mode inherits the host environment (enough for Docker).
- Do not run two agents with the same worker token.
- Port `8787` leftover from a killed Python agent:  
  `lsof -nP -i4TCP:8787 | grep LISTEN` then stop that PID.

---

## What is not in the product yet

- In-browser 3D preview (vtk.js / Glance on the job page)
- Packing `constant/polyMesh` + `case.foam` for one-click ParaView
- Running `./Allrun` or arbitrary pre-steps
- DAKOTA / design sweeps as first-class jobs
- Creating OpenFOAM jobs from the desktop app
- Splitting one case across many workers (domain decompose as multi-host)

The useful next product step for viewing is: upload the last
`cuttingPlane` / `streamLines` `.vtp` (already produced by motorBike) and
embed a small vtk.js preview. Full volume in the browser is not realistic.
