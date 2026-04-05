import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent   # repo root
FEATURES_PATH = ROOT / "brain_features.npy"
LABELS_PATH = ROOT / "balanced_emojis.npy"
OUTPUT_PATH = Path(__file__).resolve().parent / "brain_surface.html"

LEFT_VERTICES = 10242
RIGHT_VERTICES = 10242
FULL_BRAIN_VERTICES = LEFT_VERTICES + RIGHT_VERTICES
BOTH_HEMISPHERE_OFFSET = 55.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an interactive TRIBE v2 surface viewer."
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Initial sample index to show in the HTML viewer.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all sample indices and emoji labels, then exit.",
    )
    return parser.parse_args()


def load_dataset(
    features_path: Path, labels_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    brain_features = np.load(features_path)
    emoji_labels = np.load(labels_path, allow_pickle=True)

    if brain_features.ndim == 1:
        brain_features = brain_features[np.newaxis, :]
    if brain_features.ndim != 2:
        raise ValueError(
            "brain_features.npy must be a 1D or 2D array, "
            f"got shape {brain_features.shape}."
        )

    if emoji_labels.ndim != 1:
        raise ValueError(
            "balanced_emojis.npy must be a 1D array, "
            f"got shape {emoji_labels.shape}."
        )

    if brain_features.shape[0] != emoji_labels.shape[0]:
        raise ValueError(
            "Feature rows and emoji labels must align. "
            f"Got {brain_features.shape[0]} feature rows and "
            f"{emoji_labels.shape[0]} labels."
        )

    if brain_features.shape[1] != FULL_BRAIN_VERTICES:
        raise ValueError(
            f"Each feature row must have {FULL_BRAIN_VERTICES} values "
            f"for fsaverage5 both-hemisphere data. Got {brain_features.shape[1]}."
        )

    return brain_features.astype(np.float32), emoji_labels.astype(str)


def list_samples(labels: np.ndarray) -> None:
    for i, label in enumerate(labels.tolist()):
        print(f"{i:02d}: {label}")


def load_mesh_payload() -> dict:
    from nilearn import datasets
    from nilearn.surface import load_surf_mesh

    fsaverage = datasets.fetch_surf_fsaverage()
    payload = {}

    for surface_name, left_key, right_key in [
        ("inflated", "infl_left", "infl_right"),
        ("pial", "pial_left", "pial_right"),
    ]:
        left_mesh = load_surf_mesh(fsaverage[left_key])
        right_mesh = load_surf_mesh(fsaverage[right_key])

        left_coords = left_mesh.coordinates.astype(np.float32)
        left_faces = left_mesh.faces.astype(np.int32)
        right_coords = right_mesh.coordinates.astype(np.float32)
        right_faces = right_mesh.faces.astype(np.int32)

        # Spread the hemispheres apart in the combined view so they do not overlap.
        spaced_left_coords = left_coords.copy()
        spaced_right_coords = right_coords.copy()
        spaced_left_coords[:, 0] -= BOTH_HEMISPHERE_OFFSET
        spaced_right_coords[:, 0] += BOTH_HEMISPHERE_OFFSET

        both_coords = np.vstack([spaced_left_coords, spaced_right_coords])
        both_faces = np.vstack(
            [left_faces, right_faces + left_coords.shape[0]]
        ).astype(np.int32)

        payload[surface_name] = {
            "left": {
                "x": left_coords[:, 0].tolist(),
                "y": left_coords[:, 1].tolist(),
                "z": left_coords[:, 2].tolist(),
                "i": left_faces[:, 0].tolist(),
                "j": left_faces[:, 1].tolist(),
                "k": left_faces[:, 2].tolist(),
            },
            "right": {
                "x": right_coords[:, 0].tolist(),
                "y": right_coords[:, 1].tolist(),
                "z": right_coords[:, 2].tolist(),
                "i": right_faces[:, 0].tolist(),
                "j": right_faces[:, 1].tolist(),
                "k": right_faces[:, 2].tolist(),
            },
            "both": {
                "x": both_coords[:, 0].tolist(),
                "y": both_coords[:, 1].tolist(),
                "z": both_coords[:, 2].tolist(),
                "i": both_faces[:, 0].tolist(),
                "j": both_faces[:, 1].tolist(),
                "k": both_faces[:, 2].tolist(),
            },
        }

    return payload


def make_payload(
    brain_features: np.ndarray, emoji_labels: np.ndarray, initial_index: int
) -> dict:
    if not 0 <= initial_index < brain_features.shape[0]:
        raise IndexError(
            f"Index {initial_index} is out of range for {brain_features.shape[0]} samples."
        )

    left_maps = brain_features[:, :LEFT_VERTICES]
    right_maps = brain_features[:, LEFT_VERTICES:]

    return {
        "labels": emoji_labels.tolist(),
        "leftMaps": left_maps.tolist(),
        "rightMaps": right_maps.tolist(),
        "initialIndex": initial_index,
        "views": ["front", "back", "left", "right", "top", "bottom"],
        "description": (
            "Colorbar values are the TRIBE v2 feature values for the selected "
            "sample. Positive values are hotter colors, negative values are "
            "cooler colors, and 0 is the midpoint."
        ),
    }


def build_html(mesh_payload: dict, data_payload: dict) -> str:
    mesh_json = json.dumps(mesh_payload)
    data_json = json.dumps(data_payload)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TRIBE v2 Brain Surface Viewer</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #f4f0e8;
      --panel: #fffaf2;
      --ink: #171411;
      --muted: #5b544e;
      --line: #d8ccbf;
      --accent: #c85c2f;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff7ea 0, transparent 28%),
        radial-gradient(circle at top right, #f2d4c8 0, transparent 24%),
        linear-gradient(180deg, #f6f0e6 0%, #ece4d8 100%);
    }}
    .shell {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }}
    .header {{
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(2rem, 4vw, 3.3rem);
      line-height: 1;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .sub {{
      margin: 0;
      max-width: 900px;
      font-size: 1rem;
      line-height: 1.5;
      color: var(--muted);
    }}
    .grid {{
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: 20px;
      align-items: start;
    }}
    .panel {{
      background: color-mix(in srgb, var(--panel) 92%, white);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 14px 50px rgba(56, 35, 17, 0.08);
    }}
    .controls {{
      padding: 18px;
      position: sticky;
      top: 18px;
    }}
    .controls h2 {{
      margin: 0 0 14px;
      font-size: 1rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--accent);
    }}
    .field {{
      margin-bottom: 14px;
    }}
    .field label {{
      display: block;
      margin-bottom: 6px;
      font-size: 0.83rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .field select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 14px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }}
    .meta {{
      margin-top: 16px;
      padding-top: 16px;
      border-top: 1px solid var(--line);
      display: grid;
      gap: 10px;
    }}
    .meta-row {{
      display: grid;
      grid-template-columns: 96px 1fr;
      gap: 10px;
      font-size: 0.95rem;
    }}
    .meta-key {{
      color: var(--muted);
    }}
    .plot-wrap {{
      padding: 10px;
    }}
    #plot {{
      min-height: 78vh;
    }}
    @media (max-width: 980px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
      .controls {{
        position: static;
      }}
      #plot {{
        min-height: 62vh;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="header">
      <h1>TRIBE v2 Brain Surface Viewer</h1>
      <p class="sub">{data_payload["description"]}</p>
    </div>
    <div class="grid">
      <div class="panel controls">
        <h2>Controls</h2>
        <div class="field">
          <label for="sample-select">Sample</label>
          <select id="sample-select"></select>
        </div>
        <div class="field">
          <label for="surface-select">Surface</label>
          <select id="surface-select">
            <option value="pial">Pial (realistic)</option>
            <option value="inflated">Inflated</option>
          </select>
        </div>
        <div class="field">
          <label for="hemi-select">Hemisphere</label>
          <select id="hemi-select">
            <option value="both">Both sides</option>
            <option value="left">Left side</option>
            <option value="right">Right side</option>
          </select>
        </div>
        <div class="field">
          <label for="view-select">View</label>
          <select id="view-select">
            <option value="front">Front</option>
            <option value="back">Back</option>
            <option value="left">Left</option>
            <option value="right">Right</option>
            <option value="top">Top</option>
            <option value="bottom">Bottom</option>
          </select>
        </div>
        <div class="meta">
          <div class="meta-row">
            <div class="meta-key">Emoji</div>
            <div id="emoji-label"></div>
          </div>
          <div class="meta-row">
            <div class="meta-key">Range</div>
            <div id="range-label"></div>
          </div>
          <div class="meta-row">
            <div class="meta-key">Meaning</div>
            <div>Negative values are cooler colors, positive values are hotter colors.</div>
          </div>
        </div>
      </div>
      <div class="panel plot-wrap">
        <div id="plot"></div>
      </div>
    </div>
  </div>
  <script>
    const MESHES = {mesh_json};
    const DATA = {data_json};

    const cameras = {{
      left: {{ eye: {{ x: -1.8, y: 0, z: 0 }}, up: {{ x: 0, y: 0, z: 1 }}, center: {{ x: 0, y: 0, z: 0 }} }},
      right: {{ eye: {{ x: 1.8, y: 0, z: 0 }}, up: {{ x: 0, y: 0, z: 1 }}, center: {{ x: 0, y: 0, z: 0 }} }},
      top: {{ eye: {{ x: 0, y: 0, z: 1.8 }}, up: {{ x: 0, y: 1, z: 0 }}, center: {{ x: 0, y: 0, z: 0 }} }},
      bottom: {{ eye: {{ x: 0, y: 0, z: -1.8 }}, up: {{ x: 0, y: 1, z: 0 }}, center: {{ x: 0, y: 0, z: 0 }} }},
      front: {{ eye: {{ x: 0, y: 1.8, z: 0 }}, up: {{ x: 0, y: 0, z: 1 }}, center: {{ x: 0, y: 0, z: 0 }} }},
      back: {{ eye: {{ x: 0, y: -1.8, z: 0 }}, up: {{ x: 0, y: 0, z: 1 }}, center: {{ x: 0, y: 0, z: 0 }} }},
    }};

    const sampleSelect = document.getElementById("sample-select");
    const surfaceSelect = document.getElementById("surface-select");
    const hemiSelect = document.getElementById("hemi-select");
    const viewSelect = document.getElementById("view-select");
    const emojiLabel = document.getElementById("emoji-label");
    const rangeLabel = document.getElementById("range-label");

    function buildSampleOptions() {{
      DATA.labels.forEach((label, index) => {{
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = `${{String(index).padStart(2, "0")}}: ${{label}}`;
        sampleSelect.appendChild(option);
      }});
      sampleSelect.value = String(DATA.initialIndex);
    }}

    function getCurrentValues(index, hemisphere) {{
      const leftValues = DATA.leftMaps[index];
      const rightValues = DATA.rightMaps[index];
      if (hemisphere === "left") {{
        return leftValues;
      }}
      if (hemisphere === "right") {{
        return rightValues;
      }}
      return leftValues.concat(rightValues);
    }}

    function makeTrace(mesh, values, absMax) {{
      return {{
        type: "mesh3d",
        x: mesh.x,
        y: mesh.y,
        z: mesh.z,
        i: mesh.i,
        j: mesh.j,
        k: mesh.k,
        intensity: values,
        colorscale: "RdBu",
        reversescale: true,
        cmin: -absMax,
        cmax: absMax,
        colorbar: {{
          title: "Value",
          tickfont: {{ size: 13 }},
          titlefont: {{ size: 14 }},
          len: 0.72,
        }},
        flatshading: false,
        lighting: {{
          ambient: 0.55,
          diffuse: 0.7,
          specular: 0.15,
          roughness: 0.8,
        }},
        hovertemplate: "Value: %{{intensity:.4f}}<extra></extra>",
      }};
    }}

    function render() {{
      const index = Number(sampleSelect.value);
      const surfaceType = surfaceSelect.value;
      const hemisphere = hemiSelect.value;
      const view = viewSelect.value;

      const mesh = MESHES[surfaceType][hemisphere];
      const values = getCurrentValues(index, hemisphere);
      const absMax = Math.max(...values.map(v => Math.abs(v)));
      const trace = makeTrace(mesh, values, absMax);

      const layout = {{
        title: {{
          text: `TRIBE v2 sample ${{index}}: ${{DATA.labels[index]}} (${{surfaceType}})`,
          font: {{ size: 24, color: "#171411" }},
        }},
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        margin: {{ l: 0, r: 0, t: 56, b: 0 }},
        scene: {{
          camera: cameras[view],
          xaxis: {{ visible: false }},
          yaxis: {{ visible: false }},
          zaxis: {{ visible: false }},
          aspectmode: "data",
        }},
      }};

      Plotly.react("plot", [trace], layout, {{
        displaylogo: false,
        responsive: true,
        modeBarButtonsToRemove: ["hoverClosest3d"],
      }});

      emojiLabel.textContent = DATA.labels[index];
      rangeLabel.textContent = `${{(-absMax).toFixed(3)}} to ${{absMax.toFixed(3)}}`;
    }}

    buildSampleOptions();
    sampleSelect.addEventListener("change", render);
    surfaceSelect.addEventListener("change", render);
    hemiSelect.addEventListener("change", render);
    viewSelect.addEventListener("change", render);
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    brain_features, emoji_labels = load_dataset(FEATURES_PATH, LABELS_PATH)

    if args.list:
        list_samples(emoji_labels)
        return

    mesh_payload = load_mesh_payload()
    data_payload = make_payload(brain_features, emoji_labels, args.index)
    html = build_html(mesh_payload, data_payload)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Saved interactive viewer to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
