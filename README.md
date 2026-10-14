# Blindspot

A simulated Franka Panda arm that follows natural-language pick-and-place instructions. The project builds from a MuJoCo scene and scripted expert through imitation learning, vision-language-action models, reinforcement learning, and an LLM/VLM agent. See [the project idea](docs/idea.md) and [the roadmap](docs/roadmap.md).

## Setup

Install `uv` and Python 3.12, then run:

```sh
make setup
make test
make lint
```

The test suite is configured with a `rendering` marker. CI excludes rendering tests because they require a graphics backend; they can be run locally when the required MuJoCo rendering dependencies are available.

## Third-party licenses

| Component | License information |
| --- | --- |
| Project code | MIT; see [LICENSE](LICENSE). |
| MuJoCo Menagerie | Per-directory terms are retained in the [Menagerie license file](third_party/mujoco_menagerie/LICENSE); check the license for each model used. |
| Franka Emika Panda model | Apache-2.0; see the upstream [Panda license](third_party/mujoco_menagerie/franka_emika_panda/LICENSE). |
| Model weights | No model weights are bundled in this phase. Record each checkpoint's license and use terms before adding or downloading it. |

The [MuJoCo Menagerie submodule](third_party/mujoco_menagerie) is pinned by Git to a specific commit. Dependencies retain their respective upstream licenses. No third-party model weights are redistributed by this repository at this stage.