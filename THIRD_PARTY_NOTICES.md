# Licenses and third-party attribution

Scene resources described below are distributed in the separate Hugging Face environment repository. Downloading them into `scenes/` restores the paths referenced in these notices.

The root MIT license applies to ROAM-Bench code, task annotations, and benchmark-authored scene configuration, generated object meshes, and generated textures. Third-party meshes, materials, software, and model assets retain their source licenses.

| Component | Source and license | Distribution |
|---|---|---|
| Scene-generation toolkit | [SceneSmith](https://github.com/nepfaff/scenesmith), MIT | License notice retained; obtain the generator upstream |
| Generated object meshes and textures | ROAM-Bench, MIT | Included with author-authorized redistribution |
| Articulated meshes and textures | [ArtVIP](https://huggingface.co/datasets/x-humanoid-robomind/ArtVIP), Apache 2.0 | Converted SDF/glTF resources are included; copyright and license notices retained |
| Room PBR materials | [AmbientCG](https://ambientcg.com/), CC0 1.0 | Included where required by scene sources |
| 3D generation tooling | [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects), SAM License | Generated scene assets retain their author rights; model weights and tooling are obtained upstream |
| Simulator | [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/index.html), NVIDIA terms | Obtain separately; the Dockerfile references the official image |
| Robot and hands | Unitree G1 and Inspire assets in the Isaac Sim asset library | Obtain separately under their source terms |

ArtVIP resources are represented as glTF visual meshes and Drake-compatible SDF models with collision resources. They retain their original source identifiers in scene metadata. `scenes/asset_provenance.json` lists the ArtVIP instances and their source asset IDs. These representation and integration changes do not relicense the underlying assets.

Copyright (c) 2026 Nicholas Pfaff — SceneSmith.

Copyright [2024] [X-Humanoid] — ArtVIP.

Applicable notices and license texts are in [code/licenses](code/licenses). Generated object assets are distributed under MIT with author-authorized redistribution. Generation software and model weights are separate upstream resources governed by their own terms.
