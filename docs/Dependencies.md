# External Dependencies & Licenses

This document lists all external packages used by SAM3DBody2abc, their licenses, and commercial use implications.

## License Summary

| Package | License | Commercial Use | Notes |
|---------|---------|----------------|-------|
| SAM-3D-Body | Meta Research | ✅ With attribution | Core model |
| GroundLink | MIT | ✅ Yes | Physics foot contact |
| TAPNet/TAPIR | Apache 2.0 | ✅ Yes | Point tracking |
| LightGlue | Apache 2.0 | ✅ Yes | Feature matching |
| Kornia | Apache 2.0 | ✅ Yes | CV operations |
| LoFTR | Apache 2.0 | ✅ Yes | Feature matching |
| PyTorch3D | BSD-3-Clause | ✅ Yes | Differentiable rendering |
| SMPL/SMPL-X | MPI License | ⚠️ Requires license | Body model |
| Ultralytics YOLO | AGPL-3.0 | ⚠️ Copyleft | Person detection |
| OpenCV | Apache 2.0 | ✅ Yes | Image processing |
| NumPy | BSD | ✅ Yes | Numerical computing |
| SciPy | BSD | ✅ Yes | Scientific computing |
| PyTorch | BSD | ✅ Yes | Deep learning |

---

## Core Dependencies

### SAM-3D-Body (Meta Research)

**Repository**: https://github.com/facebookresearch/sam-3d-body  
**License**: [Meta Research License](https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE)  
**Usage**: Core 3D body reconstruction from single images

**License Summary**:
- ✅ Research and non-commercial use: Allowed
- ✅ Commercial use: Allowed with attribution
- ✅ Modification: Allowed
- ✅ Distribution: Allowed with license notice

**Attribution Required**: Yes - include license notice in derivative works.

---

### GroundLink

**Repository**: https://github.com/HerocatUED/GroundLink  
**Paper**: "GroundLink: Physics-Informed Multi-Terrain Motion Estimation" (SIGGRAPH Asia 2024)  
**License**: MIT  
**Usage**: Physics-based foot contact detection using Ground Reaction Force prediction

**License Summary**:
- ✅ Commercial use: Yes
- ✅ Modification: Yes
- ✅ Distribution: Yes
- ✅ Private use: Yes

**Citation**:
```bibtex
@inproceedings{groundlink2024,
  title={GroundLink: Physics-Informed Multi-Terrain Motion Estimation},
  author={...},
  booktitle={SIGGRAPH Asia 2024},
  year={2024}
}
```

---

### TAPNet / TAPIR (Google DeepMind)

**Repository**: https://github.com/google-deepmind/tapnet  
**Paper**: "TAPIR: Tracking Any Point with per-frame Initialization and temporal Refinement" (ICCV 2023)  
**License**: Apache 2.0  
**Usage**: Point tracking for foot contact and camera motion estimation

**License Summary**:
- ✅ Commercial use: Yes
- ✅ Modification: Yes
- ✅ Distribution: Yes
- ✅ Patent use: Yes (explicit grant)

**Citation**:
```bibtex
@inproceedings{doersch2023tapir,
  title={TAPIR: Tracking Any Point with per-frame Initialization and temporal Refinement},
  author={Doersch, Carl and Yang, Yi and Veber, Mel and Guber, Dilara and Bauer, Daniel and Rubinstein, Michael and Tompson, Jonathan J and Zisserman, Andrew and Fleet, David J},
  booktitle={ICCV},
  year={2023}
}
```

---

### LightGlue

**Repository**: https://github.com/cvg/LightGlue  
**Paper**: "LightGlue: Local Feature Matching at Light Speed" (ICCV 2023)  
**License**: Apache 2.0  
**Usage**: Feature matching for camera calibration and motion estimation

**License Summary**:
- ✅ Commercial use: Yes
- ✅ Modification: Yes
- ✅ Distribution: Yes

**Citation**:
```bibtex
@inproceedings{lindenberger2023lightglue,
  title={LightGlue: Local Feature Matching at Light Speed},
  author={Lindenberger, Philipp and Sarlin, Paul-Edouard and Pollefeys, Marc},
  booktitle={ICCV},
  year={2023}
}
```

---

### Kornia

**Repository**: https://github.com/kornia/kornia  
**License**: Apache 2.0  
**Usage**: Differentiable computer vision operations, LoFTR feature matching

**License Summary**:
- ✅ Commercial use: Yes
- ✅ Modification: Yes
- ✅ Distribution: Yes

---

## Optional Dependencies

### PyTorch3D (Meta)

**Repository**: https://github.com/facebookresearch/pytorch3d  
**License**: BSD-3-Clause  
**Usage**: Differentiable mesh rendering for Silhouette Refiner

**License Summary**:
- ✅ Commercial use: Yes
- ✅ Modification: Yes
- ✅ Distribution: Yes

**Installation**:
```bash
# Conda (recommended)
conda install pytorch3d -c pytorch3d

# Pip (may need to build from source)
pip install pytorch3d
```

---

### SMPL / SMPL-X (MPI)

**Website**: https://smpl-x.is.tue.mpg.de/  
**License**: Custom MPI License  
**Usage**: Body model for Silhouette Refiner (optional)

**License Summary**:
- ✅ Research/Academic use: Free
- ⚠️ Commercial use: Requires separate license from MPI
- Download requires registration

**To Use**:
1. Register at https://smpl-x.is.tue.mpg.de/
2. Download body models
3. Place in designated folder
4. For commercial use, contact MPI for licensing

**Installation**:
```bash
pip install smplx
```

---

## Dependencies with Special Considerations

### Ultralytics YOLO

**Repository**: https://github.com/ultralytics/ultralytics  
**License**: AGPL-3.0  
**Usage**: Person detection in Camera Solver

**License Summary**:
- ✅ Personal/Research use: Yes
- ⚠️ Commercial use: AGPL-3.0 requires source code disclosure
- 💰 Commercial license available from Ultralytics

**AGPL-3.0 Implications**:
If you distribute software that uses YOLO:
- You must release your source code under AGPL-3.0
- Or obtain a commercial license from Ultralytics

**Commercial License**: https://ultralytics.com/license

**Workaround for Commercial Use**:
The YOLO dependency is only used in the Camera Solver for person detection/masking. You can:
1. Disable Camera Solver features that use YOLO
2. Provide your own masks instead of auto-detection
3. Purchase Ultralytics commercial license

---

## Python Package Dependencies

These are installed via `requirements.txt`:

| Package | License | Purpose |
|---------|---------|---------|
| numpy | BSD | Numerical arrays |
| torch | BSD | Deep learning |
| torchvision | BSD | Vision models |
| opencv-python | Apache 2.0 | Image processing |
| scipy | BSD | Scientific computing |
| einops | MIT | Tensor operations |
| timm | Apache 2.0 | Vision models |
| huggingface-hub | Apache 2.0 | Model downloads |
| hydra-core | MIT | Configuration |
| omegaconf | BSD | Configuration |
| pytorch-lightning | Apache 2.0 | Training framework |

All of these are permissively licensed (MIT, BSD, Apache 2.0) and safe for commercial use.

---

## Commercial Use Checklist

For commercial deployment, ensure you have:

1. ✅ **SAM-3D-Body**: Include Meta Research License notice
2. ✅ **GroundLink**: MIT - no restrictions
3. ✅ **TAPNet/TAPIR**: Apache 2.0 - no restrictions
4. ✅ **LightGlue**: Apache 2.0 - no restrictions
5. ⚠️ **SMPL/SMPL-X**: Contact MPI for commercial license (if using Silhouette Refiner with SMPL)
6. ⚠️ **Ultralytics YOLO**: Purchase commercial license OR disable YOLO features OR provide your own masks

---

## Minimal Commercial Configuration

To use SAM3DBody2abc commercially with zero licensing concerns:

```
✅ Use:
- Core SAM3DBody processing (with attribution)
- GroundLink foot contact (MIT)
- TAPNet tracking (Apache 2.0)
- Multi-camera triangulation

⚠️ Avoid or license:
- Camera Solver with auto-masking (uses YOLO) → provide your own masks
- Silhouette Refiner with SMPL → use skeleton_hull mode instead
```

---

## Citation

If you use this package in academic work, please cite the relevant papers:

```bibtex
@misc{sam3dbody2abc,
  title={SAM3DBody2abc: Video to Animated FBX Export},
  year={2024},
  url={https://github.com/your-repo/ComfyUI-SAM3DBody2abc}
}

@article{sam3dbody,
  title={SAM-3D-Body: ...},
  author={Meta AI Research},
  year={2024}
}

@inproceedings{groundlink2024,
  title={GroundLink: Physics-Informed Multi-Terrain Motion Estimation},
  booktitle={SIGGRAPH Asia 2024},
  year={2024}
}
```
