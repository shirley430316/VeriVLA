## VeriVLA: Adaptive Test-Time Verification for Flow-Based VLAs with Latent World Models

VeriVLA proposes a latent world model verifier for action ranking to improve vision-language-action (VLA) policies at inference time. The verifier predicts future visual latents and task progress increments for candidate action chunks, selecting the candidate with the highest cumulative predicted progress. A two-axis trigger activates verification based on within-chunk motion variation and cross-sample disagreement.

### Demo
https://github.com/user-attachments/assets/0d046b16-7fa2-40bd-93b3-8265ed93cd12

### TODO

- [x] model structure
- [ ] training code
- [ ] LIBERO & Robotwin integration
- [ ] real-world robot integration

### Acknowledgements

VeriVLA builds on [V-JEPA 2](https://github.com/facebookresearch/vjepa2) and [OpenPI](https://github.com/Physical-Intelligence/openpi). Much appreciation!
