<div align="center">

# VeriVLA: Adaptive Test-Time Verification for Flow-Based VLAs with Latent World Models

<p>
  <strong><a href="https://github.com/shirley430316">Xueli Sun</a><sup>1</sup></strong>
  &nbsp;&nbsp;
  <strong><a href="https://innovator-zero.github.io/">Yuxiang Lu</a><sup>1</sup></strong>
  &nbsp;&nbsp;
  <strong><a href="https://www.linkedin.com/in/yantong-liu-10aa7734b/">Yantong Liu</a><sup>1</sup></strong>
  &nbsp;&nbsp;
  <strong><a href="https://www.linkedin.com/in/xianzhefan/">Xianzhe Fan</a><sup>1</sup></strong>
  &nbsp;&nbsp;
  <strong><a href="https://xavierchen34.github.io/">Xi Chen</a><sup>1</sup></strong>
  &nbsp;&nbsp;
  <strong><a href="https://happinesslz.github.io/">Zhe Liu</a><sup>1†</sup></strong>
  &nbsp;&nbsp;
  <strong><a href="https://i.cs.hku.hk/~hszhao/">Hengshuang Zhao</a><sup>1*</sup></strong>
</p>

<p>
  <sup>1</sup> The University of Hong Kong
  <br>
  <sup>*</sup> Corresponding author &nbsp;&nbsp; <sup>†</sup> Project lead
</p>

<p>
  <a href="https://shirley430316.github.io/projects/verivla/"><img alt="Project Page" src="https://img.shields.io/badge/Project_Page-VeriVLA-82B366.svg"></a>
</p>

</div>

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
