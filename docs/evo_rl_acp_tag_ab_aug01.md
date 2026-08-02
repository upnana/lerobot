# ACP tag A/B（2026-08-01）：v2 / v3 · with vs without tag

> 承接 `docs/evo_rl_bowls_tray_v3_day_note_jul29.md`。  
> 目的：对照部署时是否喂 `Advantage: positive`，并记录脚本、视频与 value overlay 结论。

---

## 一句话结论

**Without tag（`n10_acp_none`）是更干净的闭环基线：策略不依赖开局硬喂 positive。**  
今晚 with-tag **没有**提高子任务完成度；v3 相对 v2 略好（纸巾偶发进盘），口红仍是共同瓶颈。部署默认保持 **without tag**。

---

## 文件变更（本轮新增 / 相关）

| 文件 | 作用 |
|------|------|
| `infer_v2_acp_tag.sh` | v2 真机 infer，默认 `PRESET=n10`（**with tag**），默认 ckpt **120000** |
| `infer_v3_acp_tag.sh` | v3 真机 infer，默认 `PRESET=n10`（**with tag**），默认 ckpt **100000** |
| `infer_v2_ckpt_120k.sh` | v2 @120k 对照；默认仍是 `n10_acp_none`（**without tag**） |
| `infer_smolvla_acp_bowls_tray_3cam.sh` | 共用入口：`PRESET` → `ACP_TAG`；`OVERLAY_VALUE=1` 录完叠 value |
| `overlay_value_on_eval.sh` | 对 eval 数据集跑 value infer + 导出 overlay mp4 |
| `docs/assets/value_overlay_demos/` | GitHub 上的 H.264 demo（见下节） |

### Preset 语义（infer）

| `PRESET` | Prompt | 含义 |
|----------|--------|------|
| `n10_acp_none`（默认推荐） | 仅任务文本 | **without tag** |
| `n10` | 任务文本 + `Advantage: positive` | **with tag** |

Value overlay 用的是对应数据版本的 value ckpt：

- v2 eval → `Evo-RL/outputs/value_train/bowls_tray_3cam_v2`
- v3 eval → `Evo-RL/outputs/value_train/bowls_tray_3cam_v3`

字段：`complementary_info.value_bowls_tray_3cam_v{2,3}`（曲线是 \(V(s)\)，不是 advantage 尖峰）。

---

## 视频（已推 GitHub，H.264）

目录：[`docs/assets/value_overlay_demos/`](https://github.com/upnana/lerobot/tree/main/docs/assets/value_overlay_demos)

| 文件 | 条件 | 说明 |
|------|------|------|
| [`v2_step120k_ep0_value_overlay.mp4`](https://github.com/upnana/lerobot/blob/main/docs/assets/value_overlay_demos/v2_step120k_ep0_value_overlay.mp4) | v2 120k **without tag** | 甜区基线（注意：该 ep 中段有人手，A/B 需谨慎） |
| [`v2_step140k_ep0_value_overlay.mp4`](https://github.com/upnana/lerobot/blob/main/docs/assets/value_overlay_demos/v2_step140k_ep0_value_overlay.mp4) | v2 140k without tag | 中后期对照 |
| [`v2_step120k_acp_positive_ep0_value_overlay.mp4`](https://github.com/upnana/lerobot/blob/main/docs/assets/value_overlay_demos/v2_step120k_acp_positive_ep0_value_overlay.mp4) | v2 120k **with tag** | 2026-08-01 晚 |
| [`v3_step100k_acp_positive_ep0_value_overlay.mp4`](https://github.com/upnana/lerobot/blob/main/docs/assets/value_overlay_demos/v3_step100k_acp_positive_ep0_value_overlay.mp4) | v3 100k **with tag** | 2026-08-01 晚 |

本地 eval 根目录（含 raw 四路视频 + `_value_viz`）：

```text
Evo-RL/outputs/eval/smolvla_acp_bowls_tray_3cam_v2/
  step-120000_preset-n10_acp_none_acp-none_n10_20260801_203236[_value_viz]
  step-120000_preset-n10_acp-positive_n10_20260801_220529[_value_viz]
Evo-RL/outputs/eval/smolvla_acp_bowls_tray_3cam_v3/
  step-100000_preset-n10_acp-positive_n10_20260801_221731[_value_viz]
```

原始 overlay 为 AV1；入库前转 H.264：

```bash
ffmpeg -y -i <av1.mp4> -c:v libx264 -pix_fmt yuv420p -crf 23 -preset medium -an <h264.mp4>
```

---

## Value infer 结果对比

> N=1 / 条件，定性为主；不是正式 SR。时长均约 **110s**。

| 跑次 | Value mean | 首 100 帧 → 末 100 帧 | 末帧 \(V\) | 子任务观感 |
|------|------------|------------------------|------------|------------|
| v2 120k **without tag** | -0.35 | -0.36 → -0.33（+0.02） | ≈ -0.50 | 中段有人手；口红/纸巾未完成 |
| v2 120k **with tag** | -0.37 | -0.38 → -0.22（**+0.16**） | ≈ -0.20 | 黄盘空；仍在白盘上叠碗；口红/纸巾失败 |
| v3 100k **with tag** | **-0.16** | -0.39 → -0.09（**+0.31**） | ≈ -0.08 | **纸巾进黄盘**；口红仍在桌上；碗未入 tray |

### Value 侧读法

1. **With-tag 末段抬升更明显，但任务未完成** → 典型假进度 / 乐观偏差（靠近、伸爪被当成进展，grasp 不敏感）。  
2. **v3 整体 \(V\) 更高**，与「纸巾进盘」一致，但对口红缺失仍偏乐观。  
3. **Without tag 说明什么：** 闭环能力来自任务条件 + 视觉/状态，不依赖部署时贴 positive；硬喂 tag 今晚没有换成更高子任务完成度。

动作幅度（‖Δaction‖ mean）三跑接近（≈5.7–6.0），未见 with-tag「明显更敢动」。

---

## 总对比表（v2 / v3 × tag）

| | v2 without tag | v2 with tag | v3 with tag | v3 without tag（同夜） |
|--|----------------|-------------|-------------|------------------------|
| 叠碗 | 有进展，未稳定入 tray | 未入 tray | 部分叠好，未入 tray | 缺干净完整对照 ep |
| 纸巾 | 未完成 | 未完成 | **进盘** | 历史中期：做不完整 |
| 口红 | 未完成 | 未完成 | **未完成** | 历史：主瓶颈 |
| Tag 增益 | 基线 | **无整任务增益** | 相对 v2 有纸巾进步（数据），非 tag | — |

---

## 部署与下一步

| 项 | 决定 |
|----|------|
| 默认 infer | **`PRESET=n10_acp_none`**（without tag） |
| With-tag 用途 | 诊断 / 对照，不是默认增益开关 |
| 下一刀数据 | 口红成功 + 近失败；少堆已会的叠碗 |
| Value | 压 grasp 敏感度；继续用 overlay 对齐视频，不单看曲线斜率 |
| 评估 | 补 N≥20 分阶段 SR；v3 **without tag** 甜区 ckpt 干净 A/B |

### 复现命令

```bash
# without tag（推荐默认）
bash /home/rxn/lerobot/infer_v2_ckpt_120k.sh
# 或
PRESET=n10_acp_none bash /home/rxn/lerobot/infer_smolvla_acp_bowls_tray_3cam.sh 120000

# with tag
bash /home/rxn/lerobot/infer_v2_acp_tag.sh          # v2 @120k
bash /home/rxn/lerobot/infer_v3_acp_tag.sh          # v3 @100k
```

---

## 面试可用表述

> ACP 标签是训练时的条件信号；部署默认 **不喂** `Advantage: positive`。我们做了 v2/v3 × with/without tag 的真机对照：without tag 说明策略不依赖开局硬喂 advantage；with tag 没有提高口红/整任务完成度，value 末段反而更容易虚高。v3 相对 v2 在纸巾上有一点数据迭代收益，瓶颈仍在口红抓放——所以下一轮优先补抓取数据与 value 校准，而不是继续拧 tag 开关。
