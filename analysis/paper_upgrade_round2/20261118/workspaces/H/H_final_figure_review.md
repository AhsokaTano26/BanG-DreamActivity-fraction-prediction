# Figure QA Report — 二轮18张冻结图

**Detected type:** SVG/vector PDF。**Target journal:** generic / Entertainment Computing manuscript。

## Programmatic findings

- **来源与输出：** F交付清单全部文件SHA、每图输入SHA、18组54资产SHA与G复制件逐一相符。冻结figure manifest为`0413a951…`，F delivery manifest为`c7d205e5…`。
- **矢量与文字：** H独立解析18个SVG，全部保留live text与原生path，ID唯一，图像对象为0；18个PDF的`pdfimages -list`也均为0。SVG最小显式字号为7.5用户单位，稿件中的实际字号仍随排版缩放检查。
- **字体：** H重新执行18次`pdffonts`，所有字体嵌入且有Unicode映射。普通文字与数学字形的矢量路径身份分别保留。
- **几何：** H重新执行18次Poppler bbox提取，全部PDF字框位于页面内。外部figure-qa helper在项目`.venv`缺少`lxml`，18次返回2；原stderr保存在`H_final_svg_programmatic/`，该helper的几何检查记为不可用。自有XML、PDF字框及视觉核验另有实际结果。
- **剖面：** 原始900点的比值有66个浮点键，按12位等价键得到61组。H逐组复算最小2ΔNLL及格数，派生CSV全部相符；支持点17个保持原值。SVG中独立识别900个网格散点、17支持散点和四组各24个模拟点；图例复制与真实数据点分开计数。
- **单位和目标：** 18张图的科学目标、图注与原生来源逐一对应。公平图保留共享减独立方向及精确参数/近似MAC的区别；迁移图保留phase1的120案例、事后身份与失败方向；可靠性图区分point/完整组/整活动及宽度；未来图保留0期、20期pilot和80期待激活计划。
- **Palette / Text / Resolution：** 连续色阶未套类别色allow-list；本阶段未执行OCR或新渲染。输入为实际矢量输出，已有400dpi PNG用于观察。

## VLM judgment (1–5 each)

实际查看F已有彩色、灰度、protanopia、deuteranopia全部联系图，并查看迁移、尾部区间和锚点差异的单图PDF渲染。其余单图由联系图、字框与来源记录共同核验；最终主文排版另验。

|Dimension|Score|Note|
|---|---:|---|
|Clarity|4|面板标题、对照方向和单位清楚，事后迁移及未来等待身份可见。|
|Hierarchy|4|定量点线是主要视觉对象，注释承担条件和支持说明。|
|Alignment|4|联系图与重点单图未见阻碍解释的遮挡；轴及色条文字的实际页面边界完整。|
|Palette coherence|4|灰度及两种色觉预览仍以线型、marker、数值和直接标签区分。|
|Journal-fit|4|统计图保持矢量，机制图可编辑，单图具备论文阅读所需的来源与限制。|

## Recommendation

**Status: ship（独立图资产）**。可进入稿件；最终PDF页面缩放、图注邻接和阅读顺序待独立验收。F原9条启发警告继续保留：8条涉及旋转轴/色条字串的水平估宽，1条涉及账本纸张的有意叠层。H的实际PDF字框与视觉检查支持这些解释。本阶段未把外部helper的exit2写成干净结果。

审计者首版曾仅按SVG `<use>`计数，将24个独立三角形path漏计；修正为带数据clip的原生marker计数后核对通过。初始失败快照保存在`H_final_figure_evidence_initial.json`，原F资产未改变。

```json
{
  "file": "F/figures (18 frozen SVG/PDF groups)",
  "type": "svg",
  "journal": "generic",
  "status": "ship",
  "findings": [
    {
      "check": "geometry",
      "severity": "info",
      "message": "External helper unavailable in project venv because lxml is missing; independent PDF word bounds and visual checks were performed.",
      "action": "none",
      "hint": "Keep helper exit2 and stderr explicit; evaluate final manuscript layout in its own PDF window."
    }
  ],
  "measurements": {
    "figures": 18,
    "assets": 54,
    "svg_image_objects": 0,
    "pdf_image_objects": 0,
    "pdf_text_outside_page": 0,
    "profile_raw_float_keys": 66,
    "profile_rounded_keys": 61,
    "profile_support_points": 17
  },
  "vlm": {
    "clarity": 4,
    "hierarchy": 4,
    "alignment": 4,
    "palette_coherence": 4,
    "journal_fit": 4
  }
}
```
