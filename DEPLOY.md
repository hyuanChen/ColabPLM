# ColabPLM — Colab 链接部署指南

本仓库本地已完整跑通，`notebooks/ColabPLM.ipynb` 内已保留全部训练日志与结果。
要拿到一个可分享的 Google Colab 链接（deliverable 之一），只需把 notebook 推到
GitHub 或 Gist 即可，不需要在 Colab 上重新训练。

下面三种方式任选其一。**推荐方式 A（GitHub repo）**，因为它最稳，且 reviewer
也能直接看到你的源码。

---

## 方式 A — 推到 GitHub 仓库（推荐）

### A.1 浏览器操作（最简单，无需命令行）
1. 打开 https://github.com/new ，新建一个 **public** repo，例如 `ColabPLM`。
2. 把本机 `/home/hongyuan/pretrained/homework/pretain/ColabPLM/` 整个目录拖到
   GitHub 上传页面（"uploading an existing file"）。注意排除大目录：
   - **不要上传** `hf_cache/`（~400 MB，模型权重，Colab 会自己下载）
   - 其他都上传，包括 `notebooks/`, `colab_plm/`, `screenshots/`, `report/`,
     `data/`, `outputs/`, `scripts/`, `README.md`, `DEPLOY.md`
3. Commit。
4. Colab 链接（把 `<USER>` 和 `<REPO>` 改成你的）：

   ```
   https://colab.research.google.com/github/<USER>/<REPO>/blob/main/notebooks/ColabPLM.ipynb
   ```

### A.2 命令行操作
本机暂时没有 `gh` CLI。先装：

```bash
sudo apt update && sudo apt install -y gh
gh auth login              # 选 GitHub.com → HTTPS → 浏览器授权
```

然后：

```bash
cd /home/hongyuan/pretrained/homework/pretain/ColabPLM

# .gitignore：排除大文件
cat > .gitignore <<'EOF'
hf_cache/
**/__pycache__/
*.pyc
EOF

git init -b main
git add .
git commit -m "ColabPLM: initial commit"
gh repo create ColabPLM --public --source=. --remote=origin --push
```

最后一行执行成功后，`gh` 会直接打印仓库 URL。Colab 链接：

```
https://colab.research.google.com/github/<你的用户名>/ColabPLM/blob/main/notebooks/ColabPLM.ipynb
```

---

## 方式 B — Gist（只想分享 notebook 单文件）

1. 打开 https://gist.github.com 。
2. Filename 写 `ColabPLM.ipynb`。
3. 把 `notebooks/ColabPLM.ipynb` 的全部内容粘贴进去（不要去掉 outputs）。
4. 选 **Create public gist**。
5. URL 形如 `https://gist.github.com/<USER>/<GIST_ID>`。Colab 链接：

   ```
   https://colab.research.google.com/gist/<USER>/<GIST_ID>/ColabPLM.ipynb
   ```

⚠️ Gist 方式只有 notebook 本体；reviewer 看不到 `colab_plm/` 源码、报告和
screenshots。如果作业要求"完整代码可访问"，请用方式 A。

---

## 方式 C — Google Drive + Colab（不需要 GitHub）

1. 把 `notebooks/ColabPLM.ipynb` 上传到 Google Drive 任意位置。
2. 在 Drive 里右键 → "用 Google Colab 打开"。
3. 在 Colab 顶栏 **Share → Anyone with the link → Viewer**，复制链接。
4. 这就是 Colab 链接。代码模块（`colab_plm/*.py`）也已经内嵌在 notebook 的
   cell 中，Colab 上 run 不需要任何额外文件。

---

## 提交清单（一次性核对）

把以下 4 项交给老师/批改系统：

| Deliverable | 文件 / 链接 |
|------|------|
| Technical Report | `report/TECHNICAL_REPORT.pdf`（或 `.md` / `.html`） |
| Screenshots（4 张）| `screenshots/training_curve.png`<br>`screenshots/confusion_matrix.png`<br>`screenshots/training_log.png`<br>`screenshots/notebook_summary.png` |
| Colab notebook 链接 | 上面方式 A/B/C 任一拿到的 URL |
| （建议附）GitHub repo | 方式 A 的仓库 URL，方便 reviewer 看完整代码 |

---

## 常见问题

**Q1. Colab 打开后没有显示已有结果？**
确认上传的是 `notebooks/ColabPLM.ipynb`（135 KB，带 outputs），不是
`ColabPLM.clean.ipynb`（22 KB，无 outputs）。

**Q2. Reviewer 想在 Colab 上重跑怎么办？**
直接点 Runtime → Run all。Cell 2 装包 ~30 秒，cell 5 拉 DeepLoc ~10 秒，
cell 6 训练 3 个 epoch ~4 分钟（T4），完整流程 5 分钟内跑完。

**Q3. 大模型怎么换？**
打开 notebook → cell 3 → 把 `MODEL_NAME` 改成
`facebook/esm2_t33_650M_UR50D` 或 `westlake-repl/SaProt_35M_AF2` 等，其他
不用动。注意 650M 在 Colab T4 上要把 `BATCH_SIZE` 降到 4。

**Q4. 提交之后链接失效？**
GitHub repo 要保持 **public**；Gist 同理；Drive notebook 的 sharing 要保持
"Anyone with the link"。

---

如果你装了 `gh` 并 `gh auth login` 之后告诉我，我可以直接帮你把 repo 推上
GitHub 并打印出最终 Colab 链接。
