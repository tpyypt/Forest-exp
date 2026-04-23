# Pointcept 科研 Git 工作流（实操版）

本文档用于当前仓库，目标是：

1. 安全提交代码；
2. 尝试新思路失败后可以快速、干净回退；
3. 为好结果代码打标签，方便长期复现。


## 1. 初始化与仓库清理

仓库根目录执行：

```bash
bash scripts/setup_git_workflow.sh
```

该脚本会自动完成：

1. 设置提交模板 `.gitmessage-research.txt`；
2. 配置本地 Git 常用参数（`fetch.prune`, `rerere` 等）；
3. 添加常用别名：`git st`, `git lg`。

同时，本仓库已配置科研专用 `.gitignore`，会过滤：

1. 数据目录（`data/`, `datasets/`）；
2. 模型权重（`.pth/.pt/.ckpt`）；
3. 实验输出和日志（`exp/`, `wandb/`, `tensorboard/` 等）；
4. Conda/venv 及 Python 缓存文件。


## 2. 日常提交规范（推荐）

每次实验改动建议遵循：

```bash
# 1) 看改动
git st
git diff

# 2) 只添加你确认要提交的文件
git add <file1> <file2>

# 3) 检查暂存区
git diff --staged

# 4) 提交
git commit
```

提交消息规范：`type(scope): summary`

1. `feat(model): add xxx block`
2. `fix(train): avoid oom in ...`
3. `exp(loss): tune focal gamma 2->1.5`
4. `exp(config): reduce point_max to 220k`
5. `docs: update forest quickstart`


## 3. 版本回退与对比（核心）

先查看历史：

```bash
git lg
```

### 场景 A：临时切回旧版本测试，不破坏当前开发

方式 1（最常用，detach）：

```bash
# 当前有未提交改动时先暂存
git stash -u

# 临时切到旧提交
git switch --detach <good_commit_hash>

# 跑测试/验证
# ...

# 回到原分支
git switch -

# 恢复改动
git stash pop
```

方式 2（推荐做长期对比，不污染当前目录）：

```bash
git worktree add ../pointcept-old-run <good_commit_hash>
cd ../pointcept-old-run
# 在此目录运行历史版本实验
```

### 场景 B：确认最近提交全错，彻底回到好版本

先做保险分支：

```bash
git branch backup/before-hard-reset-$(date +%Y%m%d-%H%M)
```

再硬回退：

```bash
git reset --hard <good_commit_hash>
```

如果错误提交已经 push 到远端：

```bash
git push --force-with-lease origin <your_branch>
```

说明：`--force-with-lease` 比 `--force` 更安全。


## 4. 为基线/好结果打 Tag（可复现）

当你拿到稳定指标（例如高 mIoU/mAP）：

```bash
git tag -a v1.0-baseline -m "for_instancev2 semseg, miou=xx.x, seed=3407"
git push origin v1.0-baseline
```

查看标签：

```bash
git show v1.0-baseline
```

按标签复现：

```bash
git switch --detach v1.0-baseline
```


## 5. 最小实践建议

1. 新实验先开分支：`git switch -c exp/<topic>`。
2. 每次改动保持小步提交，不要攒大提交。
3. 每个可复现好结果都打 Tag。
4. 重大风险操作（`reset --hard`）前先建 `backup/*` 分支。
