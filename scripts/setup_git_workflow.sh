#!/usr/bin/env bash

set -euo pipefail

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[ERROR] 当前目录不是 Git 仓库，请先进入项目根目录。"
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "[INFO] 在仓库 $REPO_ROOT 配置本地 Git 工作流..."

# Commit 模板
git config --local commit.template .gitmessage-research.txt

# 科研项目常用的安全与可读性设置
git config --local pull.rebase false
git config --local fetch.prune true
git config --local rerere.enabled true

# 常用别名（仅仓库本地生效）
git config --local alias.st "status -sb"
git config --local alias.lg "log --oneline --graph --decorate --all -n 30"
git config --local alias.unstage "restore --staged"
git config --local alias.last "log -1 --stat"

echo "[OK] Git 工作流配置完成。"
echo ""
echo "下一步建议："
echo "  1) 查看状态: git st"
echo "  2) 查看日志: git lg"
echo "  3) 提交改动: git add <files> && git commit"
echo ""
echo "中文实操文档：pointcept/docs/git_workflow_zh.md"
