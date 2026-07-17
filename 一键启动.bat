@echo off
chcp 65001 >nul
title AI足球反馈系统 - 启动器

echo ============================================
echo   AI足球反馈系统 一键启动脚本
echo ============================================
echo.
echo [1/2] 正在点火 后端视觉大脑 (api_server.py) ...
start "后端视觉大脑 - api_server.py" cmd /k "cd /d %~dp0 && python api_server.py"

echo [2/2] 正在点火 前端交互面板 (npm run dev) ...
start "前端交互面板 - npm run dev" cmd /k "cd /d %~dp0AI-Football-Web && npm run dev"

echo.
echo 两个服务窗口已弹出，请等待约 3 秒待程序就绪...
timeout /t 3 /nobreak >nul

echo.
echo 启动完成！后端与前端窗口保持独立运行，关闭本窗口不会影响它们。
echo 若要停止服务，请直接关闭对应的命令行窗口。
echo.
pause
