"""示例: 使用 fealpy.geometry.visualize.visulizer 可视化 CAD 文件."""

from __future__ import annotations

import sys
from pathlib import Path

# 直接运行时确保 src/ 在 sys.path 中
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fealpy.geometry.visualize.visulizer import show_file

# show_file("/path/to/file.step")  # STEP 文件可视化
# show_file("/path/to/file.dxf")   # DXF 文件可视化

if __name__ == "__main__":
    show_file('/home/zac/桌面/项目/三维转二维/案例1/zhou1.stp')
    show_file('/home/zac/桌面/项目/三维转二维/案例2/初始图2.dxf')
    show_file('/home/zac/桌面/项目/三维转二维/案例3/齿轮图.dxf')
    show_file('/home/zac/桌面/项目/三维转二维/案例4/二级行星轮.STEP')
    show_file('/home/zac/桌面/项目/三维转二维/案例4/三级行星轮.STEP')
