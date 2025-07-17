import triton
import os
import tempfile
import shutil
import re

files = []

if triton.__version__[0] == '3':
    files = [
        os.path.join(triton.__path__[0], "backends/amd/driver.py"),
        os.path.join(triton.__path__[0], "backends/nvidia/driver.py"),
        os.path.join(triton.__path__[0], "backends/driver.py"),
    ]
elif triton.__version__[0:3] == '2.3':
    files = [
        os.path.join(triton.__path__[0], "runtime/driver.py"),
        os.path.join(triton.__path__[0], "common/build.py"),
    ]
else:
    raise Exception('Unsupported Triton version: {}'.format(triton.__version__))


def make_triton_compatible_with_paddle():
    if triton.__version__[0:3] == '2.3':
        link_file = os.path.join(triton.__path__[0], "tools/link.py")
        new_all_lines = []
        with open(link_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.replace(
                    "(int)sizeof({meta.orig_kernel_name}_kernels);",
                    "(int)(sizeof({meta.orig_kernel_name}_kernels) / sizeof({meta.orig_kernel_name}_kernels[0]));"
                )
                new_all_lines.append(line)
        with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8') as tmpf:
            tmpf.writelines(new_all_lines)
            tmpf_path = tmpf.name
        shutil.move(tmpf_path, link_file)

    for file in files:
        has_add_patch = False
        new_all_lines = []

        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            if "import use_triton_in_paddle as torch" in line:
                has_add_patch = True
                break

        if not has_add_patch:
            for idx, line in enumerate(lines):
                if re.match(r'^\s*import torch', line):
                    indent = re.match(r'^(\s*)', line).group(1)
                    # 替换成 try-import fallback 块
                    new_all_lines.extend([
                        indent + "try:\n",
                        indent + "    import torch\n",
                        indent + "except:\n",
                        indent + "    print(\"No module named 'torch', we will use_triton_in_paddle as torch inside triton\")\n",
                        indent + "    import use_triton_in_paddle as torch\n"
                    ])
                else:
                    new_all_lines.append(line)

            try:
                with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8') as tmpf:
                    tmpf.writelines(new_all_lines)
                    tmpf_path = tmpf.name
                shutil.move(tmpf_path, file)
            except Exception as e:
                print(f"Failed to patch {file}: {e}")
