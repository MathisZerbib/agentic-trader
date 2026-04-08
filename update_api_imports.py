import glob

def fix_files(files):
    for filepath in files:
        with open(filepath, "r") as f:
            content = f.read()
            
        if "from typing" not in content:
            content = "from typing import List, Optional\n" + content
        else:
            content = content.replace("from typing import List", "from typing import List, Optional")
            
        if "import os\n" not in content:
            content = "import os\n" + content
            
        if "import json\n" not in content:
            content = "import json\n" + content
            
        with open(filepath, "w") as f:
            f.write(content)

fix_files(glob.glob("backend/api/*.py"))
