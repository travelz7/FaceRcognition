#!"D:\PyCharm Community Edition 2019.2.3\基于PCA的人脸识别\venv\Scripts\python.exe"
# EASY-INSTALL-ENTRY-SCRIPT: 'user-agent==0.1.9','console_scripts','ua'
__requires__ = 'user-agent==0.1.9'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('user-agent==0.1.9', 'console_scripts', 'ua')()
    )
