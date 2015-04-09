#!/usr/bin/env python
"""
python-dnn uses python (and theano) to implement major Deep Learing Networks.
It currently supports:
        CNN
        SDA
        DBN
        A general DNN Finetune kit with maxout and dropout.
"""

from setuptools import setup, find_packages,Command
import subprocess
import os 


DOCLINES = __doc__.split("\n")


CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: Apache v2.0 License
Programming Language :: C
Programming Language :: Python
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: POSIX

"""

MAJOR = 1
MINOR = 0
MICRO = 1
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)




class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info ./src/pythonDnn/version.py ./src/*.egg-info')


# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
    	out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
    	GIT_REVISION = out.strip().decode('ascii')
    except OSError:
    	GIT_REVISION = "Unknown"

    return GIT_REVISION


def get_version_info():
    # Adding the git rev number needs to be done inside
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev-' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION


def write_version_py(filename='src/pythonDnn/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version' : FULLVERSION,
                       'git_revision' : GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()

if __name__ == '__main__':
    # Rewrite the version file every time
    write_version_py()

    try:
        import theano
        requires=[]
    except ImportError:
    	requires=['theano>=0.7.0']

    metadata = dict(
        name = 'pythonDnn',
        maintainer = "Abil N George,Sudharshan GK",
        maintainer_email = "mail@abilng.in,sudharpun90@gmail.com",
        description = DOCLINES[0],
        long_description = "\n".join(DOCLINES[2:]),
        url = "https://github.com/IITM-DONLAB/python-dnn",
        download_url = "https://github.com/IITM-DONLAB/python-dnn/zipball/master",
        license = 'Apache v2.0 License',
        packages = [ 
        	'pythonDnn','pythonDnn.io_modules', 'pythonDnn.layers', 'pythonDnn.models',
        	'pythonDnn.run', 'pythonDnn.utils'],
        package_dir = {'pythonDnn': 'src/pythonDnn'},
        install_requires = requires,
        zip_safe=True,
        cmdclass={'clean': CleanCommand,},
    )
    FULLVERSION, GIT_REVISION = get_version_info()
    metadata['version'] = FULLVERSION
    setup(**metadata)
