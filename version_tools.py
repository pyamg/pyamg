# https://github.com/numpy/numpy/commits/master/setup.py
# Return the git revision as a string
import os
import subprocess


def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
        out = _minimal_ext_cmd(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
        GIT_BRANCH = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"
        GIT_BRANCH = ""

    return GIT_REVISION, GIT_BRANCH


def set_version_info(VERSION, ISRELEASED):
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of numpy.version messes up the build under Python 3.

    if os.path.exists('.git'):
        GIT_REVISION, GIT_BRANCH = git_version()
    elif os.path.exists('pyamg/version.py'):
        # must be a source distribution, use existing version file
        try:
            from pyamg.version import git_revision as GIT_REVISION
            from pyamg.version import git_branch as GIT_BRANCH
        except ImportError:
            raise ImportError("Unable to import git_revision. Try removing "
                              "pyamg/version.py and the build directory "
                              "before building.")
    else:
        GIT_REVISION = "Unknown"
        GIT_BRANCH = ""

    FULLVERSION = VERSION
    if not ISRELEASED:
        FULLVERSION += '.dev0' + '+' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION, GIT_BRANCH


def write_version_py(VERSION,
                     FULLVERSION,
                     GIT_REVISION,
                     GIT_BRANCH,
                     ISRELEASED,
                     filename='pyamg/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
git_branch = '%(git_branch)s'
release = %(isrelease)s
if not release:
    version = full_version
"""

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'git_branch': GIT_BRANCH,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()
