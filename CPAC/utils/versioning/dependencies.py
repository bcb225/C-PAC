# Copyright (C) 2022  C-PAC Developers

# This file is part of C-PAC.

# C-PAC is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.

# C-PAC is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
# License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with C-PAC. If not, see <https://www.gnu.org/licenses/>.
"""Gather and report dependency versions alphabetically"""
try:
    from importlib.metadata import distributions
except ModuleNotFoundError:
    from importlib_metadata import distributions
from pathlib import Path
from subprocess import PIPE, Popen, STDOUT
import sys

__all__ = ['PYTHON_PACKAGES', 'REPORTED', 'REQUIREMENTS']


def cli_version(command, dependency=None, in_result=True, delimiter=' ',
                formatting=None):
    """Collect a version from a CLI

    Parameters
    ----------
    command : str

    dependency : str, optional
        software name to report if not included in result

    in_result : boolean, optional
        parse software name from result?

    delimiter : str, optional
        if parsing software name from result, what's the delimiter?

    formatting : func, optional
        if result needs any formatting, function to do so

    Returns
    -------
    dict
        {software: version}
    """
    with Popen(command, stdout=PIPE, stderr=STDOUT, shell=True) as _command:
        _version = _command.stdout.read().decode('utf-8')
        if int(_command.poll()) == 127:  # handle missing command
            return {}
    if formatting is not None:
        _version = formatting(_version)
    if in_result:
        return dict([tuple(_version.split(delimiter, 1))])
    return {dependency: _version}


def first_line(stdout):
    """Return first line of stdout"""
    if '\n' in stdout:
        return stdout.split('\n', 1)[0]
    return stdout


def requirements() -> dict:
    """Create a dictionary from requirements.txt"""
    import CPAC
    reqs = {}
    with open(Path(CPAC.__path__[0]).parent.joinpath('requirements.txt'), 'r',
              encoding='utf8') as _req:
        for line in _req.readlines():
            for delimiter in ['==', ' @ ']:
                if delimiter in line:
                    key, value = line.split(delimiter, 1)
                    reqs[key.strip()] = value.strip()
                    continue
    return reqs


def _version_sort(_version_item):
    """Key to report by case-insensitive dependecy name"""
    return _version_item[0].lower()


PYTHON_PACKAGES = dict(sorted({
  d.name: d.version for d in list(distributions())}.items(),
  key=_version_sort))
REPORTED = dict(sorted({
  **cli_version('ldd --version', formatting=first_line),
  'Python': sys.version.replace('\n', ' ').replace('  ', ' ')
}.items(), key=_version_sort))
REQUIREMENTS = requirements()
