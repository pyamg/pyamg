PyAMG is distributed under LICENSE.txt:

Copyright (c) 2008-2022 PyAMG Developers
License: MIT, http://opensource.org/licenses/MIT

Guidelines
---

We welcome bug reports and general improvements to the code.  Please see the
following for general guidelines for contributing (and refer to
[organization.md](organization.md) for details on developer roles):
  - For general questions or bugs on PyAMG, performance problems, install failures, etc., please file an [issue](https://github.com/pyamg/pyamg/issues);
  - For proposed changes to the code, please first open an issue followed by a [pull request](https://github.com/scikit-hep/awkward/pulls) that refers to a specific issue number.
  - To initiate a pull request:
    - fork the project
    - create a descriptive branch, e.g. `git checkout -b fix-issue-999`
    - initiate the pull request to `pyamg:main`
  - Branches should avoid merge commits from main; instead please rebase before issuing a PR if possible.
  - Commit messages should attempt to follow https://www.conventionalcommits.org/en/v1.0.0/
      ```
      <type>(<scope>): <description>
      fix      bug fix
      feat     new feature
      build    build system
      chore    general maintenance
      ci       ci configuration
      docs     documentation
      perf     performance improvements
      refactor refactoring code
      style    linting and code formatting
      test     add/modify testing
      ```
  - Write commit messages in [imperative mood](https://git.kernel.org/pub/scm/git/git.git/tree/Documentation/SubmittingPatches#n183)

Contributing
---

Contributing code to this project assumes that

- you implicitly transfer the copyright of your contribution to the PyAMG
  Developers or that your contributions are not significant enough to claim
  copyright

- you agree to the Developer Certificate of Origin for
  your contributing portion as described here (and below):
  http://developercertificate.org/

- there is no assumption that contributions will be retained or
  included in a release.

Certificate of Origin
---

"""
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
"""
