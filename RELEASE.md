# Releasing `stellarphot`

This document describes the process of releasing a new version of `stellarphot`. The release
process is automated using GitHub Actions using the workflow defined
in `.github/workflows/publish.yml`.

## Notes on the setup

1. We are publishing to PyPI using the "Trusted Publisher" setup, so no tokens need to be stored
   in the repository.
1. Uploads are done from the `publish` GitHUb Actions environment (available in the project settings).
   If we wanted to we could use this to restrict who can trigger the workflow, but for now it's open to
   everyone who can commit to the protected branch, main.
1. The version number is set when a GitHub release is made via the UI. The release should be done on
   a commit that is tagged with the version.

## Step by step

1. Make sure all issue for the milestone have been closed and that all PRs milestoned to the release
   have been merged.
1. **On your local copy of the repository**, make sure you are on the main branch.
1. Pull the latest changes from the main branch so that you are at the commit intended for the release.
1. Clean up your local copy with `git clean -fxd` to make sure you are not including any extraneous
   files.
1. Run the tests locally with `pytest --remote-data any stellarphot`. This only tests on your
   platform, but that is fine. Our testing on GitHub takes care of the rest.
1. Try building the release locally. The "blessed" way to do this is to use `pipx run build` from the
   root of the repository. This will create a wheel and a source distribution in the `dist` directory.
   Running `python -m build` will also work, and create both. Apparently `pipx` runs in an isolated
   environment.
    1. It is fine to ignore warnings here.
    2. What you build here is not what will be uploaded to PyPI. The GitHub Actions workflow will build the package in a clean environment. This step is just to catch errors.

1. Clean up the repository again with `git clean -fxd`.
1. Check [PyPI](https://pypi.org/project/stellarphot/#history) for the next release number.
   [PyPI](https://pypi.org/project/stellarphot/#history) is the authoritative source for the current version. A "yanked" released still counts as a release...you cannot use a release number that has been used before even if it was yanked.
1. Tag the commit with the version number. The version number should be formatted as `X.Y.Z`, where `X`
   is the major version number, `Y` is the minor version number, and `Z` is the patch version number.
   For an alpha release add `.alphaN` where `N` is the alpha number. For example, `1.0.0.alpha1`. For a
   beta release add `.betaN` where `N` is the beta number. For example, `1.0.0.beta1`.
1. Build the package again to make sure the version number is correct in the files it builds in
   `dist`.
1. Push the tag to the upstream GitHub repo https://github.com/feder-observatory/stellarphot.git
1. Go to https://github.com/feder-observatory/stellarphot. Pushing the tag should have a triggered
   a run of all of the CI tests. Wait until they pass.
1. Near the top of the file listing on GitHub, click the "Tags" link -- it will be next to "Branches"
   and will have a number in front of it. The tag you pushed should be at the top of the list. If it
   is not, figure out why. You cannot proceed until the tag is listed. (Probably you forgot to push
   the tag or you pushed it to the wrong repository.)
1. Click on the tag. This will take you to a page that shows the assets (tarball, Zip file)
   associated with the tag.
1. Click the "Create release from tag" button.
    1. The tag should be filled for you.
    1. The release title should be the same as the tag.
    1. Select the "Previous tag" -- the "auto" option may not work if the previous tag was a pre-release. For example, the eventual release of `2.0.0` should have a previous release of `1.4.14`, not whatever the most recent testing release was.
    1. Click the "Generate release notes" button. This will fill in the release notes with the
       commit messages since the last release.
    1. Review the release notes.
    1. **IF THIS IS A TESTING (ALPHA OR BETA) RELEASE** then check the box "Set as a pre-release" and
       uncheck the box "Set as the latest release".
    1. **OTHERWISE** just leave the boxes the way they are.
    1. Click the "Publish release" button.
1. Wait for the GitHub Actions workflow to complete. This will build the package and upload it to PyPI.
1. Check [PyPI](https://pypi.org/project/stellarphot/) for the new release. It should be there. If
it is not, figure out why.
1. Go celebrate your success! 🎉
