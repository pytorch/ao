# Releasing New Documentation Version

## idding docs for a new release

Official docs for a release are built and pushed when a final tag
is created via GitHub. When -rc tags a built, you should see a directory
named after the tag but with the -rc suffix and the prefix "v"
truncated. For example, v0.1.0-rc1 would be come just 0.1. This way
you can preview your documentation before the release.

After the final tag is created, an action will move all the docs from
`release/xx` branch to a directory named after your release version.

### Updating `stable` versions

The stable directory is a symlink to the latest released version.
On the day of the release, you need to update the symlink to the
release version. For example:

```
git checkout gh-pages
git pull
rm stable # remove the existing symlink. **Do not** edit!
ln -s 0.1 stable   # substitute the correct version number here
git add stable
git commit -m "Update stable to 0.1"
git push -u origin
```

### Adding version to dropdown

In addition to updating stable, you need to update the dropdown to include
the latest version of docs.

In `versions.txt`, add this line in the list
(substituting the correct version number here):

```
<li class="toctree-l1">
  <a class="reference internal" href="0.1/">v0.1.0 (stable)</a>
</li>
```

### Adding a <noindex> tag to old versions

You don't want your old documentation to be discoverable by search
engines. Therefore, you can run the following script to add a 
`noindex` tag to all .html files in the old version of the docs.
For exampla, when releasing 0.2, you want to add noindex tag to all
0.1 documentation. Here is the script:

```
#!/bin/bash

# Adds <meta name="robots" content="noindex"> tags to all html files in a
# directory (recursively)
#
# Usage:
# ./add_noindex_tags.sh <directory>
#
# Example (from the root directory if previous release was 0.3)
# ./scripts/add_noindex_tags.sh 0.3
if [ "$1" == "" ]; then
  echo "Incorrect usage. Correct Usage: add_no_index_tags.sh <directory>"
  exit 1
fi
find $1 -name "*.html" -print0 | xargs -0 sed -i '/<head>/a \ \ <meta name="robots" content="noindex">'
```

1. Checkout the `gh-pages` branch.
2. Create a new branch out of `gh-pages`.
3. Create a new branch out of `gh-pages`.
4. Save the above script into a file called `add_noindex_tags.sh`.
6. Run against the old documentation directory using the usage instruction in the script comments. (you may need to `chmod +x scripts/add_noindex_tags.sh`)
1. Submit a PR and merge into the `gh-pages` branch.
