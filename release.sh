#!/bin/sh

rm -r build
python setup.py build
pytest || exit 1

CURR_VER=$(python -c 'import gdstk; print(gdstk.__version__)')

if ! grep "## $CURR_VER - " CHANGELOG.md > /dev/null 2>&1; then
    echo "Version $CURR_VER not found in the release notes of CHANGELOG.md"
    exit 1
fi

if ! grep "version = \"$CURR_VER\"" pyproject.toml > /dev/null 2>&1; then
    echo "Version $CURR_VER not set in pyproject.toml"
    exit 1
fi

if ! grep "Gdstk $CURR_VER" README.md > /dev/null 2>&1; then
    echo "Version $CURR_VER not found in the benchmark in README.md"
    echo "Continue anyway [y/n]?"
    read -r GOON
    if [ "$GOON" != 'y' ] ; then
        exit 1
    fi
fi

echo "Release version $CURR_VER [y/n]?"
read -r GOON

if [ "$GOON" = 'y' ] ; then
    LAST_VER=$(git tag -l | tail -n 1)

    if [ "$LAST_VER" = "v$CURR_VER" ]; then
        echo "Version $CURR_VER (from package) already tagged. Skipping the creation of a new commit."
    else
        git commit -s -m "Release v$CURR_VER"
        git tag -am "Release v$CURR_VER" "v$CURR_VER"
    fi

    echo "Review the status and 'git push' to finish release."
fi
