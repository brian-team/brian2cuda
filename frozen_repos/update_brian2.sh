# Stash brian2 changes
(
    cd brian2
    git stash
)

if [ -n "$1" ]; then
    # If commit is given as cmd argument, do checkout and recreate brian2.diff
    cd brian2
    git checkout "$1"
    git stash pop && git diff > ../brian2.diff && cd - && git add brian2 brian2.diff
else
    # If not, just update to tracked submodule
    git submodule update brian2
    # Apply brian2.diff
    cd brian2
    git apply -v ../brian2.diff && cd -
fi

