# Stash brian2genn changes
(
    cd brian2genn
    git stash
)
git submodule update brian2genn
# Apply brian2genn.diff
(
    cd brian2genn
    git apply -v ../brian2genn.diff
)
