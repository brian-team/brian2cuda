#!/bin/bash
# Initialize all submodules and update them with diff files if they exist

for submodule in brian2 brian2genn genn; do

    if [ -n "$(ls -A $submodule)" ]; then
        read -p "frozen_repos/$submodule is not empty. Do you want to delete and reinitialize it? [Y/n]" answer
        if [ $answer = "Y" ]; then
            echo "Deleting frozen_repos/$submodule ..."
            rm -rf $submodule
            mkdir $submodule
        else
            echo "Leaving frozen_repos/$submodule unchanged"
            continue
        fi
    fi

    echo "Updating frozen_repos/$submodule ..."
    git submodule update --init $submodule
    cd $submodule >/dev/null || (echo "ERROR: 'cd $submodule' failed, exiting" && exit)
    if [ -f ../$submodule.diff ]; then
        echo "Applying $submodule.diff ..."
        git apply -v ../$submodule.diff
    else
        echo "No diff file found for $submodule"
    fi
    cd - >/dev/null || (echo "ERROR: 'cd -' failed, exiting" && exit)
done
