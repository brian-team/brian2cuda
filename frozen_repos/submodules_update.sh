#!/bin/bash
# Initialize all submodules and update them with diff files if they exist

(
    # cd into git root directory, needed for sumbodule update --init
    cd $(git rev-parse --show-toplevel)
    for sub in brian2 brian2genn genn; do
   	submodule=frozen_repos/"$sub"
        if [ -n "$(ls -A $submodule)" ]; then
            read -p "$submodule is not empty. Do you want to delete and reinitialize it? [Y/n]" answer
            if [ $answer = "Y" ]; then
                echo "Deleting $submodule ..."
                rm -rf $submodule
                mkdir $submodule
            else
                echo "Leaving $submodule unchanged"
                continue
            fi
        fi
    
        echo "Updating $submodule ..."
        git submodule update --init $submodule
        (
            cd $submodule
            if [ -f ../$sub.diff ]; then
                echo "Applying $sub.diff ..."
                git apply -v ../$sub.diff
            else
                echo "No diff file found for $submodule"
            fi
        )
    done
)
