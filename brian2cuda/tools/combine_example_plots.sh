# combine the same plots from different subdirectories into one image

# the base directory where the example plots are
base_dir="brian2_examples/"

# Set the direcotry to take the image names from. Will only generate images for
# png files present in this direcotry.
first_dir="cuda/default_preferences/"

cd $base_dir
echo "Generating plots in $base_dir"

dir=`find -type d -exec sh -c '[ -f "$0"/CUBA.1.png ]' '{}' \; -print`

set -- $first_dir/*
for plot in $@; do
    png_name=`basename $plot`
    png_array=( )
    for d in ${dir[@]}; do
        png_array+=( "$d/$png_name" )
    done
    out_name="combined.$png_name"
    convert +append ${png_array[@]} "$out_name"
    echo "generated $out_name"
done

file_name="plot_order.txt"
echo "The plot configurations as in the plot order are:" > $file_name
n=1
for d in ${dir[@]}; do
    echo "$n: $d" >> $file_name
    n=$((n+1))
done

cat $file_name
