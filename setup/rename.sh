a=1211
for file in allParams-1_ECG_VCG_{1..607}_dump.txt; do
    echo "version%04d.txt" "$a"

    # Require a 3 digit padding
    new=$(printf "version%04d.txt" "$a")

    # Change the name
    mv "$file" "$new"
    let a=a+1
done
