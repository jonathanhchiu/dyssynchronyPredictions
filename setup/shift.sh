a=583
for file in allParams-1_ECG_VCG_{584..607}_dump.txt; do

    echo "allParams-1_ECG_VCG_%d_dump.txt" "$a"
    # Require a 3 digit padding
    new=$(printf "allParams-1_ECG_VCG_%d_dump.txt" "$a")

    # Change the name
    mv "$file" "$new"
    let a=a+1
done
