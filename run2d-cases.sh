# keep low freq (detail editing)
python run2d.py --image_path "data/stones.png" --source_prompt "A stack of stone" --target_prompt "A Buddha statue" --dwt_dds --use_dds --J 3 --num_iters 500 --gs 7.5 --keep_low --disable_wavelet

python run2d.py --image_path "data/girls2bear.png" --source_prompt "a painting of girls" --target_prompt "a painting of bears" --dwt_dds --use_dds --J 3 --num_iters 500 --gs 7.5 --keep_low

# keep high freq (color editing)
python run2d.py --image_path "data/Coffee2.png" --source_prompt "a cup of coffee" --target_prompt "a cup of matcha" --dwt_dds --use_dds --J 3 --num_iters 500 --gs 7.5 --keep_high --disable_wavelet
