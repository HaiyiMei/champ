source /usr/local/lib/miniconda3/etc/profile.d/conda.sh
export PYOPENGL_PLATFORM=osmesa

# Constatnt
video_base_dir=driving_videos
ref_img_dir=reference_imgs

# Read args
driving_video_path=$1
ref_img_path=$2

# judge if it's not null
if [ -z $driving_video_path ] || [ -z $ref_img_path ]; then
    echo "Please input:
    1. the driving video path
    2. reference image path"
    exit 1
fi

# Use this conda env to do SMPL Fitting, Transfer, and Rendering
conda activate /mnt/aigc/meihaiyi/envs/4D-humans

# -----  Step 1, Process Data ----- #
# Process the args & Define Constants
video_full_name=$(basename $driving_video_path)
video_name="${video_full_name%.*}"  # S01E01_11046_11185
video_dir=$video_base_dir/$video_name  # driving_videos/S01E01_11046_11185
image_path=$video_dir/images  # driving_videos/S01E01_11046_11185/images
mkdir -p $image_path
cp $driving_video_path $video_dir
driving_video_path=$video_dir/$video_full_name  # driving_videos/S01E01_11046_11185/S01E01_11046_11185.mp4

echo "--------------------------------------"
echo "Converting video to images..."
echo "--------------------------------------"
ffmpeg -i $driving_video_path -c:v png $image_path/%04d.png

echo "--------------------------------------"
echo "Fitting SMPL"
echo "--------------------------------------"
python -m scripts.data_processors.smpl.generate_smpls --reference_imgs_folder $ref_img_dir --driving_video_path $video_dir

echo "--------------------------------------"
echo "Smoothing SMPL"
echo "--------------------------------------"
/mnt/aigc/meihaiyi/programs/blender-3.6.13-linux-x64/blender --background --python scripts/data_processors/smpl/smooth_smpls.py --smpls_group_path $video_dir/smpl_results/smpls_group.npz --smoothed_result_path $video_dir/smpl_results/smpls_group_smoothed.npz

echo "--------------------------------------"
echo "Transfering SMPL"
echo "--------------------------------------"
ref_img_full_name=$(basename $ref_img_path)
ref_img_name="${ref_img_full_name%.*}"
python -m scripts.data_processors.smpl.smpl_transfer --reference_path $ref_img_dir/smpl_results/$ref_img_name.npy --driving_path $video_dir --output_folder $video_dir --figure_transfer --view_transfer
ref_img_path=$video_dir/reference_img/0001.png

echo "--------------------------------------"
echo "Rendering SMPL"
echo "--------------------------------------"
/mnt/aigc/meihaiyi/programs/blender-3.6.13-linux-x64/blender scripts/data_processors/smpl/blend/smpl_rendering.blend --background --python scripts/data_processors/smpl/render_condition_maps.py --driving_path $video_dir/smpl_results_transferred --reference_path $ref_img_path

# Render DWPose
echo "--------------------------------------"
echo "Rendering DWPose"
echo "--------------------------------------"
conda activate /mnt/aigc/meihaiyi/envs/champ
python -m scripts.data_processors.dwpose.generate_dwpose --input $video_dir/normal --output $video_dir/dwpose

# champ
# -----  Step 2, Process Data ----- #
src_yaml_path=configs/inference/inference.yaml
dst_yaml_path=$video_dir/inference.yaml
echo "import os; \
    from omegaconf import OmegaConf; \
    cfg = OmegaConf.load('$src_yaml_path'); \
    frame_len = len(os.listdir('$image_path')); \
    cfg['data'] = {'ref_image_path': '$ref_img_path', 'guidance_data_folder': '$video_dir', 'frame_range': [0, frame_len - 1]}; \
    OmegaConf.save(cfg, '$dst_yaml_path')" | python
python inference.py --config $dst_yaml_path
