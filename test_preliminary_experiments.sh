# ========== GENRERATE TEST IMAGES AND RESULTS ==========

# ==== 1st experiment ====

# CycleGAN with standard configuration
python CycleGAN/test.py --dataroot /vol/biomedic3/hjr119/US_GEN_W_CUT/CUT/datasets/ct2us_hp --checkpoints_dir /vol/bitbucket/cm1320/CT2US/checkpoints/ --results_dir /vol/bitbucket/cm1320/CT2US/test/ --name CycleGAN_standard --model cycle_gan --input_nc 1 --output_nc 1

# CUT with standard configuration
python CUT/test.py --dataroot /vol/biomedic3/hjr119/US_GEN_W_CUT/CUT/datasets/ct2us_hp --checkpoints_dir /vol/bitbucket/cm1320/CT2US/checkpoints/ --results_dir /vol/bitbucket/cm1320/CT2US/test/ --name CUT_standard --model cut --CUT_mode CUT --input_nc 1 --output_nc 1

# ==== 2nd experiment ====

# CycleGAN without identity loss
python CycleGAN/test.py --dataroot /vol/biomedic3/hjr119/US_GEN_W_CUT/CUT/datasets/ct2us_hp --checkpoints_dir /vol/bitbucket/cm1320/CT2US/checkpoints/ --results_dir /vol/bitbucket/cm1320/CT2US/test/ --name CycleGAN_noIdtLoss --model cycle_gan --input_nc 1 --output_nc 1

# CUT without identity loss
python CUT/test.py --dataroot /vol/biomedic3/hjr119/US_GEN_W_CUT/CUT/datasets/ct2us_hp --checkpoints_dir /vol/bitbucket/cm1320/CT2US/checkpoints/ --results_dir /vol/bitbucket/cm1320/CT2US/test/ --name CUT_noIdtLoss --model cut --CUT_mode FastCUT --input_nc 1 --output_nc 1

# ==== 3rd experiment ====

# use LPIPS as a cycle consistency loss both for the model with and without the idt loss
python CycleGAN/test.py --dataroot /vol/biomedic3/hjr119/US_GEN_W_CUT/CUT/datasets/ct2us_hp --checkpoints_dir /vol/bitbucket/cm1320/CT2US/checkpoints/ --results_dir /vol/bitbucket/cm1320/CT2US/test/ --name CycleGAN_LPIPS --model cycle_gan --input_nc 1 --output_nc 1
python CycleGAN/test.py --dataroot /vol/biomedic3/hjr119/US_GEN_W_CUT/CUT/datasets/ct2us_hp --checkpoints_dir /vol/bitbucket/cm1320/CT2US/checkpoints/ --results_dir /vol/bitbucket/cm1320/CT2US/test/ --name CycleGAN_LPIPS_noIdtLoss --model cycle_gan --input_nc 1 --output_nc 1

# ==== 4th experiment ====

# CUT with standard configuration and removed layer 0 from PatchNCE loss
python CUT/test.py --dataroot /vol/biomedic3/hjr119/US_GEN_W_CUT/CUT/datasets/ct2us_hp --checkpoints_dir /vol/bitbucket/cm1320/CT2US/checkpoints/ --results_dir /vol/bitbucket/cm1320/CT2US/test/ --name CUT_standard_noLayer0 --model cut --CUT_mode CUT --input_nc 1 --output_nc 1
# CUT without identity loss and removed layer 0 from PatchNCE loss
python CUT/test.py --dataroot /vol/biomedic3/hjr119/US_GEN_W_CUT/CUT/datasets/ct2us_hp --checkpoints_dir /vol/bitbucket/cm1320/CT2US/checkpoints/ --results_dir /vol/bitbucket/cm1320/CT2US/test/ --name CUT_noIdtLoss_noLayer0 --model cut --CUT_mode FastCUT --input_nc 1 --output_nc 1

# ========== CALCULATE FID ==========

# cycle_gan test.py does not store files in the respective folders but stores them as <*_real_A.png> or <*_real_B.png> etc.
# pythorch_fid expects files in folders. CUT test.py already stores files in folders.
bash rearrange_files_cycle_gan.sh /vol/bitbucket/cm1320/CT2US/test/CycleGAN_standard/test_latest/images/
bash rearrange_files_cycle_gan.sh /vol/bitbucket/cm1320/CT2US/test/CycleGAN_noIdtLoss/test_latest/images/
bash rearrange_files_cycle_gan.sh /vol/bitbucket/cm1320/CT2US/test/CycleGAN_LPIPS/test_latest/images/
bash rearrange_files_cycle_gan.sh /vol/bitbucket/cm1320/CT2US/test/CycleGAN_LPIPS_noIdtLoss/test_latest/images/

# calculate FID for each model
echo -e "CycleGAN_standard:\t" >> /vol/bitbucket/cm1320/CT2US/test/FID_scores.txt
python -m pytorch_fid /vol/bitbucket/cm1320/CT2US/test/CycleGAN_standard/test_latest/images/real_B /vol/bitbucket/cm1320/CT2US/test/CycleGAN_standard/test_latest/images/fake_B >> /vol/bitbucket/cm1320/CT2US/test/FID_scores.txt 2>> /vol/bitbucket/cm1320/CT2US/test/stderr.txt
echo -e "\nCUT_standard:\t" >> /vol/bitbucket/cm1320/CT2US/test/FID_scores.txt
python -m pytorch_fid /vol/bitbucket/cm1320/CT2US/test/CUT_standard/test_latest/images/real_B /vol/bitbucket/cm1320/CT2US/test/CUT_standard/test_latest/images/fake_B >> /vol/bitbucket/cm1320/CT2US/test/FID_scores.txt 2>> /vol/bitbucket/cm1320/CT2US/test/stderr.txt
echo -e "\nCycleGAN_noIdtLoss:\t" >> /vol/bitbucket/cm1320/CT2US/test/FID_scores.txt
python -m pytorch_fid /vol/bitbucket/cm1320/CT2US/test/CycleGAN_noIdtLoss/test_latest/images/real_B /vol/bitbucket/cm1320/CT2US/test/CycleGAN_noIdtLoss/test_latest/images/fake_B >> /vol/bitbucket/cm1320/CT2US/test/FID_scores.txt 2>> /vol/bitbucket/cm1320/CT2US/test/stderr.txt
echo -e "\nCUT_noIdtLoss:\t" >> /vol/bitbucket/cm1320/CT2US/test/FID_scores.txt
python -m pytorch_fid /vol/bitbucket/cm1320/CT2US/test/CUT_noIdtLoss/test_latest/images/real_B /vol/bitbucket/cm1320/CT2US/test/CUT_noIdtLoss/test_latest/images/fake_B >> /vol/bitbucket/cm1320/CT2US/test/FID_scores.txt 2>> /vol/bitbucket/cm1320/CT2US/test/stderr.txt
echo -e "\nCycleGAN_LPIPS:\t" >> /vol/bitbucket/cm1320/CT2US/test/FID_scores.txt
python -m pytorch_fid /vol/bitbucket/cm1320/CT2US/test/CycleGAN_LPIPS/test_latest/images/real_B /vol/bitbucket/cm1320/CT2US/test/CycleGAN_LPIPS/test_latest/images/fake_B >> /vol/bitbucket/cm1320/CT2US/test/FID_scores.txt 2>> /vol/bitbucket/cm1320/CT2US/test/stderr.txt
echo -e "\nCycleGAN_LPIPS_noIdtLoss:\t" >> /vol/bitbucket/cm1320/CT2US/test/FID_scores.txt
python -m pytorch_fid /vol/bitbucket/cm1320/CT2US/test/CycleGAN_LPIPS_noIdtLoss/test_latest/images/real_B /vol/bitbucket/cm1320/CT2US/test/CycleGAN_LPIPS_noIdtLoss/test_latest/images/fake_B >> /vol/bitbucket/cm1320/CT2US/test/FID_scores.txt 2>> /vol/bitbucket/cm1320/CT2US/test/stderr.txt
echo -e "\nCUT_standard_noLayer0:\t" >> /vol/bitbucket/cm1320/CT2US/test/FID_scores.txt
python -m pytorch_fid /vol/bitbucket/cm1320/CT2US/test/CUT_standard_noLayer0/test_latest/images/real_B /vol/bitbucket/cm1320/CT2US/test/CUT_standard_noLayer0/test_latest/images/fake_B >> /vol/bitbucket/cm1320/CT2US/test/FID_scores.txt 2>> /vol/bitbucket/cm1320/CT2US/test/stderr.txt
echo -e "\nCUT_noIdtLoss_noLayer0:\t" >> /vol/bitbucket/cm1320/CT2US/test/FID_scores.txt
python -m pytorch_fid /vol/bitbucket/cm1320/CT2US/test/CUT_noIdtLoss_noLayer0/test_latest/images/real_B /vol/bitbucket/cm1320/CT2US/test/CUT_noIdtLoss_noLayer0/test_latest/images/fake_B >> /vol/bitbucket/cm1320/CT2US/test/FID_scores.txt 2>> /vol/bitbucket/cm1320/CT2US/test/stderr.txt