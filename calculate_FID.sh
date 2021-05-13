# FID on CUT simulated US images
echo "FID achieved by CUT simulating US images"
python -m pytorch_fid /vol/bitbucket/cm1320/CUT/test/experiment1/test_latest/images/real_B /vol/bitbucket/cm1320/CUT/test/experiment1/test_latest/images/fake_B/

# FID in CycleGAn simulated US images
echo "FID achieved by CycleGAN simulating US images"
python -m pytorch_fid /vol/bitbucket/cm1320/CycleGAN/test/experiment1/test_latest/images/real_B /vol/bitbucket/cm1320/CycleGAN/test/experiment1/test_latest/images/fake_B