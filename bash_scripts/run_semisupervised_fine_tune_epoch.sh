 python federated_training/scripts/run_semisupervised_fine_tune_epoch.py --checkpoint local_data/checkpoints/checkpoint.pth \
  --input_img_directory local_data/images/ \
   --output_annotation_directory local_data/annotations/ \
    --label_file local_data/label_file.txt \
    --server_address 192.168.100.43 \
    --server_port 8080
