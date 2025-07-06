I used this script for training:

python pytorch_train_update.py \
  --data_volumes "vol:bloodvessel/image/x1y5z1.hdf5:data" \
  --label_volumes "validation1:bloodvessel/mask/x1y5z1.hdf5:data" \
  --coordinates_pkl "training_coords.pkl" \
  --train_dir "./pytorch_output_updated" \
  --model_args '{"fov_size": [32, 64, 64], "num_classes": 1}' \
  --learning_rate 0.001 \
  --batch_size 1 \
  --max_steps 100000 \
  --max_samples 100000

And this for inference:

python pytorch_inference_updated.py   --input_volume "bloodvessel/image/x1y5z1.hdf5:data"   --output_volume "predictions_updated/x1y5z1_output.h5:predictions"   --model_checkpoint "pytorch_output_updated/checkpoint_096000.pt"   --model_args '{"fov_size": [32, 64, 64], "num_classes": 1}'   --batch_size 4


But in the resulting segmentation I see veritcal lines running through it as though there is a stride in the chunks? or some issue that causes small sections to be missed. Could you identify why that might be?
