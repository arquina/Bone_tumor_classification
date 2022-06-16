# Bone_tumor_classification
School project

ResNet model

```
For 5 fold cross validation (ResNet)

python -m script.__main__ --rootdir <dataset_rootdir> --df_dir <metadata_dir> --batch_size <batch_size> --epoch <epoch> --task class --lr <lr> 
--weight_decay <weight_decay> --fold <fold> --classnum 3 --cuda <cuda device number> -- result <result_dir>

For train the model(ResNet)

python -m script.__main__ --rootdir <dataset_rootdir> --df_dir <metadata_dir> --batch_size <batch_size> --epoch <epoch> --task class --lr <lr> 
--weight_decay <weight_decay> --fold 0 --classnum 3 --cuda <cuda device number> -- result <result_dir>

For test the ResNet

python -m script.__main__ --external_dir <external_dataset_rootdir> --external_data <metadata of external data> --batch_size <batch_size> --epoch <epoch> --task class
--external --classnum 3 --cuda <cuda device number> -- result <result_dir>

```

Mask-R-CNN model
```
For train the model(Mask-R-CNN)

python -m script.__main__ --rootdir <dataset_rootdir> --df_dir <metadata_dir> --batch_size <batch_size> --epoch <epoch> --task mask --classnum 3 --cuda <cuda device number> 
--custom --result <result_dir>

For train the multitask model 

python -m script.__main__ --rootdir <dataset_rootdir> --df_dir <metadata_dir> --batch-size <batch_size> --epoch <epoch> --task mask --classnum 3 --cuda <cuda device number>
--custom --fracture --subtype --modality <Select the combination> --result <result_dir>

For test the Mask-R-CNN model with external dataset

python -m script.__main__ --external_dir <external_dataset_rootdir> --external_data <metadata of external data> --batch_size <batch_size> --epoch <epoch> --task mask
--external --custom --checkpoint <checkpoint path for mask r cnn model> --classnum 3 --cuda <cuda device number> -- result <result_dir>

```

