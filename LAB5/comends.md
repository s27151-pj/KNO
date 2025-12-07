Dense:
python main.py --model-type dense

CNN:
python main.py --model-type cnn

Z augmentacją:
python main.py --model-type cnn --use-augmentation

Dense model:
python predict.py --model artifacts/fashion_dense.keras --image "test_samples\test1.jpg"

CNN model:
python predict.py --model artifacts/fashion_cnn.keras --image "test_samples\test1.jpg"

Create Tests jpg
python generate_test_images.py