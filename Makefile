train:
	python train.py ./flowers --gpu

test:
	python train.py ./flowers --test --gpu 

predict:
	python predict.py ./flowers/test/1/image_06743.jpg ./vgg16_checkpoint.pth --gpu
