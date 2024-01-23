# python cifar_finetune.py -a resnet --depth 20 --model checkpoints/cifar10/W4A4-mp/model_best.pth.tar --Nbits 4 --act 4 --bin --evaluate
# python cifar_finetune.py -a resnet --depth 20 --model checkpoints/cifar10/W4A4-origin-mp/checkpoint.pth.tar --Nbits 4 --act 4 --bin --evaluate
# python cifar_finetune.py -a resnet --depth 20 --model checkpoints/cifar10/W4A4-train/checkpoint.pth.tar --Nbits 8 --act 4 --bin --evaluate
python cifar_finetune.py -a resnet --depth 20 --model checkpoints/cifar10/W4A4-mp/checkpoint.pth.tar  --Nbits 4 --act 4 --evaluate --bin
