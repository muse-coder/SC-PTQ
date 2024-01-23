# python convert.py -a resnet --depth 20 --model checkpoints/cifar10/resnet-20/model_best.pth.tar  --checkpoint checkpoints/cifar10/W4A4-origin-mp --Nbits 4 --act 4 >W4A4-origin-mp.txt
# python convert.py -a resnet --depth 20 --model checkpoints/cifar10/resnet-20/model_best.pth.tar  --checkpoint checkpoints/cifar10/W4A4-SC-mp --Nbits 4 --act 4 >W4A4-SC-mp.txt
# python convert.py -a resnet --depth 20 --model checkpoints/cifar10/resnet-20/model_best.pth.tar  --checkpoint checkpoints/cifar10/W3A3-mp --Nbits 3 --act 3 >W3A3-mp.txt
# python convert.py -a resnet --depth 20 --model checkpoints/cifar10/resnet-20/model_best.pth.tar  --checkpoint checkpoints/cifar10/W8A8-mp --Nbits 8 --act 8 >W8A8-mp.txt
python convert.py -a resnet --depth 20 --model checkpoints/cifar10/resnet-20/model_best.pth.tar  --checkpoint checkpoints/cifar10/W4A4-mp --Nbits 4 --act 4 >W4A4-mp.txt
