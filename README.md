# AutoBNN

This is the implementation of [Searching for Accurate Binary Neural Architectures](http://openaccess.thecvf.com/content_ICCVW_2019/papers/NeurArch/Shen_Searching_for_Accurate_Binary_Neural_Architectures_ICCVW_2019_paper.pdf)

## Network 

The implementation of VGG_Small and ResNet18 is in vgg_small.py and resnet18.py

|  Network   | Expansion Ratio |
|  ----  | ----  |
| VGG-Auto-A | [0.5, 2, 1, 1, 1, 0.5]
| VGG-Auto-B  | [2, 2, 4, 2, 4, 0.5] |
| Res18-Auto-A | [2, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4] |
| Res18-Auto-B | [3, 4, 4, 3, 1, 4, 4, 4, 3, 3, 3, 3] |


## Citation

    @inproceedings{shen2019searching, 
        title={Searching for accurate binary neural architectures},
        author={Shen, Mingzhu and Han, Kai and Xu, Chunjing and Wang, Yunhe},
        booktitle={Proceedings of the IEEE International Conference on Computer Vision Workshops},
        pages={0--0},
        year={2019}
    }
