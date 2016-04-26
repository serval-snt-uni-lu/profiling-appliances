# profiling-appliances

This repository is to support our paper submitted to ICIT 2016:

Daoyuan Li, Tegawendé F. Bissyandé, Sylvain Kubler, Jacques Klein, and Yves Le Traon. [Profiling household appliance electricity usage with n-gram language modeling](http://orbilu.uni.lu/handle/10993/24163). In The 2016IEEE International Conference on Industrial Technology (ICIT 2016), Taipei, March 2016.IEEE.

## How the code works

* Prepare for dataset
* Convert real-valued time series into texts
* Build per-class corpus from texts
* Calculate fitness scores (using the text segmentation techniques)
* Make classification predictions based on segmentation fitness scores
