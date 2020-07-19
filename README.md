# `sushirank`
Educational implementation of pointwise and pairwise learning-to-rank models

[Slides](https://www.canva.com/design/DAECZGAOAyg/nr62T6NHenTi67-8hSVGHw/view?utm_content=DAECZGAOAyg&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink)

Repository for [Shopee x Data Science BKK Dive into Learning-to-rank ใครไม่แร้งค์ เลินนิ่งทูแร้งค์](https://www.facebook.com/events/282648886314674)
* `sandbox.ipynb` - notebook for workshop
* `sushirank/datasets.py` - Pytorch datasets for pointwise and pairwise models
* `sushirank/finetuners.py` -  pointwise and pairwise models implemented in Pytorch-lightning
* `sushirank/metrics.py` - common metrics for ranking problems

## Resources for Implementation
* [LGBMRanker](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html)
* [allRank](https://github.com/allegro/allRank)
* [tensorflow/ranking](https://github.com/tensorflow/ranking)

## Research Papers
* [RankNet](https://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf)
* [From RankNet to LambdaRank to LambdaMART: An Overview](https://www.microsoft.com/en-us/research/uploads/prod/2016/02/MSR-TR-2010-82.pdf)
