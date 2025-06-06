# SDDObench: A Benchmark for Streaming Data-Driven Optimization with Concept Drift  

SDDObench is the first benchmark tailored for evaluating and comparing the streaming data-driven evolutionary algorithms  (SDDEAs). SDDObench comprises two sets of objective functions combined with five different types of concept drifts, which offer the benefit of being inclusive in generating data streams that mimic various real-world situations, while also facilitating straightforward description and analysis.  

## Required packages

The code has been tested running under python 3.10.13, with the following packages installed (along with their dependencies):

- numpy==1.26.0

## Quick start

Please run the 'example.py'.

## Recommended parameters setting range

Please refer to the relevant descriptions provided in the manuscript for a comprehensive understanding of the total parameters. Below are the suggested ranges of values for the user-defined parameters:

| parameter                                                    | range of values   |
| ------------------------------------------------------------ | ----------------- |
| changing Intensity for $\phi$: $\mathcal{I}_{\phi}$          | $[0,1]$           |
| the number of randomly selected points in $D_2$: $\|\varLambda\|$ | $[[0.1T],T]$      |
| offset of the noise in $D_5$                                 | $[0.1,0.5]$       |
| change rate in F2: $r_c$                                     | $[0.1,1]$         |
| recurrent period: $P$                                        | $[[0.1T],[0.5T]]$ |

Where $T$ denotes the maximum change time. 

## Citation
The PDF version of the paper is available [here](https://www.researchgate.net/publication/382249402_SDDObench_A_Benchmark_for_Streaming_Data-Driven_Optimization_with_Concept_Drift#fullTextFileContent). If you find our SDDObench has given you some help, please cite it in your publications or projects:

```latex
@inproceedings{zhong2024sddobench,
  title={SDDObench: A Benchmark for Streaming Data-Driven Optimization with Concept Drift},
  author={Zhong, Yuanting and Wang, Xincan and Sun, Yuhong and Gong, Yue-Jiao},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference},
  address = {Melbourne VIC Australia},
  pages={59--67},
  year={2024}
}
```

