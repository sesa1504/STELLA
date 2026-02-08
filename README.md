# STELLA - a modular framework for StatioTemporal Event-based Lagrangian particLe trAcking
We introduce STELLA (v1.0.0), a modular framework for **s**tatio**t**emporal **e**vent-based **L**agrangian partic**l**e tr**a**cking in fluid flows. The framework is implemented as a GUI in python and takes the raw event stream obtained from an event-based camera as input. Once the data is loaded, the processing is done in four steps: Preprocessing, Detection, Tracking, Validation. In preprocessing, a ROI can be set in time and space and the filtered events can be saved. Subsequently, different algorithms for direct processing and image-based detection can be used to identify clustered events associated to individual particles. Based on the clustered events, particle tracks (position, velocity) can be derived by using a Kalman filter, spline fitting or hybrid approaches. Finally, a track quality filter and a neighborhood filter can be applied to reject spurious tracks during validation. In every step, the evaluation results can be saved and loaded in a way that also just single modules of STELLA can be used. For further information, please find our paper here: [STELLA](https://www.youtube.com/watch?v=dQw4w9WgXcQ).

If you use any of this code, please cite the following publication:
```bibtex
@article{Sachs2026STELLA,
  title={STELLA: A modular framework for SpatioTemporal Event-based Lagrangian particLe trAcking},
  author={Sachs, Sebastian and Jung, Steffen and Kahl, Max and Willert, Christ and Keuper, Margret and Cierpka, Christian},
  pages = {X},
  volume = {X},
  number = {X},
  journal = {Experiments in Fluids},
  year={2026}
}
```
# Requirements
-
# Install
1) install STELLA
clone git
navidate to main folder

`pip install -r requirements.txt`<br>

2) OpenEB
To read .raw or .dat files, please install the open source SDK [OpenEB](https://github.com/prophesee-ai/openeb) from prophesee by following the instructions given in the git. Once OpenEB is successfully installed, make sure to add the following files to your python paths:

```
[path to openeb folder]\openeb\sdk\modules\core\python\pypkg
[path to openeb folder]\openeb\build\py3\Release
```
# Datasets
example data
larger datasets
# Quick start guide
1) Preprocessing

2) Detection

3) Tracking

4) Validation
# Handling output
-
