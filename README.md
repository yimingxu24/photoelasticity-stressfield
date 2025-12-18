# photoelasticity-stressfield
From In-Situ High-Throughput Failure Investigation to Informed Circularity Decisions

## Code structure
| *file*  |                         description                          |
| :-------: | :----------------------------------------------------------: |
|   Bg_Remove.py    |      Extract frames from videos and remove the background, keeping only the specimen.       |
|   Calibration.py    | Extract RGB birefringence from specimen images and establish the stress–RGB relationship. |
|   preprocessing.py    | Segment specimen images from photoelasticity images. |
|   stress_cie.py    | Convert the stress–RGB relationship to the stress–CIE relationship and perform data augmentation. |
|   StressField.py    | Reconstruct stress fields from specimen RGB images. |


## Dependencies

- Python 3.12.9
- numpy 1.21.6
- pandas 1.3.5
- scipy 1.7.3
- matplotlib 3.5.2
- opencv-python 4.7.0.72
