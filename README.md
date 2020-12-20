## Face mask detector for UConn's Learning community Innovation Zone

## Prerequisites

All the dependencies and required libraries are included in the file <code>requirements.txt</code>  
Use the following command to install all the dependencies
```
$ pip3 install -r requirements.txt
```

## Installation
Clone the repo:
```
$ git clone https://github.com/ABraik-bit/mask-detector-LCIZ
```
Use the following command to install all the dependencies:
```
$ pip3 install -r requirements.txt
```


## Working

To re-train the ML module
```
$ python3 train_mask_detector.py --dataset dataset
```

2. To detect face masks in an image type the following command: 
```
$ python3 detect_mask_image.py --image images/pic1.jpeg
```

3. To detect face masks in real-time video streams use the following command:
```
$ python3 detect_mask_video.py 
```
## Testing
![Mask On](https://github.com/ABraik-bit/mask-detector-LCIZ/blob/main/Mask_on.JPG)
![Mask Off](https://github.com/ABraik-bit/mask-detector-LCIZ/blob/main/No_mask.JPG)
